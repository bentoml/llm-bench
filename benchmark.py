import os
import argparse
import functools
import multiprocessing
from collections import defaultdict
from transformers import AutoTokenizer
from userdef import UserDef as BaseUserDef
from collections import defaultdict
from truefoundry.ml import get_client, ArtifactPath


TFY_ML_REPO = os.environ.get("TFY_ML_REPO")
RUN_NAME = 'test'

client = get_client()
run = client.create_run(ml_repo=TFY_ML_REPO, run_name=RUN_NAME)

try:
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS"))
except (TypeError, ValueError):
    MAX_TOKENS = 512

print(f"max_tokens set to {MAX_TOKENS}")

MODEL = os.environ.get("MODEL", "nousresearch-meta-llama-3-1-8b-instruct")
MODEL_HF = os.environ.get("MODEL_HF", "NousResearch/Meta-Llama-3.1-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(MODEL_HF)

default_system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

if os.environ.get("SYSTEM_PROMPT") == "1":
    system_prompt = default_system_prompt
    system_prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
    if system_prompt_file is not None:
        with open(system_prompt_file) as f:
            system_prompt = f.read().strip()
else:
    system_prompt = ""

base_url = os.environ.get("BASE_URL", "http://localhost:8000")


@functools.lru_cache(maxsize=8)
def get_prompt_set(min_input_length=0, max_input_length=500):
    """
    return a list of prompts with length between min_input_length and max_input_length
    """
    import json
    import requests
    import os

    # check if the dataset is cached
    if os.path.exists("databricks-dolly-15k.jsonl"):
        print("Loading cached dataset")
        with open("databricks-dolly-15k.jsonl", "r") as f:
            dataset = [json.loads(line) for line in f.readlines()]
    else:
        print("Downloading dataset")
        raw_dataset = requests.get(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
        )
        content = raw_dataset.content
        open("databricks-dolly-15k.jsonl", "wb").write(content)
        dataset = [json.loads(line) for line in content.decode().split("\n")]
        print("Dataset downloaded")

    for d in dataset:
        d["question"] = d["context"] + d["instruction"]
        d["input_tokens"] = len(tokenizer(d["question"])["input_ids"])
        d["output_tokens"] = len(tokenizer(d["response"]))
    system_prompt_len = len(tokenizer(system_prompt)["input_ids"])
    return [
        {"prompt": d["question"], "input_tokens": system_prompt_len + d["input_tokens"]}
        for d in dataset
        if min_input_length <= d["input_tokens"] <= max_input_length
    ]

prompts = get_prompt_set(200, 1000)


class OpenAIChatStreaming(BaseUserDef):
    BASE_URL = base_url
    PROMPTS = prompts

    @classmethod
    def make_request(cls):
        import json
        import random

        prompt = random.choice(cls.PROMPTS)
        headers = {"accept": "application/json", "content-type": "application/json"}
        url = f"{cls.BASE_URL}/v1/chat/completions"
        data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt["prompt"]},
            ],
            "max_tokens": MAX_TOKENS,
            "stream": True,
        }
        if len(MODEL) > 2:
            data['model'] = MODEL
        return url, headers, json.dumps(data), prompt

    @staticmethod
    def parse_response(chunk: bytes):
        import json

        data = chunk.decode("utf-8").strip()
        output = []
        for line in data.split("\n"):
            if line.strip():
                if len(line.split(":", 1)) == 2:
                    line = line.split(":", 1)[1].strip()
                    if line == "[DONE]":
                        continue
                    try:
                        text = json.loads(line)["choices"][0]["delta"]["content"]
                        output += tokenizer.encode(text, add_special_tokens=False)
                    except Exception as e:
                        print(line)
                        print(e)
                        continue
                else:
                    print(line)
        return output

# Global queue to communicate with the logging process
log_queue = None
log_process = None

# Function to process log entries in the background
def log_worker(queue, filename, write_to_mlrepo=True):
    """Background worker that writes log entries to the JSONL file and ML Repo"""
    log_data = defaultdict(dict)

    while True:
        # Get the log entry from the queue
        string = queue.get()

        # If we receive the 'STOP' signal, exit the loop
        if string == "STOP":
            if log_data:  # Write any remaining data before exiting
                write_to_json(log_data, filename)
            break

        split_string = string.split(":", 1)

        if len(split_string) == 2:
            key, value = split_string
            key, value = key.strip(), value.strip()  # Removing extra spaces
            value = eval(value)
            if key == "Time":
                if log_data:  # Write the previous group before resetting
                    write_to_json(log_data, filename)
                    if write_to_mlrepo:
                        run.log_metrics({
                                k: v for k,v in log_data.items() if isinstance(v, float) or isinstance(v, int)
                            }, step=int(value))
                log_data.clear()  # Clear for the new group of key-value pairs

            log_data[key] = value


def write_to_json(data, filename):
    """Write the collected key-value pairs as a row in a JOSNL file."""
    import json
    with open(filename, "a") as jsonfile:
        json.dump(data, jsonfile)
        jsonfile.write("\n")


# Function to log messages by adding them to the global queue
def logger(string):
    """Put log messages into the global queue."""
    log_queue.put(string)
    print(string)


# Start the logging process with a global queue and process
def start_logging_process(filename="log.jsonl"):
    """Start the background logging process using a global queue."""
    global log_queue, log_process
    log_queue = multiprocessing.Queue()
    log_process = multiprocessing.Process(target=log_worker, args=(log_queue, filename))
    log_process.start()


# Stop the logging process gracefully
def stop_logging_process():
    """Stop the background logging process gracefully."""
    global log_queue, log_process
    log_queue.put("STOP")  # Signal the worker to stop
    log_process.join()  # Wait for the worker to finish


if __name__ == "__main__":
    import asyncio
    from common import start_benchmark_session

    # arg parsing
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--max_users", type=int, required=True)
    parser.add_argument("--session_time", type=float, default=None)
    parser.add_argument("--ping_correction", action="store_true")
    args = parser.parse_args()
    run.log_params(vars(args))
    # Start the logging process
    start_logging_process()
    asyncio.run(start_benchmark_session(args, OpenAIChatStreaming, logger=logger))
    # Stop the logging process
    stop_logging_process()
    artifact_version = client.log_artifact(
        ml_repo=TFY_ML_REPO,
        name="results",
        artifact_paths=[
            ArtifactPath(src="log.jsonl"),
            ArtifactPath(src="final_report.pkl"),
        ],
        description="Output of llm-perf",
    )
    print(artifact_version.fqn)
    run.end()
    