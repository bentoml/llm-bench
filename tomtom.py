from openllm_llama2_20_prompt import UserDef as BaseUserDef
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


class UserDef(BaseUserDef):
    # BASE_URL = f'http://44.197.196.96:3000/query'  # A10G
    BASE_URL = f"http://35.175.106.206:3000/query"  # A10G x4
    # BASE_URL = f'http://45.77.229.137:3000/query'  # A100

    @classmethod
    def make_request(cls):
        import random

        prompt = random.choice(cls.PROMPTS)
        headers = {"Content-Type": "text/plain"}
        url = f"{cls.BASE_URL}/query"  # A10G x4
        return url, headers, prompt

    @staticmethod
    def parse_response(chunk: bytes):
        import json

        d = json.loads(chunk)["action"]
        return tokenizer(json.dumps(d))["input_ids"][1:]


if __name__ == "__main__":
    import asyncio
    from common import start_benchmark_session

    asyncio.run(start_benchmark_session(UserDef))
