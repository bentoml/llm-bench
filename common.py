from asyncio.tasks import Task
import argparse
import asyncio
import numpy as np
import aiohttp
import time
import collections
import contextlib
import math
import functools
import csv
import os
import matplotlib.pyplot as plt
import psutil  # For CPU utilization
import GPUtil  # For GPU utilization
import nvidia_smi # pip install nvidia-ml-py3

class MetricsCollector:
    def __init__(self, user_def, model_name, ping_latency=0.0):
        self.start_time = math.floor(time.time())
        self.response_word_bucket = collections.defaultdict(int)
        self.response_head_latency_bucket = collections.defaultdict(list)
        self.response_latency_bucket = collections.defaultdict(list)
        self.on_going_requests = 0
        self.response_bucket = collections.defaultdict(int)
        self.total_requests = 0
        self.on_going_users = 0
        self.status_bucket = collections.defaultdict(int)
        self.user_def = user_def
        self.ping_latency = ping_latency
        self.model_name = model_name

    def collect_response_chunk(self, chunk: list):
        self.response_word_bucket[math.floor(time.time())] += len(chunk)

    def collect_response_status(self, status):
        self.status_bucket[status] += 1

    def collect_response_head_latency(self, latency):
        self.response_head_latency_bucket[math.floor(time.time())] += [
            latency - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_http_request(self):
        start_time = time.time()
        self.on_going_requests += 1
        yield
        self.on_going_requests -= 1
        self.response_bucket[math.floor(time.time())] += 1
        self.response_latency_bucket[math.floor(time.time())] += [
            time.time() - start_time - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_user(self):
        self.on_going_users += 1
        yield
        self.on_going_users -= 1

    async def report_loop(self, session_time, time_window=5):
        """
        Each bucket is in 1s. This function will report the avg metrics in the past time_window seconds.
        """
        metrics = []
        
        while True:
            await asyncio.sleep(time_window)
            now = math.floor(time.time())
            time_elapsed = now - self.start_time
            
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_usage = info.used/info.total
            nvidia_smi.nvmlShutdown()
            
            current_metrics = {
                'time_elapsed': time_elapsed,
                'active_users': self.on_going_users,
                'requests_per_second': sum(self.response_bucket[i] for i in range(now - time_window, now)) / time_window,
                'total_requests': self.total_requests,
                'active_requests': self.on_going_requests,
                'cpu_utilization': psutil.cpu_percent(),
                'gpu_utilization': GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0,  # Assumes one GPU
                'gpu_memory_usage': gpu_memory_usage,
            }
            
            head_latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_head_latency_bucket[i]
            ]
            current_metrics['response_head_latency'] = np.mean(head_latency_bucket) if head_latency_bucket else 0
            
            latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_latency_bucket[i]
            ]
            
            current_metrics['response_latency'] = np.mean(latency_bucket) if latency_bucket else 0
                
            current_metrics['response_tokens_per_second'] = sum(self.response_word_bucket[i] for i in range(now - time_window, now)) / time_window

            # Record metrics
            metrics.append(current_metrics)

            # Print current metrics
            for key, value in current_metrics.items():
                print(f"{key}: {value}")
            print(f"status: {self.status_bucket}")
            print()
            if (session_time - time_elapsed) <= time_window:
                print("---------- Final Result ----------")
                
                average_period = time_elapsed - 60 # Ignore the first 60 seconds
                print(f"Request/s: {sum(self.response_bucket[i] for i in range(now - average_period, now)) / average_period}")
                print(f"Response Tokens/s: {sum(self.response_word_bucket[i] for i in range(now - average_period, now)) / average_period}")
                print(f"Response Latency: {np.mean(latency_bucket)}")

                # Output to CSV
                csv_file = f'data/{self.model_name}-benchmark-openllm-user={self.on_going_users}_time={session_time}_utc={math.floor(time.time())}.csv'
                with open(csv_file, 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=metrics[0].keys())
                    writer.writeheader()
                    for data in metrics:
                        writer.writerow(data)

                # Plotting
                self.plot_metrics(metrics, session_time, self.model_name)
    
    def plot_metrics(self, metrics, session_time, model_name):
        # Prepare the data for plotting
        times = [m['time_elapsed'] for m in metrics]
        total_requests = [m['total_requests'] for m in metrics]
        requests_per_second = [m['requests_per_second'] for m in metrics]
        response_latency = [m['response_latency'] for m in metrics if 'response_latency' in m]
        response_tokens_per_second = [m['response_tokens_per_second'] for m in metrics]
        gpu_utilization = [m['gpu_utilization'] for m in metrics]
        cpu_utilization = [m['cpu_utilization'] for m in metrics]
        gpu_memory_usage = [m['gpu_memory_usage'] for m in metrics]

        # Create a 3x2 grid of subplots
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'OpenLLM Benchmarking with {model_name}')

        # Plot each metric in its subplot
        axs[0, 0].plot(times, requests_per_second, label='Requests/s')
        axs[0, 0].set_title('Requests per Second')
        axs[0, 0].set_xlabel('Time Elapsed (s)')
        axs[0, 0].set_ylabel('Requests/s')

        axs[0, 1].plot(times, response_latency, label='Response Latency')
        axs[0, 1].set_title('Response Latency')
        axs[0, 1].set_xlabel('Time Elapsed (s)')
        axs[0, 1].set_ylabel('Latency')

        axs[0, 2].plot(times, response_tokens_per_second, label='Response Tokens/s')
        axs[0, 2].set_title('Response Tokens per Second')
        axs[0, 2].set_xlabel('Time Elapsed (s)')
        axs[0, 2].set_ylabel('Tokens/s')

        axs[1, 0].plot(times, gpu_utilization, label='GPU Utilization')
        axs[1, 0].set_title('GPU Utilization')
        axs[1, 0].set_xlabel('Time Elapsed (s)')
        axs[1, 0].set_ylabel('Utilization (%)')

        axs[1, 1].plot(times, cpu_utilization, label='CPU Utilization')
        axs[1, 1].set_title('CPU Utilization')
        axs[1, 1].set_xlabel('Time Elapsed (s)')
        axs[1, 1].set_ylabel('Utilization (%)')

        axs[1, 2].plot(times, gpu_memory_usage, label='GPU Memory Usage')
        axs[1, 2].set_title('GPU Memory Usage')
        axs[1, 2].set_xlabel('Time Elapsed (s)')
        axs[1, 2].set_ylabel('Utilization (%)')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'graph/{model_name}-benchmark-metrics-openllm-user={self.on_going_users}_time={session_time}_utc={math.floor(time.time())}.png')
        plt.show()
        
def linear_regression(x, y):
    x = tuple((i, 1) for i in x)
    y = tuple(i for i in y)
    a, b = np.linalg.lstsq(x, y, rcond=None)[0]
    return a, b


class UserSpawner:
    def __init__(
        self,
        user_def,
        collector: MetricsCollector,
        target_user_count=None,
        target_time=None,
    ):
        self.target_user_count = 1 if target_user_count is None else target_user_count
        self.target_time = time.time() + 10 if target_time is None else target_time

        self.data_collector = collector
        self.user_def = user_def

        self.user_list: list[Task] = []

    async def sync(self):
        while True:
            if self.current_user_count == self.target_user_count:
                return
            await asyncio.sleep(0.1)

    @property
    def current_user_count(self):
        return len(self.user_list)

    async def user_loop(self):
        with self.data_collector.collect_user():
            cookie_jar = aiohttp.DummyCookieJar()
            try:
                async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
                    while True:
                        url, headers, data = self.user_def.make_request()
                        self.data_collector.total_requests += 1
                        with self.data_collector.collect_http_request():
                            req_start = time.time()
                            async with session.post(
                                url,
                                headers=headers,
                                data=data,
                            ) as response:
                                self.data_collector.collect_response_status(
                                    response.status
                                )
                                self.data_collector.collect_response_head_latency(
                                    time.time() - req_start
                                )
                                try:
                                    if response.status != 200:
                                        continue
                                    async for data in response.content.iter_chunked(128000):
                                        result = self.user_def.parse_response(data)
                                        if not result: break
                                        self.data_collector.collect_response_chunk(
                                            result
                                        )
                                except Exception as e:
                                    self.data_collector.collect_response_status(str(e))
                                    raise e
                        await self.user_def.rest()
            except asyncio.CancelledError:
                pass

    def spawn_user(self):
        self.user_list.append(asyncio.create_task(self.user_loop()))

    async def cancel_all_users(self):
        try:
            user = self.user_list.pop()
            user.cancel()
        except IndexError:
            pass
        await asyncio.sleep(0)

    async def spawner_loop(self):
        while True:
            current_users = len(self.user_list)
            if current_users == self.target_user_count:
                await asyncio.sleep(0.1)
            elif current_users < self.target_user_count:
                self.spawn_user()
                sleep_time = max(
                    (self.target_time - time.time())
                    / (self.target_user_count - current_users),
                    0,
                )
                await asyncio.sleep(sleep_time)
            elif current_users > self.target_user_count:
                self.user_list.pop().cancel()
                sleep_time = max(
                    (time.time() - self.target_time)
                    / (current_users - self.target_user_count),
                    0,
                )
                await asyncio.sleep(sleep_time)

    async def aimd_loop(
        self,
        adjust_interval=5,
        sampling_interval=5,
        ss_delta=1,
    ):
        """
        Detect a suitable number of users to maximize the words/s.
        """
        while True:
            while True:
                # slow start
                now = math.floor(time.time())
                words_per_seconds = [
                    self.data_collector.response_word_bucket[i]
                    for i in range(now - sampling_interval, now)
                ]
                slope = linear_regression(
                    range(len(words_per_seconds)), words_per_seconds
                )[0]
                if slope >= -0.01:
                    # throughput is increasing
                    cwnd = self.current_user_count
                    target_cwnd = max(int(cwnd * (1 + ss_delta)), cwnd + 1)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    print(f"SS: {cwnd} -> {target_cwnd}")
                    await asyncio.sleep(adjust_interval)
                else:
                    # throughput is decreasing, stop slow start
                    cwnd = self.current_user_count
                    target_cwnd = math.ceil(cwnd * 0.5)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    print(f"SS Ended: {target_cwnd}")
                    break

            await self.sync()
            await asyncio.sleep(min(adjust_interval, sampling_interval, 10))
            return 0


async def start_benchmark_session(user_def):
    # arg parsing
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--session_time", type=float, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--ping_correction", action="store_true")
    args = parser.parse_args()

    # ping server
    response_times = []
    async with aiohttp.ClientSession() as session:
        async with session.get(user_def.ping_url()) as response:
            assert response.status == 200
        await asyncio.sleep(0.3)

        for _ in range(5):
            time_start = time.time()
            async with session.get(user_def.ping_url()) as response:
                assert response.status == 200
            response_times.append(time.time() - time_start)
            await asyncio.sleep(0.3)
    ping_latency = sum(response_times) / len(response_times)
    print(f"Ping latency: {ping_latency}. ping correction: {args.ping_correction}")

    # init
    collector = MetricsCollector(
        user_def, args.name, ping_latency - 0.005 if args.ping_correction else 0
    )
    user_spawner = UserSpawner(
        user_def, collector, args.max_users, target_time=time.time() + 20
    )
    asyncio.create_task(user_spawner.spawner_loop())
    asyncio.create_task(collector.report_loop(args.session_time))
    if args.max_users is None:
        asyncio.create_task(user_spawner.aimd_loop())

    if args.session_time is not None:
        await asyncio.sleep(args.session_time)
    else:
        await asyncio.wait(user_spawner.user_list)

    await user_spawner.cancel_all_users()
    return 0


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def _tokenizer(text):
        return tokenizer(text)["input_ids"][1:]

    return _tokenizer


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

    tokenizer = get_tokenizer()
    for d in dataset:
        d["input_tokens"] = len(tokenizer(d["instruction"]))
        d["output_tokens"] = len(tokenizer(d["response"]))
    return [
        d["instruction"]
        for d in dataset
        if min_input_length <= d["input_tokens"] <= max_input_length
    ]
