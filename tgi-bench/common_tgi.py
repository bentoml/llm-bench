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
import nvidia_smi
import sys
sys.path.append('../')
from common import MetricsCollector as BaseMetricsCollector
from common import UserSpawner as BaseUserSpawner
from common import get_prompt_set

class MetricsCollector(BaseMetricsCollector):
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
                'gpu_utilization': GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0,  # Assumes one GPU,
                'gpu_memory_usage': gpu_memory_usage * 100,
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
                
                average_period = time_elapsed - 4 * time_window
                print(f"Request/s: {sum(self.response_bucket[i] for i in range(now - average_period, now)) / average_period}")
                print(f"Response Tokens/s: {sum(self.response_word_bucket[i] for i in range(now - average_period, now)) / average_period}")
                print(f"Response Latency: {np.mean(latency_bucket)}")

                # Output to CSV
                csv_file = f'data/benchmark-tgi-user={self.on_going_users}_time={session_time}.csv'
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
        fig.suptitle(f'TGI Benchmarking with {model_name}')

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
        plt.savefig(f'graph/{model_name}-benchmark-metrics-tgi-user={self.on_going_users}_time={session_time}_{math.floor(time.time())}.png')
        plt.show()

def linear_regression(x, y):
    x = tuple((i, 1) for i in x)
    y = tuple(i for i in y)
    a, b = np.linalg.lstsq(x, y, rcond=None)[0]
    return a, b


class UserSpawner(BaseUserSpawner):
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
                                    async for data, end_of_http_chunk in response.content.iter_chunks():
                                        result = self.user_def.parse_response(data)
                                        self.data_collector.collect_response_chunk(
                                            result
                                        )
                                        if not end_of_http_chunk:
                                            break
                                except Exception as e:
                                    self.data_collector.collect_response_status(str(e))
                                    raise e
                        await self.user_def.rest()
            except asyncio.CancelledError:
                pass


async def start_benchmark_session(user_def):
    # arg parsing
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--max_users", type=int, default=None)
    parser.add_argument("--session_time", type=float, default=None)
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
        user_def, ping_latency - 0.005 if args.ping_correction else 0
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


