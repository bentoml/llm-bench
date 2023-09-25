import openllm
import re
import random
import argparse
import asyncio
import aiohttp
import json
import time
import collections
import contextlib
import math

WORD = re.compile(r'\w+')

PROMPTS = ['Increase the car temperature to 72 degrees.',
 'Turn off air conditioning.',
 'Set the temperature to 72 degrees',
 'Raise the car temperature',
 'Make the car cooler',
 'Control car temperature to 22 degrees',
 'Turn on the AC',
 'Can you adjust the temperature to 72 degrees?',
 "I'm feeling cold, increase the temperature",
 'Can you adjust the temperature to 22 degrees?',
 'Adjust the car temperature to 22 degrees?',
 'Make it warmer inside the car',
 'Increase the temperature in the car',
 'Could you please lower the temperature?',
 'Turn on the air conditioner',
 'Lower the car temperature to 22 degrees',
 'Make the car cooler',
 'Can you cool the car please?',
 'Please adjust the temperature to 23 degrees.',
 "I'm feeling cold. Can you increase the temperature?",
 'Lower the car temperature',
 'Turn on the air conditioning',
 'Turn on the air conditioner',
 'Please increase the temperature to 72 degrees',
 'Can you make the car cooler',
 'Adjust the temperature to 21 degrees',
 'Can you lower the temperature to 22 degrees?',
 "Turn on the vehicle's air conditioning",
 'Set the car temperature to 22 degrees Celsius',
 'Turn on the air conditioning',
 'Increase the car temperature',
 'Turn on the air conditioning',
 'Change the temperature in the car to 21 degrees',
 'Increase the cabin temperature to 23 degrees',
 'Set the temperature to 22 degrees',
 'Can you increase the temperature to 75 degrees?',
 'Set temperature to 72 degrees',
 "I'm feeling cold, increase the temperature",
 'Can you turn up the AC?',
 'Set climate control to 72 degrees.',
 'Can you increase the temperature to 25 degrees?',
 'Adjust the car temperature to 22 degrees',
 'I am feeling cold. Increase the temperature',
 'Set the temperature to 22 degrees',
 'Can you turn on the AC',
 'Can you set the temperature to 22 degrees',
 'Heat up the car to 72 degrees',
 'Increase the vehicle temperature',
 'Increase the car temperature',
 'Set climate control to 72 degrees',
 'Lower the temperature',
 'Turn up the heater',
 'Lower the temperature in the car',
 'Increase the car temperature',
 'Turn on the air conditioning',
 'Can you adjust the temperature to 72 degrees',
 'Please reduce the car temperature to 22 degrees.',
 'Adjust the temperature to 72 degrees',
 'Make car cooler',
 'Turn on the air conditioning',
 'Can you cool down the car to 20 degrees?',
 'Change the temperature to 22 degrees',
 'Change the temperature to 24 degrees',
 'Can I turn on the air conditioning?',
 'Can you set the temperature to 22 degrees?',
 'Set the temperature to 72 degrees',
 'Set the temperature to 22 degrees.',
 'Change climate to 72 degrees.',
 'Turn on the AC please',
 'Turn on the AC',
 'Turn up the heat',
 'Cool down the car',
 'Can you increase the temperature to 23 degrees?',
 'Set the car temperature to 21 degrees',
 'Lower the car temperature',
 'Turn on the climate control',
 'Turn up the temperature to 25 degrees',
 'Turn on the air conditioning',
 "It's too cold. Can you turn up the heat?",
 "I'm feeling hot. Can you turn on the AC?",
 'Set the temperature to 22 degrees',
 'Can you turn on the air conditioning?',
 'Set the car temperature to 22 degrees',
 'Increase the temperature to 72 degrees',
 'Set climate control to 72 degrees',
 'Could you turn up the heat?',
 'I want to set the temperature to 24 degrees',
 'Can I set the vehicle temperature to 22 degrees?',
 'Can I set the car temperature to 24 degrees?',
 'Turn the heat up in the car',
 'Increase the car temperature',
 'Heat up the car please.',
 'Can you turn up the heat?',
 'Turn on the AC and set it to 22 degrees',
 'Set the car temperature to 72 degrees',
 'Turn on the air conditioning',
 'Adjust the temperature to 72 degrees',
 'Turn the AC to 22 degrees',
 'Turn on the AC',
 'Adjust the car temperature to 22 degrees',
 'Turn off the climate control',
 'Turn the A/C to 22 degrees',
 'Set the temperature to 72 degrees',
 'Lower the cabin temperature',
 'Can you increase the temperature to 24 degrees?',
 "I'm feeling hot. Can you lower the temperature?",
 'Can you make it warmer inside the car?',
 'Please lower the temperature to 22 degrees',
 'Turn on climate control',
 'Lower the temperature to 18 degrees']


def make_request(prompt):
    headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
    config = openllm.AutoConfig.for_model("llama").model_construct_env(max_new_tokens=400, top_p=0.21).model_dump()
    data = {'prompt': prompt, 'llm_config': config, 'adapter_name': None}
    url = f'http://llama27bchat-org-ss-org-1--aws-us-east-1.mt2.bentoml.ai/v1/generate_stream'
    return url, headers, json.dumps(data)


def parse_response(chunk):
    return chunk

async def rest():
    await asyncio.sleep(1)

def generate_prompt():
    return random.choice(PROMPTS)


def tokenize(text):
    words = WORD.findall(text)
    return words


class MetricsCollector:
    def __init__(self):
        self.start_time = math.floor(time.time())
        self.word_bucket = collections.defaultdict(int)
        self.on_going_requests = 0
        self.request_bucket = collections.defaultdict(int)
        self.total_requests = 0
        self.on_going_users = 0
        self.status_bucket = collections.defaultdict(int)

    def collect_response_chunk(self, chunk: str):
        self.word_bucket[math.floor(time.time())] += len(tokenize(chunk))

    def collect_response_status(self, status):
        self.status_bucket[status] += 1

    @contextlib.contextmanager
    def collect_http_request(self):
        self.on_going_requests += 1
        yield
        self.on_going_requests -= 1
        self.request_bucket[math.floor(time.time())] += 1

    @contextlib.contextmanager
    def collect_user(self):
        self.on_going_users += 1
        yield
        self.on_going_users -= 1

    async def report_realtime_metrics(self, time_window=5):
        '''
        Each bucket is in 1s. This function will report the avg metrics in the past time_window seconds.
        '''
        while True:
            await asyncio.sleep(time_window)
            now = math.floor(time.time())
            print(f"Time: {now - self.start_time}")
            print(f"Requests/s: {sum(self.request_bucket[i] for i in range(now - time_window, now)) / time_window}")
            print(f"Words/s: {sum(self.word_bucket[i] for i in range(now - time_window, now)) / time_window}")
            print(f"Ongoing requests: {self.on_going_requests}")
            print(f"Ongoing users: {self.on_going_users}")
            print(f"Total requests: {self.total_requests}")
            print(f"Status: {self.status_bucket}")
            print()


async def start_user(generate_prompt, rest, collector: MetricsCollector):
    with collector.collect_user():
        cookie_jar = aiohttp.DummyCookieJar()
        async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
            while True:
                prompt = generate_prompt()
                url, headers, data = make_request(prompt)
                collector.total_requests += 1
                with collector.collect_http_request():
                    async with session.post(url, headers=headers, data=data) as response:
                        collector.collect_response_status(response.status)
                        try:
                            if response.status != 200:
                                continue
                            async for data, end_of_http_chunk in response.content.iter_chunks():
                                result = parse_response(data)
                                collector.collect_response_chunk(result.decode('utf-8'))
                                if not end_of_http_chunk:
                                    break
                        except Exception as e:
                            collector.collect_response_status(str(e))
                            continue
                await rest()


async def start_benchmark_session(max_users=2, spawn_interval=0.2, session_time=60):
    collector = MetricsCollector()
    asyncio.create_task(collector.report_realtime_metrics())
    user_list = []
    for _ in range(max_users):
        user_list.append(asyncio.create_task(start_user(generate_prompt, rest, collector)))
        await asyncio.sleep(spawn_interval)
    await asyncio.sleep(session_time)
    for user in user_list:
        user.cancel()
    return 0


async def main(args) -> int:
    '''
    start a benchmark session with default parameters if not specified.
    '''
    return await start_benchmark_session(args.max_users, args.spawn_interval, args.session_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark llama")
    parser.add_argument('--max_users', type=int, default=2)
    parser.add_argument('--spawn_interval', type=float, default=0.1)
    parser.add_argument('--session_time', type=float, default=60)
    args = parser.parse_args()
    asyncio.run(main(args))
