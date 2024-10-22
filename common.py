from asyncio.tasks import Task
import argparse
import asyncio
import numpy as np
import aiohttp
import pickle
import time
import collections
import contextlib
import math


def print_aggregations(d, logger):
    end_time = max([t for u, t in d["response_bucket"]])
    total_time = end_time - d["start_time"]
    print("total_time_seconds:", total_time)
    print("total_users:", len(set([u for u, t in d["response_bucket"]])))
    print(f"current_users: {d['on_going_users']}")
    # Calculate total requests
    total_requests = sum(d["response_bucket"].values())

    # Calculate average requests per second
    avg_requests_per_second = total_requests / total_time if total_time > 0 else 0

    # Prepare data for quantile calculations
    head_latencies = [
        lat for lats in d["response_head_latency_bucket"].values() for lat in lats if lat > 0
    ]
    response_times = [
        lat for lats in d["response_latency_bucket"].values() for lat in lats if lat > 0
    ]

    # Calculate total tokens
    total_output_tokens = sum(d["response_word_bucket"].values())
    total_input_tokens = sum(d["input_word_bucket"].values())

    # Calculate tokens per second per user
    output_tokens_per_second_per_user = collections.defaultdict(list)
    input_tokens_per_second_per_user = collections.defaultdict(list)

    for (user, time), tokens in d["response_word_bucket"].items():
        user_response_time = sum(d["response_latency_bucket"][(user, time)])
        if user_response_time > 0 and tokens:
            output_tokens_per_second_per_user[user] += [tokens / user_response_time]

    for (user, time), tokens in d["input_word_bucket"].items():
        user_head_latency = sum(d["response_head_latency_bucket"][(user, time)])
        if user_head_latency > 0 and tokens:
            input_tokens_per_second_per_user[user] += [tokens / user_head_latency]
    
    avg_input_tokens_per_second_per_user = {}
    for user in input_tokens_per_second_per_user:
        avg_input_tokens_per_second_per_user[user] = sum(input_tokens_per_second_per_user[user])/len(input_tokens_per_second_per_user[user])

    avg_output_tokens_per_second_per_user = {}
    for user in output_tokens_per_second_per_user:
        avg_output_tokens_per_second_per_user[user] = sum(output_tokens_per_second_per_user[user])/len(output_tokens_per_second_per_user[user])

    # Calculate total tokens per second (for all users)
    total_output_tokens_per_second = total_output_tokens / total_time if total_time > 0 else 0
    total_input_tokens_per_second = sum(avg_input_tokens_per_second_per_user.values())

    # Calculate input/output tokens per request (user)
    input_tokens_per_request = collections.defaultdict(list)
    output_tokens_per_request = collections.defaultdict(list)

    for (user, time), tokens in d["input_word_bucket"].items():
        if tokens:
            input_tokens_per_request[user].append(tokens)

    for (user, time), tokens in d["response_word_bucket"].items():
        if tokens:
            output_tokens_per_request[user].append(tokens)

    input_tokens_per_request_flat = [
        token
        for user_tokens in input_tokens_per_request.values()
        for token in user_tokens
    ]
    output_tokens_per_request_flat = [
        token
        for user_tokens in output_tokens_per_request.values()
        for token in user_tokens
    ]

    # Define quantiles
    quantiles = [0, 50, 90, 95, 99, 100]

    # Print aggregations
    logger(f"total_requests: {total_requests}")
    logger(f"average_requests_per_second: {avg_requests_per_second:.2f}")

    for q in quantiles:
        value = np.percentile(head_latencies, q)
        logger(f"response_time_first_token_second_quantile_{q}: {value:.2f}")

    for q in quantiles:
        value = np.percentile(response_times, q)
        logger(f"response_time_second_quantile_{q}: {value:.2f}")

    for q in quantiles:
        value = np.percentile(list(avg_output_tokens_per_second_per_user.values()), q)
        logger(f"output_tokens_per_second_per_user_quantile_{q}: {value:.2f}")

    for q in quantiles:
        value = np.percentile(list(avg_input_tokens_per_second_per_user.values()), q)
        logger(f"input_tokens_per_second_per_user_quantile_{q}: {value:.2f}")

    logger(f"total_output_tokens_per_second_all_users: {total_output_tokens_per_second:.2f}")
    logger(f"total_input_tokens_per_second_all_users: {total_input_tokens_per_second:.2f}")

    for q in quantiles:
        value = np.percentile(input_tokens_per_request_flat, q)
        logger(f"input_tokens_per_request_quantile_{q}: {value:.2f}")

    for q in quantiles:
        value = np.percentile(output_tokens_per_request_flat, q)
        logger(f"output_tokens_per_request_quantile_{q}: {value:.2f}")

    logger(f"total_output_tokens: {total_output_tokens}")
    logger(f"total_input_tokens: {total_input_tokens}")


class MetricsCollector:
    def __init__(self, user_def, logging_function, session_time=None, ping_latency=0.0):
        self.start_time = math.floor(time.time())
        self.input_word_bucket = collections.defaultdict(int)
        self.response_word_bucket = collections.defaultdict(int)
        self.response_head_latency_bucket = collections.defaultdict(list)
        self.response_latency_bucket = collections.defaultdict(list)
        self.requests_through_time_bucket = collections.defaultdict(int)
        self.on_going_requests = 0
        self.response_bucket = collections.defaultdict(int)
        self.total_requests = 0
        self.on_going_users = 0
        self.status_bucket = collections.defaultdict(int)
        self.user_def = user_def
        self.max_users = 0
        self.session_time = session_time
        self.ping_latency = ping_latency
        self.logging_function = logging_function
        print("Logger: ", self.logging_function)

    def collect_response_chunk(
        self, chunk: list, input_tokens: int, user_id: int, time_key: int
    ):
        self.response_word_bucket[user_id, time_key] += len(chunk)
        self.input_word_bucket[user_id, time_key] += input_tokens
        self.requests_through_time_bucket[user_id, time_key] += 1
        self.max_users = max(self.max_users, user_id)

    def collect_response_status(self, status):
        self.status_bucket[status] += 1

    def collect_response_head_latency(self, latency, user_id, time_key):
        self.response_head_latency_bucket[user_id, time_key] += [
            latency - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_http_request(self, user_id, time_key):
        start_time = time.time()
        self.on_going_requests += 1
        yield
        self.on_going_requests -= 1
        self.response_bucket[user_id, time_key] += 1
        self.response_latency_bucket[user_id, time_key] += [
            time.time() - start_time - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_user(self):
        self.on_going_users += 1
        yield
        self.on_going_users -= 1

    async def report_loop(self, report_time_window=30):
        """
        Each bucket is in 1s. This function will report the avg metrics in the past time_window seconds.
        """
        while True:
            await asyncio.sleep(report_time_window)
            now = math.floor(time.time())
            # Log the metrics
            self.logging_function(f"Time: {now - self.start_time}")
            print_aggregations({
                "start_time": self.start_time,
                "input_word_bucket": self.input_word_bucket,
                "response_word_bucket": self.response_word_bucket,
                "response_head_latency_bucket": self.response_head_latency_bucket,
                "response_latency_bucket": self.response_latency_bucket,
                "requests_through_time_bucket": self.requests_through_time_bucket,
                "response_bucket": self.response_bucket,
                "total_requests": self.total_requests,
                "status_bucket": self.status_bucket,
                "max_users": self.max_users,
                "ping_latency": self.ping_latency,
                "on_going_users": self.on_going_users
            }, self.logging_function)
            self.logging_function("")

            if self.session_time and (now - self.start_time) >= self.session_time:
                break

    def report_final(self):
        data_to_pickle = {
            "start_time": self.start_time,
            "input_word_bucket": self.input_word_bucket,
            "response_word_bucket": self.response_word_bucket,
            "response_head_latency_bucket": self.response_head_latency_bucket,
            "response_latency_bucket": self.response_latency_bucket,
            "requests_through_time_bucket": self.requests_through_time_bucket,
            "response_bucket": self.response_bucket,
            "total_requests": self.total_requests,
            "status_bucket": self.status_bucket,
            "max_users": self.max_users,
            "ping_latency": self.ping_latency,
            "on_going_users": self.on_going_users
        }
        with open("final_report.pkl", "wb") as f:
            pickle.dump(data_to_pickle, f)

        print("=================== Final Report ====================")
        print_aggregations(data_to_pickle, self.logging_function)


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
        self.current_id = -1
        self.user_list: list[Task] = []

    async def sync(self):
        while True:
            if self.current_user_count == self.target_user_count:
                return
            await asyncio.sleep(0.1)

    @property
    def current_user_count(self):
        return len(self.user_list)

    async def user_loop(self, user_id=0):
        with self.data_collector.collect_user():
            cookie_jar = aiohttp.DummyCookieJar()
            try:
                async with aiohttp.ClientSession(cookie_jar=cookie_jar) as session:
                    while True:
                        url, headers, data, input_data = self.user_def.make_request()
                        time_key = math.floor(time.time())
                        with self.data_collector.collect_http_request(
                            user_id, time_key
                        ):
                            req_start = time.time()
                            async with session.post(
                                url,
                                headers=headers,
                                data=data,
                            ) as response:
                                try:
                                    if response.status != 200:
                                        continue

                                    first = True
                                    result = []
                                    async for data, end_of_http_chunk in response.content.iter_chunks():
                                        output = self.user_def.parse_response(data)
                                        result += output
                                        if first and output:
                                            first = False
                                            self.data_collector.collect_response_head_latency(
                                                time.time() - req_start,
                                                user_id,
                                                time_key,
                                            )
                                        if not end_of_http_chunk:
                                            break
                                    self.data_collector.collect_response_chunk(
                                        result,
                                        input_data["input_tokens"],
                                        user_id,
                                        time_key,
                                    )
                                except Exception as e:
                                    self.data_collector.collect_response_status(str(e))
                                    raise e

                        await self.user_def.rest()
                        self.data_collector.total_requests += 1
                        self.data_collector.collect_response_status(response.status)
            except asyncio.CancelledError:
                pass

    def spawn_user(self):
        user_id = self.current_id + 1
        self.user_list.append(asyncio.create_task(self.user_loop(user_id)))
        self.current_id = self.current_id + 1

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


async def start_benchmark_session(args, user_def, logger=print):
    # ping server
    response_times = []
    async with aiohttp.ClientSession() as session:
        async with session.get(user_def.ping_url()) as response:
            print(response.status)
            assert response.status in [200, 404]
        await asyncio.sleep(0.3)

        for _ in range(5):
            time_start = time.time()
            async with session.get(user_def.ping_url()) as response:
                assert response.status in [200, 404]
            response_times.append(time.time() - time_start)
            await asyncio.sleep(0.3)
    ping_latency = sum(response_times) / len(response_times)
    print(f"Ping latency: {ping_latency}. ping correction: {args.ping_correction}")
    # init
    collector = MetricsCollector(
        user_def, logger, args.session_time, ping_latency if args.ping_correction else 0
    )
    user_spawner = UserSpawner(
        user_def, collector, args.max_users, target_time=time.time() + args.ramp_up_time
    )
    asyncio.create_task(user_spawner.spawner_loop())
    asyncio.create_task(collector.report_loop())
    if args.max_users is None:
        asyncio.create_task(user_spawner.aimd_loop())

    if args.session_time is not None:
        await asyncio.sleep(args.session_time + 1)
    else:
        await asyncio.wait(user_spawner.user_list)
    collector.logging_function((f"Time: {1 +  math.floor(time.time()) - collector.start_time}"))
    collector.report_final()
    await user_spawner.cancel_all_users()
    return 0
