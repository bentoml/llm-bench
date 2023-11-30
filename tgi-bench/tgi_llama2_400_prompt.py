import asyncio
from tgi_llama2_20_prompt import UserDef as BaseUserDef
from common_tgi import get_prompt_set, start_benchmark_session


class UserDef(BaseUserDef):
    @classmethod
    def make_request(cls):
        """
        return url, headers, body
        """
        import json
        import random

        prompt = random.choice(get_prompt_set(80, 120))

        headers = {"accept": "application/json", "Content-Type": "application/json"}
        data = {
            "inputs": prompt, 
            "parameters": {"max_new_tokens": 100, "top_p":0.21}
        }
        url = f"{cls.BASE_URL}/generate_stream"
        return url, headers, json.dumps(data)

if __name__ == "__main__":
    asyncio.run(start_benchmark_session(UserDef))
