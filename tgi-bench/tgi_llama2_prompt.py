from common import start_benchmark_session, get_tokenizer, get_prompt_set
import asyncio
import json

class UserDef:
    # BASE_URL = "http://llama27bchat-org-ss-org-1--aws-us-east-1.mt2.bentoml.ai"
    # BASE_URL= "http://llama2-7b-org-ss-org-1--aws-us-east-1.mt2.bentoml.ai"
    # BASE_URL = "http://llama2-13b-org-ss-org-1--aws-us-east-1.mt2.bentoml.ai"
    # BASE_URL = "http://184.105.5.107:3000"
    BASE_URL = "http://184.105.6.4:8080"

    @classmethod
    def ping_url(cls):
        return f"{cls.BASE_URL}"

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

    @classmethod
    def parse_response(cls, chunk):
        """
        take chunk and return list of tokens, used for token counting
        """
        response = chunk.decode("utf-8").strip()[5:]
        data = json.loads(response)
        try:
            return [data['token']['id']]
        except:
            return []

    @staticmethod
    async def rest():
        import asyncio

        await asyncio.sleep(0.01)

if __name__ == "__main__":
    asyncio.run(start_benchmark_session(UserDef))
