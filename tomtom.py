from openllm_llama2_20_prompt import UserDef as BaseUserDef
from transformers import LlamaTokenizer


tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


class UserDef(BaseUserDef):
    # BASE_URL = f'http://44.197.196.96:3000/query'  # A10G
    BASE_URL = f"http://35.175.106.206:3000/query"  # A10G x4
    # BASE_URL = f'http://45.77.229.137:3000/query'  # A100
    PROMPTS = [
        "Increase the car temperature to 72 degrees.",
        "Turn off air conditioning.",
        "Set the temperature to 72 degrees",
        "Raise the car temperature",
        "Make the car cooler",
        "Control car temperature to 22 degrees",
        "Turn on the AC",
        "Can you adjust the temperature to 72 degrees?",
        "I'm feeling cold, increase the temperature",
        "Can you adjust the temperature to 22 degrees?",
        "Adjust the car temperature to 22 degrees?",
        "Make it warmer inside the car",
        "Increase the temperature in the car",
        "Could you please lower the temperature?",
        "Turn on the air conditioner",
        "Lower the car temperature to 22 degrees",
        "Make the car cooler",
        "Can you cool the car please?",
        "Please adjust the temperature to 23 degrees.",
        "I'm feeling cold. Can you increase the temperature?",
        "Lower the car temperature",
        "Turn on the air conditioning",
        "Turn on the air conditioner",
        "Please increase the temperature to 72 degrees",
        "Can you make the car cooler",
        "Adjust the temperature to 21 degrees",
        "Can you lower the temperature to 22 degrees?",
        "Turn on the vehicle's air conditioning",
        "Set the car temperature to 22 degrees Celsius",
        "Turn on the air conditioning",
        "Increase the car temperature",
        "Turn on the air conditioning",
        "Change the temperature in the car to 21 degrees",
        "Increase the cabin temperature to 23 degrees",
        "Set the temperature to 22 degrees",
        "Can you increase the temperature to 75 degrees?",
        "Set temperature to 72 degrees",
        "I'm feeling cold, increase the temperature",
        "Can you turn up the AC?",
        "Set climate control to 72 degrees.",
        "Can you increase the temperature to 25 degrees?",
        "Adjust the car temperature to 22 degrees",
        "I am feeling cold. Increase the temperature",
        "Set the temperature to 22 degrees",
        "Can you turn on the AC",
        "Can you set the temperature to 22 degrees",
        "Heat up the car to 72 degrees",
        "Increase the vehicle temperature",
        "Increase the car temperature",
        "Set climate control to 72 degrees",
        "Lower the temperature",
        "Turn up the heater",
        "Lower the temperature in the car",
        "Increase the car temperature",
        "Turn on the air conditioning",
        "Can you adjust the temperature to 72 degrees",
        "Please reduce the car temperature to 22 degrees.",
        "Adjust the temperature to 72 degrees",
        "Make car cooler",
        "Turn on the air conditioning",
        "Can you cool down the car to 20 degrees?",
        "Change the temperature to 22 degrees",
        "Change the temperature to 24 degrees",
        "Can I turn on the air conditioning?",
        "Can you set the temperature to 22 degrees?",
        "Set the temperature to 72 degrees",
        "Set the temperature to 22 degrees.",
        "Change climate to 72 degrees.",
        "Turn on the AC please",
        "Turn on the AC",
        "Turn up the heat",
        "Cool down the car",
        "Can you increase the temperature to 23 degrees?",
        "Set the car temperature to 21 degrees",
        "Lower the car temperature",
        "Turn on the climate control",
        "Turn up the temperature to 25 degrees",
        "Turn on the air conditioning",
        "It's too cold. Can you turn up the heat?",
        "I'm feeling hot. Can you turn on the AC?",
        "Set the temperature to 22 degrees",
        "Can you turn on the air conditioning?",
        "Set the car temperature to 22 degrees",
        "Increase the temperature to 72 degrees",
        "Set climate control to 72 degrees",
        "Could you turn up the heat?",
        "I want to set the temperature to 24 degrees",
        "Can I set the vehicle temperature to 22 degrees?",
        "Can I set the car temperature to 24 degrees?",
        "Turn the heat up in the car",
        "Increase the car temperature",
        "Heat up the car please.",
        "Can you turn up the heat?",
        "Turn on the AC and set it to 22 degrees",
        "Set the car temperature to 72 degrees",
        "Turn on the air conditioning",
        "Adjust the temperature to 72 degrees",
        "Turn the AC to 22 degrees",
        "Turn on the AC",
        "Adjust the car temperature to 22 degrees",
        "Turn off the climate control",
        "Turn the A/C to 22 degrees",
        "Set the temperature to 72 degrees",
        "Lower the cabin temperature",
        "Can you increase the temperature to 24 degrees?",
        "I'm feeling hot. Can you lower the temperature?",
        "Can you make it warmer inside the car?",
        "Please lower the temperature to 22 degrees",
        "Turn on climate control",
        "Lower the temperature to 18 degrees",
    ]

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
