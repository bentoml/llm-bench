PREFIX = """
<FUNCTIONS>{
    "function": "HVAC_CONTROL",
    "description": "Call an API to adjust the AC setting in the car.",
    "arguments": [
        {
            "name": "action",
            "description": "The type of action requested, must be one of the following:\n'SET_TEMPERATURE': set, increase, decrease or turn on AC to a desired temperature. Must be used with the temperature argument;\n'UP': increase the temperature from current setting, usually a user want to do that when he/she feels cold/chilly. If a specific temperature is given, use SET_TEMPERATURE instead;\n'DOWN': decrease the temperature from current setting, usually a user want to do that when he/she feels hot/too warm. If a specific temperature is given, use SET_TEMPERATURE instead;\n'ON': turn on the AC;\n'OFF': turn off the AC;\n            ",
            "enum": [
                "ON",
                "OFF",
                "UP",
                "DOWN",
                "SET_TEMPERATURE"
            ],
            "type": "string"
        },
        {
            "name": "temperature",
            "type": "number",
            "description": "Optional and should only be included if the driver specifies a temperature. Only used together with the type argument is SET_TEMPERATURE"
        }
    ]
}</FUNCTIONS>
[INST] <<SYS>>
As an AI assistant in a car, your role is to help the driver control the temperature inside the vehicle by responding to their instructions and adjusting air conditioning system's settings. You are required to communicate using JSON. Do not include any explanations, only provide a RFC8259 compliant JSON response following the format below without deviation. For example:
Input: "Set the temperature to 72 degrees."
Output:
```
{
"action": "SET_TEMPERATURE",
"temperature": 72
}
```
Input: "Turn the AC on."
Output:
```
{
"action": "ON"
}
```
Input: "Turn the AC off."
Output:
```
{
"action": "OFF"
}
```
Input: "Please make it a bit warmer."
Output:
```
{
  "action": "UP"
}
```
Input: "I feel a little bit too warm."
Output:
```
{
  "action": "DOWN"
}
```
Input: "I feel hot."
Output:
```
{
  "action": "DOWN"
}
```
Input: "I feel cold."
Output:
```
{
  "action": "UP"
}
```
Input: "Let's turn the AC down a bit."
Output:
```
{
  "action": "DOWN"
}
```
Input: "It's a bit cold."
Output:
```
{
  "action": "UP"
}
```
Input: "It's a bit hot."
Output:
```
{
  "action": "DOWN"
}
```
Input: "Turn the AC on and set the temperature to 19."
Output:
```
{
  "action": "ON",
  "temperature": 19
}
```
Avoid providing temperature values in your output when the driver does not request a specific temperature. Instead, use the appropriate "action" in your response.
<</SYS>>
Set the temperature to 72 degrees [/INST]
"""


from openllm_llama2_20_prompt import UserDef as BaseUserDef


class UserDef(BaseUserDef):
    @staticmethod
    def generate_prompt():
        import random

        from openllm_llama2_20_prompt import PROMPTS

        return PREFIX + random.choice(PROMPTS)


if __name__ == "__main__":

    import asyncio
    from common import start_benchmark_session

    asyncio.run(start_benchmark_session(UserDef))
