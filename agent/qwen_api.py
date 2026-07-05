import os
import dashscope
from dashscope import Generation


class QwenClient:
    def __init__(self, api_key=None, model="qwen-plus"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        dashscope.base_http_api_url = "https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1"

    def chat(self, messages):
        response = Generation.call(
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            result_format="message",
        )
        if response.status_code != 200:
            raise RuntimeError(f"{response.code}: {response.message}")
        return response.output.choices[0].message.content