import os
import dashscope
from dashscope import Generation

dashscope.base_http_api_url = 'https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1'

def call_qwen(prompt: str, model: str = "qwen-plus") -> dict:
    """
    调用 Qwen API，返回答案和 token 消耗。
    args:
        prompt: 输入的提示词
        model: 使用的模型名称
    returns:
        dict 包含 answer, prompt_tokens, completion_tokens, total_tokens
    """
    messages = [
        {'role': 'system', 'content': '你是一个金融专家，请严格根据提供的证据材料回答问题。'},
        {'role': 'user', 'content': prompt}
    ]

    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        messages=messages,
        result_format="message"
    )

    if response.status_code == 200:
        answer = response.output.choices[0].message.content.strip()
        usage = response.usage
        return {
            "answer": answer,
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens
        }
    else:
        print(f"API错误: {response.code} - {response.message}")
        return {
            "answer": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }