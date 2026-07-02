import os
from dashscope import Generation
import dashscope

# 以下为华北2（北京）地域的URL，各地域的URL不同。调用时请将WorkspaceId替换为真实的业务空间ID。
dashscope.base_http_api_url = 'https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1'
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant.'},
    {'role': 'user', 'content': '怎么做番茄炒蛋'}
]
response = Generation.call(
    # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key = "sk-xxx",
    
    api_key= os.getenv("DASHSCOPE_API_KEY"), 
    model="qwen-plus",   # 模型列表：https://help.aliyun.com/model-studio/getting-started/models
    messages=messages,
    result_format="message"
)

if response.status_code == 200:
    print(response.output.choices[0].message.content)
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")