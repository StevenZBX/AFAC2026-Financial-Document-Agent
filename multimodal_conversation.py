import os
import dashscope 
from dashscope import MultiModalConversation

# 以下为华北2（北京）地域的URL，调用时请将WorkspaceId替换为真实的业务空间ID，各地域的URL不同。
dashscope.base_http_api_url = "https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1"

messages = [
    {
        "role": "user",
        "content": [
            {
                "image": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20251031/ownrof/f26d201b1e3f4e62ab4a1fc82dd5c9bb.png"
            },
            {"text": "请问图片展现了有哪些商品？"},
        ],
    }
]
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3.6-plus',   # 可按需更换为其它多模态模型，并修改相应的 messages
    messages=messages)
print(f"模型第一轮输出：{response.output.choices[0].message.content[0]['text']}")

messages.append(response['output']['choices'][0]['message'])
user_msg = {"role": "user", "content": [{"text": "它们属于什么风格？"}]}
messages.append(user_msg)
response = MultiModalConversation.call(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv('DASHSCOPE_API_KEY'),
    model='qwen3-vl-plus',
    messages=messages)
    
print(f"模型第二轮输出：{response.output.choices[0].message.content[0]['text']}")