import os
import dashscope

# 以下为华北2（北京）地域的URL，调用时请将WorkspaceId替换为真实的业务空间ID，各地域的URL不同。
dashscope.base_http_api_url = "https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1"

messages = []
conversation_idx = 1
while True:
    print("=" * 20 + f"第{conversation_idx}轮对话" + "=" * 20)
    conversation_idx += 1
    user_msg = {"role": "user", "content": input("请输入你的消息：")}
    messages.append(user_msg)
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
        api_key=os.getenv('DASHSCOPE_API_KEY'),
         # 此处以qwen-plus为例，可按需更换为其它深度思考模型
        model="qwen-plus", 
        messages=messages,
        enable_thinking=True,
        result_format="message",
        stream=True,
        incremental_output=True
    )
    # 定义完整思考过程
    reasoning_content = ""
    # 定义完整回复
    answer_content = ""
    # 判断是否结束思考过程并开始回复
    is_answering = False
    print("=" * 20 + "思考过程" + "=" * 20)
    for chunk in response:
        # 如果思考过程与回复皆为空，则忽略
        if (chunk.output.choices[0].message.content == "" and 
            chunk.output.choices[0].message.reasoning_content == ""):
            pass
        else:
            # 如果当前为思考过程
            if (chunk.output.choices[0].message.reasoning_content != "" and 
                chunk.output.choices[0].message.content == ""):
                print(chunk.output.choices[0].message.reasoning_content, end="",flush=True)
                reasoning_content += chunk.output.choices[0].message.reasoning_content
            # 如果当前为回复
            elif chunk.output.choices[0].message.content != "":
                if not is_answering:
                    print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                    is_answering = True
                print(chunk.output.choices[0].message.content, end="",flush=True)
                answer_content += chunk.output.choices[0].message.content
    # 将模型回复的content添加到上下文中
    messages.append({"role": "assistant", "content": answer_content})
    print("\n")
    # 如果您需要打印完整思考过程与完整回复，请将以下代码解除注释后运行
    # print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
    # print(f"{reasoning_content}")
    # print("=" * 20 + "完整回复" + "=" * 20 + "\n")
    # print(f"{answer_content}")