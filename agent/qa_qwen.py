from qwen_api import QwenClient


class QAAgent:
    def __init__(self):
        self.client = QwenClient()

    def answer(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个金融文档问答助手，请基于给定信息准确回答。"},
            {"role": "user", "content": question},
        ]
        return self.client.chat(messages)
    
if __name__ == "__main__":
    qa = QAAgent()
    print(qa.answer("请给我写出金融最基本的几大要素"))