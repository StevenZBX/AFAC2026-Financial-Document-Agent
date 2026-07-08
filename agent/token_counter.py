class TokenCounter:
    """统计所有 API 调用的 Token 消耗"""

    def __init__(self):
        """初始化，三个计数器都从0开始"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add(self, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        """
        每次调用 API 之后，把这次的 token 消耗加进来。

        args:
            prompt_tokens: 这次发出去的 token 数
            completion_tokens: 这次收到的 token 数
            total_tokens: 这次总共花的 token 数
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += total_tokens

    def summary(self) -> dict:
        """
        返回目前为止所有题目的总 token 消耗。

        returns:
            dict 包含三项统计
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        }