import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

import dashscope
from dashscope import Generation


@dataclass
class ChatResult:
    """
    Format of history of chatting
    content: prompt and question
    reasoning: thinking process
    model: the model version
    """
    content: str
    reasoning: str
    usage: Dict[str, Any]
    model: str

    @property
    def answer(self) -> str:
        return self.content

    @property
    def prompt_tokens(self) -> int:
        return self._usage_int("input_tokens", "prompt_tokens")

    @property
    def completion_tokens(self) -> int:
        return self._usage_int("output_tokens", "completion_tokens")

    @property
    def total_tokens(self) -> int:
        total = self._usage_int("total_tokens")
        if total:
            return total
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "answer": self.answer,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }
        )
        return data

    def __getitem__(self, key: str) -> Any:
        if key == "answer":
            return self.answer
        if key == "prompt_tokens":
            return self.prompt_tokens
        if key == "completion_tokens":
            return self.completion_tokens
        if key == "total_tokens":
            return self.total_tokens
        return getattr(self, key)

    def _usage_int(self, *keys: str) -> int:
        for key in keys:
            value = (self.usage or {}).get(key)
            if value is not None:
                return int(value)
        return 0


class QwenClient:
    """
    The API of the Qwen model
    Parameters:
        api_key: the api of the model
        model: the version of the model
        enable_thinking: allow the model thinking to obtain more precise response
    """
    def __init__(self, api_key=None, model="qwen-plus", enable_thinking=False):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        self.enable_thinking = enable_thinking
        dashscope.base_http_api_url = "https://ws-yf86n1x92c5y03kj.cn-beijing.maas.aliyuncs.com/api/v1"

    def chat_with_trace(self, messages: List[Dict[str, str]]) -> ChatResult:
        call_args = {
            "api_key": self.api_key,
            "model": self.model,
            "seed": 12138,
            "messages": messages,
            "result_format": "message",
            "enable_thinking": self.enable_thinking,
        }

        response = Generation.call(
            **call_args,
        )
        if response.status_code != 200:
            raise RuntimeError(f"{response.code}: {response.message}")

        message = response.output.choices[0].message
        content = self._get_value(message, "content", "")
        reasoning, content = self._extract_reasoning_and_content(message, content)

        return ChatResult(
            content=content,
            reasoning=reasoning,
            usage=self._to_plain(self._get_value(response, "usage", {})),
            model=self.model,
        )

    @classmethod
    def _extract_reasoning_and_content(cls, message: Any, content: str) -> Tuple[str, str]:
        for key in ("reasoning_content", "reasoning", "thought", "thinking"):
            value = cls._get_value(message, key)
            if value:
                return str(value), content

        return cls._split_think_block(content)

    @staticmethod
    def _split_think_block(content: str) -> Tuple[str, str]:
        start_tag = "<think>"
        end_tag = "</think>"
        if not content or start_tag not in content or end_tag not in content:
            return "", content

        start = content.find(start_tag) + len(start_tag)
        end = content.find(end_tag, start)
        if end == -1:
            return "", content

        reasoning = content[start:end].strip()
        answer = (content[: content.find(start_tag)] + content[end + len(end_tag) :]).strip()
        return reasoning, answer

    @classmethod
    def _get_value(cls, obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _to_plain(cls, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            return {str(k): cls._to_plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_plain(item) for item in value]
        if hasattr(value, "to_dict"):
            return cls._to_plain(value.to_dict())
        if hasattr(value, "__dict__"):
            return cls._to_plain(vars(value))
        return str(value)
