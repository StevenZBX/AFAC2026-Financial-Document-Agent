import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from agent.qwen_api import ChatResult, QwenClient


class QAAgent:
    def __init__(self, trace_path=None, enable_thinking=False):
        self.client = QwenClient(enable_thinking=enable_thinking)
        self.trace_dir = self._resolve_trace_dir(trace_path)
        self.run_time = datetime.now().astimezone()

    def answer_messages(
        self,
        messages: List[Dict[str, str]],
        trace_extra: Optional[Dict[str, Any]] = None,
    ) -> ChatResult:
        """
        Answers from the model, return the ChatResult from qwen_api.py
        """
        result = self.client.chat_with_trace(messages)
        self._save_trace(messages, result.to_dict(), trace_extra or {})
        return result

    @staticmethod
    def _default_trace_dir() -> Path:
        """
        The path for storing logs
        """
        return Path(__file__).resolve().parents[1] / "logs"

    @classmethod
    def _resolve_trace_dir(cls, trace_path) -> Path:
        if trace_path is None:
            return cls._default_trace_dir()

        path = Path(trace_path)
        if path.suffix:
            return path.parent
        return path

    def _save_trace(self, messages, result, trace_extra):
        """
        Function for storing each response of ther model
        Including the date, response, answer, etc.
        """
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now().astimezone()
        record = {
            "timestamp": now.isoformat(),
            "messages": messages,
            "answer": result.get("content", ""),
            "reasoning": result.get("reasoning", ""),
            "usage": result.get("usage", {}),
            "model": result.get("model", ""),
        }
        record.update(trace_extra)

        log_path = self._build_trace_path(record, now)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    def _build_trace_path(self, record: Dict[str, Any], now: datetime) -> Path:
        """
        Function for building the log folder
        Logic:
            each conversation will be grouped together first,
            then logs are separated by domain and run batch
        """
        question = record.get("question")
        domain = record.get("domain") or self._get_nested(question, "domain") or "unknown"
        qid = record.get("qid") or self._get_nested(question, "qid") or "no_qid"
        total_tokens = self._total_tokens(record.get("usage", {}))

        run_date_part = self.run_time.strftime("%Y%m%d")
        run_time_part = self.run_time.strftime("%H%M%S%f")
        answer_time_part = now.strftime("%H%M%S%f")
        safe_domain = self._safe_filename(domain)
        safe_qid = self._safe_filename(qid)
        folder = (
            self.trace_dir
            / f"{run_date_part}-{run_time_part}-Conversation"
            / f"{run_date_part}-{safe_domain}"
        )
        filename = f"{answer_time_part}_{safe_qid}_tokens-{total_tokens}.json"
        return folder / filename

    @staticmethod
    def _get_nested(value: Any, key: str) -> Any:
        if isinstance(value, dict):
            return value.get(key)
        return None

    @staticmethod
    def _total_tokens(usage: Any) -> int:
        if not isinstance(usage, dict):
            return 0
        total = usage.get("total_tokens")
        if total is not None:
            return int(total)
        prompt = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        completion = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return int(prompt) + int(completion)

    @staticmethod
    def _safe_filename(value: Any) -> str:
        text = str(value or "unknown")
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
        return text.strip("._") or "unknown"


if __name__ == "__main__":
    qa = QAAgent(enable_thinking=True)
