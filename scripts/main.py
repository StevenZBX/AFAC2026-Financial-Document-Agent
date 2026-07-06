import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import QAAgent


QUESTION_PATH = "public_dataset_upload/questions/group_a/financial_contracts_questions.json"
EVIDENCE_PATH = "retriever/chunks_raw/evidence.json"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(question, evidence_item):
    options = question.get("options") or {}
    option_text = "\n".join(f"{key}. {value}" for key, value in options.items())

    evidence_blocks = []
    for i, ev in enumerate(evidence_item.get("evidence", []), start=1):
        page = (ev.get("metadata") or {}).get("page", "")
        evidence_blocks.append(
            f"[证据{i}]\n"
            f"doc_id: {ev.get('doc_id', '')}\n"
            f"page: {page}\n"
            f"text: {ev.get('text', '')}"
        )

    evidence_text = "\n\n".join(evidence_blocks)

    return f"""
你是金融文档问答助手。请只根据证据回答选择题。

题目ID：{question.get("qid")}
题型：{question.get("answer_format")}

题目：
{question.get("question")}

选项：
{option_text}

证据：
{evidence_text}

请直接给出正确答案字母，并简单说明理由。
如果是多选题，答案按 ABCD 顺序输出，不要加逗号。
""".strip()


def main():
    questions = load_json(QUESTION_PATH)
    evidences = load_json(EVIDENCE_PATH)
    evidence_by_qid = {item["qid"]: item for item in evidences if item.get("qid")}

    qa = QAAgent(enable_thinking=False)

    for question in questions:
        qid = question["qid"]
        evidence_item = evidence_by_qid.get(qid)
        if not evidence_item:
            raise ValueError(
                f"No evidence found for {qid}. Check whether evidence.json matches the question file."
            )

        messages = [
            {
                "role": "system",
                "content": "你是金融文档问答助手，只能根据用户提供的证据回答。",
            },
            {
                "role": "user",
                "content": build_prompt(question, evidence_item),
            },
        ]

        result = qa.answer_messages(
            messages,
            trace_extra={
                "qid": qid,
                "domain": question.get("domain"),
                "question": question,
                "evidence": evidence_item,
            },
        )

        print("qid:", qid)
        print("answer:")
        print(result.content)


if __name__ == "__main__":
    main()
