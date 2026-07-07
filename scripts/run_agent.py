import json
import csv
import os
import sys
from pathlib import Path


# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# from qwen_client import call_qwen
from agent.qa_qwen import QAAgent
from agent.prompts import build_prompt
from agent.answer_parser import parse_answer
from agent.token_counter import TokenCounter

def load_questions(path: str) -> list:
    """
    读取题目文件

    args:
        path: 题目 json 文件路径
    returns:
        题目列表
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_evidence(path: str) -> dict:
    """
    读取证据文件，转成 qid -> 证据文本 的字典

    args:
        path: evidence.json 路径
    returns:
        qid 到证据文本的字典
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evidence_map = {}
    for item in data:
        qid = item["qid"]
        evidence_blocks = []
        for i, chunk in enumerate(item.get("evidence", []), start=1):
            page = (chunk.get("metadata") or {}).get("page", "")
            evidence_blocks.append(
                f"[证据{i}]\n"
                f"doc_id: {chunk.get('doc_id', '')}\n"
                f"page: {page}\n"
                f"text: {chunk.get('text', '')}"
            )
        evidence_map[qid] = "\n\n".join(evidence_blocks)

    return evidence_map


def save_answer_csv(results: list, counter: TokenCounter, output_path: str):
    """
    生成 answer.csv 文件

    args:
        results: 每道题的答案列表
        counter: Token 统计对象
        output_path: 输出路径
    """
    summary = counter.summary()

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "answer", "prompt_tokens", "completion_tokens", "total_tokens"])

        # 第一行写总 summary
        writer.writerow([
            "summary",
            "",
            summary["prompt_tokens"],
            summary["completion_tokens"],
            summary["total_tokens"]
        ])

        # 每道题的答案
        for row in results:
            writer.writerow([
                row["qid"],
                row["answer"],
                row["prompt_tokens"],
                row["completion_tokens"],
                row["total_tokens"]
            ])

    print(f"answer.csv 已生成：{output_path}")


def run(questions_path, evidence_path: str, output_path: str):
    """
    主运行函数，串联所有步骤

    args:
        questions_path: 题目文件路径或题目文件路径列表
        evidence_path: 证据文件路径
        output_path: 输出文件路径
    """
    print("读取题目...")
    questions = []
    if isinstance(questions_path, (str, Path)):
        questions_path = [questions_path]
    for path in questions_path:
        loaded_questions = load_questions(path)
        questions.extend(loaded_questions)
        print(f"{path}: {len(loaded_questions)} 道题")
    print(f"共 {len(questions)} 道题")

    print("读取证据...")
    evidence_map = load_evidence(evidence_path)

    counter = TokenCounter()
    results = []

    qa = QAAgent(enable_thinking=True)
    for i, q in enumerate(questions):
        qid = q["qid"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]

        # 获取证据
        evidence = evidence_map.get(qid, "")

        # 构建 prompt
        prompt = build_prompt(question, options, evidence, answer_format)
        
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        # 调用 Qwen
        # result = call_qwen(prompt)
        result = qa.answer_messages(messages,
                                    trace_extra={
                                        "qid": qid,
                                        "domain":q["domain"],
                                        "question": q,
                                        "evidence": evidence
                                    })

        # 解析答案
        final_answer = parse_answer(result["answer"], answer_format)

        # 统计 token
        counter.add(
            result["prompt_tokens"],
            result["completion_tokens"],
            result["total_tokens"]
        )

        results.append({
            "qid": qid,
            "answer": final_answer,
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens": result["total_tokens"]
        })

        print(f"[{i+1}/{len(questions)}] {qid}: {final_answer}  (tokens: {result['total_tokens']})")

    save_answer_csv(results, counter, output_path)
    print(f"总 token 消耗: {counter.summary()['total_tokens']}")


if __name__ == "__main__":
    run(
        questions_path=[
            str(ROOT / "public_dataset_upload" / "questions" / "group_a" / "financial_contracts_questions.json"),
            str(ROOT / "public_dataset_upload" / "questions" / "group_a" / "financial_reports_questions.json"),
            str(ROOT / "public_dataset_upload" / "questions" / "group_a" / "insurance_questions.json"),
        ],
        evidence_path=str(ROOT / "retriever" / "chunks_raw" / "evidence.json"),
        output_path=str(ROOT / "answer.csv")
    )
