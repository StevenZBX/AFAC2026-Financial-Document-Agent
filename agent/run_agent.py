import json
import csv
import os
import sys


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qwen_client import call_qwen
from prompts import build_prompt
from answer_parser import parse_answer
from token_counter import TokenCounter

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
        texts = []
        for chunk in item.get("evidence", []):
            texts.append(chunk.get("text", ""))
        evidence_map[qid] = "\n\n".join(texts)

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


def run(questions_path: str, evidence_path: str, output_path: str):
    """
    主运行函数，串联所有步骤

    args:
        questions_path: 题目文件路径
        evidence_path: 证据文件路径
        output_path: 输出文件路径
    """
    print("读取题目...")
    questions = load_questions(questions_path)
    print(f"共 {len(questions)} 道题")

    print("读取证据...")
    evidence_map = load_evidence(evidence_path)

    counter = TokenCounter()
    results = []

    for i, q in enumerate(questions):
        qid = q["qid"]
        question = q["question"]
        options = q["options"]
        answer_format = q["answer_format"]

        # 获取证据
        evidence = evidence_map.get(qid, "")

        # 构建 prompt
        prompt = build_prompt(question, options, evidence, answer_format)

        # 调用 Qwen
        result = call_qwen(prompt)


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
        questions_path="../insurance_questions.json",
        evidence_path="../evidence.json",
        output_path="../answer.csv"
    )