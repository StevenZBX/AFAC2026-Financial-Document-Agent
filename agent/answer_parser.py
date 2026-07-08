import re

def parse_answer(raw: str, answer_format: str) -> str:
    """
    从模型的原始输出中提取出规范的答案
    args:
        raw: 模型原始输出，比如 "答案是A和C" 或 "AC" 或 "选B"
        answer_format: 题型 mcq/multi/tf
    returns:
        规范化的答案字符串，比如 "A" 或 "AC" 或 "B"
    """
    # 从原始输出里找出所有大写字母 A B C D
    letters = re.findall(r'[A-D]', raw.upper())

    if not letters:
        return ""

    if answer_format == "mcq" or answer_format == "tf":
        # 单选题和判断题只取第一个字母
        return letters[0]

    elif answer_format == "multi":
        # 多选题：去重 + 排序
        unique_letters = sorted(set(letters))
        return "".join(unique_letters)

    return letters[0]