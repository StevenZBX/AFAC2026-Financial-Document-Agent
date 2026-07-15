def build_prompt(question: str, options: dict, evidence: str, answer_format: str) -> str:
    # 把选项字典拼成文字
    options_text = ""
    for key, value in options.items():
        options_text += f"{key}. {value}\n"

    if answer_format == "mcq":
        format_instruction = "只输出一个大写字母，例如：A"
    elif answer_format == "multi":
        format_instruction = "只输出所有正确答案的大写字母，按字母顺序，不加空格或逗号，例如：AC"
    elif answer_format == "tf":
        format_instruction = "只输出A或B"
    else:
        format_instruction = "只输出答案字母"

    # 有证据时用证据，没有证据时直接问
    if evidence:
        prompt = f"""

【证据材料】
{evidence}

【题目】
{question}

【选项】
{options_text}
【要求】
严格依据证据材料作答。
{format_instruction}

答案："""
    else:
        prompt = f"""

【题目】
{question}

【选项】
{options_text}
【要求】
{format_instruction}

答案："""

    return prompt