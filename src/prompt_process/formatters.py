import pandas as pd
from typing import List, Callable
from .m3exam_utils import generate_prompt

# ---------- Default format ----------
def format_default(
    line: pd.Series,
    choices: List[str],
    include_answer: bool = True
) -> str:
    example = "Question: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nAnswer: " + line["answer_text"] + "\n\n"
    else:
        example += "\nAnswer:"
    return example

# ---------- MMLU-Thai format ----------
def format_mmlu_thai(
    line: pd.Series,
    choices: List[str],
    include_answer: bool = True
) -> str:
    example = "คำถาม: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nคำตอบ: " + line["answer_text"] + "\n\n"
    else:
        example += "\nคำตอบ:"
    return example

# ---------- XCOPA format ----------
def format_xcopa(
    line: pd.Series,
    choices: List[str],
    include_answer: bool = True
) -> str:
    example = (
        "คำถาม: เลือกเหตุผลที่ถูกต้องที่สุดจากสถานการณ์ที่กำหนดให้ "
        + line["question"]
        + " เป็นเพราะอะไร"
    )
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nเพราะ: " + line["answer_text"] + "\n\n"
    else:
        example += "\nเพราะ:"
    return example


# ---------- M6EXAM Format ------------
def format_m6exam(
    line: pd.Series,
    choices: List[str] = [],
    include_answer: bool = True
) -> str:
    example = "ข้อ\n"+ line['"no"']+line["instruction"]+"\n"+ line["input"]

    if include_answer:
        example += "\nตอบ:" + line["answer_text"] + "\n\n"
    else:
        example += "\nตอบ:"
    return example

# ---------- ThaiExam Format ------------
def format_thai_exam(
    line: dict,
    choices: List[str] = [],
    include_answer: bool = True,
) -> str:
    exam_type = line["subject"]
    if exam_type != "ic":
        prompt = f"\n{line['question']}\na. {line['a']}\nb. {line['b']}\nc. {line['c']}\nd. {line['d']}\ne. {line['e']}\nคำตอบ:"
    else:
        prompt = f"\n{line['question']}\na. {line['a']}\nb. {line['b']}\nc. {line['c']}\nd. {line['d']}\nคำตอบ:"
    
    if include_answer:
        prompt += str(line["answer_text"])
    return prompt

def format_choices(choices: List[str]) -> str:
    """Format choices for display."""
    return '\n'.join([f"The answer is: \\boxed{{{choice}}}" for i, choice in enumerate(choices)])

FORMATTERS: dict[str, Callable[[pd.Series, List[str], bool], str]] = {
    "mmlu": format_default,
    "mmlu_thai": format_mmlu_thai,
    "xcopa": format_xcopa,
    "xnli": format_mmlu_thai,
    "belebele": format_mmlu_thai,
    "m3exam": generate_prompt,
    "m6exam": format_m6exam,
    "thai_exam": format_thai_exam,
    "choices": format_choices
}

ANSWER_TYPES: dict[str, str] ={
    "mmlu": "Answer:",
    "mmlu_thai": "คำตอบ:",
    "xcopa": "เพราะ:",
    "xnli": "คำตอบ:",
    "belebele": "คำตอบ:",
    "m3exam": "คำตอบ:",
    "m6exam": "คำตอบ:",
    "thai_exam": "คำตอบ:"
}

ANSWER_CHOICES: dict[str, List[str]] = {
    "mmlu": ["A", "B", "C", "D"],
    "mmlu_thai": ["A", "B", "C", "D"],
    "xcopa": ["A", "B"],
    "xnli": ["A", "B", "C"],
    "belebele": ["A", "B", "C", "D"],
    #"m3exam": ["A", "B", "C", "D", "E"] special case
    "m6exam": ["1", "2", "3", "4", "5"],
    "thai_exam": ["a","b","c","d","e"]
}