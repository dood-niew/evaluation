import pandas as pd
from typing import List, Callable
from collections import defaultdict
import logging
import json
import os

ROOT = "/project/lt200258-aithai/llm/code/Eval_LLM/evaluation"

logging.basicConfig(level=logging.INFO)
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

def preprocess_xcopa(df):
    df["choices"] = np.array([df["choice1"],df["choice2"]]).T.tolist()
    df.rename(columns={"question":"task","premise":"question","label":"answer"}, inplace=True)
    df = df[['question','choices','answer']]
    return df


# ---------- M3EXAM Format ----------
# original from https://github.com/DAMO-NLP-SG/M3Exam
def generate_dev_examples(dev_questions, lang, method):
    dev_example_dict = defaultdict(lambda: defaultdict(list))
    for q in dev_questions:
        level = q['level']
        cate = q['subject_category']
        dev_string = generate_one_example(q, lang, method, fill_answer=True)
        dev_example_dict[level][cate].append(dev_string)
    
    return dev_example_dict

def generate_one_example(question, lang, method, fill_answer=False):
    answer_word = {
        'english': "Answer:", 'chinese': '答案：', 'vietnamese': 'Câu trả lời:', 'thai': 'คำตอบ:', 
        'italian': 'La risposta:', 'javanese': 'Wangsulan:', 'swahili': 'Jibu:', 
        'afrikaans': 'Antwoord:', 'portuguese': 'Responder:'
    }
    background = '\n' + '\n'.join(question['background_description']) if question['background_description'] else ''
    if method == 'default':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\n{answer_word[lang]}'
    elif method == 'en-instruct':
        prompt = background + '\n' + question['question_text'] + '\n' + '\n'.join(question['options']) + f'\nAnswer:'
    elif method == 'en-trans':
        prompt = question['background_description_english'] + '\n' + question['question_text_english'] + '\n' + question['options_english'] + f'\nAnswer:'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if fill_answer:
        prompt += str(question['answer_text'])
    
    return prompt

# added utility function for m3exam to get choices based on answer_text
def get_choices_m3exam(row : pd.Series):
    if row["answer_text"] in ['1','2','3','4','5']:
        return ['1','2','3','4','5'][:len(row["options"])]
    elif row["answer_text"] in ['๑', '๒', '๓', '๔', '๕']:
        return ['๑','๒','๓','๔','๕'][:len(row["options"])]
    else:
        logging.error(f"Unexpected answer_text: {row['answer_text']}. Please issue a support ticket.")
        return []
    
def generate_prompt(lang, method, setting, model, test_question, dev_question):
    subject2target = {
        'english': {'language': 'English', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'english4all': {'language': 'Language', 'math': "Math", 'social-science': "Social Science", 'natural-science': 'Natural Science'},
        'chinese': {'language': '语文', 'math': "数学", 'social-science': "社会科学", 'natural-science': '自然科学'},
        'javanese': {'language': 'Bahasa Jawa'},
        'swahili': {'language': 'KISWAHILI'},
        'thai': {'language': 'ภาษาไทย', 'math': 'คณิตศาสตร์', 'social-science': 'สังคมศึกษา', 'natural-science': 'วิทยาศาสตร์'},
        'vietnamese': {'language': 'Tiếng Việt', 'math': "Toán", 'social-science': "Khoa học xã hội", 'natural-science': 'Khoa học tự nhiên'},
        'italian': {'language': 'Italiano', 'math': "Matematica", 'social-science': "Scienze sociali", 'natural-science': 'Scienze naturali'},
        'afrikaans': {'language': 'Afrikaans Huistaal', 'math': "Wiskunde", 'social-science': "Sosiale Wetenskappe", 'natural-science': 'Natuurwetenskap'},
        'portuguese': {'language': 'Linguagens', 'math': 'Matemática', 'social-science': 'Ciências Humanas', 'natural-science': 'Ciências da Natureza'},
    }
    subject = subject2target[lang][test_question['subject_category']]

    if method == 'default':
        hint = f"The following is a multiple choice question about {subject}."
        if model in ['chat', 'fake'] or setting == 'zero-shot':
            hint += ' Please only give the correct option, without any other details or explanations.'
    elif method in ['en-instruct', 'en-trans']:
        subject = subject2target['english4all'][test_question['subject_category']]
        hint = f"The following is a multiple choice question about {subject}."
        hint += ' Please only give the correct option, without any other details or explanations.'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    if setting == 'zero-shot':
        prompt = hint + '\n\n' + generate_one_example(test_question, lang, method)
    elif setting == 'few-shot':
        dev_questions_list = dev_question[test_question['level']][test_question['subject_category']]
        prompt = hint + '\n\n' + '\n\n'.join(dev_questions_list) + '\n\n' + generate_one_example(test_question, lang, method)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    return prompt

    
DEV_EXAMPLES_CACHE = None
def load_dev_examples_once(lang="thai", method="default"):
    """Load dev examples into global cache if not already loaded."""
    global DEV_EXAMPLES_CACHE
    if DEV_EXAMPLES_CACHE is None:
        json_path = f"{ROOT}/data/m3exam/text-question/thai-questions-dev.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dev questions file not found: {json_path}")

        with open(json_path, "r") as f:
            dev_questions = json.load(f)

        DEV_EXAMPLES_CACHE = generate_dev_examples(dev_questions, lang, method)
        logging.info(f"Loaded dev examples: {sum(len(cats) for cats in DEV_EXAMPLES_CACHE.values())} categories")
    return DEV_EXAMPLES_CACHE


# ---------- M6EXAM Format ------------
def preprocess_m6exam() -> pd.DataFrame:
    subjects = ["english.csv", "math.csv", "science.csv", "social.csv", "thai.csv"]
    subsets = ["train","test"]
    
    save_df = pd.DataFrame()
    
    for subset in subsets:
        for subject in subjects:
            df = pd.read_csv(f"{ROOT}/data/thai-onet-m6-exam/data/{subset}/{subject}")
            df = df[(df['isAnswerable'] == True) & (df['isSingleChoiceSolution'] == True) & (df['isMultipleChoice'] == True)]
            df["answer_text"] =  df.apply(lambda x: x["result"][0], axis=1)
            subject_name = subject.split('.')[0]
            df["subject"] = [subject_name] * df.shape[0]
            df["subset"] = [subset] * df.shape[0]
            save_df = pd.concat([save_df, df], ignore_index=True)
    
    logging.info(f"Preprocessed M6EXAM data: {save_df.shape[0]} rows")
    save_df.to_csv(f"{ROOT}/data/thai-onet-m6-exam/data/preprocessed_m6exam.csv", index=False)
    

def format_m6exam(
    line: pd.Series,
    choices: List[str],
    include_answer: bool = True
) -> str:
    example = "ข้อ\n"+ line['"no"']+line["instruction"]+"\n"+ line["input"]

    if include_answer:
        example += "\nตอบ:" + line["answer_text"] + "\n\n"
    else:
        example += "\nตอบ:"
    return example


# ---------- ThaiExam Format ------------
def preprocess_thai_exam() -> pd.DataFrame:
    exam_type = ["a_level", "ic", "onet", "tgat", "tpat1"]
    subsets = ["train", "test"]
    save_df = pd.DataFrame()
    for exam in exam_type:
        for subset in subsets:
            with open(f"{ROOT}/data/thai_exam/data/{exam}/{exam}_{subset}.jsonl", "r") as f:
                df = [json.loads(line) for line in f]
                df = pd.DataFrame(df)
            if exam == "ic":
                df["e"] = [""] * df.shape[0]

            df["exam_type"] = [exam] * df.shape[0]
            df["subset"] = [subset] * df.shape[0]
            save_df = pd.concat([save_df, df], ignore_index=True)
    
    logging.info(f"Preprocessed ThaiExam data: {save_df.shape[0]} rows")
    save_df.to_csv(f"{ROOT}/data/thai_exam/preprocessed_thai_exam.csv", index=False)

def format_thai_exam(
    line: dict,
    choices: List[str] = [],
    include_answer: bool = True,
) -> str:
    exam_type = line["exam_type"]
    if exam_type != "ic":
        prompt = f"\n{line['question']}\na. {line['a']}\nb. {line['b']}\nc. {line['c']}\nd. {line['d']}\ne. {line['e']}\nคำตอบ:"
    else:
        prompt = f"\n{line['question']}\na. {line['a']}\nb. {line['b']}\nc. {line['c']}\nd. {line['d']}\nคำตอบ:"
    
    if include_answer:
        prompt += str(line["answer_text"])
    return prompt


# --------- XNLI -----------
def preprocess_xnli(df):
    df["choices"] = [["สอดคล้อง", "เป็นกลาง", "โต้แย้ง"] for _ in range(len(df))]
    df["question"] = "หลักฐาน : " +df["premise"] + "\nสมมติฐาน : " + df["hypothesis"] + "\nสมมติฐานนี้และหลักฐานดังกล่าวมีความสัมพันธ์กันอย่างไร"
    df.rename(columns={"label":"answer"}, inplace=True)
    df = df[["question","choices","answer"]]
    return df


# --------- UTIL -----------
def generate_few_shot_prompt(num_shots: int, dev_df: pd.DataFrame, instuction: str, task_name: str, choices: List[str]):
    prompt = instuction + "\n\n"

    for shot in range(min(num_shots, dev_df.shape[0])):
        prompt += FORMATTERS[task_name](
            dev_df.iloc[shot, :],
            choices,
            include_answer=True,
        ) + "\n"

    return prompt

def restructure_dataframe(df: pd.DataFrame, have_meta_data:bool = False, choices: List[str] = ['A', 'B', 'C', 'D']) -> pd.DataFrame:
    """
    Restructures a DataFrame by splitting the 'choices' column into separate columns
    and mapping the 'answer' column to letter values.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with columns 'question', 'subject', 'choices', and 'answer'
    
    Returns:
    pd.DataFrame: Restructured DataFrame with columns 'question', 'subject', 'A', 'B', 'C', 'D', and 'answer'
    """
    df_new = df.copy()
    df_new[choices] = df_new['choices'].apply(pd.Series)
    answer_map = {i: choice for i, choice in enumerate(choices)}
    
    df_new['answer'] = df_new['answer'].map(answer_map)
    df_new = df_new.rename(columns={"answer": "answer_text"})  
    df_new = df_new.drop('choices', axis=1)
    if have_meta_data:
        df_new["subject"] = df_new.apply(lambda x: x["metadata"]["subject"] if "metadata" in x and "subject" in x["metadata"] else "other", axis=1)
    return df_new


# ---------- Task format mapping ----------
FORMATTERS: dict[str, Callable[[pd.Series, List[str], bool], str]] = {
    "mmlu": format_default,
    "mmlu_thai": format_mmlu_thai,
    "xcopa": format_xcopa,
    "xnli": format_mmlu_thai,
    "belebele": format_mmlu_thai,
    "m3exam": generate_prompt,
    "m6exam": format_m6exam,
    "thai_exam": format_thai_exam   
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

def get_prompt(
    task_name: str,
    line: pd.Series,
    few_shot_line: pd.DataFrame = None,
    instuction: str = "",
    num_shots: int = 5,
    include_answer: bool = True
) -> str:
    if task_name not in FORMATTERS:
        logging.error(f"Task name '{task_name}' is not supported.")

    if task_name != "m3exam":
        if num_shots > 0 and few_shot_line is not None:
            return generate_few_shot_prompt(min(5,num_shots), few_shot_line, instuction, task_name, ANSWER_CHOICES[task_name])+FORMATTERS[task_name](line, ANSWER_CHOICES[task_name], False), ANSWER_CHOICES[task_name], ANSWER_TYPES[task_name], line["answer_text"]
        elif num_shots > 0 and few_shot_line is None:
            logging.error("num_shots > 0 but few_shot_line is None. Please provide a valid few_shot_line DataFrame.")
            raise ValueError("num_shots > 0 but few_shot_line is None. Please provide a valid few_shot_line DataFrame.")
        
        return FORMATTERS[task_name](line, ANSWER_CHOICES[task_name], include_answer), ANSWER_CHOICES[task_name], ANSWER_TYPES[task_name], line["answer_text"]
    else:
        choices = get_choices_m3exam(line)
        lang = 'thai'
        method = 'default'
        setting = 'few-shot'
        dev_examples = load_dev_examples_once(lang, method)
        return FORMATTERS[task_name](lang, method, setting, None, line, dev_examples), choices, ANSWER_TYPES[task_name], line["answer_text"]

