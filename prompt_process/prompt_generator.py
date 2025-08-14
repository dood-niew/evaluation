import pandas as pd
from typing import List
from .m3exam_utils import load_dev_examples_once, get_choices_m3exam
from .formatters import FORMATTERS, ANSWER_CHOICES, ANSWER_TYPES
import logging

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