import pandas as pd
from typing import Callable, List
from .constant import MMLU_EXAM_TYPE
def eval_m3exam(df: pd.DataFrame) -> dict:
    subject_list = ['math', 'science', 'social', 'thai']
    level_list = ['high', 'low', 'mid']
    years = ['2009','2010','2013','2014','2015','2016','2017','2019','2020','2022']
    report_json = {
        "subject": {},
        "level": {},
        "year": {},
        "overall": 0
    }
    for subject in subject_list:
        filter_df = df[(df['subject'] == subject)]
        report_json["subject"][subject] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
    
   
    for level in level_list:
        try:
            filter_df = df[(df['level'] == level)]
            report_json["level"][level] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
        except:
            report_json["level"][level] = "N/A" # it can be possible when you have a low sample size for a specific level (divine by zeros)

    for year in years:
        try:
            filter_df = df[(df['year'] == year)]
            report_json["year"][year] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
        except:
            report_json["year"][year] = "N/A"

    report_json["overall"] = round(df["correctness"].to_list().count(1) / df.shape[0], 4)
    
    return report_json


def eval_m6exam(df: pd.DataFrame) -> dict:
    subject_list = ['english', 'math', 'science', 'social', 'thai']
    years = [2016, 2017, 2018, 2019, 2020, 2021]
    report_json = {
        "subject": {},
        "year": {},
        "overall": 0
    }
    try:
        for subject in subject_list:
            filter_df = df[(df['subject'] == subject)]
            report_json["subject"][subject] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
    except:
        report_json["subject"] = "N/A"
    try:
        for year in years:
            filter_df = df[(df['year'] == year)]
            report_json["year"][year] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
    except:
        report_json["year"] = "N/A"
    report_json["overall"] = round(df["correctness"].to_list().count(1) / df.shape[0], 4)
    
    return report_json

def eval_thai_exam(df: pd.DataFrame) -> dict:
    exam_type = ["a_level", "ic", "onet", "tgat", "tpat1"]
    report_json = {
        "exam": {},
        "overall": 0
    }
    try:
        for exam in exam_type:
            filter_df = df[(df['subject'] == exam)]
            report_json["exam"][exam] = round(filter_df["correctness"].to_list().count(1) / filter_df.shape[0],4)
    except:
        report_json["exam"] = "N/A"
    report_json["overall"] = round(df["correctness"].to_list().count(1) / df.shape[0], 4)
    
    return report_json

def eval_mmlu(df: pd.DataFrame) -> dict:
    exam_type = MMLU_EXAM_TYPE
    report_json = {
        "exam": {},
        "overall": 0
    }
    for exam in exam_type:
        filter_df = df[(df['exam'] == exam)]
        report_json["exam"][exam] = round(filter_df["is_correct"].to_list().count(1) / filter_df.shape[0],4)

    report_json["overall"] = round(df["is_correct"].to_list().count(1) / df.shape[0], 4)
    
    return report_json

def default_eval(df: pd.DataFrame) -> dict:
    """
    Default evaluation function that returns the overall accuracy.
    """
    report_json = {
        "overall": round(df["correctness"].to_list().count(1) / df.shape[0], 4)
    }
    return report_json

EVAL: dict[str, Callable[[pd.Series, List[str], bool], str]] = {
    "mmlu": default_eval,
    "mmlu_thai": default_eval,
    "xcopa": default_eval,
    "xnli": default_eval,
    "belebele": default_eval,
    "m3exam": eval_m3exam,
    "m6exam": eval_m6exam,
    "thai_exam": eval_thai_exam,   
}
