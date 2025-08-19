import json
from ..download_data import (download_m3exam, download_thai_exam, download_m6exam, 
                             download_mmlu, download_mmlu_thai, download_belebele,
                             download_xnli, download_xcopa)
from .constant import MMLU_EXAM_TYPE
import pandas as pd
import numpy as np
import datasets
from ..prompt_process.preprocessors import (preprocess_xnli, preprocess_xcopa, preprocess_thai_exam,
                                            preprocess_m6exam)
from ..prompt_process.prompt_generator import restructure_dataframe
from ..prompt_process.formatters import ANSWER_CHOICES
import logging
import os

def m3exam_recipe():
    download_m3exam()
    with open("./data/m3exam/text-question/thai-questions-test.json", "r") as f:
        test_questions = json.load(f)
    test_questions_df = pd.DataFrame(test_questions)
    yield test_questions_df, [], None

def m6exam_recipe():
    download_m6exam()
    if os.path.exists("./data/thai-onet-m6-exam/data/preprocessed_m6exam.csv"):
        logging.info("preprocessed_m6exam dataset already exists, skipping preprocessing.")
    else:
        logging.info("Preprocessing m6exam dataset...")
        preprocess_m6exam()
    df = pd.read_csv(f"./data/thai-onet-m6-exam/data/preprocessed_m6exam.csv")
    test_questions_df = df[df["subset"] == "test"].reset_index(drop=True)
    dev_questions_df = df[df["subset"] == "train"].reset_index(drop=True)
    yield test_questions_df, dev_questions_df, None

def thai_exam_recipe():
    download_thai_exam()
    print("current path:", os.getcwd())
    
    if os.path.exists("./data/thai_exam/preprocessed_thai_exam.csv"):
        logging.info("preprocessed_thai_exam dataset already exists, skipping preprocessing.")
    else:
        logging.info("Preprocessing thai_exam dataset...")
        preprocess_thai_exam()
    df = pd.read_csv(f"./data/thai_exam/preprocessed_thai_exam.csv")
    test_questions_df = df[df["subset"] == "test"].reset_index(drop=True)
    dev_questions_df = df[df["subset"] == "train"].reset_index(drop=True)
    yield test_questions_df, dev_questions_df, None

def mmlu_recipe():
    download_mmlu()
    for exam_type in MMLU_EXAM_TYPE:
        mmlu_dataset = datasets.load_dataset("./data/mmlu/", exam_type)
        dev_questions_df = restructure_dataframe(pd.DataFrame(mmlu_dataset["dev"]))
        test_questions_df = restructure_dataframe(pd.DataFrame(mmlu_dataset["test"]))
        yield test_questions_df, dev_questions_df, exam_type

def mmlu_thai_recipe():
    download_mmlu_thai()
    for exam_type in MMLU_EXAM_TYPE:
        mmlu_dataset = datasets.load_dataset("./data/seaexam/mmlu-thai")
        dev_questions_df = restructure_dataframe(pd.DataFrame(mmlu_dataset["validation"]), have_meta_data=True)
        test_questions_df = restructure_dataframe(pd.DataFrame(mmlu_dataset["test"]), have_meta_data=True)
        yield test_questions_df, dev_questions_df, exam_type
        
def belebele_recipe():
    download_belebele()
    df = datasets.load_dataset("./data/belebele", "tha_Thai")
    df = pd.DataFrame(df["test"])
    df["choices"] = np.array([df["mc_answer1"],df["mc_answer2"],df["mc_answer3"],df["mc_answer4"]]).T.tolist()
    df["answer"] = (df["correct_answer_num"]).apply(lambda x: int(x)-1)
    df["question"] = df["flores_passage"]+"\n"+df["question"]
    df = df[["question","choices","answer"]]
    dev_questions_df = df.iloc[-5:]
    test_questions_df = df.iloc[:-5]
    dev_questions_df = restructure_dataframe(dev_questions_df)
    test_questions_df = restructure_dataframe(test_questions_df).iloc[:]
    yield test_questions_df, dev_questions_df, None
    
def xnli_recipe():
    download_xnli()
    xnli = datasets.load_dataset("./data/xnli/", "th")
    test_questions_df = pd.DataFrame(xnli["test"])
    test_questions_df = preprocess_xnli(test_questions_df)
    ev_questions_df = pd.DataFrame(xnli["validation"])[-5:]
    ev_questions_df = preprocess_xnli(ev_questions_df)    
    ev_questions_df = restructure_dataframe(ev_questions_df,choices=ANSWER_CHOICES["xnli"])
    test_questions_df = restructure_dataframe(test_questions_df,choices=ANSWER_CHOICES["xnli"])
    yield test_questions_df, ev_questions_df, None
    
def xcopa_recipe():
    download_xcopa()
    with open("./data/xcopa/xcopa_thai_test.jsonl", "r") as f:
        test = []
        for line in f:
            test.append(json.loads(line))
        test_df = pd.DataFrame(test)

    with open("./data/xcopa/xcopa_thai_val.jsonl", "r") as f:
        val = []
        for line in f:
            val.append(json.loads(line))
        val_df = pd.DataFrame(val)
    val_df = preprocess_xcopa(val_df)
    test_df = preprocess_xcopa(test_df)
    val_df = restructure_dataframe(val_df, choices=ANSWER_CHOICES["xcopa"])
    test_df = restructure_dataframe(test_df, choices=ANSWER_CHOICES["xcopa"])
    yield test_df, val_df, None
    
RECIPE = {
    "m3exam": m3exam_recipe,
    "m6exam": m6exam_recipe,
    "thai_exam": thai_exam_recipe,
    "mmlu": mmlu_recipe,
    "mmlu_thai": mmlu_thai_recipe,
    "belebele": belebele_recipe,
    "xnli": xnli_recipe,
    "xcopa": xcopa_recipe
    }