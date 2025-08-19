import pandas as pd
import logging
import json
import numpy as np
import os
def preprocess_xcopa(df):
    df["choices"] = np.array([df["choice1"],df["choice2"]]).T.tolist()
    df.rename(columns={"question":"task","premise":"question","label":"answer"}, inplace=True)
    df = df[['question','choices','answer']]
    return df

def preprocess_m6exam() -> pd.DataFrame:
    if os.path.exists("./data/thai-onet-m6-exam/data/preprocessed_m6exam.csv"):
        logging.info("Preprocessed M6EXAM data already exists, skipping preprocessing.")
        return True
    
    subjects = ["english.csv", "math.csv", "science.csv", "social.csv", "thai.csv"]
    subsets = ["train","test"]
    
    save_df = pd.DataFrame()
    
    for subset in subsets:
        for subject in subjects:
            df = pd.read_csv(f"./data/thai-onet-m6-exam/data/{subset}/{subject}")
            df = df[(df['isAnswerable'] == True) & (df['isSingleChoiceSolution'] == True) & (df['isMultipleChoice'] == True)]
            df["answer_text"] =  df.apply(lambda x: x["result"][0], axis=1)
            subject_name = subject.split('.')[0]
            df["subject"] = [subject_name] * df.shape[0]
            df["subset"] = [subset] * df.shape[0]
            save_df = pd.concat([save_df, df], ignore_index=True)
    
    logging.info(f"Preprocessed M6EXAM data: {save_df.shape[0]} rows")
    save_df.to_csv(f"./data/thai-onet-m6-exam/data/preprocessed_m6exam.csv", index=False)
    
def preprocess_thai_exam() -> pd.DataFrame:
    if os.path.exists("./data/thai_exam/preprocessed_thai_exam.csv"):
        logging.info("Preprocessed ThaiExam data already exists, skipping preprocessing.")
        return True
    
    exam_type = ["a_level", "ic", "onet", "tgat", "tpat1"]
    subsets = ["train", "test"]
    save_df = pd.DataFrame()
    for exam in exam_type:
        for subset in subsets:
            with open(f"./data/thai_exam/data/{exam}/{exam}_{subset}.jsonl", "r") as f:
                df = [json.loads(line) for line in f]
                df = pd.DataFrame(df)
            if exam == "ic":
                df["e"] = [""] * df.shape[0]

            df["exam_type"] = [exam] * df.shape[0]
            df["subset"] = [subset] * df.shape[0]
            save_df = pd.concat([save_df, df], ignore_index=True)
    
    logging.info(f"Preprocessed ThaiExam data: {save_df.shape[0]} rows")
    save_df.drop(columns=["subject"], inplace=True)
    save_df.rename(columns={"answer":"answer_text", "exam_type":"subject"}, inplace=True)
    save_df.to_csv(f"./data/thai_exam/preprocessed_thai_exam.csv", index=False)
    
def preprocess_xnli(df):
    df["choices"] = [["สอดคล้อง", "เป็นกลาง", "โต้แย้ง"] for _ in range(len(df))]
    df["question"] = "หลักฐาน : " +df["premise"] + "\nสมมติฐาน : " + df["hypothesis"] + "\nสมมติฐานนี้และหลักฐานดังกล่าวมีความสัมพันธ์กันอย่างไร"
    df.rename(columns={"label":"answer"}, inplace=True)
    df = df[["question","choices","answer"]]
    return df