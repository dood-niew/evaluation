import gdown
import zipfile
from huggingface_hub import snapshot_download
import logging
import requests
import os

# ---------- M3EXAM --------------
def download_m3exam():
    url = "https://drive.google.com/uc?id=1eREETRklmXJLXrNPTyHxQ3RFdPhq_Nes"
    output = "./data/m3exam.zip"
    extract_to = "./data/m3exam"
    password = "12317"  

    if not os.path.exists(output):
        logging.info("Downloading M3EXAM dataset...")
        gdown.download(url, output, quiet=False)
    else:
        logging.info("M3EXAM dataset already exists, skipping download.")

    if not os.path.exists(extract_to):
        logging.info("Extracting files from M3EXAM dataset...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to, pwd=password.encode())
    else:
        logging.info("Extracted folder already exists, skipping extraction.")


### if can't connect to internet with backend node
"""bash
gdown "https://drive.google.com/uc?id=1eREETRklmXJLXrNPTyHxQ3RFdPhq_Nes" -O ./data/m3exam.zip
cd ../data
unzip -P 12317 m3exam.zip -d m3exam
"""


# ---------- M6EXAM --------------
def download_m6exam():
    dataset_dir = "./data/thai-onet-m6-exam/data/preprocessed_m6exam.csv"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading ThaiEXAM dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="openthaigpt/thai-onet-m6-exam",
            repo_type="dataset",
            local_dir=dataset_dir
        )
        logging.info("ThaiExam Dataset downloaded to: %s", local_dir)
    else:
        logging.info("ThaiExam Dataset already exists at: %s", dataset_dir)
        
       
### if can't connect to internet with backend node
"""bash
hf download openthaigpt/thai-onet-m6-exam --local-dir ./data/thai-onet-m6-exam --repo-type dataset
"""

# ---------- ThaiEXAM --------------
def download_thai_exam():
    dataset_dir = "./data/thai_exam"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading ThaiEXAM dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="scb10x/thai_exam",
            repo_type="dataset",
            local_dir="./data/thai_exam"
        )
        logging.info("ThaiExam Dataset downloaded to: %s", local_dir)
    else:
        logging.info("ThaiExam Dataset already exists at: %s", dataset_dir)
### if can't connect to internet with backend node
"""bash
hf download scb10x/thai_exam --local-dir ./data/thai_exam --repo-type dataset
"""

# ---------- MMLU --------------
def download_mmlu():
    dataset_dir = "./data/mmlu"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading MMLU dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="cais/mmlu",
            repo_type="dataset",
            local_dir="./data/mmlu"
        )
        logging.info("MMLU Dataset downloaded to: %s", local_dir)
    else:
        logging.info("MMLU Dataset already exists at: %s", dataset_dir)

### if can't connect to internet with backend node
"""bash
hf download cais/mmlu --local-dir ./data/mmlu --repo-type dataset
"""

# ---------- MMLU-Thai --------------
def download_mmlu_thai():
    dataset_dir = "./data/seaexam/mmlu-thai"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading MMLU-Thai dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="cais/mmlu_thai",
            repo_type="dataset",
            local_dir="./data/seaexam/mmlu-thai"
        )
        logging.info("MMLU-Thai Dataset downloaded to: %s", local_dir)
    else:
        logging.info("MMLU-Thai Dataset already exists at: %s", dataset_dir)

"""bash
hf download SeaLLMs/SeaExam --local-dir ./data/seaexam --repo-type dataset
"""
# ---------- xnli --------------
def download_xnli():
    dataset_dir = "./data/xnli"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading XNLI dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="facebook/xnli",
            repo_type="dataset",
            local_dir="./data/xnli",
        )
        logging.info("XNLI Dataset downloaded to: %s", local_dir)
    else:
        logging.info("XNLI Dataset already exists at: %s", dataset_dir)

"""bash
hf download facebook/xnli --local-dir ./data/xnli --repo-type dataset
"""
# ---------- belebele --------------
def download_belebele():
    dataset_dir = "./data/belebele"
    if not os.path.exists(dataset_dir):
        logging.info("Downloading Belebele dataset from Hugging Face...")
        local_dir = snapshot_download(
            repo_id="facebook/belebele",
            repo_type="dataset",
            local_dir="./data/belebele",
        )
        logging.info("Belebele Dataset downloaded to: %s", local_dir)
    else:
        logging.info("Belebele Dataset already exists at: %s", dataset_dir)

"""bash
hf download facebook/belebele --local-dir ./data/belebele --repo-type dataset
"""
# ---------- XCOPA Thai --------------
def download_xcopa():
    dataset_dir = "./data/xcopa"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
        logging.info("Downloading XCOPA Thai dataset...")
        urls = [
            "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/test.th.jsonl",
            "https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/val.th.jsonl"
        ]
        names = ["xcopa_thai_test.jsonl", "xcopa_thai_val.jsonl"]
        for url, name in zip(urls, names):
            response = requests.get(url)
            response.raise_for_status()
            with open(f"./data/xcopa/{name}", "w", encoding="utf-8") as f:
                f.write(response.text)
        logging.info("XCOPA Thai dataset downloaded and saved.")
    else:
        logging.info("XCOPA Thai dataset already exists, skipping download.")

"""bash
mkdir -p ./data/xcopa
wget https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/test.th.jsonl -O ./data/xcopa/xcopa_thai_test.jsonl
wget https://raw.githubusercontent.com/cambridgeltl/xcopa/master/data/th/val.th.jsonl -O ./data/xcopa/xcopa_thai_val.jsonl
"""

if __name__ == "__main__":
    download_m3exam()
    download_m6exam()
    download_thai_exam()
    download_mmlu()
    download_mmlu_thai()
    download_xnli()
    download_belebele()
    download_xcopa()