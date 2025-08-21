import argparse
from transformers.trainer_utils import set_seed
from .model import HFModel
from .recipe import RECIPE, EVAL
from .evaluator.evaluator import Evaluator
import os
import pandas as pd
import json 

def main(args):
    model_obj = HFModel(args.model_path)
    model_name = args.model_path.split("/")[-1]
    current_time = pd.Timestamp.now()
    eval = Evaluator(model_obj)
    if model_name[:len("checkpoint")] == "checkpoint":
        model_name = args.model_path.split("/")[-2] + "_" + model_name
    
    print(f"arg.data = {args.data}")
    
    for dataset_name in args.data:
        if dataset_name not in RECIPE:
            raise ValueError(f"Dataset {dataset_name} is not supported. Available datasets: {list(RECIPE.keys())}")
        for gen_test_df,gen_def_df,gen_exam_type in RECIPE[dataset_name]():
            test_df = gen_test_df
            if args.debug:
                test_df = test_df.iloc[:10]
            def_df = gen_def_df
            exam_type = gen_exam_type
            if exam_type is None:
                exam_type = "default"
            output = eval.eval_pretrain(
                test_df=test_df,
                dev_df=def_df,
                task_name=dataset_name,
                exam_type=exam_type,
                num_shots=args.num_shots,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len,
                save_result_dir=args.save_result_dir,
                model_name=model_name,
                current_time=current_time,
                instuction="ต่อไปนี้เป็นข้อสอบปรนัยจงเลือกคำตอบที่ถูกต้องที่สุด",
                debug=args.debug,
            )
        
            print(f"Evaluation for {dataset_name} completed.")    
            save_output = EVAL[dataset_name](output)
            if not os.path.exists(os.path.join(args.save_result_dir,model_name+"_"+str(current_time)[:19], "summary")):
                os.makedirs(os.path.join(args.save_result_dir,model_name+"_"+str(current_time)[:19], "summary"))
            with open(os.path.join(args.save_result_dir,model_name+"_"+str(current_time)[:19], "summary", f"{dataset_name}_report.json"), "w") as f:
                json.dump(save_output, f, ensure_ascii=False, indent=2)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on LLMs")
    parser.add_argument('--model-path', 
                        type=str, 
                        required=True, 
                        help='Model name or model path to evaluate')
    parser.add_argument(
        "--save-result-dir",
        type=str,
        default="./results",
        help="Path to save results",
    )
    parser.add_argument(
        '--data',
        nargs="+",
        default=["mmlu", "mmlu_thai", "belebele", "xcopa", "xnli", "m3exam", "thai_exam", "m6exam"],
        help='list of datasets to evaluate on (space-separated)'
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="pt",
        choices=["pt", "sft"],
        help="Mode of the model, either 'pt' for Pretrain or 'sft' for chat-based models"
    )
    
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    parser.add_argument("-s", "--seed", type=int, default=47, help="Random seed")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--max-seq-len", type=int, default=8192, help="Size of the output generated text")
    parser.add_argument("--num-shots", type=int, default=5, help="Number of shots for few-shot learning")
    
    args = parser.parse_args()
    set_seed(args.seed)
    main(args=args)