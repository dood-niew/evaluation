import torch
from ..prompt_process.prompt_generator import get_prompt
from tqdm import tqdm
import numpy as np
import pandas as pd
from ..model import HFModel
import os

class Evaluator:
    def __init__(self, model_obj: HFModel):
        self.model = model_obj.model
        self.model_obj = model_obj
    @torch.no_grad()
    def eval_pretrain(
        self,
        test_df : pd.DataFrame,
        dev_df: pd.DataFrame,
        task_name: str,
        exam_type: str, 
        num_shots=5,
        batch_size=1,
        max_seq_len=8192,
        save_result_dir=None,
        model_name:str =None,
        current_time: str=None,
        instuction: str="ต่อไปนี้เป็นข้อสอบปรนัยจงเลือกคำตอบที่ถูกต้องที่สุด",
        debug: bool = False,
        **kwargs,
    ):
        result = []
        score = []
        all_prompt_list = []
        
        
        prompt, choices, answer_types, answer = get_prompt(
            task_name,
            test_df.iloc[0],
            few_shot_line=dev_df,
            num_shots=num_shots,
            include_answer=False,
            instuction=instuction
        )
        
        
        all_probs = {f"prob_{i}":[] for i in range(len(choices))}
        if task_name == "m3exam":
            all_probs = {f"prob_{i}":[] for i in range(5)}
        # tmp = "torch.tensor(["
        
        idx_list = list(range(0, len(test_df), batch_size))
        for i in tqdm(idx_list):
            full_prompt_list = []
            answer_list = []
            for index,row in test_df.iloc[i:i+batch_size].iterrows():
                prompt, choices, answer_types, answer = get_prompt(
                    task_name,
                    row,
                    few_shot_line=dev_df,
                    num_shots=num_shots,
                    include_answer=False,
                    instuction=instuction
                )
                full_prompt_list.append(prompt)
                all_prompt_list.append(prompt)
                answer_list.append(answer)

            # choice_ids = tmp
            # for choice in choices:
            #     choice_ids += f'tokenizer(" {choice}", add_special_tokens=False)["input_ids"][-1],'
            # choice_ids = choice_ids[:-1] + "]).unsqueeze(0).to(model.device)" 
            # choice_ids = eval(choice_ids)
            
            choice_ids = torch.tensor(
                [self.model_obj.tokenizer(f" {c}", add_special_tokens=False)["input_ids"][-1] for c in choices]
            ).unsqueeze(0).to(self.model.device)

            
            logits, input_info = self.model_obj.get_logits(full_prompt_list, max_seq_len)
            softval = logits.gather(1, choice_ids.expand(logits.size(0), -1)).softmax(1)
            if softval.dtype in {torch.bfloat16, torch.float16}:
                softval = softval.to(dtype=torch.float32)
            probs = softval.detach().cpu().numpy()

            for i in range(len(probs)):
                for j, choice in enumerate(choices):
                    all_probs[f"prob_{j}"].append(probs[i][j])
                pred = dict(zip(range(len(choices)),choices))[np.argmax(probs[i])]

                if answer_list != []:
                    correct = 1 if pred == answer_list[i] else 0
                    score.append(correct)
                result.append(pred)    
        
        if save_result_dir:
            test_df = test_df.copy()
            test_df.loc[:,"predict"] = result
            test_df = test_df.copy()
            test_df.loc[:,"prompt"] = all_prompt_list
            for i, _ in enumerate(choices):
                test_df = test_df.copy()
                test_df.loc[:, f"prob_{i}"] = all_probs[f"prob_{i}"]
            if score:
                test_df["correctness"] = score
            os.makedirs(save_result_dir, exist_ok=True)
            if os.path.exists(os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19],task_name)) == False:
                os.makedirs(os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19],task_name))
            
            test_df.to_csv(
                os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19], task_name, f"{exam_type}_result.csv"),
                encoding="utf-8",
                index=False,
            )
        return test_df