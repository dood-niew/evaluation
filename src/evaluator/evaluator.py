import torch
from ..prompt_process.prompt_generator import get_prompt, format_instructions
from ..prompt_process.postprocess import PostProcessor
from tqdm import tqdm
import numpy as np
import pandas as pd
from ..model import HFModel
import os

class Evaluator:
    def __init__(self, model_obj: HFModel):
        self.model = model_obj.model
        self.model_obj = model_obj
        self.tokenizer = model_obj.tokenizer
    @torch.no_grad()
    def eval_pt(
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
        
        idx_list = list(range(0, len(test_df), batch_size))
        for i in tqdm(idx_list):
            full_prompt_list = []
            answer_list = []
            for _, row in test_df.iloc[i:i+batch_size].iterrows():
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
    @torch.no_grad()
    def eval_it(
        self,
        test_df: pd.DataFrame,
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
        thinking: bool = True,
        **kwargs,
    ):
        result = []
        score = []
        all_prompt_list = []
        thinks = []
        responses = []
        idx_list = list(range(0, len(test_df), batch_size))
        for i in tqdm(idx_list):
            full_prompt_list = []
            answer_list = []
            for _, row in test_df.iloc[i:i+batch_size].iterrows():
                prompt, choices, answer_types, answer = get_prompt(
                    task_name,
                    row,
                    few_shot_line=dev_df,
                    num_shots=num_shots,
                    include_answer=False,
                    instuction=instuction
                )
                
                chat_template_apply = format_instructions(
                    self.tokenizer,
                    prompt,
                    choices,
                    thinking_mode=thinking
                )
                
                full_prompt_list.append(chat_template_apply)
                all_prompt_list.append(chat_template_apply)
                answer_list.append(answer)
            
            model_inputs = self.tokenizer(full_prompt_list, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
            generated_ids = self.model_obj.generate(
                **model_inputs,
                max_new_tokens=max_seq_len,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            
            for i, input_ids in enumerate(model_inputs.input_ids):
                output_ids = generated_ids[i][len(input_ids):].tolist()
                decoded_output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                processor = PostProcessor(decoded_output)
                thinks.append(processor.extract_think())
                responses.append(processor.extract_answer())
                pred = processor.extract_boxed_number()
                result.append(pred)
                score.append(1 if pred == answer_list[i] else 0)
            
        if save_result_dir:
            test_df = test_df.copy()
            test_df.loc[:,"predict"] = result
            test_df = test_df.copy()
            test_df.loc[:,"prompt"] = all_prompt_list
            test_df = test_df.copy()
            test_df.loc[:,"think"] = thinks
            test_df = test_df.copy()
            test_df.loc[:,"response"] = responses
            if score:
                test_df["correctness"] = score
            os.makedirs(save_result_dir, exist_ok=True)
            if os.path.exists(os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19],task_name)) == False:
                os.makedirs(os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19],task_name))
            
            test_df.to_csv(
                os.path.join(save_result_dir,model_name+"_"+str(current_time)[:19], task_name,f"{exam_type}_result.csv"),
                encoding="utf-8",
                index=False,
            )
        return test_df
        
        