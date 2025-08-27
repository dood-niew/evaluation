from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import torch
from typing import List 
from abc import ABC, abstractmethod
import openai
from concurrent.futures import ThreadPoolExecutor
from .prompt_process.formatters import FORMATTERS
from typing import List
class BaseModel(ABC):
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass



class HFModel(BaseModel):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            pad_token_id=self.tokenizer.pad_token_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval()
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.checkpoint_path,
            pad_token_id=self.tokenizer.pad_token_id,
            trust_remote_code=True
        )
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def get_logits(self, inputs: List[str], max_seq_len: int = 8192):
        input_ids = self.tokenizer(inputs, padding='longest', return_tensors='pt')["input_ids"]
        input_ids = input_ids.to(self.model.device)

        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, -max_seq_len:]

        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs["logits"][:, -1, :]
        log_probs = torch.nn.functional.softmax(logits, dim=-1)
        return log_probs, {"tokens": {"input_ids": input_ids}}
    def format_instructions(self, prompt: str, choices: List[str], thinking_mode=True) -> str:
        messages = [
            {"role": "user", "content": f"{prompt} \nMultiple Choice of {choices}\nPlease put your answer in a \\boxed, e.g. \n {FORMATTERS['choices'](choices)}"},
        ]
        if thinking_mode != None:
            text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode)
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False)
        return text


class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def generate_prompt(self, prompt, **kwargs):
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            **kwargs
        )
        return response.choices[0].message.content

    def generate(self, prompts, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda p: self.generate_prompt(p, **kwargs), prompts))
        return results

    def format_instructions(self, prompt: str, choices: List[str], thinking_mode=True):
        messages = [
            {"role": "user", "content": f"{prompt} \nMultiple Choice of {choices}\nPlease put your answer in a \\boxed, e.g. \n {FORMATTERS['choices'](choices)}"}
        ]
        return messages
    def load_model(self):
        pass
