from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import torch
from typing import List 
from abc import ABC, abstractmethod
import openai
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
    
    
class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs):
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response["choices"][0]["message"]["content"]
