from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import torch
from typing import List 
from abc import ABC, abstractmethod
import openai
from concurrent.futures import ThreadPoolExecutor
from .prompt_process.formatters import FORMATTERS
from typing import List
import requests
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


class VLLMModel(BaseModel):
    def __init__(self, base_url: str, model_name: str, api_key: str = None):
        """
        Initialize VLLM Model
        
        Args:
            base_url: Base URL of vLLM API server (e.g., "http://example/thaillm-inference")
            model_name: Model name (e.g., "thaillm-8b")
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.model_name = model_name
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    def load_model(self):
        """vLLM API doesn't need to load model locally"""
        pass
    
    def generate_prompt(self, prompt, **kwargs):
        """
        Generate response for single prompt
        
        Args:
            prompt: List of messages or string prompt
            **kwargs: Additional parameters like temperature, top_p, etc.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            raise ValueError("Prompt must be string or list of messages")
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Unexpected response format: {e}")
            return None
    
    def generate(self, prompts, max_workers: int = 5, **kwargs):
        """
        Generate responses for multiple prompts with parallel processing
        
        Args:
            prompts: Single prompt or list of prompts
            max_workers: Number of parallel workers
            **kwargs: Additional parameters like temperature, top_p, etc.
        """
        if isinstance(prompts, (str, list)) and not isinstance(prompts[0] if isinstance(prompts, list) and prompts else None, dict):
            if isinstance(prompts, str):
                prompts = [prompts]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda p: self.generate_prompt(p, **kwargs), 
                prompts
            ))
        
        return results
    
    def generate_with_completions_api(self, prompt: str, **kwargs):
        """
        Alternative method using completions API (if available)
        
        Args:
            prompt: String prompt
            **kwargs: Additional parameters
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return None
    
    def format_instructions(self, prompt: str, choices: List[str], thinking_mode=True):
        """
        Format instructions for multiple choice questions
        
        Args:
            prompt: Question prompt
            choices: List of answer choices
            thinking_mode: Whether to enable thinking mode (not used in vLLM API)
        """
        messages = [
            {
                "role": "user", 
                "content": f"{prompt} \nMultiple Choice of {choices}\nPlease put your answer in a \\boxed, e.g. \n {FORMATTERS['choices'](choices)}"
            }
        ]
        return messages
    
    def get_model_info(self):
        """
        Get information about available models
        """
        try:
            response = requests.get(
                f"{self.base_url}/v1/models",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get model info: {e}")
            return None
    
    def test_connection(self):
        """
        Test connection to vLLM API server
        """
        try:
            test_response = self.generate_prompt("Hello", max_tokens=10)
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False