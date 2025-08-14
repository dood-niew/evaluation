from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import torch
from typing import List 
def load_models_tokenizer(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        device_map="balanced",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).eval()
    model.generation_config = GenerationConfig.from_pretrained(
        checkpoint_path,
        pad_token_id=tokenizer.pad_token_id,
        trust_remote_code=True
    )
    return model, tokenizer

def get_logits(tokenizer, model, inputs: List[str], max_seq_len: int = 8192):
    input_ids = tokenizer(inputs, padding='longest')["input_ids"]
    input_ids = torch.tensor(input_ids, device=model.device)

    if input_ids.shape[1] > max_seq_len:
        input_ids = input_ids[:, input_ids.shape[1] - max_seq_len + 1 :]
    tokens = {"input_ids": input_ids}
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    outputs = model(input_ids, attention_mask=attention_mask)["logits"]
    logits = outputs[:, -1, :]
    log_probs = torch.nn.functional.softmax(logits, dim=-1)
    return log_probs, {"tokens": tokens}