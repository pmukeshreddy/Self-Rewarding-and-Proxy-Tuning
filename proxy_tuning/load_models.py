import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model
from config import small_base_model_name, large_model_name, lora_config

def load_small_base_model():
    tokenizer = AutoTokenizer.from_pretrained(small_base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        small_base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    return model, tokenizer

def load_small_tuned_model(small_base_model):
    return get_peft_model(small_base_model, LoraConfig(**lora_config))

def load_large_model():
    tokenizer = AutoTokenizer.from_pretrained(large_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        large_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer
