import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Assuming test_data from data_preparation.py

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    device_map="auto",
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base_model, "./m1_model")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

model = model.merge_and_unload()

test_prompts = test_data[:100]['question']

def generate_responses(model, tokenizer, prompt, num_responses=4, max_length=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    responses = []
    for _ in range(num_responses):
        output = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append(response)
    return responses

generated_data = []
for prompt in test_prompts:
    responses = generate_responses(model, tokenizer, prompt)
    generated_data.append({'prompt': prompt, 'responses': responses})

judge_prompt_template = """
Evaluate the following response to the math question on a scale of 1-5 (5 best) for accuracy and clarity.
Question: {prompt}
Response: {response}
Score:
"""

def self_reward(model, tokenizer, prompt, response):
    judge_input = judge_prompt_template.format(prompt=prompt, response=response)
    inputs = tokenizer(judge_input, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    score_text = tokenizer.decode(output[0], skip_special_tokens=True)
    try:
        score = int(score_text.strip().split()[-1])
    except:
        score = 1
    return score
