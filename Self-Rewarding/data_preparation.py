import json
import random
from datasets import load_dataset

dataset = load_dataset("livecodebench/math_competition", split="test")
dataset.to_json("math.jsonl")

with open("math.jsonl", "r") as f:
    math_data = [json.loads(line) for line in f]

ift_data = []
for problem in math_data:
    prompt = problem["problem"]
    steps = []
    current = prompt
    for i in range(random.randint(1, 5)):
        step_prompt = f"Continue solving: {current}"
        step_response = "Placeholder step"  # In real, generate or use
        steps.append(step_response)
        current += "\n" + step_response
    answer = "Placeholder answer"
    ift_data.append({"prompt": prompt, "steps": steps, "answer": answer})

test_data = math_data
