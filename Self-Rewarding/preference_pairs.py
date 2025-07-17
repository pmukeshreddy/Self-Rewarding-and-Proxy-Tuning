# Assuming generated_data from response_generation.py

preference_pairs = []
for item in generated_data:
    prompt = item['prompt']
    scored_responses = [(resp, self_reward(model, tokenizer, prompt, resp)) for resp in item['responses']]
    scored_responses.sort(key=lambda x: x[1], reverse=True)
    if len(scored_responses) >= 2:
        chosen = scored_responses[0][0]
        rejected = scored_responses[-1][0]
        preference_pairs.append({'prompt': prompt, 'chosen': chosen, 'rejected': rejected})
