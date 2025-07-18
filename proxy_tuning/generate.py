import torch

def proxy_generate(large_model, large_tokenizer, small_base_model, small_tuned_model, prompt, max_new_tokens=256, alpha=0.1, temperature=0.7):
    inputs = large_tokenizer(prompt, return_tensors="pt").to(large_model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        with torch.no_grad():
            large_outputs = large_model(input_ids=generated_ids, attention_mask=attention_mask)
            large_logits = large_outputs.logits[:, -1, :]

        with torch.no_grad():
            small_base_outputs = small_base_model(input_ids=generated_ids, attention_mask=attention_mask)
            small_base_logits = small_base_outputs.logits[:, -1, :]

        with torch.no_grad():
            small_tuned_outputs = small_tuned_model(input_ids=generated_ids, attention_mask=attention_mask)
            small_tuned_logits = small_tuned_outputs.logits[:, -1, :]

        proxy_delta = small_tuned_logits - small_base_logits
        adjusted_logits = large_logits + alpha * proxy_delta

        probs = torch.softmax(adjusted_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

        if next_token.item() == large_tokenizer.eos_token_id:
            break

    response = large_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return response
