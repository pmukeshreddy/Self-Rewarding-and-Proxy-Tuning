from load_models import load_small_base_model, load_small_tuned_model, load_large_model
from generate import proxy_generate

# Uncomment to train
# from train import train_small_model
# train_small_model()

small_base_model, _ = load_small_base_model()
small_tuned_model = load_small_tuned_model(small_base_model)
large_model, large_tokenizer = load_large_model()

test_prompt = "Solve: 2 + 2 = ?"
response = proxy_generate(large_model, large_tokenizer, small_base_model, small_tuned_model, test_prompt)
print(response)
