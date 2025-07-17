from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

dpo_dataset = Dataset.from_list(preference_pairs)

dpo_config = DPOConfig(
    output_dir="./m2_checkpoint",
    per_device_train_batch_size=4,
        beta=0.1,
    num_train_epochs=1,
    learning_rate=1e-6,
    fp16=True,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    report_to="none"
)

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", device_map="auto", torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, lora_config)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=dpo_config,
    train_dataset=dpo_dataset
)
trainer.train()
trainer.save_model("./m2_model")
