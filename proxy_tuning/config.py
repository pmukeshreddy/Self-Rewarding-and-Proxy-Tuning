small_base_model_name = "google/gemma-2-2b-it"
large_model_name = "google/gemma-2-9b-it"

lora_config = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

sft_config = {
    "output_dir": "./small_tuned_checkpoint",
    "per_device_train_batch_size": 2,
    "num_train_epochs": 1,
    "learning_rate": 1e-5,
    "fp16": True,
    "gradient_accumulation_steps": 4,
    "optim": "adamw_8bit",
    "report_to": "none",
    "dataset_text_field": "question"
}
