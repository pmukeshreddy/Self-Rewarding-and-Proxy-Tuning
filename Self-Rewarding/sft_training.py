import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b-it",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager"
)
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def format_example(example):
    content = f"Question: {example['prompt']}\n" + "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(example['steps'])]) + f"\nFinal answer: {example['answer']}"
    chat = [{"role": "user", "content": example['prompt']}, {"role": "assistant", "content": content}]
    input_ids = tokenizer.apply_chat_template(chat, tokenize=True, max_length=512, truncation=True)
    return {'input_ids': input_ids}

train_data = ift_data[:800]
eval_data = ift_data[800:1000]

tokenized_train = [format_example(ex) for ex in train_data]
tokenized_eval = [format_example(ex) for ex in eval_data]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

args = TrainingArguments(
    output_dir="./m1_checkpoint",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    learning_rate=1e-6,
    fp16=True,
    gradient_accumulation_steps=8,
    save_strategy="epoch",
    optim="adamw_8bit",
    report_to="none",
    logging_strategy="steps",
    logging_steps=1,
    eval_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator
)
trainer.train()
trainer.save_model("./m1_model")
