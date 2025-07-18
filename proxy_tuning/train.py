from datasets import load_dataset
from trl import SFTTrainer
from config import sft_config
from load_models import load_small_base_model, load_small_tuned_model

def train_small_model():
    small_base_model, _ = load_small_base_model()
    small_tuned_model = load_small_tuned_model(small_base_model)
    
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset['train']
    
    trainer = SFTTrainer(
        model=small_tuned_model,
        train_dataset=train_data,
        args=SFTConfig(**sft_config)
    )
    trainer.train()
    trainer.save_model("./small_tuned_model")
