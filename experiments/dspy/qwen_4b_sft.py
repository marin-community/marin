
import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def train():
    print("🚀 Starting Qwen-4B Training (HF Stack)...")
    
    # 1. Data Preparation (Pipeline Step 1)
    # The pipeline expects data at this specific path.
    # If using real data, ensure 'process_data.py' has run before this.
    data_path = "experiments/dspy/format_adaptation_dataset.jsonl"
    
    if not os.path.exists(data_path):
        print("⚠️ Dataset not found at expected path. Creating dummy data for verification...")
        os.makedirs("experiments/dspy", exist_ok=True)
        dummy_data = [
            {"text": "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\nOK<|im_end|>"},
            {"text": "<|im_start|>user\nClaim: True<|im_end|>\n<|im_start|>assistant\nVerdict: True<|im_end|>"}
        ]
        with open(data_path, "w") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
    else:
        print(f"✅ Found dataset at: {data_path}")

    # 2. Model & Tokenizer Loading (Pipeline Step 2)
    # Loading Qwen-4B-Chat with 4-bit quantization for memory efficiency.
    model_id = "Qwen/Qwen1.5-4B-Chat"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"📥 Loading Model: {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config if torch.cuda.is_available() else None, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.config.use_cache = False
    
    # Prepare model for LoRA training (k-bit training)
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    # Universal fix for TRL version mismatches: set max_length on tokenizer directly
    # This avoids arg errors in both SFTTrainer and SFTConfig
    tokenizer.model_max_length = 512

    # 3. LoRA Configuration (Pipeline Step 3)
    # Targeting specific linear layers for Qwen architecture.
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM", 
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Dataset Loading & Formatting (Pipeline Step 4)
    dataset = load_dataset("json", data_files=data_path, split="train")

    def formatting_func(example):
        # Simply returns the pre-formatted text field
        return example["text"]

    # 5. Training Execution (Pipeline Step 5)
    print("🔄 Initializing Trainer...")
    
    # Using SFTConfig as required by newer TRL versions
    training_args = SFTConfig(
        output_dir="checkpoints/qwen-4b-pipeline",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=10, # Adjustable based on dataset size
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
        # max_seq_length removed to avoid version conflicts
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=training_args,
    )

    trainer.train()
    
    # 6. Saving Artifacts (Pipeline Step 6)
    output_path = "saved_models/qwen-4b-final"
    trainer.save_model(output_path)
    print(f"✅ Training Finished! Model saved to {output_path}")

if __name__ == "__main__":
    train()
