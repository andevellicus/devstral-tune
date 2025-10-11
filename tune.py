#!/usr/bin/env python3
"""
finetune_devstral.py
--------------------
QLoRA fine-tuning for Devstral on statistical analysis traces.

Usage:
    python finetune_devstral.py \
        --train-data organized_traces/splits/train.jsonl \
        --val-data organized_traces/splits/val.jsonl \
        --model-name mistralai/Devstral-small-2505 \
        --tokenizer-name mistralai/Mistral-Small-3.1-24B-Base-2503 \
        --output-dir ./devstral-finetuned \
        --epochs 3 \
        --batch-size 1 \
        --gradient-accumulation 8 \
        --max-length 16384 \
        --lora-r 128 \
        --lora-alpha 16 \/
        --use-flash-attention
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_conversation(messages: List[Dict], tokenizer, max_length: int = 8192) -> Dict:
    """
    Format conversation for training with loss only on assistant responses.
    Preserves XML structure (<python>, <memory>, etc.) from traces.
    """
    input_ids = []
    labels = []
    
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "").strip()
        
        if not content:
            continue
        
        if role == "system":
            # System prompt: include in context, no loss
            if i == 0:  # First message
                tokens = tokenizer.encode(f"<s>[INST] {content} [/INST]", add_special_tokens=False)
            else:
                tokens = tokenizer.encode(f"[INST] {content} [/INST]", add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))
            
        elif role == "user":
            # User message: include in context, no loss
            tokens = tokenizer.encode(f"[INST] {content} [/INST]", add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))
            
        elif role == "tool":
            # Tool execution results: include in context, no loss
            tokens = tokenizer.encode(f"\n{content}\n", add_special_tokens=False)
            input_ids.extend(tokens)
            labels.extend([-100] * len(tokens))
            
        elif role == "assistant":
            # Assistant response: compute loss on this
            # Preserve XML structure like <python>...</python>
            tokens = tokenizer.encode(content, add_special_tokens=False)
            
            # Add EOS if this is the last message or before a user message
            is_last = (i == len(messages) - 1)
            next_is_user = (i + 1 < len(messages) and messages[i + 1].get("role") == "user")
            
            if is_last or next_is_user:
                tokens.append(tokenizer.eos_token_id)
            
            input_ids.extend(tokens)
            labels.extend(tokens)  # Train on assistant outputs
    
    # Truncate if needed
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids)
    }


def preprocess_function(examples: Dict, tokenizer, max_length: int = 8192) -> Dict:
    """Tokenize and prepare training examples."""
    all_input_ids = []
    all_labels = []
    all_attention_masks = []
    
    for messages in examples["messages"]:
        formatted = format_conversation(messages, tokenizer, max_length)
        all_input_ids.append(formatted["input_ids"])
        all_labels.append(formatted["labels"])
        all_attention_masks.append(formatted["attention_mask"])
    
    return {
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_masks
    }


def create_qlora_config(
    r: int = 64,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None
) -> LoraConfig:
    """Create QLoRA configuration."""
    if target_modules is None:
        # Default target modules for Mistral-based models
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )


def load_model_and_tokenizer(
    model_name: str,
    tokenizer_name: Optional[str] = None,
    use_4bit: bool = True,
    use_flash_attention: bool = True
):
    """Load model and tokenizer with quantization."""
    
    # BitsAndBytes config for 4-bit quantization
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
    else:
        bnb_config = None
    
    # Use separate tokenizer if specified (e.g., base model tokenizer for fine-tuned models)
    if tokenizer_name is None:
        tokenizer_name = model_name
    
    print(f"Loading tokenizer from: {tokenizer_name}")
    
    # Load tokenizer - should work fine with the base model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Load model
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "trust_remote_code": True
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    print(f"Loading model from: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    # Prepare model for k-bit training
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer


@dataclass
class CustomDataCollator(DataCollatorForLanguageModeling):
    """Custom collator that handles variable-length sequences with proper padding."""
    
    def __call__(self, features):
        # Extract sequences
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for ids, lbls in zip(input_ids, labels):
            pad_len = max_len - len(ids)
            
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
            padded_labels.append(lbls + [-100] * pad_len)  # -100 is ignore index
            attention_masks.append([1] * len(ids) + [0] * pad_len)
        
        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Training JSONL file")
    parser.add_argument("--val-data", required=True, help="Validation JSONL file")
    parser.add_argument("--model-name", default="mistralai/Mistral-Small-3.1-24B", 
                       help="Base model name (e.g., mistralai/Devstral-Small-2507)")
    parser.add_argument("--tokenizer-name", default=None,
                       help="Tokenizer model name (defaults to model-name). Use base model for fine-tuned models.")
    parser.add_argument("--output-dir", default="./devstral-finetuned", 
                       help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--use-flash-attention", action="store_true", 
                       help="Use Flash Attention 2")
    args = parser.parse_args()
    
    print("Loading data...")
    train_data = load_jsonl(args.train_data)
    val_data = load_jsonl(args.val_data)
    
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        tokenizer_name=args.tokenizer_name,
        use_flash_attention=args.use_flash_attention
    )
    
    print("Applying QLoRA...")
    lora_config = create_qlora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )
    
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )
    
    # Custom data collator that preserves -100 labels
    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=["tensorboard"],
        logging_dir=f"{args.output_dir}/logs",
        early_stopping_patience=2,
        early_stopping_threshold=0.01,
        max_grad_norm=1.0
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()



