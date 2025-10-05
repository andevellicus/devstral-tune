#!/usr/bin/env python3
"""
merge_lora.py
-------------
Merge LoRA adapters into the base model to create a standalone fine-tuned model.

Usage:
    python merge_lora.py \
        --base-model mistralai/Devstral-Small-2507 \
        --adapter-path ./output \
        --output-path ./devstral-merged \
        --tokenizer-name mistralai/Mistral-Small-3.1-24B-Base-2503
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora_adapters(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
    tokenizer_name: str = None,
    device_map: str = "auto",
    torch_dtype = torch.bfloat16
):
    """
    Merge LoRA adapters into base model and save the result.
    
    Args:
        base_model_name: HuggingFace model ID or path to base model
        adapter_path: Path to the LoRA adapter weights (output from training)
        output_path: Where to save the merged model
        tokenizer_name: Tokenizer to use (defaults to base_model_name)
        device_map: Device map for model loading
        torch_dtype: Data type for model weights
    """
    
    if tokenizer_name is None:
        tokenizer_name = base_model_name
    
    print(f"Loading base model from: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map=device_map
    )
    
    print("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    
    print(f"Loading tokenizer from: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )
    
    print(f"Saving tokenizer to: {output_path}")
    tokenizer.save_pretrained(output_path)
    
    print(f"\n✓ Successfully merged model saved to: {output_path}")
    print(f"\nYou can now load it with:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_path}')")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model"
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base model name or path (e.g., mistralai/Devstral-Small-2507)"
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to LoRA adapter weights (training output directory)"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Where to save the merged model"
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Tokenizer to use (defaults to base-model). Use for models with different tokenizer."
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map for model loading (default: auto)"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 instead of bfloat16"
    )
    
    args = parser.parse_args()
    
    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16
    
    merge_lora_adapters(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        tokenizer_name=args.tokenizer_name,
        device_map=args.device_map,
        torch_dtype=torch_dtype
    )


if __name__ == "__main__":
    main()
