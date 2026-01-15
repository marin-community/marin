#!/usr/bin/env python3
"""
Script to transform HuggingFace datasets:
1. Remove 'response_seed' column (if it exists)
2. Remove 'messages' column (if it exists)
3. Add 'conversations' column with the specified format

Usage:
    pip install datasets huggingface_hub
    huggingface-cli login  # Login with your HF token
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated

Examples:
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-16384-tokens
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated-32768-tokens --force_overwrite
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-science-qwen3-32b-annotated-32768-tokens --force_overwrite
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/open-thoughts-4-30k-code-qwen3-32b-annotated --force_overwrite
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/another-dataset
    python reformat_marin_openthoughts_data.py --dataset_name marin-community/another-dataset --response_column my_response_column
"""

import argparse
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi


def main(dataset_name: str, response_column: str, force_overwrite: bool = False):
    # Load the dataset
    print(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, split="train")
    
    print(f"Original columns: {ds.column_names}")
    print(f"Number of samples: {len(ds)}")
    
    # Check if conversations column already exists
    if "conversations" in ds.column_names:
        if not force_overwrite:
            user_input = input(f"Warning: 'conversations' column already exists in dataset '{dataset_name}'. It will be overwritten. Continue? (y/n): ")
            if user_input.lower() != 'y':
                print("QUITTING.")
                exit(0)
        else:
            print(f"Warning: 'conversations' column already exists. Overwriting due to --force_overwrite flag.")
        ds = ds.remove_columns(["conversations"])
    
    # Check if response_column exists
    if response_column not in ds.column_names:
        print(f"Error: Response column '{response_column}' not found in dataset. Available columns: {ds.column_names}")
        exit(1)
    
    # Define transform function with the specified response column
    def transform_row(example):
        """Transform a single row to add the conversations column."""
        conversation = [
            {
                "from": "human",
                "value": example["instruction_seed"]
            },
            {
                "from": "gpt",
                "value": example[response_column]
            }
        ]
        return {"conversations": conversation}
    
    # Add the conversations column
    print(f"Adding 'conversations' column (using '{response_column}' for gpt value)...")
    ds = ds.map(transform_row, desc="Adding conversations")
    
    # Remove the specified columns (only if they exist, skip silently if not)
    columns_to_remove = [col for col in ["response_seed", "messages"] if col in ds.column_names]
    if columns_to_remove:
        print(f"Removing columns: {columns_to_remove}")
        ds = ds.remove_columns(columns_to_remove)
    
    print(f"Final columns: {ds.column_names}")
    
    # Verify the transformation
    print("\nSample conversation structure:")
    sample = ds[0]["conversations"]
    print(f"  Number of turns: {len(sample)}")
    print(f"  First turn 'from': {sample[0]['from']}")
    print(f"  First turn 'value' (first 100 chars): {sample[0]['value'][:100]}...")
    print(f"  Second turn 'from': {sample[1]['from']}")
    print(f"  Second turn 'value' (first 100 chars): {sample[1]['value'][:100]}...")
    
    # Push to HuggingFace Hub
    print(f"\nPushing to HuggingFace Hub: {dataset_name}")
    ds.push_to_hub(
        dataset_name,
        private=False,  # Set to True if you want the dataset to be private
        commit_message="Transform dataset: remove response_seed and messages columns, add conversations column"
    )
    
    print(f"Done! Dataset '{dataset_name}' has been updated on HuggingFace Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform a HuggingFace dataset by removing response_seed and messages columns, and adding a conversations column."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The HuggingFace dataset name (e.g., 'marin-community/open-thoughts-4-30k-math-qwen3-32b-annotated')"
    )
    parser.add_argument(
        "--response_column",
        type=str,
        default="generated_text",
        help="The column to use for the gpt response value (default: 'generated_text')"
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="If set, automatically overwrite 'conversations' column without prompting"
    )

    args = parser.parse_args()
    main(args.dataset_name, args.response_column, args.force_overwrite)