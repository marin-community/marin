#!/usr/bin/env python3
"""
Demonstration script for Tulu dataset filtering.

This script shows how to use the data filtering system to process the 
allenai/tulu-3-sft-mixture dataset and remove/replace identity branding content.

Usage:
    python experiments/filters/demo_tulu_filter.py --strategy remove
    python experiments/filters/demo_tulu_filter.py --strategy replace  
    python experiments/filters/demo_tulu_filter.py --strategy obfuscate
"""

import argparse
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets import load_dataset
from rich.progress import track
from marin.processing.data_filter import apply_data_filter
from experiments.filters.tulu_config import (
    TULU_REMOVE_FILTER,
    TULU_REPLACE_FILTER, 
    TULU_OBFUSCATE_FILTER
)

def load_tulu_sample(sample_size=None):
    """Load a sample of the Tulu dataset for demonstration."""
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    examples = []
    
    total = sample_size if sample_size else 939343
    dataset_iter = enumerate(dataset)
    
    for i, example in track(dataset_iter, description="Loading examples", total=total):
        if sample_size and i >= sample_size:
            break
        examples.append(example)
        
    return examples

def demonstrate_filtering(examples, strategy):
    """Demonstrate the filtering with the specified strategy."""
    print(f"Strategy: {strategy}")
    
    if strategy == "remove":
        filter_config = TULU_REMOVE_FILTER
    elif strategy == "replace":
        filter_config = TULU_REPLACE_FILTER
    elif strategy == "obfuscate":
        filter_config = TULU_OBFUSCATE_FILTER
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Find first few problematic examples for before/after comparison
    problematic_examples = []
    for example in examples:
        if filter_config.should_filter_example(example) and len(problematic_examples) < 5:
            problematic_examples.append(example)
        if len(problematic_examples) >= 5:
            break
    
    icon = "✓" if len(problematic_examples) > 0 else "✗"
    print(f"{icon} Found {len(problematic_examples)} problematic examples in sample of {len(examples)}")
    
    # Apply filtering
    _ = apply_data_filter(examples, filter_config)
    
    # Show before filtering
    if problematic_examples:
        print("\nBefore filtering:")
        for example in problematic_examples:
            for message in example.get('messages', []):
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    # Find and show problematic parts
                    for pattern, _, _ in filter_config._compiled_patterns:
                        match = pattern.search(content)
                        if match:
                            start = max(0, match.start() - 50)
                            end = min(len(content), match.end() + 50)
                            snippet = content[start:end].replace('\n', ' ').replace('\r', '')
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(content):
                                snippet = snippet + "..."
                            # Highlight the matched portion in red
                            match_text = match.group()
                            highlighted_snippet = snippet.replace(match_text, f"\033[91m{match_text}\033[0m")
                            print(f"  {highlighted_snippet}")
                            break
    
    # Show after filtering for replace/obfuscate strategies
    if strategy in ["replace", "obfuscate"] and problematic_examples:
        print("\nAfter filtering:")
        for orig_example in problematic_examples:
            filtered_example = filter_config.apply_filter(orig_example)
            if filtered_example:
                for message in filtered_example.get('messages', []):
                    if message.get('role') == 'assistant':
                        content = message.get('content', '')
                        # Show same region as before
                        for pattern, replacement, _ in filter_config._compiled_patterns:
                            # Find where the original match was
                            orig_content = next(msg.get('content', '') for msg in orig_example.get('messages', []) if msg.get('role') == 'assistant')
                            orig_match = pattern.search(orig_content)
                            if orig_match:
                                start = max(0, orig_match.start() - 50)
                                end = min(len(content), orig_match.start() + 100)
                                snippet = content[start:end].replace('\n', ' ').replace('\r', '')
                                if start > 0:
                                    snippet = "..." + snippet
                                if end < len(content):
                                    snippet = snippet + "..."
                                # Highlight the replacement text in green
                                if replacement and replacement in snippet:
                                    highlighted_snippet = snippet.replace(replacement, f"\033[92m{replacement}\033[0m")
                                else:
                                    highlighted_snippet = snippet
                                print(f"  {highlighted_snippet}")
                                break

def main():
    parser = argparse.ArgumentParser(description="Demonstrate Tulu dataset filtering")
    parser.add_argument(
        "--strategy", 
        choices=["remove", "replace", "obfuscate"],
        default="replace",
        help="Filtering strategy to demonstrate"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of examples to sample from dataset (default: process entire dataset)"
    )
    
    args = parser.parse_args()
    
    print("Tulu Dataset Filter Demonstration")
    
    # Load examples
    examples = load_tulu_sample(args.sample_size)
    
    print(f"\nFilter: TULU_{args.strategy.upper()}_FILTER")
    print(f"Config: experiments/filters/tulu_config.py")
    
    # Demonstrate filtering
    demonstrate_filtering(examples, args.strategy)

if __name__ == "__main__":
    main()