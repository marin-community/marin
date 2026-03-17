#!/usr/bin/env python3
"""Download Hendrycks MATH train set (minus MATH-500 test) to a fixed JSONL file.

Produces a deterministic, shuffled dataset that any agent can load identically.

Usage:
  python prepare_hendrycks_math.py --output /path/to/hendrycks_math_train.jsonl
  python prepare_hendrycks_math.py --output /path/to/hendrycks_math_train.jsonl --seed 42

Output format (one JSON per line):
  {"idx": 0, "problem": "...", "answer": "...", "prompt": "...", "config": "algebra", "split": "train"}
"""

import argparse
import json
import random
import re

from datasets import get_dataset_config_names, load_dataset


def extract_boxed(text: str) -> str:
    i = text.find("\\boxed")
    if i == -1:
        return ""
    i += 6
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != "{":
        return ""
    i += 1
    start = i
    depth = 1
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip() if depth == 0 else ""


QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def main():
    parser = argparse.ArgumentParser(description="Download Hendrycks MATH train to fixed JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    args = parser.parse_args()

    # Load MATH-500 test to exclude
    print("Loading MATH-500 test set for decontamination...")
    test_ds = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    test_problems = {ex["problem"] for ex in test_ds}
    print(f"  {len(test_problems)} MATH-500 test problems to exclude")

    # Load all Hendrycks MATH configs, both splits
    print("Loading Hendrycks MATH (all configs, train+test)...")
    configs = sorted(get_dataset_config_names("EleutherAI/hendrycks_math"))
    print(f"  Configs: {configs}")

    all_examples = []
    for cfg in configs:
        for split in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", name=cfg, split=split)
                count_before = len(ds)
                for ex in ds:
                    if ex["problem"] in test_problems:
                        continue
                    answer = extract_boxed(ex["solution"]) if "\\boxed" in ex["solution"] else ex["solution"].strip()
                    all_examples.append({
                        "problem": ex["problem"],
                        "answer": answer,
                        "prompt": ex["problem"] + QUESTION_SUFFIX,
                        "config": cfg,
                        "split": split,
                    })
                print(f"  {cfg}/{split}: {count_before} raw → {len([e for e in all_examples if e['config'] == cfg and e['split'] == split])} after decontam")
            except Exception as e:
                print(f"  {cfg}/{split}: FAILED ({e})")

    print(f"\nTotal examples before shuffle: {len(all_examples)}")

    # Deterministic shuffle
    random.seed(args.seed)
    random.shuffle(all_examples)

    # Add sequential index
    for i, ex in enumerate(all_examples):
        ex["idx"] = i

    # Write JSONL
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(all_examples)} examples to {args.output}")
    print(f"Seed: {args.seed}")
    print(f"First 3 problems:")
    for ex in all_examples[:3]:
        print(f"  [{ex['idx']}] ({ex['config']}/{ex['split']}) {ex['problem'][:80]}...")
    print(f"Last 3 problems:")
    for ex in all_examples[-3:]:
        print(f"  [{ex['idx']}] ({ex['config']}/{ex['split']}) {ex['problem'][:80]}...")


if __name__ == "__main__":
    main()
