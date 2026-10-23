# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Length diagnostics and a base-model GSM8K control with eight Q/A examples."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.weight_merging.playground.evaluation import numeric_answer, prompt
from experiments.weight_merging.playground.prepare import DATASET_REVISIONS


def fewshot_texts(items: list[dict]) -> tuple[list[str], list[int]]:
    train = load_dataset("openai/gsm8k", "main", split="train", revision=DATASET_REVISIONS["openai/gsm8k"])
    indices = random.Random(1234).sample(range(len(train)), 8)
    prefix = "\n\n".join(f"Question: {train[i]['question']}\nAnswer: {train[i]['answer']}" for i in indices)
    return [prefix + f"\n\nQuestion: {item['prompt']}\nAnswer:" for item in items], indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["length_probe", "base_fewshot"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    items = json.loads(args.panel.read_text())["items"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if args.mode == "base_fewshot":
        selected = [item for item in items if item["dataset"] == "gsm8k"]
        texts, indices = fewshot_texts(selected)
        params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, stop=["Question:"])
    else:
        rng = random.Random(290)
        selected = []
        for dataset, count in (("gsm8k", 128), ("mbpp", 32)):
            subset = [item for item in items if item["dataset"] == dataset]
            selected.extend(rng.sample(subset, count))
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt(item)}], tokenize=False, add_generation_prompt=True
            )
            for item in selected
        ]
        indices = []
        params = SamplingParams(
            temperature=0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        )
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.tokenizer),
        dtype="bfloat16",
        max_model_len=max(8192, args.max_new_tokens + 4096),
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
        swap_space=0,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    outputs = llm.generate(texts, params)
    rows = []
    for item, text, output in zip(selected, texts, outputs, strict=True):
        completion = output.outputs[0]
        row = {
            "item_id": item["id"],
            "dataset": item["dataset"],
            "prompt": text,
            "completion": completion.text,
            "tokens": list(completion.token_ids),
            "finish_reason": completion.finish_reason,
        }
        if args.mode == "base_fewshot":
            row["correct"] = numeric_answer(completion.text) == numeric_answer(item["answer"])
            row["has_answer_marker"] = "####" in completion.text
        rows.append(row)
    result = {
        "mode": args.mode,
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "fewshot_train_indices": indices,
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for dataset in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == dataset]
        lengths = sorted(len(row["tokens"]) for row in subset)
        summary = {
            "dataset": dataset,
            "n": len(subset),
            "max_tokens": args.max_new_tokens,
            "truncated": sum(row["finish_reason"] == "length" for row in subset),
            "p95_tokens": lengths[int(0.95 * (len(lengths) - 1))],
            "max_observed_tokens": lengths[-1],
        }
        if args.mode == "base_fewshot":
            summary["correct"] = sum(row["correct"] for row in subset)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
