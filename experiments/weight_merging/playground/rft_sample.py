# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample the pinned Instruct model; verify math references and execute code."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.weight_merging.playground.evaluation import prompt, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True)
    items = json.loads(args.tasks.read_text())["items"]
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt(item)}], tokenize=False, add_generation_prompt=True
        )
        for item in items
    ]
    llm = LLM(
        model=str(args.parent),
        tokenizer=str(args.parent),
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=64,
        gpu_memory_utilization=0.8,
        swap_space=0,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    params = [
        SamplingParams(
            n=4,
            temperature=0.7,
            top_p=0.95,
            max_tokens=1024,
            seed=290 + i,
            stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")],
        )
        for i in range(len(items))
    ]
    outputs = llm.generate(texts, params)
    pairs = [
        (item, j, completion)
        for item, output in zip(items, outputs, strict=True)
        for j, completion in enumerate(output.outputs)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        grades = list(
            pool.map(
                lambda pair: score(
                    {**pair[0], "tests": pair[0].get("verification_tests", pair[0]["tests"])}, pair[2].text
                ),
                pairs,
            )
        )
    rows = []
    for (item, j, completion), grade in zip(pairs, grades, strict=True):
        formatted = "####" in completion.text if item["dataset"] == "gsm8k" else "```" in completion.text
        rows.append(
            {
                "item_id": item["id"],
                "sample_index": j,
                "completion": completion.text,
                "finish_reason": completion.finish_reason,
                "tokens": len(completion.token_ids),
                "accepted": grade["correct"] and formatted and completion.finish_reason != "length",
                **grade,
            }
        )
    metadata = {
        "parent": str(args.parent),
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "n": 4,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 1024,
        "seed_policy": "290 + task index",
        "rows": rows,
    }
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"samples": len(rows), "accepted": sum(row["accepted"] for row in rows)}), flush=True)


if __name__ == "__main__":
    main()
