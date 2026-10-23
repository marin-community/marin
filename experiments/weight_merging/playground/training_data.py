# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare matched, assistant-loss-only math and code SFT datasets."""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from experiments.weight_merging.playground.prepare import DATASET_REVISIONS

MATH_DATA = "meta-math/MetaMathQA"
CODE_DATA = "ise-uiuc/Magicoder-OSS-Instruct-75K"
SEED = 290


def normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def shingles(text: str, width: int = 13) -> set[str]:
    words = normalized(text).split()
    return {" ".join(words[start : start + width]) for start in range(len(words) - width + 1)}


def encode_example(tokenizer, question: str, answer: str, max_length: int) -> dict | None:
    messages = [{"role": "user", "content": question}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    # Tokenize the continuation independently: BPE can otherwise merge its
    # first characters into the final newline of the generation prefix.
    continuation = [*tokenizer.encode(answer.strip(), add_special_tokens=False), tokenizer.eos_token_id]
    full = prefix + continuation
    if len(full) > max_length:
        return None
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prefix) + full[len(prefix) :]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=8000)
    parser.add_argument("--validation-count", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    panel = json.loads(args.panel.read_text())["items"]
    forbidden_exact = {normalized(item["prompt"]) for item in panel}
    forbidden_shingles = set().union(*(shingles(item["prompt"]) for item in panel))
    tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True)
    revisions = {
        MATH_DATA: "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
        CODE_DATA: "5f839b1f368a76b161028bb9edff055db34022b2",
    }
    revisions.update(DATASET_REVISIONS)
    gsm = load_dataset("openai/gsm8k", "main", split="train", revision=revisions["openai/gsm8k"])
    metamath = load_dataset(MATH_DATA, split="train", revision=revisions[MATH_DATA])
    magicoder = load_dataset(CODE_DATA, split="train", revision=revisions[CODE_DATA])
    raw_math = [("gsm8k", i, r["question"], r["answer"], r["question"]) for i, r in enumerate(gsm)]
    # Combine GSM training references with a fixed MetaMath supplement.
    metamath_indices = list(range(len(metamath)))
    random.Random(SEED).shuffle(metamath_indices)
    for index in metamath_indices:
        row = metamath[index]
        answer = re.sub(r"The answer is:\s*", "#### ", row["response"])
        raw_math.append(("metamath", index, row["query"], answer, row.get("original_question", row["query"])))
        if len(raw_math) >= 30000:
            break
    raw_code = [
        ("magicoder", i, r["problem"], r["solution"], r["problem"])
        for i, r in enumerate(magicoder)
        if r["lang"] == "python"
    ]
    metadata = {
        "seed": SEED,
        "dataset_revisions": revisions,
        "parent_path": str(args.parent),
        "panel_sha256": hashlib.sha256(args.panel.read_bytes()).hexdigest(),
        "max_length": args.max_length,
        "train_count": args.train_count,
        "validation_count": args.validation_count,
        "domains": {},
    }
    args.output.mkdir(parents=True)
    for domain, rows in (("math", raw_math), ("code", raw_code)):
        random.Random(SEED).shuffle(rows)
        encoded = []
        records = []
        seen = set()
        rejected = {"test_overlap": 0, "duplicate": 0, "over_length": 0}
        for source, index, question, answer, original in rows:
            identity = normalized(original)
            if (
                identity in forbidden_exact
                or shingles(question) & forbidden_shingles
                or shingles(original) & forbidden_shingles
            ):
                rejected["test_overlap"] += 1
                continue
            if identity in seen:
                rejected["duplicate"] += 1
                continue
            seen.add(identity)
            suffix = (
                "\nSolve step by step. End with '#### <number>'."
                if domain == "math"
                else "\nReturn only a Python code block."
            )
            record = encode_example(tokenizer, question + suffix, answer, args.max_length)
            if record is None:
                rejected["over_length"] += 1
                continue
            encoded.append(record)
            records.append(
                {"dataset": source, "row": index, "question_sha256": hashlib.sha256(identity.encode()).hexdigest()}
            )
            if len(encoded) == args.train_count + args.validation_count:
                break
        if len(encoded) != args.train_count + args.validation_count:
            raise ValueError(f"Not enough eligible {domain} examples: {len(encoded)}")
        dataset = Dataset.from_list(encoded)
        dataset.select(range(args.validation_count)).save_to_disk(str(args.output / domain / "validation"))
        dataset.select(range(args.validation_count, len(dataset))).save_to_disk(str(args.output / domain / "train"))
        (args.output / domain / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        metadata["domains"][domain] = {
            "rejected": rejected,
            "supervised_tokens": sum(sum(label != -100 for label in item["labels"]) for item in encoded),
            "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
