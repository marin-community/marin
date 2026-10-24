# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze train-only RFT prompts and build matched, prompt-disjoint datasets."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from experiments.weight_merging.playground.evaluation import prompt
from experiments.weight_merging.playground.prepare import DATASET_REVISIONS
from experiments.weight_merging.playground.training_data import encode_example, normalized, shingles


def split_responses(items, responses, tokenizer, train_count=512, validation_count=128, max_length=1024):
    """Split source questions before selecting any accepted continuations."""
    result = {}
    records = {}
    stats = {}
    for domain, dataset in (("math", "gsm8k"), ("code", "mbpp")):
        questions = [item for item in items if item["dataset"] == dataset]
        random.Random(290).shuffle(questions)
        validation_ids = {item["id"] for item in questions[:80]}
        lookup = {item["id"]: item for item in questions}
        pools = {"train": [], "validation": []}
        counts = Counter()
        for response in responses:
            if response["item_id"] not in lookup:
                continue
            counts["sampled"] += 1
            if not response["accepted"]:
                continue
            counts["verified"] += 1
            item = lookup[response["item_id"]]
            encoded = encode_example(tokenizer, prompt(item), response["completion"], max_length)
            if encoded is None:
                counts["over_length"] += 1
                continue
            split = "validation" if item["id"] in validation_ids else "train"
            pools[split].append((encoded, {"item_id": item["id"], "sample_index": response["sample_index"]}))
        result[domain], records[domain] = {}, {}
        for split, count in (("train", train_count), ("validation", validation_count)):
            random.Random(290).shuffle(pools[split])
            if len(pools[split]) < count:
                raise ValueError(f"Insufficient {domain}/{split} RFT yield: {len(pools[split])} < {count}")
            chosen = pools[split][:count]
            result[domain][split] = [x[0] for x in chosen]
            records[domain][split] = [x[1] for x in chosen]
            counts[f"eligible_{split}"] = len(pools[split])
        stats[domain] = dict(counts)
    return result, records, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--panel", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    build = sub.add_parser("build")
    build.add_argument("--tasks", type=Path, required=True)
    build.add_argument("--samples", type=Path, required=True)
    build.add_argument("--parent", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "prepare":
        forbidden = json.loads(args.panel.read_text())["items"]
        exact = {normalized(item["prompt"]) for item in forbidden}
        grams = set().union(*(shingles(item["prompt"]) for item in forbidden))
        gsm = load_dataset("openai/gsm8k", "main", split="train", revision=DATASET_REVISIONS["openai/gsm8k"])
        mbpp = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            split="train",
            revision=DATASET_REVISIONS["google-research-datasets/mbpp"],
        )
        candidates = []
        for i, row in enumerate(gsm):
            candidates.append(
                {
                    "id": f"gsm8k/train/{i}",
                    "dataset": "gsm8k",
                    "prompt": row["question"],
                    "answer": row["answer"].split("####")[-1].strip(),
                    "tests": [],
                    "setup": [],
                }
            )
        for row in mbpp:
            candidates.append(
                {
                    "id": f"mbpp/full/train/{row['task_id']}",
                    "dataset": "mbpp",
                    "prompt": row["text"],
                    "answer": row["code"],
                    "tests": row["test_list"],
                    "verification_tests": row["test_list"] + row["challenge_test_list"],
                    "setup": row.get("test_setup_code", ""),
                }
            )
        items = []
        rejected = Counter()
        for dataset, limit in (("gsm8k", 1024), ("mbpp", len(mbpp))):
            selected = [item for item in candidates if item["dataset"] == dataset]
            random.Random(290).shuffle(selected)
            seen = set()
            accepted = 0
            for item in selected:
                identity = normalized(item["prompt"])
                if identity in exact or shingles(item["prompt"]) & grams:
                    rejected[dataset] += 1
                    continue
                if identity in seen:
                    continue
                seen.add(identity)
                items.append(item)
                accepted += 1
                if accepted == limit:
                    break
        args.output.write_text(
            json.dumps(
                {
                    "items": items,
                    "dataset_revisions": DATASET_REVISIONS,
                    "test_overlap_rejected": dict(rejected),
                    "seed": 290,
                },
                indent=2,
            )
            + "\n"
        )
    else:
        tasks = json.loads(args.tasks.read_text())
        samples = json.loads(args.samples.read_text())
        tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True)
        encoded, records, stats = split_responses(tasks["items"], samples["rows"], tokenizer)
        args.output.mkdir(parents=True)
        for domain, splits in encoded.items():
            for split, rows in splits.items():
                Dataset.from_list(rows).save_to_disk(str(args.output / domain / split))
        (args.output / "records.json").write_text(json.dumps(records, indent=2) + "\n")
        metadata = {
            "recipe": "rejection-sampled self-distillation",
            "domains": stats,
            "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
            "samples_sha256": hashlib.sha256(args.samples.read_bytes()).hexdigest(),
            "parent": str(args.parent),
            "train_count": 512,
            "validation_count": 128,
            "max_length": 1024,
            "validation_prompt_count": 80,
            "seed": 290,
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
