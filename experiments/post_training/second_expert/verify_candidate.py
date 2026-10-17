# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Independently verify MMLU-Pro provenance, grouped selection and native SFT masks."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from rigging.filesystem.storage_path import StoragePath
from transformers import AutoTokenizer

from experiments.post_training.cruxeval_transfer.constants import STUDENT_MODEL as MODEL
from experiments.post_training.cruxeval_transfer.constants import STUDENT_REVISION as MODEL_REVISION
from experiments.post_training.cruxeval_transfer.rendering import token_counts, training_row
from experiments.post_training.second_expert.constants import (
    CANONICAL,
    CANONICAL_SHA256,
    CAPTURE,
    DATASET_REVISION,
    SUBSET_SHA256,
)
from experiments.post_training.second_expert.grading import extracted_answer


def rendered_prompt(row: dict, base: str) -> str:
    options = [option for option in row["options"] if option != "N/A"]
    block = "".join(f"{chr(65 + index)}. {option}\n" for index, option in enumerate(options))
    return (
        base.replace("{$}", row["category"])
        + "\nQuestion:\n"
        + row["question"]
        + "\nOptions:\n"
        + block
        + "Answer: Let's think step by step."
    )


def indexed_rows(content: bytes) -> dict[str, dict]:
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    indexed = {row["custom_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate custom_id")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Verification artifacts are immutable; use a new prefix")
    canonical_bytes = (StoragePath(CANONICAL) / "canonical.jsonl").read_bytes()
    subset_bytes = (StoragePath(CANONICAL) / "subset-groups-1600-seed20261009.json").read_bytes()
    if hashlib.sha256(canonical_bytes).hexdigest() != CANONICAL_SHA256:
        raise ValueError("Canonical artifact changed")
    if hashlib.sha256(subset_bytes).hexdigest() != SUBSET_SHA256:
        raise ValueError("Subset artifact changed")
    canonical = indexed_rows(canonical_bytes)
    subset = json.loads(subset_bytes)
    if subset["revision"] != DATASET_REVISION or subset["canonical_sha256"] != CANONICAL_SHA256:
        raise ValueError("Subset provenance mismatch")
    source_hashes = {}
    source = {}
    for name in ("requests.jsonl", "raw-responses.jsonl", "audit.jsonl"):
        content = (StoragePath(CAPTURE) / name).read_bytes()
        source_hashes[name] = hashlib.sha256(content).hexdigest()
        source[name] = indexed_rows(content)
        if set(source[name]) != set(canonical) or len(source[name]) != 12032:
            raise ValueError(f"Source IDs do not reconcile: {name}")
    base = Path(__file__).with_name("initial_prompt.txt").read_text()
    source_rows = list(load_dataset("TIGER-Lab/MMLU-Pro", revision=DATASET_REVISION, split="test"))
    by_prompt = defaultdict(list)
    for row in source_rows:
        by_prompt[rendered_prompt(row, base)].append(row)
    identity_mismatches = []
    for identity, record in canonical.items():
        messages = source["requests.jsonl"][identity]["b_request"]["messages"]
        matches = by_prompt.get(messages[0]["content"], [])
        candidates = sorted(row["question_id"] for row in matches)
        recorded = [record["question_id"]]
        if candidates != recorded:
            identity_mismatches.append({"custom_id": identity, "exact": candidates, "recorded": recorded})
    if identity_mismatches:
        raise ValueError("Full-prompt identity mismatches: " + json.dumps(identity_mismatches))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    groups = defaultdict(list)
    token_audit = []
    probes = []
    counts = Counter()
    selected_ids = set(subset["custom_ids"])
    for identity, record in canonical.items():
        request = source["requests.jsonl"][identity]
        messages = request["b_request"]["messages"]
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError(f"Unexpected prompt shape: {identity}")
        matches = by_prompt.get(messages[0]["content"], [])
        if not matches or len({(row["answer"], row["category"]) for row in matches}) != 1:
            raise ValueError(f"Full rendered prompt lacks a label-equivalent exact match: {identity}")
        matched = min(matches, key=lambda row: row["question_id"])
        candidates = sorted(row["question_id"] for row in matches)
        if [record["question_id"]] != candidates:
            raise ValueError(
                f"Canonical source IDs differ: {identity}: exact={candidates}, " f"recorded={[record['question_id']]}"
            )
        group = hashlib.sha1(matched["question"].encode()).hexdigest()[:16]
        if (record["canonical_gold"], record["category"], record["group"]) != (
            matched["answer"],
            matched["category"],
            group,
        ):
            raise ValueError(f"Canonical labels or grouping differ: {identity}")
        groups[group].append(record)
        raw = source["raw-responses.jsonl"][identity]
        if raw["status_code"] != 200:
            raise ValueError(f"Unexpected failed source response: {identity}")
        choice = raw["body"]["choices"][0]
        message = choice["message"]
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        pred = extracted_answer(message.get("content") or "")
        correct = pred == matched["answer"]
        usable = correct and choice["finish_reason"] == "stop" and bool(reasoning.strip())
        if (pred, correct, usable) != (record["pred"], record["correct_canonical"], record["usable_canonical"]):
            raise ValueError(f"Raw-response grading does not reproduce canonical audit: {identity}")
        counts["correct"] += correct
        counts["usable"] += usable
        if not usable:
            continue
        rendered = training_row(
            {
                "id": identity,
                "problem_id": str(matched["question_id"]),
                "messages": [
                    *messages,
                    {"role": "assistant", "reasoning_content": reasoning, "content": message["content"]},
                ],
            }
        )
        total, supervised = token_counts(rendered, tokenizer)
        if total > 32768:
            raise ValueError(f"Exact native training row exceeds context: {identity}: {total}")
        token_audit.append(
            {
                "custom_id": identity,
                "selected": identity in selected_ids,
                "group": group,
                "category": matched["category"],
                "tokens": total,
                "supervised_tokens": supervised,
            }
        )
        if identity in selected_ids and len(probes) < 4:
            encoded = tokenizer.apply_chat_template(
                rendered["messages"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                **rendered["chat_template_kwargs"],
            )
            probes.append(rendered | {"input_ids": encoded["input_ids"], "assistant_mask": encoded["assistant_masks"]})
    group_categories = {
        group: Counter(row["category"] for row in rows).most_common(1)[0][0] for group, rows in groups.items()
    }
    category_counts = Counter(row["category"] for row in canonical.values())
    rng = random.Random(subset["seed"])
    reproduced = set()
    for category in sorted(category_counts):
        ordered = sorted(group for group in groups if group_categories[group] == category)
        rng.shuffle(ordered)
        quota = subset["target_docs"] * category_counts[category] / len(canonical)
        count = 0
        for group in ordered:
            if count >= quota:
                break
            reproduced.add(group)
            count += len(groups[group])
    reproduced_ids = {row["custom_id"] for group in reproduced for row in groups[group]}
    if reproduced != set(subset["groups"]) or reproduced_ids != selected_ids:
        raise ValueError("Grouped draw cannot be reproduced independently")
    selected = [row for row in token_audit if row["selected"]]
    report = {
        "canonical_sha256": CANONICAL_SHA256,
        "subset_sha256": SUBSET_SHA256,
        "source_hashes": source_hashes,
        "dataset_revision": DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "template_sha256": hashlib.sha256(tokenizer.get_chat_template().encode()).hexdigest(),
        "counts": dict(counts),
        "selected_docs": len(selected_ids),
        "selected_usable": len(selected),
        "held_out_docs": len(canonical) - len(selected_ids),
        "groups": len(groups),
        "multi_category_groups": sum(len({row["category"] for row in rows}) > 1 for rows in groups.values()),
        "all_usable_supervised_tokens": sum(row["supervised_tokens"] for row in token_audit),
        "selected_supervised_tokens": sum(row["supervised_tokens"] for row in selected),
        "selected_input_tokens": sum(row["tokens"] for row in selected),
        "longest_selected_row": max(row["tokens"] for row in selected),
        "longest_usable_row": max(row["tokens"] for row in token_audit),
        "selection_status": "candidate verified; no training data materialized or launch authorized",
    }
    output.mkdirs()
    for name, rows in (("token-audit.jsonl", token_audit), ("row-probes.jsonl", probes)):
        (output / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print("CANDIDATE_VERIFIED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
