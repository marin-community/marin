# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Choose a junkgate cutoff on held-apart, model-assisted replay annotations."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def metrics(rows: list[dict], cutoff: float) -> dict:
    positive = [row for row in rows if row["low_value"] is True]
    negative = [row for row in rows if row["low_value"] is False]
    tp = sum(row["score"] <= cutoff for row in positive)
    fp = sum(row["score"] <= cutoff for row in negative)
    return {
        "documents": len(rows),
        "low_value": len(positive),
        "natural": len(negative),
        "true_drops": tp,
        "false_drops": fp,
        "recall": tp / len(positive) if positive else None,
        "false_drop_rate": fp / len(negative) if negative else None,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--false-drop-limit", type=float, default=0.05)
    args = parser.parse_args()
    scores_file = args.root / "replay-scores-v1.json"
    scores = {row["sample_id"]: row for row in json.loads(scores_file.read_text())["scores"]}
    originals = {}
    receipt_hashes = {}
    for path in sorted((args.root / "replay-labels-v1").glob("sample-*.json")):
        if path.name.endswith(".started.json"):
            continue
        value = json.loads(path.read_text())
        receipt_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if value["status"] == "completed":
            originals[value["sample_id"]] = value["label"]
    independent = [json.loads(line) for line in (args.root / "replay-crosscheck-v1.jsonl").read_text().splitlines()]
    disagreements = [
        row["sample_id"]
        for row in independent
        if row["sample_id"] in originals and row["low_value"] != originals[row["sample_id"]]["low_value"]
    ]
    adjudications = [json.loads(line) for line in (args.root / "replay-adjudication-v1.jsonl").read_text().splitlines()]
    labels = {**originals, **{row["sample_id"]: row for row in adjudications}}
    rows = [
        {"sample_id": index, **label, "score": scores[index]["score"]}
        for index, label in sorted(labels.items())
        if label["low_value"] is not None
    ]
    rng = random.Random(20261115)
    order = list(range(len(rows)))
    rng.shuffle(order)
    heldout = frozenset(order[: round(len(rows) * 0.2)])
    training = [row for i, row in enumerate(rows) if i not in heldout]
    validation = [row for i, row in enumerate(rows) if i in heldout]
    candidates = [-1.0, *sorted({row["score"] for row in training})]
    feasible = [cutoff for cutoff in candidates if metrics(training, cutoff)["false_drop_rate"] <= args.false_drop_limit]
    cutoff = max(feasible)
    report = {
        "annotation": (
            "Together zai-org/GLM-5.3-Flash, temperature0.7; independent GPT-6 Astra crosscheck and blinded adjudication"
        ),
        "seed": 20261115,
        "cutoff": cutoff,
        "drop_comparison": "score <= cutoff",
        "selection": "Largest calibration-set cutoff with natural-document false-drop rate <= configured limit",
        "false_drop_limit": args.false_drop_limit,
        "training_ids": [row["sample_id"] for row in training],
        "heldout_ids": [row["sample_id"] for row in validation],
        "training": metrics(training, cutoff),
        "heldout": metrics(validation, cutoff),
        "all_labeled_descriptive_only": metrics(rows, cutoff),
        "labels": len(rows),
        "abstentions_or_invalid": len(receipt_hashes) - len(rows),
        "categories": dict(Counter(label["category"] for label in labels.values())),
        "blinded_crosscheck": {"documents": len(independent), "initial_binary_disagreements": disagreements},
        "adjudicated_ids": [row["sample_id"] for row in adjudications],
        "score_file_sha256": hashlib.sha256(scores_file.read_bytes()).hexdigest(),
        "model_receipt_sha256": receipt_hashes,
        "limits": [
            "Model-assisted labels are not human ground truth.",
            "Long-document labels see head/tail only; junkgate sees complete-document BME windows.",
            "Equal cache sampling estimates macro-stratum performance, not population prevalence.",
            "The 388-stratum frozen frame has 35,255,949 rows; missing historical strata are not silently substituted.",
            "An independent 20-document crosscheck is small; descriptive and heldout uncertainty remain substantial.",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ("cutoff", "training", "heldout", "labels")}))


if __name__ == "__main__":
    main()
