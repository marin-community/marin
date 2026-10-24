# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze a dataset-order prefix using measured throughput and H100 lengths."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--throughput", type=Path, required=True)
    parser.add_argument("--h100-items", type=Path, required=True)
    parser.add_argument("--eval-minutes", type=int, default=200)
    parser.add_argument("--margin", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.eval_minutes <= 200 or not 0.20 <= args.margin < 1:
        parser.error("Use at most200 eval minutes and at least20% time margin")
    measurements = json.loads(args.throughput.read_text())
    phase = next(row for row in measurements["phases"] if row["concurrency"] == 8)
    if phase["greedy_mismatches"] or not any(int(size) > 1 and n for size, n in phase["successful_batch_sizes"].items()):
        parser.error("Require a valid warmed measurement of genuine batching")
    rate = phase["tokens_per_second"]
    rows = sorted(json.loads(args.h100_items.read_text()), key=lambda row: row["id"])
    if [row["id"] for row in rows] != list(range(500)) or rate <= 0:
        parser.error("Require all500 H100 source indices and positive measured throughput")
    budget = args.eval_minutes * 60 * (1 - args.margin)
    selected = []
    tokens = 0
    for row in rows:
        if (tokens + row["completion_tokens"]) / rate > budget:
            break
        selected.append(row)
        tokens += row["completion_tokens"]
    if not selected:
        parser.error("No complete dataset-order prefix fits the budget")
    ids = [row["id"] for row in selected]
    full_tokens = sum(row["completion_tokens"] for row in rows)
    plan = dict(
        n=len(ids),
        ids=ids,
        seed=19311,
        selection="first N by source dataset order; no duration sorting or score-dependent selection",
        eval_minutes=args.eval_minutes,
        time_margin_fraction=args.margin,
        tokens_per_second=rate,
        throughput_concurrency=8,
        server=measurements["server"],
        h100_reference_sha256=hashlib.sha256(args.h100_items.read_bytes()).hexdigest(),
        h100_full_tokens=full_tokens,
        serial_full_hours_at_13_8=full_tokens / 13.8 / 3600,
        projected_full_seconds=full_tokens / rate,
        selected_reference_tokens=tokens,
        projected_subset_seconds=tokens / rate,
        paired_h100_correct=sum(row["correct"] for row in selected),
        paired_h100_accuracy=sum(row["correct"] for row in selected) / len(selected),
        paired_h100_length_finishes=sum(row["finish_reason"] == "length" for row in selected),
        note="Length-scaled estimate, not a guaranteed runtime; retain checkpoints and report actual truncations.",
    )
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
