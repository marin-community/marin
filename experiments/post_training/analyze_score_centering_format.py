# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact boxed GSM8K answers missed by a format-specific rule grader.

Example::

    python -m experiments.post_training.analyze_score_centering_format \
      --responses s3://bucket/run/exports/dumped_evals/global_step_0_evals/val-gsm8k.jsonl \
      --output /tmp/snowball-format.json

The boxed comparison is deliberately strict: it counts only the last boxed
number after the thinking turn when that number exactly matches ground truth.
It is a lower bound on correct responses, not a replacement reward rule.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.post_training.analyze_score_centering import ACCEPTED_STOPS, _filesystem, _membership_hash
from experiments.post_training.curriculum_rl.pool import boxed_answer


def summarize(responses: str, s3_endpoint: str) -> dict:
    fs, path = _filesystem(responses, s3_endpoint)
    with fs.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"no responses in {responses}")
    completed = [row for row in rows if row["stop_reason"] in ACCEPTED_STOPS]
    boxed = []
    for row in completed:
        final_turn = row["output_response"].split("<|end_think|>")[-1]
        answer = boxed_answer(final_turn)
        if answer is not None:
            boxed.append((row, answer))
    exact = [
        row
        for row, answer in boxed
        if answer.strip().replace(",", "")
        == str(row["env_extras"]["reward_spec"]["ground_truth"]).strip().replace(",", "")
    ]
    return {
        "responses": responses,
        "membership_sha256": _membership_hash(rows),
        "questions": len(rows),
        "completed": len(completed),
        "final_turn_boxed": len(boxed),
        "final_turn_boxed_exact_ground_truth": len(exact),
        "final_turn_boxed_exact_unrewarded": sum(row["score"] <= 0 for row in exact),
        "rewarded_correct": sum(row["score"] > 0 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    args = parser.parse_args()
    result = summarize(args.responses, args.s3_endpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
