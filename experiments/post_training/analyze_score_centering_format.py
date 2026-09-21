# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit exact boxed answers missed by format-specific rule graders.

Example::

    python -m experiments.post_training.analyze_score_centering_format \
      --responses s3://bucket/run/exports/dumped_evals/global_step_0_evals/val-gsm8k.jsonl \
      --output /tmp/snowball-format.json

The default GSM8K comparison ignores number commas. Exact-string mode is a
conservative Math500 diagnostic: LaTeX-equivalent strings may still differ.
The terminal-boxed subset further
requires the box to end the final turn, avoiding an earlier correct box
followed by a different final answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.post_training.analyze_score_centering import ACCEPTED_STOPS, _filesystem, _membership_hash
from experiments.post_training.curriculum_rl.pool import boxed_answer


def summarize(responses: str, s3_endpoint: str, match_mode: str = "numeric") -> dict:
    if match_mode not in ("numeric", "exact_string"):
        raise ValueError(f"unknown match mode: {match_mode}")
    fs, path = _filesystem(responses, s3_endpoint)
    with fs.open(path, "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"no responses in {responses}")
    completed = [row for row in rows if row["stop_reason"] in ACCEPTED_STOPS]
    boxed = []
    terminal_boxed = []
    for row in completed:
        final_turn = row["output_response"].split("<|end_think|>")[-1]
        answer = boxed_answer(final_turn)
        if answer is not None:
            boxed.append((row, answer))
            start = final_turn.rfind("\\boxed{") + len("\\boxed")
            depth = 0
            for index in range(start, len(final_turn)):
                if final_turn[index] == "{":
                    depth += 1
                elif final_turn[index] == "}":
                    depth -= 1
                    if depth == 0:
                        if final_turn[index + 1 :].strip() in ("", "<|eot_id|>", "<|im_end|>"):
                            terminal_boxed.append((row, answer))
                        break

    def is_exact(row: dict, answer: str) -> bool:
        truth = str(row["env_extras"]["reward_spec"]["ground_truth"])
        if match_mode == "numeric":
            return answer.strip().replace(",", "") == truth.strip().replace(",", "")
        return answer.strip() == truth.strip()

    exact = [row for row, answer in boxed if is_exact(row, answer)]
    terminal_exact = [row for row, answer in terminal_boxed if is_exact(row, answer)]
    rewarded_completed = sum(row["score"] > 0 for row in completed)
    exact_unrewarded = sum(row["score"] <= 0 for row in exact)
    terminal_exact_unrewarded = sum(row["score"] <= 0 for row in terminal_exact)
    return {
        "responses": responses,
        "match_mode": match_mode,
        "membership_sha256": _membership_hash(rows),
        "questions": len(rows),
        "completed": len(completed),
        "final_turn_boxed": len(boxed),
        "final_turn_boxed_exact_ground_truth": len(exact),
        "final_turn_boxed_exact_unrewarded": exact_unrewarded,
        "terminal_boxed_exact_ground_truth": len(terminal_exact),
        "terminal_boxed_exact_unrewarded": terminal_exact_unrewarded,
        "rewarded_correct": sum(row["score"] > 0 for row in rows),
        "rewarded_correct_completed": rewarded_completed,
        "completed_rewarded_or_exact_boxed": rewarded_completed + exact_unrewarded,
        "completed_rewarded_or_terminal_boxed": rewarded_completed + terminal_exact_unrewarded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--s3-endpoint", default="https://cwobject.com")
    parser.add_argument("--match-mode", choices=("numeric", "exact_string"), default="numeric")
    args = parser.parse_args()
    result = summarize(args.responses, args.s3_endpoint, args.match_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
