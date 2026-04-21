#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build DPO rejecteds from cached M1 N=10 generations for the pilot points.

For each pilot point, loads the 10 M1 responses + their gpt-5.1 paired-rubric
scores, tags each sample with (joint_satisfied, failed_side, min_side_score),
clusters by failure mode, and selects 1-2 rejecteds meeting the margin rule:

    rejected.joint_satisfied == False
    AND rejected.min_side_score <= MAX_REJECTED_FAILED_SIDE

Failure-mode heterogeneity: within a point, if samples cluster into distinct
failure modes (e.g., A-fails vs B-fails vs both-fail), sample one rejected
per cluster so training covers the distinct failure shapes. If a single
cluster dominates, sample one rejected from it.

Output: `rejecteds_pilot_10pt.jsonl`, one record per selected rejected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

JUDGE_THRESHOLD = 7
MAX_REJECTED_FAILED_SIDE = 5  # margin rule: rejected's failed side must be clearly below threshold
MAX_REJECTEDS_PER_POINT = 2
FAILED_BOTH_WEIGHT = 1.0
FAILED_SINGLE_WEIGHT = 1.0


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify_sample(score_a: int, score_b: int) -> dict:
    joint = score_a >= JUDGE_THRESHOLD and score_b >= JUDGE_THRESHOLD
    fa = score_a < JUDGE_THRESHOLD
    fb = score_b < JUDGE_THRESHOLD
    if fa and fb:
        failed = "both"
    elif fa:
        failed = "A"
    elif fb:
        failed = "B"
    else:
        failed = "none"
    return {
        "joint_satisfied": joint,
        "failed_side": failed,
        "min_side_score": min(score_a, score_b),
        "score_A": score_a,
        "score_B": score_b,
    }


def select_rejecteds(samples_with_tags: list[dict], max_rejecteds: int) -> list[dict]:
    """Pick rejecteds honoring margin rule + failure-mode heterogeneity."""
    qualifying = [
        s for s in samples_with_tags if not s["joint_satisfied"] and s["min_side_score"] <= MAX_REJECTED_FAILED_SIDE
    ]
    if not qualifying:
        return []

    # Partition by failure mode.
    by_mode: dict[str, list[dict]] = {}
    for s in qualifying:
        by_mode.setdefault(s["failed_side"], []).append(s)

    # Sort each mode's samples by min_side_score ascending (worst first).
    for mode in by_mode:
        by_mode[mode].sort(key=lambda x: x["min_side_score"])

    # Heterogeneity rule: if multiple modes present, take one from each (up to budget).
    # Otherwise take the bottom-K from the single mode.
    picked: list[dict] = []
    mode_order = sorted(by_mode.keys(), key=lambda m: -len(by_mode[m]))
    if len(mode_order) > 1 and max_rejecteds > 1:
        for mode in mode_order[:max_rejecteds]:
            picked.append(by_mode[mode][0])
    else:
        # Single mode, or budget==1
        only_mode = by_mode[mode_order[0]]
        for s in only_mode[:max_rejecteds]:
            picked.append(s)
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-index",
        type=Path,
        default=Path("/tmp/n10_gate/pilot_10pt_index.json"),
    )
    parser.add_argument(
        "--gens",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/bcg_M1_seed_n10/generations.jsonl"),
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/bcg_M1_seed_n10/scores.jsonl"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/pilot_dpo/rejecteds_pilot_10pt.jsonl"),
    )
    parser.add_argument("--max-per-point", type=int, default=MAX_REJECTEDS_PER_POINT)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_index.read_text())

    gens = load_jsonl(args.gens)
    scores = load_jsonl(args.scores)

    # Index gens by (pair, tp, sample_idx), scores by (pair, tp, sample_idx, side).
    gen_by = {(g["pair_id"], g["tension_point_idx"], g["sample_idx"]): g for g in gens}
    score_by = {(s["pair_id"], s["tension_point_idx"], s["sample_idx"], s["side"]): s for s in scores}

    args.out.parent.mkdir(parents=True, exist_ok=True)

    out_records: list[dict] = []
    per_point_summary: list[dict] = []

    for point in pilot:
        pid = point["pair_id"]
        tp = point["tension_point_idx"]
        tagged: list[dict] = []
        for sample_idx in range(10):
            gen = gen_by.get((pid, tp, sample_idx))
            sa = score_by.get((pid, tp, sample_idx, "A"), {})
            sb = score_by.get((pid, tp, sample_idx, "B"), {})
            if gen is None or not sa or not sb:
                continue
            tag = classify_sample(sa["score"], sb["score"])
            tagged.append(
                {
                    "pair_id": pid,
                    "tension_point_idx": tp,
                    "sample_idx": sample_idx,
                    "response": gen["response"],
                    "prompt": gen["prompt"],
                    "judge_explanation_A": sa.get("explanation", ""),
                    "judge_explanation_B": sb.get("explanation", ""),
                    **tag,
                }
            )

        picked = select_rejecteds(tagged, args.max_per_point)
        mode_counts = Counter(s["failed_side"] for s in tagged)
        per_point_summary.append(
            {
                "pair_id": pid,
                "tension_point_idx": tp,
                "tension_name": point["tension_name"],
                "coverage": point.get("coverage"),
                "n_samples": len(tagged),
                "n_joint_satisfied": sum(1 for s in tagged if s["joint_satisfied"]),
                "failure_modes": dict(mode_counts),
                "n_qualifying_rejecteds": sum(
                    1 for s in tagged if not s["joint_satisfied"] and s["min_side_score"] <= MAX_REJECTED_FAILED_SIDE
                ),
                "n_selected": len(picked),
                "selected_sample_indices": [p["sample_idx"] for p in picked],
                "selected_min_side_scores": [p["min_side_score"] for p in picked],
            }
        )
        for p in picked:
            out_records.append(
                {
                    **p,
                    "tension_name": point["tension_name"],
                    "coverage": point.get("coverage"),
                }
            )

    with args.out.open("w") as f:
        for rec in out_records:
            f.write(json.dumps(rec) + "\n")

    summary_path = args.out.parent / "rejecteds_pilot_10pt_summary.json"
    summary_path.write_text(json.dumps(per_point_summary, indent=2))

    print(f"Wrote {len(out_records)} rejected records to {args.out}")
    print(f"Wrote per-point summary to {summary_path}")
    print()
    for row in per_point_summary:
        print(
            f"  {row['pair_id'][:55]:<55} tp={row['tension_point_idx']}  "
            f"modes={row['failure_modes']}  qualify={row['n_qualifying_rejecteds']}/10  "
            f"picked={row['n_selected']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
