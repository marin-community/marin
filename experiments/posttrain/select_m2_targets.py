#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select retained M2 tension targets from the full-atlas comparison summary.

This script turns the corrected `comparison_full.json` output into:

- a broad feasible residual candidate pool
- a review set for manual inspection
- a provisional seed slice for the first `D_tension` build

Selection logic:

- feasible oracle slice only (mechanical requirement: a joint-satisfying chosen
  response only exists where the oracle itself can solve it)
- minimum weakest marginal to stay out of the global-failure trap (points where
  M1 is failing both rubrics independently are comprehension gaps, not
  trade-off tests)
- oracle-vs-M1 JSR gap threshold to focus review on clear residuals
- rank by gap descending, then lower M1 JSR, then lower M1 BJS
- seed slice caps points per pair for training-data diversity
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DEFAULT_INPUT = Path("experiments/posttrain/stage4_output/comparison_full.json")
DEFAULT_OUTPUT_DIR = Path("experiments/posttrain/stage4_output")

DEFAULT_MIN_WEAKEST = 3.0
DEFAULT_MIN_GAP = 2 / 3
DEFAULT_REVIEW_TOP_K = 50
DEFAULT_SEED_TOP_K = 40
DEFAULT_MAX_PER_PAIR = 2


def jsr_gap(point: dict) -> float:
    return point["oracle_joint_satisfaction_rate"] - point["joint_satisfaction_rate"]


def point_sort_key(point: dict) -> tuple:
    return (
        -jsr_gap(point),
        point["joint_satisfaction_rate"],
        point["balanced_joint_score"],
        point["weakest_marginal_score"],
        point["pair_id"],
        point["tension_point_idx"],
    )


def pair_summary(points: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for point in points:
        buckets.setdefault(point["pair_id"], []).append(point)

    rows = []
    for pair_id, pair_points in buckets.items():
        pair_points = sorted(pair_points, key=point_sort_key)
        best = pair_points[0]
        rows.append(
            {
                "pair_id": pair_id,
                "n_points": len(pair_points),
                "max_gap": round(max(jsr_gap(p) for p in pair_points), 3),
                "min_m1_jsr": min(p["joint_satisfaction_rate"] for p in pair_points),
                "min_m1_bjs": round(min(p["balanced_joint_score"] for p in pair_points), 3),
                "best_tension_name": best["tension_name"],
                "best_tension_point_idx": best["tension_point_idx"],
            }
        )
    rows.sort(key=lambda row: (-row["max_gap"], row["min_m1_jsr"], row["min_m1_bjs"], row["pair_id"]))
    return rows


def build_seed_slice(
    candidates: list[dict],
    seed_top_k: int,
    max_per_pair: int,
) -> list[dict]:
    kept = []
    pair_counts: Counter[str] = Counter()
    for point in candidates:
        if pair_counts[point["pair_id"]] >= max_per_pair:
            continue
        kept.append(point)
        pair_counts[point["pair_id"]] += 1
        if len(kept) >= seed_top_k:
            break
    return kept


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_candidates_csv(path: Path, points: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "pair_id",
                "tension_point_idx",
                "tension_name",
                "oracle_jsr",
                "m1_jsr",
                "jsr_gap",
                "m1_bjs",
                "weakest_marginal",
            ]
        )
        for rank, point in enumerate(points, start=1):
            writer.writerow(
                [
                    rank,
                    point["pair_id"],
                    point["tension_point_idx"],
                    point["tension_name"],
                    point["oracle_joint_satisfaction_rate"],
                    point["joint_satisfaction_rate"],
                    round(jsr_gap(point), 3),
                    point["balanced_joint_score"],
                    point["weakest_marginal_score"],
                ]
            )


def write_pair_summary_csv(path: Path, pairs: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "n_points",
                "max_gap",
                "min_m1_jsr",
                "min_m1_bjs",
                "best_tension_name",
                "best_tension_point_idx",
            ],
        )
        writer.writeheader()
        writer.writerows(pairs)


def write_review_markdown(
    path: Path,
    candidates: list[dict],
    review_points: list[dict],
    pairs: list[dict],
    seed_slice: list[dict],
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# M2 Target Selection")
    lines.append("")
    lines.append("Generated by `experiments/posttrain/select_m2_targets.py`.")
    lines.append("")
    lines.append("## Current filter")
    lines.append("")
    lines.append("- feasible oracle slice only (`feasibility_slice == feasible`)")
    lines.append(f"- `weakest_marginal_score >= {args.min_weakest:.1f}`")
    lines.append(f"- `oracle JSR - M1 JSR >= {args.min_gap:.3f}`")
    lines.append("- ranking: JSR gap descending, then lower `M1` JSR, then lower `M1` BJS")
    lines.append("")
    lines.append("## Pool size")
    lines.append("")
    lines.append(f"- candidate points: **{len(candidates)}**")
    lines.append(f"- candidate statement pairs: **{len(pairs)}**")
    lines.append(f"- review set size: **{len(review_points)}**")
    lines.append(f"- provisional seed size: **{len(seed_slice)}**")
    lines.append("")
    lines.append("## Top statement pairs in the filtered pool")
    lines.append("")
    lines.append("| pair | n_points | max gap | min M1 JSR | min M1 BJS | best tension |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in pairs[:30]:
        lines.append(
            f"| `{row['pair_id']}` | {row['n_points']} | {row['max_gap']:.3f} | "
            f"{row['min_m1_jsr']:.3f} | {row['min_m1_bjs']:.3f} | "
            f"{row['best_tension_name'][:80]} |"
        )
    lines.append("")
    lines.append("## Review set")
    lines.append("")
    lines.append("| rank | pair | tp | tension | oracle JSR | M1 JSR | gap | M1 BJS | weakest |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for rank, point in enumerate(review_points, start=1):
        lines.append(
            f"| {rank} | `{point['pair_id']}` | {point['tension_point_idx']} | "
            f"{point['tension_name'][:80]} | {point['oracle_joint_satisfaction_rate']:.3f} | "
            f"{point['joint_satisfaction_rate']:.3f} | {jsr_gap(point):.3f} | "
            f"{point['balanced_joint_score']:.3f} | {point['weakest_marginal_score']:.3f} |"
        )
    lines.append("")
    lines.append("## Provisional seed slice")
    lines.append("")
    lines.append("| rank | pair | tp | tension | oracle JSR | M1 JSR | gap | M1 BJS | weakest |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|")
    for rank, point in enumerate(seed_slice, start=1):
        lines.append(
            f"| {rank} | `{point['pair_id']}` | {point['tension_point_idx']} | "
            f"{point['tension_name'][:80]} | {point['oracle_joint_satisfaction_rate']:.3f} | "
            f"{point['joint_satisfaction_rate']:.3f} | {jsr_gap(point):.3f} | "
            f"{point['balanced_joint_score']:.3f} | {point['weakest_marginal_score']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-weakest", type=float, default=DEFAULT_MIN_WEAKEST)
    parser.add_argument("--min-gap", type=float, default=DEFAULT_MIN_GAP)
    parser.add_argument("--review-top-k", type=int, default=DEFAULT_REVIEW_TOP_K)
    parser.add_argument("--seed-top-k", type=int, default=DEFAULT_SEED_TOP_K)
    parser.add_argument("--max-per-pair", type=int, default=DEFAULT_MAX_PER_PAIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(args.input.read_text())

    feasible_residual = [
        point
        for point in data["M1"]["per_point"]
        if point["feasibility_slice"] == "feasible"
        and point["weakest_marginal_score"] >= args.min_weakest
        and jsr_gap(point) >= args.min_gap
    ]
    feasible_residual.sort(key=point_sort_key)

    review_points = feasible_residual[: args.review_top_k]
    pairs = pair_summary(feasible_residual)
    seed_slice = build_seed_slice(feasible_residual, args.seed_top_k, args.max_per_pair)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "m2_candidate_pool.json", feasible_residual)
    write_candidates_csv(output_dir / "m2_candidate_pool.csv", feasible_residual)
    write_json(output_dir / "m2_seed_slice.json", seed_slice)
    write_pair_summary_csv(output_dir / "m2_pair_summary.csv", pairs)
    write_review_markdown(
        output_dir / "m2_target_review.md",
        feasible_residual,
        review_points,
        pairs,
        seed_slice,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
