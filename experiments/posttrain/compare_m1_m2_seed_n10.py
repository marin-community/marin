#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Side-by-side M1 vs M2 per-point comparison on the 40-point seed at N=10.

Inputs:
    - experiments/posttrain/stage4_output/bcg_M1_seed_n10/bcg_summary.json
    - experiments/posttrain/stage4_output/bcg_M2_seed_n10/bcg_summary.json

Output: stdout table + a JSON summary file for downstream plotting.

Matches per-point on (pair_id, tension_point_idx). Computes JSR delta, BJS
delta, per-rubric marginal deltas, and classifies each point as
improved / regressed / unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_per_point(path: Path) -> dict:
    data = json.loads(path.read_text())
    return {(p["pair_id"], p["tension_point_idx"]): p for p in data["per_point"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m1",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/bcg_M1_seed_n10/bcg_summary.json"),
    )
    parser.add_argument(
        "--m2",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/bcg_M2_seed_n10/bcg_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/m2_vs_m1_seed_n10_comparison.json"),
    )
    args = parser.parse_args()

    m1 = load_per_point(args.m1)
    m2 = load_per_point(args.m2)

    shared = sorted(set(m1) & set(m2))
    only_m1 = set(m1) - set(m2)
    only_m2 = set(m2) - set(m1)
    print(f"shared points: {len(shared)}; m1-only: {len(only_m1)}; m2-only: {len(only_m2)}")

    rows = []
    for k in shared:
        p1 = m1[k]
        p2 = m2[k]
        jsr_delta = p2["joint_satisfaction_rate"] - p1["joint_satisfaction_rate"]
        bjs_delta = p2["balanced_joint_score"] - p1["balanced_joint_score"]
        a_delta = p2["mean_A_score"] - p1["mean_A_score"]
        b_delta = p2["mean_B_score"] - p1["mean_B_score"]
        weak_delta = p2["weakest_marginal_score"] - p1["weakest_marginal_score"]
        if jsr_delta > 0.05:
            bucket = "improved"
        elif jsr_delta < -0.05:
            bucket = "regressed"
        else:
            bucket = "unchanged"
        rows.append(
            {
                "pair_id": k[0],
                "tension_point_idx": k[1],
                "tension_name": p1.get("tension_name"),
                "m1_jsr": p1["joint_satisfaction_rate"],
                "m2_jsr": p2["joint_satisfaction_rate"],
                "jsr_delta": jsr_delta,
                "m1_bjs": p1["balanced_joint_score"],
                "m2_bjs": p2["balanced_joint_score"],
                "bjs_delta": bjs_delta,
                "m1_mean_A": p1["mean_A_score"],
                "m2_mean_A": p2["mean_A_score"],
                "a_delta": a_delta,
                "m1_mean_B": p1["mean_B_score"],
                "m2_mean_B": p2["mean_B_score"],
                "b_delta": b_delta,
                "m1_weakest": p1["weakest_marginal_score"],
                "m2_weakest": p2["weakest_marginal_score"],
                "weakest_delta": weak_delta,
                "bucket": bucket,
            }
        )

    # Aggregate.
    improved = [r for r in rows if r["bucket"] == "improved"]
    regressed = [r for r in rows if r["bucket"] == "regressed"]
    unchanged = [r for r in rows if r["bucket"] == "unchanged"]

    m1_jsr_mean = sum(r["m1_jsr"] for r in rows) / len(rows)
    m2_jsr_mean = sum(r["m2_jsr"] for r in rows) / len(rows)
    m1_bjs_mean = sum(r["m1_bjs"] for r in rows) / len(rows)
    m2_bjs_mean = sum(r["m2_bjs"] for r in rows) / len(rows)

    print("\n=== Aggregate ===")
    print(f"  M1 mean JSR: {m1_jsr_mean:.4f}  M2 mean JSR: {m2_jsr_mean:.4f}  Δ: {m2_jsr_mean - m1_jsr_mean:+.4f}")
    print(f"  M1 mean BJS: {m1_bjs_mean:.4f}  M2 mean BJS: {m2_bjs_mean:.4f}  Δ: {m2_bjs_mean - m1_bjs_mean:+.4f}")
    print(f"  improved: {len(improved)}   regressed: {len(regressed)}   unchanged: {len(unchanged)}")
    print(f"  improve/regress ratio: {len(improved) / max(1, len(regressed)):.2f}x")

    print("\n=== Top 15 improvements (ΔJSR > 0) ===")
    for r in sorted(rows, key=lambda x: -x["jsr_delta"])[:15]:
        print(
            f"  {r['pair_id']:<60} tp={r['tension_point_idx']}  "
            f"JSR {r['m1_jsr']:.2f} → {r['m2_jsr']:.2f} (Δ{r['jsr_delta']:+.2f})  "
            f"A Δ{r['a_delta']:+.2f}  B Δ{r['b_delta']:+.2f}"
        )

    print("\n=== Top 15 regressions (ΔJSR < 0) ===")
    for r in sorted(rows, key=lambda x: x["jsr_delta"])[:15]:
        print(
            f"  {r['pair_id']:<60} tp={r['tension_point_idx']}  "
            f"JSR {r['m1_jsr']:.2f} → {r['m2_jsr']:.2f} (Δ{r['jsr_delta']:+.2f})  "
            f"A Δ{r['a_delta']:+.2f}  B Δ{r['b_delta']:+.2f}"
        )

    out = {
        "aggregate": {
            "n_shared_points": len(shared),
            "m1_mean_jsr": m1_jsr_mean,
            "m2_mean_jsr": m2_jsr_mean,
            "jsr_delta_mean": m2_jsr_mean - m1_jsr_mean,
            "m1_mean_bjs": m1_bjs_mean,
            "m2_mean_bjs": m2_bjs_mean,
            "bjs_delta_mean": m2_bjs_mean - m1_bjs_mean,
            "n_improved": len(improved),
            "n_regressed": len(regressed),
            "n_unchanged": len(unchanged),
        },
        "per_point": rows,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nWrote comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
