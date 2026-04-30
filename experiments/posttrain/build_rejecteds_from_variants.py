#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build DPO rejecteds from M1-on-variants cached outputs (Tier B).

Input:
    - M1 generations on variants (generations.jsonl with prompt_id like
      "bcg::<pair_id>/tp<NNN>v<VV>", sample_idx, response_text).
    - gpt-5.1 judge scores on those generations (scores.jsonl with side A/B).

For each (pair_id, tension_point_idx, variant_idx):
    - tag each of 10 M1 samples with (score_A, score_B, joint_satisfied, failed_side, min_side).
    - keep non-joint-satisfying samples with failed_side <= MARGIN (default 5).
    - apply failure-mode heterogeneity clustering: if multiple failure modes
      (A-only / B-only / both) are present, sample across modes; else bottom-K
      from the single mode.
    - output top-K rejecteds per variant (default 5).

Output: rejecteds_tier_b.jsonl with fields ready to crossproduct with chosens.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

JUDGE_THRESHOLD = 7
MAX_FAILED_SIDE = 5
MAX_REJECTEDS_PER_VARIANT = 5


TP_VARIANT_RE = re.compile(r"tp(\d+)v(\d+)")


def parse_config_id(config_id: str) -> tuple[int, int]:
    m = TP_VARIANT_RE.match(config_id)
    if not m:
        raise ValueError(f"bad config_id: {config_id}")
    return int(m.group(1)), int(m.group(2))


def load_jsonl(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return [json.loads(line) for line in f if line.strip()]
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(score_a: int, score_b: int) -> dict:
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


def select_per_variant(samples: list[dict], max_k: int, max_failed_side: int) -> list[dict]:
    qualifying = [s for s in samples if not s["joint_satisfied"] and s["min_side_score"] <= max_failed_side]
    if not qualifying:
        return []
    by_mode: dict[str, list[dict]] = {}
    for s in qualifying:
        by_mode.setdefault(s["failed_side"], []).append(s)
    for mode in by_mode:
        by_mode[mode].sort(key=lambda x: x["min_side_score"])
    mode_order = sorted(by_mode.keys(), key=lambda m: -len(by_mode[m]))
    picked: list[dict] = []
    if len(mode_order) > 1 and max_k > 1:
        # Round-robin across modes until max_k or exhausted.
        mode_cursors = {m: 0 for m in mode_order}
        while len(picked) < max_k:
            added_this_round = False
            for mode in mode_order:
                if mode_cursors[mode] < len(by_mode[mode]) and len(picked) < max_k:
                    picked.append(by_mode[mode][mode_cursors[mode]])
                    mode_cursors[mode] += 1
                    added_this_round = True
            if not added_this_round:
                break
    else:
        only_mode = by_mode[mode_order[0]]
        picked = only_mode[:max_k]
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generations",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/tier_b/m1_variants/generations.jsonl"),
    )
    parser.add_argument(
        "--scores", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/m1_variants/scores.jsonl")
    )
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/tier_b/variants/train_variants_tier_b.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/rejecteds_tier_b.jsonl")
    )
    parser.add_argument("--max-per-variant", type=int, default=MAX_REJECTEDS_PER_VARIANT)
    parser.add_argument("--max-failed-side", type=int, default=MAX_FAILED_SIDE)
    args = parser.parse_args()

    gens = load_jsonl(args.generations)
    scores = load_jsonl(args.scores)
    variants = load_jsonl(args.variants)
    # Index prompts by (pair, tp, variant_idx)
    variant_prompts = {}
    for v in variants:
        for vi, p in enumerate(v["variants"]):
            variant_prompts[(v["pair_id"], v["tension_point_idx"], vi)] = p

    # Index scores. Score key: (pair_id, tension_point_idx, sample_idx, side).
    # BUT for variant runs we need variant_idx too — the original stage4 score schema
    # didn't carry variant, so we encode variant_idx in the sample_idx hash if needed,
    # OR the upstream call must have used a different score custom_id. For this script
    # we assume scores carry (pair_id, tension_point_idx, variant_idx, sample_idx, side).
    score_by = {}
    for s in scores:
        key = (s["pair_id"], s["tension_point_idx"], s.get("variant_idx"), s["sample_idx"], s["side"])
        score_by[key] = s

    # Group gens by (pair, tp, variant_idx)
    by_variant: dict[tuple, list[dict]] = {}
    for g in gens:
        key = (g["pair_id"], g["tension_point_idx"], g["variant_idx"])
        by_variant.setdefault(key, []).append(g)

    out = []
    summary = []
    for (pid, tp, vi), samples in sorted(by_variant.items()):
        tagged = []
        for g in samples:
            si = g["sample_idx"]
            sa = score_by.get((pid, tp, vi, si, "A"), {}).get("score")
            sb = score_by.get((pid, tp, vi, si, "B"), {}).get("score")
            if sa is None or sb is None:
                continue
            tag = classify(sa, sb)
            tagged.append(
                {
                    "pair_id": pid,
                    "tension_point_idx": tp,
                    "variant_idx": vi,
                    "sample_idx": si,
                    "response": g["response"],
                    "prompt": variant_prompts.get((pid, tp, vi), g.get("prompt", "")),
                    **tag,
                }
            )
        picked = select_per_variant(tagged, args.max_per_variant, args.max_failed_side)
        out.extend(picked)
        mode_counts = Counter(s["failed_side"] for s in tagged)
        summary.append(
            {
                "pair_id": pid,
                "tension_point_idx": tp,
                "variant_idx": vi,
                "n_samples": len(tagged),
                "n_joint_satisfied": sum(1 for s in tagged if s["joint_satisfied"]),
                "failure_modes": dict(mode_counts),
                "n_picked": len(picked),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {len(out)} rejecteds across {len(summary)} variants to {args.out}")
    print(f"  mean rejecteds/variant: {len(out)/max(1,len(summary)):.2f}")
    print(f"  variants with 0 rejecteds: {sum(1 for r in summary if r['n_picked']==0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
