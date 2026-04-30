#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Crossproduct assembled DPO preference pairs for Tier B.

Inputs:
    - chosens_tier_b.jsonl   (top-K chosens per variant, from gen_chosens.py select)
    - rejecteds_tier_b.jsonl (bottom-M rejecteds per variant, from build_rejecteds_from_variants.py)

Output: pairs_tier_b.jsonl with up to K*M pairs per variant — prompt-matched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--chosens", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/chosens_tier_b.jsonl")
    )
    p.add_argument(
        "--rejecteds", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/rejecteds_tier_b.jsonl")
    )
    p.add_argument("--out", type=Path, default=Path("experiments/posttrain/stage4_output/tier_b/pairs_tier_b.jsonl"))
    args = p.parse_args()

    chosens = load_jsonl(args.chosens)
    rejecteds = load_jsonl(args.rejecteds)

    # Group by (pair_id, tp, variant_idx)
    chosens_by = defaultdict(list)
    for c in chosens:
        key = (c["pair_id"], c["tension_point_idx"], c["variant_idx"])
        chosens_by[key].append(c)
    rej_by = defaultdict(list)
    for r in rejecteds:
        key = (r["pair_id"], r["tension_point_idx"], r["variant_idx"])
        rej_by[key].append(r)

    pairs = []
    summary_per_variant = []
    for key in sorted(chosens_by.keys() | rej_by.keys()):
        cs = chosens_by.get(key, [])
        rs = rej_by.get(key, [])
        summary_per_variant.append(
            {
                "pair_id": key[0],
                "tp": key[1],
                "variant_idx": key[2],
                "n_chosens": len(cs),
                "n_rejecteds": len(rs),
                "n_pairs": len(cs) * len(rs),
            }
        )
        for c in cs:
            for r in rs:
                prompt = c["variant_prompt"]
                chosen_text = c["chosen_response"]
                rejected_text = r["response"]
                h = hashlib.sha256((prompt + "|C|" + chosen_text + "|R|" + rejected_text).encode("utf-8")).hexdigest()[
                    :24
                ]
                pairs.append(
                    {
                        "pair_id": key[0],
                        "tension_point_idx": key[1],
                        "variant_idx": key[2],
                        "chosen_draw_idx": c["draw_idx"],
                        "rejected_sample_idx": r["sample_idx"],
                        "prompt": prompt,
                        "chosen_response": chosen_text,
                        "rejected_response": rejected_text,
                        "chosen_score_A": c["chosen_score_A"],
                        "chosen_score_B": c["chosen_score_B"],
                        "rejected_score_A": r["score_A"],
                        "rejected_score_B": r["score_B"],
                        "rejected_failed_side": r["failed_side"],
                        "hash": h,
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")
    with args.out.with_suffix(".summary.json").open("w") as f:
        json.dump(summary_per_variant, f, indent=2)

    pair_hist = Counter(s["n_pairs"] for s in summary_per_variant)
    print(f"Wrote {len(pairs)} preference pairs across {len(summary_per_variant)} variants to {args.out}")
    print(f"  variants with 0 pairs:  {sum(1 for s in summary_per_variant if s['n_pairs']==0)}")
    print(f"  mean pairs/variant:     {len(pairs)/max(1,len(summary_per_variant)):.2f}")
    print(f"  pairs-per-variant histogram: {dict(sorted(pair_hist.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
