#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a tier_b variants file into a GCS-shard that bcg_probe_infer.py can consume.

Input:  tier_b/variants/train_variants_tier_b.jsonl (one row per (pair, tp) with
        a list of variant prompts).
Output: shard_00000.jsonl.gz matching the schema of
        gs://marin-us-east5/alignment/bcg_full_2573_prompts/shard_00000.jsonl.gz:
        {behavior_id, config_id, system_prompt, user_message, rubric,
         bcg_pair_id, bcg_tension_point_idx, bcg_tension_name}

Each variant becomes its own prompt row, with config_id encoding (tp, variant_idx)
as "tp<NNN>v<VV>" so the downstream stage can split them back apart.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/tier_b/variants/train_variants_tier_b.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/n10_gate/tier_b/shard_00000.jsonl.gz"))
    args = parser.parse_args()

    variants = [json.loads(l) for l in args.variants.read_text().splitlines() if l.strip()]
    rows = []
    for v in variants:
        for vi, prompt in enumerate(v["variants"]):
            rows.append(
                {
                    "behavior_id": f"bcg::{v['pair_id']}",
                    "config_id": f"tp{v['tension_point_idx']:03d}v{vi:02d}",
                    "system_prompt": "",
                    "user_message": prompt,
                    "rubric": "",
                    "bcg_pair_id": v["pair_id"],
                    "bcg_tension_point_idx": v["tension_point_idx"],
                    "bcg_tension_name": v["tension_name"],
                    "variant_idx": vi,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows)} variant prompts to {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
