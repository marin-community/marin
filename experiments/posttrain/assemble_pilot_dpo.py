#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine selected chosens + rejecteds into DPO preference pairs for the pilot.

Inputs:
    - chosens_pilot_10pt.jsonl  (one best chosen per (pair, tp, variant) )
    - rejecteds_pilot_10pt.jsonl (1-2 rejecteds per original pilot point,
      keyed by (pair, tp); these come from M1 N=10 responses on the ORIGINAL
      atlas prompt)

Join strategy:
    Pair every (pair, tp, variant, chosen) with every retained rejected for
    the same (pair, tp). This gives up to len(variants) * len(rejecteds)
    pairs per pilot point. The chosen is generated against the variant
    prompt; the rejected was generated against the original atlas prompt.

    NOTE: chosen and rejected in each output pair correspond to DIFFERENT
    prompts in the data we have on disk. DPO expects chosen/rejected against
    the SAME prompt. To keep the prompts matched, we emit the CHOSEN keyed
    to its variant prompt, and use the M1 rejected's OWN prompt as the
    rejected's prompt. For DPO, both entries should share the `prompt` field
    used at training time. We therefore use the VARIANT prompt as the
    canonical prompt for each pair and treat the rejected as an off-prompt
    failure-mode demonstration.

    Since the rejected was generated on the same tension corner, its
    contents still represent the failure pattern we want to train away
    from. But this is a known caveat worth flagging in the logbook.

    For a cleaner production D_tension, we would regenerate M1 responses
    on the variants themselves. That's a later extension.

Output: pilot_pairs_10pt.jsonl with records shaped for downstream merging
into the bloomv2 preference dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_chat_messages(prompt: str, response: str) -> list[dict]:
    return [
        {"role": "user", "content": prompt, "name": None, "tool_calls": None, "tool_call_id": None},
        {"role": "assistant", "content": response, "name": None, "tool_calls": None, "tool_call_id": None},
    ]


def record_hash(prompt: str, chosen_content: str, rejected_content: str) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    h.update(b"|CHOSEN|")
    h.update(chosen_content.encode("utf-8"))
    h.update(b"|REJECTED|")
    h.update(rejected_content.encode("utf-8"))
    return h.hexdigest()[:24]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chosens", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/chosens_pilot_10pt.jsonl")
    )
    parser.add_argument(
        "--rejecteds",
        type=Path,
        default=Path("experiments/posttrain/stage4_output/pilot_dpo/rejecteds_pilot_10pt.jsonl"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("experiments/posttrain/stage4_output/pilot_dpo/pilot_pairs_10pt.jsonl")
    )
    args = parser.parse_args()

    chosens = load_jsonl(args.chosens)
    rejecteds = load_jsonl(args.rejecteds)

    # Group rejecteds by (pair, tp).
    rej_by: dict[tuple, list[dict]] = {}
    for r in rejecteds:
        key = (r["pair_id"], r["tension_point_idx"])
        rej_by.setdefault(key, []).append(r)

    pairs = []
    dropped = []
    for c in chosens:
        key = (c["pair_id"], c["tension_point_idx"])
        variant_prompt = c["variant_prompt"]
        chosen_text = c["chosen_response"]
        rs = rej_by.get(key, [])
        if not rs:
            dropped.append({"reason": "no_rejected_for_point", **key, "variant_idx": c["variant_idx"]})
            continue
        for r in rs:
            rejected_text = r["response"]
            pair_record = {
                "pair_id": c["pair_id"],
                "tension_point_idx": c["tension_point_idx"],
                "variant_idx": c["variant_idx"],
                "rejected_sample_idx": r["sample_idx"],
                "prompt": variant_prompt,
                "chosen_response": chosen_text,
                "rejected_response": rejected_text,
                "chosen_score_A": c["chosen_score_A"],
                "chosen_score_B": c["chosen_score_B"],
                "rejected_score_A": r["score_A"],
                "rejected_score_B": r["score_B"],
                "rejected_failed_side": r["failed_side"],
                "coverage": r.get("coverage"),
                "tension_name": r.get("tension_name"),
                "hash": record_hash(variant_prompt, chosen_text, rejected_text),
            }
            pairs.append(pair_record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(pairs)} preference pairs to {args.out}")
    if dropped:
        print(f"Dropped {len(dropped)} chosens (no rejected available):")
        for d in dropped:
            print(f"  {d}")

    # Summary: pairs per point
    from collections import Counter

    per_point = Counter((p["pair_id"], p["tension_point_idx"]) for p in pairs)
    print("\nPairs per pilot point:")
    for (pid, tp), n in sorted(per_point.items()):
        print(f"  {pid:<55} tp={tp}  n_pairs={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
