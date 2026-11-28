# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the per-group comparison table from measurement summary files.

Reads ``summary-*.json`` outputs under the results prefix and renders the report
table: per pair x group movement, per-update values, and the RL/SFT ratio.

Usage (anywhere with read access to the results prefix):
  python experiments/active/mochi-update-magnitude/report.py \
      --results s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude/results
"""

from __future__ import annotations

import argparse
import json
import posixpath

import fsspec

GROUP_ORDER = [
    "attention",
    "expert_banks",
    "shared_dense_mlp",
    "router",
    "norm_scales",
    "embed_output",
    "attention_conv_gate",
]


def load_summaries(results_prefix: str) -> list[dict]:
    fs, _ = fsspec.url_to_fs(results_prefix)
    out = []
    for entry in sorted(fs.ls(results_prefix, detail=False)):
        name = posixpath.basename(entry)
        if name.startswith("summary-") and name.endswith(".json"):
            with fs.open(entry, "rb") as handle:
                out.append(json.load(handle))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="s3://marin-us-east-08a/marin/users/benfeuer/mochi-update-magnitude/results"
    )
    args = parser.parse_args()

    summaries = load_summaries(args.results)
    rows = [row for doc in summaries for row in doc.get("summary", []) if row.get("lands") == "as-stored"]
    by_stage: dict[str, list[dict]] = {}
    for row in rows:
        by_stage.setdefault(row["stage"], []).append(row)

    print("# RL vs SFT per-update weight movement (as-stored, group-level)\n")
    for stage in ("rl", "sft"):
        stage_rows = by_stage.get(stage, [])
        if not stage_rows:
            continue
        print(f"## {stage}\n")
        print("| pair | updates | group | rel_norm | per update | median elem rel | bf16 changed |")
        print("|---|---|---|---|---|---|---|")
        for row in sorted(
            stage_rows, key=lambda r: (r["pair"], GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99)
        ):
            print(
                f"| {row['pair']} | {row['updates']} | {row['group']} "
                f"| {row['group_rel_norm']:.3e} | {row['group_rel_norm_per_update']:.3e} "
                f"| {row.get('group_median_rel_elem')} | {row.get('group_frac_bf16_changed')} |"
            )
        print()

    # RL/SFT ratio per group, pairing each RL pair against the clean FP32 SFT pair.
    sft_rows = {(r["group"], r["updates"]): r for r in by_stage.get("sft", [])}
    rl_rows = by_stage.get("rl", [])
    if rl_rows and sft_rows:
        print("## RL/SFT per-update ratio (vs sft-exp3-0to2100)\n")
        print("| rl pair | group | rl per update | sft per update | ratio |")
        print("|---|---|---|---|---|")
        sft_by_group = {group: row for (group, updates), row in sft_rows.items() if updates == 2100}
        for row in sorted(
            rl_rows, key=lambda r: (r["pair"], GROUP_ORDER.index(r["group"]) if r["group"] in GROUP_ORDER else 99)
        ):
            sft = sft_by_group.get(row["group"])
            if sft and row.get("group_rel_norm_per_update") and sft.get("group_rel_norm_per_update"):
                ratio = row["group_rel_norm_per_update"] / sft["group_rel_norm_per_update"]
                print(
                    f"| {row['pair']} | {row['group']} | {row['group_rel_norm_per_update']:.3e} "
                    f"| {sft['group_rel_norm_per_update']:.3e} | {ratio:.2f} |"
                )
        print()


if __name__ == "__main__":
    main()
