# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the final RL-vs-SFT per-update comparison from the measurement summaries.

Reads every ``summary-*.json`` under the results prefix, keeps the as-stored
group aggregates, and renders the report: per pair x group movement, the
per-update values, and the RL/SFT ratio against the clean FP32 SFT pair.

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
GROUP_LABELS = {
    "attention": "attention projections",
    "expert_banks": "expert MLP banks",
    "shared_dense_mlp": "shared/dense MLP",
    "router": "router",
    "norm_scales": "norm scales",
    "embed_output": "embeddings + output",
    "attention_conv_gate": "attention conv/gate",
}


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
    rows = [r for doc in summaries for r in doc.get("summary", []) if r.get("lands") == "as-stored"]
    # Latest stamp wins when a pair was measured more than once.
    latest: dict[tuple, dict] = {}
    for doc in summaries:
        stamp = posixpath.basename(doc.get("_source", "")).replace("summary-", "").replace(".json", "")
        for row in doc.get("summary", []):
            if row.get("lands") != "as-stored":
                continue
            key = (row["pair"], row["group"])
            prior = latest.get(key)
            if prior is None or stamp >= prior[0]:
                latest[key] = (stamp, row)
    rows = [row for _, row in latest.values()]

    def group_index(group: str) -> int:
        return GROUP_ORDER.index(group) if group in GROUP_ORDER else 99

    print("# Mochi per-update weight movement: RL vs SFT (as-stored lands)\n")
    for stage in ("rl", "sft"):
        stage_rows = sorted((r for r in rows if r["stage"] == stage), key=lambda r: (r["pair"], group_index(r["group"])))
        if not stage_rows:
            continue
        print(f"## {stage} pairs\n")
        print("| pair | updates | group | rel_norm (stage) | per update | median |Δ|/|W| | bf16 changed |")
        print("|---|---|---|---|---|---|---|")
        for row in stage_rows:
            print(
                f"| {row['pair']} | {row['updates']} | {GROUP_LABELS.get(row['group'], row['group'])} "
                f"| {row['group_rel_norm']:.3e} | {row['group_rel_norm_per_update']:.3e} "
                f"| {row.get('group_median_rel_elem'):.3e} | {row.get('group_frac_bf16_changed'):.3f} |"
            )
        print()

    sft_by_group = {
        group: row
        for (pair, group), (_, row) in ((k, v) for k, v in ((key, val) for key, val in latest.items()))
        if pair == "sft-exp3-0to2100"
    }
    rl_pairs = sorted({pair for (pair, _) in latest if pair.startswith("rl-")})
    if rl_pairs and sft_by_group:
        print("## RL/SFT per-update ratio (per group, vs the clean FP32 SFT pair)\n")
        print("| rl pair | updates | group | rl per update | sft per update | ratio |")
        print("|---|---|---|---|---|---|")
        for pair in rl_pairs:
            for group in GROUP_ORDER:
                rl = latest.get((pair, group))
                sft = sft_by_group.get(group)
                if not rl or not sft:
                    continue
                rl_pu, sft_pu = rl["group_rel_norm_per_update"], sft["group_rel_norm_per_update"]
                if not rl_pu or not sft_pu:
                    continue
                print(
                    f"| {pair} | {rl['updates']} | {GROUP_LABELS.get(group, group)} "
                    f"| {rl_pu:.3e} | {sft_pu:.3e} | {rl_pu / sft_pu:.2f} |"
                )
            print()


if __name__ == "__main__":
    main()
