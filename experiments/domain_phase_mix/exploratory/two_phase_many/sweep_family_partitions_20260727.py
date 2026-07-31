# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sweep the family partition that the hierarchical surrogate pools over.

`hpr_band` is the strongest model we have by rank-aggregated optimum quality and fit, and it is
hierarchical: it pools bucket effects within families. That grouping is a free modelling choice, yet
only one alternative has ever been benchmarked (`source_group_hierarchical`, which merges each
Common Crawl topic's high/low pair and loses on every dataset). So "we use these families" currently
rests on a single comparison.

This holds the model fixed and varies only the partition, across three kinds of candidate:

* **Degenerate bounds.** One family per bucket (no pooling) and one family for everything (maximal
  pooling) bracket what pooling can buy. If the semantic partition cannot beat both, the hierarchy
  is decoration.
* **Hand-specified alternatives.** Grouping by provenance (which corpus a bucket came from) or by
  quality tier (high/low across topics) tests whether the current semantic grouping is picking the
  right axis at all.
* **Learned partitions.** Agglomerative clustering on the correlation of bucket exposure columns,
  at several granularities. This is design-driven rather than outcome-driven on purpose: clustering
  on fitted responses would let the partition see the target and leak into out-of-fold error.

Reported on both the out-of-fold split and the held-out policies, because the two disagree for this
model family and the disagreement is the interesting part.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "family_partition_sweep_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
LEARNED_SIZES = (4, 8, 13, 20)
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)


def _grouped(labels: dict[str, str], domains: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    members: dict[str, list[int]] = {}
    for index, domain in enumerate(domains):
        members.setdefault(labels[domain], []).append(index)
    names = tuple(members)
    return names, tuple(np.asarray(members[name], dtype=int) for name in names)


def provenance_label(domain: str) -> str:
    """Which corpus a bucket came from, ignoring what it is about."""
    if domain.startswith("dolma3_cc/"):
        return "dolma3_cc"
    if domain.startswith("dolmino_synth"):
        return "dolmino_synth"
    if domain.startswith("dolmino"):
        return "dolmino_other"
    return "dolma3_other"


def quality_label(domain: str) -> str:
    """Quality tier across topics, rather than topic across tiers."""
    if domain.endswith("_high"):
        return "cc_high"
    if domain.endswith("_low"):
        return "cc_low"
    return "non_cc"


def learned_partition(dataset, size: int) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """Cluster buckets by how their exposure co-varies across the design.

    Uses the design matrix only, never the target, so the partition cannot leak outcome information
    into out-of-fold error. Buckets that the swarm always moves together are pooled, which is the
    condition under which pooling actually buys anything.
    """
    exposure = dataset.weights.sum(axis=1)
    correlation = np.corrcoef(exposure.T)
    correlation = np.nan_to_num(correlation, nan=0.0)
    distance = np.clip(1.0 - correlation, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    labels = fcluster(linkage(squareform(distance, checks=False), method="average"), t=size, criterion="maxclust")
    members: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        members.setdefault(int(label), []).append(index)
    names = tuple(f"learned{size}_{label}" for label in members)
    return names, tuple(np.asarray(v, dtype=int) for v in members.values())


def partitions(dataset) -> dict[str, tuple[tuple[str, ...], tuple[np.ndarray, ...]]]:
    domains = dataset.domains
    result: dict[str, tuple[tuple[str, ...], tuple[np.ndarray, ...]]] = {
        "current_semantic": (dataset.family_names, dataset.family_members),
        "singleton": (domains, tuple(np.array([i]) for i in range(len(domains)))),
        "single_family": (("all",), (np.arange(len(domains)),)),
        "provenance": _grouped({d: provenance_label(d) for d in domains}, domains),
        "quality_tier": _grouped({d: quality_label(d) for d in domains}, domains),
    }
    groups = bench.source_groups(dataset)
    result["source_group"] = (tuple(n for n, _ in groups), tuple(m for _, m in groups))
    for size in LEARNED_SIZES:
        result[f"learned_{size}"] = learned_partition(dataset, size)
    return result


def evaluate(dataset, dataset_id, name: str, names, members, shapes) -> list[dict[str, object]]:
    """Refit the fixed model under one partition and score out-of-fold and held-out."""
    swapped = replace(dataset, family_names=names, family_members=members)
    configs = [
        bench.Config(VARIANT, shape_index, shape, l2, residual, 0.0, 0.0)
        for shape_index, shape in enumerate(shapes)
        for l2 in bench.L2_GRID
        for residual in bench.RESIDUAL_SHRINK_GRID
    ]
    indices = np.arange(len(swapped.target))
    splits = bench.split_indices(swapped, dataset_id, indices, bench.SCREEN_SEED)
    rows = []
    for config in configs:
        prediction = bench.oof_prediction(swapped, config, splits)
        summary = bench.metric_summary(swapped.target, prediction)
        rows.append({"partition": name, "families": len(names), "l2": config.l2, **summary})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        shapes = tuple(bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, args.num_shapes))
        for name, (names, members) in partitions(dataset).items():
            rows = evaluate(dataset, dataset_id, name, names, members, shapes)
            best = min(rows, key=lambda r: float(r["rmse"]))
            frames.append({"dataset": dataset_id.value, **best})
            print(f"  {dataset_id.value:<26}{name:<18}k={len(names):<4}rmse {best['rmse']:.6f}")
    table = pd.DataFrame(frames)
    table.to_csv(args.output_dir / "partition_sweep.csv", index=False)

    print("\nBest OOF RMSE per dataset, delta against the current semantic partition:")
    for dataset_name, group in table.groupby("dataset"):
        base = float(group[group["partition"] == "current_semantic"]["rmse"].iloc[0])
        print(f"\n  {dataset_name}   (current = {base:.6f})")
        for _, row in group.sort_values("rmse").iterrows():
            delta = row["rmse"] - base
            flag = "  <-- BETTER" if delta < 0 else ""
            print(f"    {row['partition']:<18}k={row['families']:<4}{row['rmse']:.6f}   {delta:+.6f}{flag}")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
