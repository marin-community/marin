# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test the semantic family partition against random partitions of the same shape.

An earlier sweep compared the current three-family partition against degenerate bounds, hand-specified
alternatives and learned clusterings, and the current one held. That sweep was missing the control
that isolates *semantics* from *granularity*: random partitions with the same family sizes. Without
it, "the semantic grouping wins" is confounded with "three groups of sizes 31, 6 and 2 wins", which
would be a statement about the shrinkage structure and not about the meaning of the groups.

The comparison is a one-sided question with a natural null distribution, so it is reported as a
permutation p-value: the fraction of size-matched random partitions whose out-of-fold error is at
least as low as the semantic one. A small fraction means the semantics carry information; a large one
means only the shape of the partition mattered and the labels are decoration.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "random_matched_partitions_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
DRAWS = 40
PARTITION_SEED = 20260727


def matched_random_partition(
    sizes: tuple[int, ...],
    bucket_count: int,
    generator: np.random.Generator,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    """A random partition with exactly the given family sizes."""
    order = generator.permutation(bucket_count)
    members, start = [], 0
    for size in sizes:
        members.append(np.sort(order[start : start + size]))
        start += size
    return tuple(f"random_{index}" for index in range(len(sizes))), tuple(members)


def best_oof_rmse(dataset, dataset_id, names, members, shapes) -> float:
    """Lowest out-of-fold RMSE over the shrinkage grid, for one partition."""
    swapped = replace(dataset, family_names=names, family_members=members)
    splits = bench.split_indices(swapped, dataset_id, np.arange(swapped.n), bench.SCREEN_SEED)
    best = float("inf")
    for shape_index, shape in enumerate(shapes):
        for l2 in bench.L2_GRID:
            for residual in bench.RESIDUAL_SHRINK_GRID:
                config = bench.Config(VARIANT, shape_index, shape, l2, residual, 0.0, 0.0)
                prediction = bench.oof_prediction(swapped, config, splits)
                best = min(best, float(np.sqrt(np.mean((prediction - swapped.target) ** 2))))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-shapes", type=int, default=3)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, args.num_shapes)
    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sizes = tuple(len(members) for members in dataset.family_members)
        semantic = best_oof_rmse(dataset, dataset_id, dataset.family_names, dataset.family_members, shapes)
        generator = np.random.default_rng(PARTITION_SEED)
        draws = []
        for draw in range(args.draws):
            names, members = matched_random_partition(sizes, dataset.m, generator)
            draws.append(best_oof_rmse(dataset, dataset_id, names, members, shapes))
            rows.append({"dataset": dataset_id.value, "partition": f"random_{draw}", "rmse": draws[-1]})
        rows.append({"dataset": dataset_id.value, "partition": "semantic", "rmse": semantic})
        draws_array = np.asarray(draws)
        beats = int((draws_array <= semantic).sum())
        print(f"\n{dataset_id.value}   family sizes {sizes}")
        print(f"  semantic partition            rmse {semantic:.6f}")
        print(
            f"  {args.draws} size-matched random    rmse {draws_array.min():.6f} to {draws_array.max():.6f}  "
            f"median {np.median(draws_array):.6f}"
        )
        print(
            f"  random partitions at least as good: {beats}/{args.draws}  "
            f"-> permutation p = {(beats + 1) / (args.draws + 1):.3f}"
        )

    pd.DataFrame(rows).to_csv(args.output_dir / "random_matched_partitions.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
