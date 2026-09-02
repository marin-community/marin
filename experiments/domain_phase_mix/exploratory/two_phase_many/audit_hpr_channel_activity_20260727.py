# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether HPR's named mechanisms are active over the exposures the panel actually realizes.

The audit says the benefit link eventually outgrows the replay penalty. Scanning the links showed
something stronger: the penalty is a squared softplus of ``log1p(exposure) - tau``, so with a selected
threshold near 5.1 it stays flat until exposure exceeds roughly 170. If realized exposures sit far
below that, the replay channel the promoted variant is named after contributes nothing and its fitted
coefficient is unidentified rather than small.

Also checks two invariances a mechanistic model should have and this one may not: a tied policy's
prediction should not depend on where the phase boundary was drawn, and no two design columns should
be identical. The duplicate check is run under the singleton partition as well, because singleton
families are where the family and member ledgers coincide.
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

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "hpr_channel_activity_20260727"
VARIANT = bench.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.THREE_HUNDRED_M_TABLE9,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
PROMOTED_SHAPE = bench.family_grp.Shape(
    exponent=0.33989885260566105,
    late_multiplier=6.627794351309641,
    forgetting_rate=6.14421235332821e-06,
    penalty_threshold=5.136810831800622,
)
HARM_PREFIXES = ("family_overexposure", "family_member_replay", "family_excess_replay")
DEAD_COLUMN_TOLERANCE = 1e-10
BOUNDARY_FRACTIONS = (0.5, 0.6, 0.7, 0.8, 0.9)


def channel_activity(dataset, config) -> list[dict[str, object]]:
    """Per design column: range and whether it varies at all across the panel's rows."""
    design = bench.build_design(dataset, config)
    rows = []
    for index, name in enumerate(design.names):
        column = design.values[:, index]
        spread = float(column.max() - column.min())
        rows.append(
            {
                "column": name,
                "min": float(column.min()),
                "max": float(column.max()),
                "spread": spread,
                "dead": spread < DEAD_COLUMN_TOLERANCE,
            }
        )
    return rows


def boundary_invariance(dataset, config) -> dict[str, float]:
    """How much a tied policy's predicted exposure moves when only the phase boundary changes.

    The policy is held fixed and tied across phases, so the trained model is identical; only the
    bookkeeping split of tokens between phases varies. A boundary-free policy whose prediction moves
    with the boundary means the transition parameters are absorbing the split.
    """
    tied = bench.proportional_weights(dataset)
    stacked = np.stack([tied, tied], axis=0)[None, :, :]
    total = float(np.mean(dataset.c0 + dataset.c1))
    sums = []
    for fraction in BOUNDARY_FRACTIONS:
        probe = replace(
            dataset,
            weights=stacked,
            target=np.zeros(1, dtype=float),
            c0=np.full(dataset.m, fraction * total),
            c1=np.full(dataset.m, (1.0 - fraction) * total),
        )
        sums.append(float(bench.retained_exposure(probe, config.shape).sum()))
    return {
        "tied_exposure_min": min(sums),
        "tied_exposure_max": max(sums),
        "tied_relative_swing": (max(sums) - min(sums)) / max(min(sums), 1e-12),
    }


def singleton_duplicates(dataset, config) -> int:
    """Identical column pairs once every bucket is its own family."""
    singleton = replace(
        dataset,
        family_names=dataset.domains,
        family_members=tuple(np.array([index]) for index in range(dataset.m)),
    )
    design = bench.build_design(singleton, config)
    scale = np.maximum(np.abs(design.values).max(axis=0), 1e-30)
    normalized = design.values / scale[None, :]
    pairs = 0
    for left in range(design.values.shape[1]):
        for right in range(left + 1, design.values.shape[1]):
            if float(np.abs(normalized[:, left] - normalized[:, right]).max()) < DEAD_COLUMN_TOLERANCE:
                pairs += 1
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = bench.Config(VARIANT, 0, PROMOTED_SHAPE, 0.0, 1.0, 0.0, 0.0)
    frames = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        exposure = bench.retained_exposure(dataset, PROMOTED_SHAPE)
        family_total = np.column_stack([exposure[:, m].sum(axis=1) for m in dataset.family_members])
        onset = float(np.expm1(PROMOTED_SHAPE.penalty_threshold))
        activity = channel_activity(dataset, config)
        harm = [row for row in activity if str(row["column"]).startswith(HARM_PREFIXES)]
        dead_harm = sum(1 for row in harm if row["dead"])
        print(f"\n{dataset_id.value}")
        print(
            f"  bucket exposure   : min {exposure.min():.4g}  median {np.median(exposure):.4g}  max {exposure.max():.4g}"
        )
        print(f"  family total      : min {family_total.min():.4g}  max {family_total.max():.4g}")
        print(f"  replay harm onset : exposure > {onset:.4g}  (threshold tau={PROMOTED_SHAPE.penalty_threshold:.3f})")
        print(f"  harm columns dead : {dead_harm}/{len(harm)}")
        for row in harm:
            flag = "  DEAD" if row["dead"] else ""
            print(f"    {row['column']:<44} spread {row['spread']:.3e}{flag}")
        invariance = boundary_invariance(dataset, config)
        print(
            f"  tied policy under boundaries {BOUNDARY_FRACTIONS}: exposure "
            f"{invariance['tied_exposure_min']:.4g} to {invariance['tied_exposure_max']:.4g}  "
            f"(swing {invariance['tied_relative_swing'] * 100:.1f}%)"
        )
        duplicates = singleton_duplicates(dataset, config)
        print(f"  identical column pairs under the singleton partition: {duplicates}")
        for row in activity:
            frames.append({"dataset": dataset_id.value, **row})

    print("\n" + "=" * 96)
    print("HARM ONSET ACROSS THE SHAPE GRID: how many candidate shapes put the replay onset")
    print("above the largest exposure the panel realizes, making the channel inert")
    print("=" * 96)
    shapes = bench.family_grp.shape_candidates(bench.family_grp.Variant.BUCKET_RESOLVED, 64)
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        inert = 0
        for shape in shapes:
            exposure = bench.retained_exposure(dataset, shape)
            if float(np.expm1(shape.penalty_threshold)) > float(exposure.max()):
                inert += 1
        print(f"  {dataset_id.value:<26}{inert}/{len(shapes)} shapes have an inert bucket-replay channel")

    pd.DataFrame(frames).to_csv(args.output_dir / "channel_activity.csv", index=False)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
