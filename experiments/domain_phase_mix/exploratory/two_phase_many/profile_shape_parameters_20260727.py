# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Profile each transition parameter independently, because selecting a shape row identifies a bundle.

Shape selection is a discrete argmin over a fixed Sobol library, and each row moves the exponent, the
late multiplier, the forgetting rate and the penalty threshold together. Showing that the same row is
chosen in 82 to 100 percent of outer folds therefore establishes that the *bundle* is well separated
from the other bundles on offer. It says nothing about whether any individual parameter is identified:
a row could win because its exponent is right while its forgetting rate is arbitrary, and the library
would never reveal that.

The test is a profile sweep. Hold every parameter at its selected value, vary one continuously, and
refit the linear head at each point with the ridge grid re-optimized. A parameter the panel can see
produces a profile with a clear interior minimum and a rise that exceeds the fold-to-fold noise. A
parameter the panel cannot see produces a flat profile, and its selected value is then an artefact of
whichever library row happened to win on the strength of the others.

Fold-to-fold spread of the cross-validated error is reported alongside each profile, because "the
profile rises by 3 percent" is only meaningful against how much it moves between resamples.
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    corrected_hpr_model_20260727 as corrected,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    evaluate_hpr_v2_nested_20260727 as nested,
)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "reference_outputs" / "shape_parameter_profiles_20260727"
DATASETS = (
    bench.DatasetId.THREE_HUNDRED_M_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_UNCHEATABLE,
    bench.DatasetId.DELPHI_3E18_TABLE9,
)
# Each parameter is swept over the range its own Sobol library spans, so the profile covers exactly the
# space selection could have explored.
PROFILE_GRIDS = {
    "exponent": np.exp(np.linspace(np.log(0.08), np.log(1.2), 13)),
    "late_multiplier": np.exp(np.linspace(np.log(0.75), np.log(12.0), 13)),
    "forgetting_rate": np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-5), np.log(4.0), 12))]),
    "penalty_threshold": np.linspace(0.0, 7.0, 13),
}
FOLD_SEEDS = (7152, 7157, 7163, 7171)
# A profile is called identified when its rise from the minimum exceeds this multiple of the
# fold-to-fold standard deviation of the cross-validated error at the minimum.
IDENTIFIED_MULTIPLE = 2.0


def cross_validated(dataset, dataset_id, corrections, config, seed: int) -> float:
    splits = bench.split_indices(dataset, dataset_id, np.arange(dataset.n), seed)
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = corrected.fit_corrected(dataset, config, corrections, train).predict(dataset.weights[test])
    mask = np.isfinite(prediction)
    return float(np.sqrt(np.mean((prediction[mask] - dataset.target[mask]) ** 2)))


def best_over_ridge(dataset, dataset_id, corrections, shape, shape_index: int, seed: int) -> float:
    """Lowest cross-validated error at this shape, re-optimizing the ridge grid.

    Re-optimizing matters: a profile that held the ridge fixed would attribute to the shape whatever
    the shrinkage would have absorbed.
    """
    best = float("inf")
    for l2 in bench.L2_GRID:
        for residual in bench.RESIDUAL_SHRINK_GRID:
            config = bench.Config(nested.VARIANT, shape_index, shape, l2, residual, 0.0, 0.0)
            best = min(best, cross_validated(dataset, dataset_id, corrections, config, seed))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shapes = nested.promoted_shapes()
    arms = (
        ("original", corrected.Corrections()),
        (
            "identifiable+ledger+recency",
            corrected.Corrections(
                transition=corrected.TransitionForm.RECENCY_KERNEL,
                identifiable_hierarchy=True,
                deduplicated_ledgers=True,
                normalized_family_ledger=True,
            ),
        ),
    )

    rows = []
    for dataset_id in DATASETS:
        dataset = bench.load_dataset(dataset_id)
        sigma = nested.RUN_SIGMA[dataset_id.value]
        for arm, corrections in arms:
            selected = nested.two_stage_selection(dataset, dataset_id, corrections, shapes, np.arange(dataset.n))
            anchor = selected.shape
            noise = float(
                np.std(
                    [
                        best_over_ridge(dataset, dataset_id, corrections, anchor, selected.shape_index, seed)
                        for seed in FOLD_SEEDS
                    ],
                    ddof=1,
                )
            )
            print(f"\n{dataset_id.value} / {arm}")
            print(
                f"  selected shape {selected.shape_index}: exponent {anchor.exponent:.4f}, "
                f"eta {anchor.late_multiplier:.4f}, lambda {anchor.forgetting_rate:.4g}, "
                f"tau {anchor.penalty_threshold:.4f}"
            )
            print(f"  fold-to-fold sd of the cross-validated error at that shape: {noise:.6f} ({noise / sigma:.2f}s)")
            for parameter, grid in PROFILE_GRIDS.items():
                errors = np.asarray(
                    [
                        best_over_ridge(
                            dataset,
                            dataset_id,
                            corrections,
                            replace(anchor, **{parameter: float(value)}),
                            selected.shape_index,
                            bench.SCREEN_SEED,
                        )
                        for value in grid
                    ]
                )
                argmin = int(np.argmin(errors))
                rise = float(errors.max() - errors.min())
                identified = rise > IDENTIFIED_MULTIPLE * noise
                interior = 0 < argmin < len(grid) - 1
                rows.append(
                    {
                        "dataset": dataset_id.value,
                        "arm": arm,
                        "parameter": parameter,
                        "selected_value": getattr(anchor, parameter),
                        "profile_argmin": float(grid[argmin]),
                        "profile_min": float(errors.min()),
                        "profile_max": float(errors.max()),
                        "rise_over_noise": rise / max(noise, 1e-12),
                        "interior_minimum": interior,
                        "identified": identified,
                        "fold_noise": noise,
                    }
                )
                verdict = "identified" if identified else "FLAT - not identified"
                print(
                    f"    {parameter:<20} profile argmin {grid[argmin]:>9.4g} "
                    f"(selected {getattr(anchor, parameter):>9.4g})  "
                    f"rise {rise / max(noise, 1e-12):>6.1f}x noise  "
                    f"{'interior' if interior else 'AT EDGE ':<9} -> {verdict}"
                )

    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "shape_parameter_profiles.csv", index=False)

    print("\n" + "=" * 100)
    print("WHICH TRANSITION PARAMETERS THE PANEL CAN ACTUALLY SEE")
    print("=" * 100)
    for parameter, group in table.groupby("parameter"):
        identified = int(group["identified"].sum())
        interior = int(group["interior_minimum"].sum())
        print(
            f"  {parameter:<20} identified in {identified}/{len(group)} cells, "
            f"interior minimum in {interior}/{len(group)}, "
            f"median rise {group['rise_over_noise'].median():.1f}x fold noise"
        )
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
