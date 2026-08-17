# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "scipy"]
# ///
"""Evaluate one primary-selected WSD80 temporal state across all targets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    evaluate_gated_absorption_variants_20260809 as variants,
)


def evaluate_shared_state(variant: variants.Variant, seed: int) -> list[dict[str, float | str]]:
    """Select shapes on the primary target and refit only target-specific heads."""
    outer = variants.harness.wsd80_folds(
        "random",
        variants.PANEL.weights,
        np.arange(len(variants.PRIMARY)),
        variants.N_FOLDS,
        seed,
    )
    fold_designs = []
    for train, test in outer:
        shape = variants.select(variants.PRIMARY, train, seed, variant)
        free, constrained = variants.design(variants.PANEL.weights, variant, shape)
        fold_designs.append((train, test, free, constrained))

    full_shape = variants.select(variants.PRIMARY, np.arange(len(variants.PRIMARY)), seed, variant)
    free, constrained = variants.design(variants.PANEL.weights, variant, full_shape)
    axis = np.linspace(0.0, 1.0, variants.SURFACE_GRID)
    grid_0, grid_1 = np.meshgrid(axis, axis, indexing="ij")
    flat_0, flat_1 = grid_0.ravel(), grid_1.ravel()
    grid_free, grid_constrained = variants.design(variants.wsd.grid_weights(flat_0, flat_1), variant, full_shape)
    tied_axis = np.linspace(0.0, 1.0, variants.SURFACE_GRID * variants.SURFACE_GRID // 4)
    tied_free, tied_constrained = variants.design(variants.wsd.grid_weights(tied_axis, tied_axis), variant, full_shape)

    interior_rows = np.flatnonzero(variants.INTERIOR)
    rows: list[dict[str, float | str]] = []
    for target_index, target in enumerate(variants.TARGETS.names):
        response = variants.TARGETS.values[:, target_index]
        predictions = np.empty_like(response)
        for train, test, fold_free, fold_constrained in fold_designs:
            b, a = variants.fit_head(fold_free[train], fold_constrained[train], response[train])
            predictions[test] = fold_free[test] @ b + fold_constrained[test] @ a

        observed_best = int(interior_rows[np.argmin(response[interior_rows])])
        ranked = interior_rows[np.argsort(predictions[interior_rows])]
        b, a = variants.fit_head(free, constrained, response)
        surface = grid_free @ b + grid_constrained @ a
        tied = tied_free @ b + tied_constrained @ a
        best = int(np.argmin(surface))
        rows.append(
            {
                "target": target,
                "rmse": float(np.sqrt(np.mean((predictions - response)[variants.INTERIOR] ** 2))),
                "regret_at_1": float(response[ranked[0]] - response[observed_best]),
                "phase_0_optimum": float(flat_0[best]),
                "phase_1_optimum": float(flat_1[best]),
                "predicted_gain": float(tied.min() - surface.min()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=tuple(variants.VARIANTS), default="GA-017")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = evaluate_shared_state(variants.VARIANTS[args.variant], args.seed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in rows:
        print(
            f"{row['target']}: RMSE {row['rmse']:.6f}; Regret@1 {row['regret_at_1']:.6f}; "
            f"optimum ({row['phase_0_optimum']:.3f},{row['phase_1_optimum']:.3f}); "
            f"gain {row['predicted_gain']:+.6f}"
        )


if __name__ == "__main__":
    main()
