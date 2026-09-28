# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task calibration of the surrogates in the floor region of the held-out bank.

For every task, the bank coordinates whose measured value on that task lies in its bottom decile are the floor
region: the rows a floor rule exists to get right. Each model's fold -1 per-component bank predictions (from a
selection package's shards) are compared with the corrected registry's per-coordinate measurements there; bias
and RMSE are reported per model, per target, pooled over tasks and by task group, with a source-block bootstrap
of the paired differences against WSPU. Nothing is fitted or launched.

usage: uv run python analyze_delphi_floor_region_calibration_20260907.py [--selection DIR] [--methods a,b,c]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_delphi_selection_20260906 as benchmark
from experiments.domain_phase_mix.exploratory.two_phase_many import score_delphi_selection_20260906 as scorer
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)

REFERENCE = Path(__file__).resolve().parent / "reference_outputs"
REGISTRY = REFERENCE / "single_phase_heldout_round3_corrected_20260903" / "heldout_coordinate_components.csv"
DEFAULT_SELECTION = REFERENCE / "delphi_fitted_floor_selection_20260907"
DEFAULT_METHODS = (
    "weibull_softplus_unscaled",
    "weibull_softplus_unscaled@log_deficit_bounded_link",
    "weibull_softplus_unscaled@fitted_floor_link",
)
DECILE = 0.1
BOOTSTRAP_DRAWS = 2000
SEED = 20260907


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()
    methods = tuple(args.methods.split(","))
    panel = benchmark.read_npz(args.selection / "inputs" / "panel.npz")
    components_table = pd.read_csv(REGISTRY)
    components_table = components_table[components_table.panel.eq(benchmark.PANEL)]
    rows = []
    for target in benchmark.TARGETS:
        labels = pd.read_csv(args.selection / "inputs" / f"{target}_bank_labels.csv")
        features = benchmark.read_npz(args.selection / "inputs" / f"{target}_bank_features.npz")
        coordinate_ids = [str(c) for c in features["coordinate_id"]]
        blocks, _ = scorer.source_blocks(labels.set_index("coordinate_id").loc[coordinate_ids].sources)
        measured = components_table[components_table.target.eq(target)].pivot(
            index="coordinate_id", columns="component", values="bpb_mean"
        )
        # A few bank coordinates carry only the aggregate in the registry; they cannot enter a per-task check.
        covered = np.array([coordinate in measured.index for coordinate in coordinate_ids])
        measured = measured.reindex(coordinate_ids)
        components = [str(c) for c in panel[f"{target}_components"]]
        for index, component in enumerate(components):
            values = measured[component].to_numpy(float)
            valid = covered & np.isfinite(values)
            floor_rows = np.flatnonzero(valid & (values <= np.nanquantile(values[valid], DECILE)))
            for method in methods:
                shard = benchmark.read_npz(args.selection / "baseline_shards" / method / target / f"r0_f-1_c{index}.npz")
                predicted = shard["bank_prediction"]
                for row in floor_rows:
                    rows.append(
                        {
                            "target": target,
                            "component": component,
                            "group": registry.floor_task_group(component),
                            "method": method,
                            "coordinate_id": coordinate_ids[row],
                            "source_block": int(blocks[row]),
                            "measured": float(values[row]),
                            "predicted": float(predicted[row]),
                            "residual": float(values[row] - predicted[row]),
                        }
                    )
    table = pd.DataFrame(rows)
    output = args.selection / "floor_region_calibration"
    output.mkdir(exist_ok=True)
    table.to_csv(output / "floor_region_residuals.csv", index=False)
    summary = (
        table.groupby(["target", "method"])
        .residual.agg(bias="mean", rmse=lambda r: float(np.sqrt(np.mean(np.square(r)))), rows="count")
        .reset_index()
    )
    by_group = (
        table.groupby(["target", "group", "method"])
        .residual.agg(bias="mean", rmse=lambda r: float(np.sqrt(np.mean(np.square(r)))))
        .reset_index()
    )
    # Paired source-block bootstrap of |residual| against WSPU.
    rng = np.random.default_rng(SEED)
    contrasts = []
    reference = methods[0]
    for target in benchmark.TARGETS:
        wide = table[table.target.eq(target)].pivot_table(
            index=["component", "coordinate_id", "source_block"], columns="method", values="residual"
        )
        block_ids = wide.index.get_level_values("source_block").to_numpy()
        unique = np.unique(block_ids)
        for method in methods[1:]:
            delta = (wide[method].abs() - wide[reference].abs()).to_numpy(float)
            draws = []
            for _ in range(BOOTSTRAP_DRAWS):
                chosen = rng.choice(unique, size=len(unique), replace=True)
                picked = np.concatenate([delta[block_ids == block] for block in chosen])
                draws.append(picked.mean())
            contrasts.append(
                {
                    "target": target,
                    "candidate": method,
                    "reference": reference,
                    "mean_abs_residual_delta": float(delta.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                    "source_blocks": len(unique),
                }
            )
    contrasts = pd.DataFrame(contrasts)
    summary.to_csv(output / "summary.csv", index=False)
    by_group.to_csv(output / "summary_by_group.csv", index=False)
    contrasts.to_csv(output / "paired_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print("floor region (bottom decile of each task's measured bank values), residual = measured - predicted:")
    print(summary.round(4).to_string(index=False))
    print(by_group.pivot_table(index=["target", "group"], columns="method", values="rmse").round(4).to_string())
    print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
