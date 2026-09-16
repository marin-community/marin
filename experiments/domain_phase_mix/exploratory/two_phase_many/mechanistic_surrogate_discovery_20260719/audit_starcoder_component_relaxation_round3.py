# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Falsify multi-rate component relaxation on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    component_relaxation_models as relaxation,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    potential_phase_models as potential,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_component_relaxation_round3 as mcr_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    SCRIPT_DIR.parent
    / "reference_outputs/mechanistic_surrogate_discovery_20260719/round3_component_relaxation_starcoder"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fold_potentials(panel: Any, config: potential.PotentialConfig) -> list[mcr_screen.FoldPotential]:
    result = []
    for fold, (train, test) in enumerate(starcoder.surface_folds(panel)):
        tied_train = train[panel.paired_mask[train]]
        model = potential.fit_potential(
            starcoder.geometry(panel),
            panel.aggregate_weights,
            panel.one_phase_target,
            tied_train,
            config,
        )
        result.append(mcr_screen.FoldPotential(fold, train, test, model))
    return result


def select_rates(
    panel: Any,
    potential_config: potential.PotentialConfig,
) -> tuple[relaxation.ComponentRelaxationConfig, np.ndarray, pd.DataFrame, pd.DataFrame]:
    folds = fold_potentials(panel, potential_config)
    rows = []
    predictions = {}
    configs = mcr_screen.rate_configs(len(panel.family_names))
    fold_index = np.full(panel.n, -1, dtype=int)
    for fold in folds:
        fold_index[fold.test] = fold.fold
    for config in configs:
        prediction = np.full(panel.n, np.nan, dtype=float)
        for fold in folds:
            model = relaxation.fit_component_relaxation(fold.model, panel.alpha0, config)
            prediction[fold.test] = model.predict(panel.weights[fold.test])
        predictions[config.key] = prediction
        rows.append(
            {
                "surface": panel.name,
                "config": config.key,
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism"])
    rates = tuple(float(value) for value in str(table.iloc[0]["config"]).removeprefix("rates=").split(","))
    selected = relaxation.ComponentRelaxationConfig(rates)
    fold_rows = []
    for fold in range(len(folds)):
        test = fold_index == fold
        local = []
        for config in configs:
            prediction = predictions[config.key]
            local.append(
                {
                    "fold": fold,
                    "config": config.key,
                    "rmse": float(np.sqrt(np.mean((prediction[test] - panel.two_phase_target[test]) ** 2))),
                }
            )
        fold_rows.append(pd.DataFrame(local).sort_values("rmse").iloc[0].to_dict())
    return selected, predictions[selected.key], table, pd.DataFrame(fold_rows)


def leave_region_out(
    panel: Any,
    potential_config: potential.PotentialConfig,
    config: relaxation.ComponentRelaxationConfig,
) -> list[dict[str, Any]]:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
    }
    rows = []
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        tied_train = train[panel.paired_mask[train]]
        tied_model = potential.fit_potential(
            starcoder.geometry(panel),
            panel.aggregate_weights,
            panel.one_phase_target,
            tied_train,
            potential_config,
        )
        model = relaxation.fit_component_relaxation(tied_model, panel.alpha0, config)
        prediction = model.predict(panel.weights[test])
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **paired_screen.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return rows


def fit_full(
    panel: Any,
    potential_config: potential.PotentialConfig,
    config: relaxation.ComponentRelaxationConfig,
) -> relaxation.ComponentRelaxationModel:
    tied = np.flatnonzero(panel.paired_mask)
    tied_model = potential.fit_potential(
        starcoder.geometry(panel),
        panel.aggregate_weights,
        panel.one_phase_target,
        tied,
        potential_config,
    )
    return relaxation.fit_component_relaxation(tied_model, panel.alpha0, config)


def optimum_record(panel: Any, model: relaxation.ComponentRelaxationModel) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 201)
    rare0, rare1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - rare0.ravel(), rare0.ravel()]),
            np.column_stack([1.0 - rare1.ravel(), rare1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "phase0_rare": float(rare0.ravel()[best]),
        "phase1_rare": float(rare1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                rare0.ravel()[best] - panel.weights[observed_best, 0, 1],
                rare1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    metric_rows = []
    grid_tables = []
    fold_rows = []
    region_rows = []
    optimum_rows = []
    for panel in panels:
        selected_potential, _potential_grid = starcoder.select_potential(panel)
        selected, prediction, grid, fold_selection = select_rates(panel, selected_potential)
        grid_tables.append(grid)
        metric_rows.append(
            {
                "surface": panel.name,
                "n_tied": int(panel.paired_mask.sum()),
                "selected_potential": selected_potential.key,
                "selected_rates": selected.key,
                **paired_screen.scalar_metrics(panel.two_phase_target, prediction),
            }
        )
        fold_selection.insert(0, "surface", panel.name)
        fold_rows.append(fold_selection)
        region_rows.extend(leave_region_out(panel, selected_potential, selected))
        optimum_rows.append(optimum_record(panel, fit_full(panel, selected_potential, selected)))

    metrics = pd.DataFrame(metric_rows)
    grid = pd.concat(grid_tables, ignore_index=True)
    fold_selection = pd.concat(fold_rows, ignore_index=True)
    regions = pd.DataFrame(region_rows)
    optima = pd.DataFrame(optimum_rows)
    metrics.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    grid.to_csv(args.output_dir / "rate_grid.csv", index=False)
    fold_selection.to_csv(args.output_dir / "fold_rate_selection.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out_metrics.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    report = [
        "# Multi-rate component relaxation: StarCoder falsification",
        "",
        "The tied response was selected from phase-tied rows only; family rates were selected by full-surface OOF. No Delphi historical or adversarial outcome was read.",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Foldwise rate selections",
        "",
        fold_selection.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    print(metrics.to_string(index=False))
    print("\nLeave-region-out")
    print(regions.to_string(index=False))
    print("\nOptima")
    print(optima.to_string(index=False))
    print("\nFoldwise")
    print(fold_selection.to_string(index=False))


if __name__ == "__main__":
    main()
