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
# ]
# ///
"""Test round-one candidate forms on the independent production swarm."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as models,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round1_production_transfer"
N_SPLITS = 5
SEEDS = (0, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def load_panel() -> models.PairedPanel:
    raw = pooled.load_production_dataset()
    family_names, family_members, _quality = observatory.family_partition(raw)
    return models.PairedPanel(
        name="production_uncheatable",
        target="uncheatable",
        frame=raw.frame.copy(),
        domain_names=tuple(raw.domain_names),
        family_names=family_names,
        family_members=family_members,
        weights=np.asarray(raw.weights, dtype=float),
        c0=np.asarray(raw.c0, dtype=float),
        c1=np.asarray(raw.c1, dtype=float),
        two_phase_target=np.asarray(raw.y, dtype=float),
        one_phase_target=np.full(raw.n, np.nan, dtype=float),
    )


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    slope, intercept = np.polyfit(predicted, observed, 1)
    selected = int(np.argmin(predicted))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "bias": float(np.mean(residual)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "regret_at_1": float(observed[selected] - np.min(observed)),
        "worst_optimism": float(np.max(observed - predicted)),
    }


def aggregate_configs() -> list[models.AggregateConfig]:
    return [models.AggregateConfig(power, offset, 1.0) for power in (0.25, 0.5, 1.0) for offset in (0.1, 0.3, 1.0)]


def phase_configs(family: models.PhaseFamily) -> list[models.PhaseConfig]:
    if family is models.PhaseFamily.PMVT:
        return [models.PMVTConfig(2.0, l2, True, True, "bucket", "family") for l2 in (0.01, 0.1, 1.0)]
    if family is models.PhaseFamily.TERMINAL_EQUILIBRIUM:
        return [
            models.TerminalEquilibriumConfig(ratio, l2)
            for ratio in (0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
            for l2 in (0.01, 0.1, 1.0)
        ]
    raise ValueError(f"Unsupported production candidate {family}")


def oof_prediction(
    panel: models.PairedPanel,
    aggregate_config: models.AggregateConfig,
    family: models.PhaseFamily,
    phase_config: models.PhaseConfig,
    seed: int,
) -> np.ndarray:
    prediction = np.full(panel.n, np.nan, dtype=float)
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for train, test in splitter.split(np.arange(panel.n)):
        model = models.fit_joint(panel, train, aggregate_config, family, phase_config)
        prediction[test] = model.predict_weights(panel.weights[test])
    return prediction


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    rows: list[dict[str, Any]] = []
    predictions: dict[str, list[np.ndarray]] = {}
    for family in (models.PhaseFamily.PMVT, models.PhaseFamily.TERMINAL_EQUILIBRIUM):
        for aggregate_config in aggregate_configs():
            for phase_config in phase_configs(family):
                seed_predictions = [
                    oof_prediction(panel, aggregate_config, family, phase_config, seed) for seed in SEEDS
                ]
                prediction = np.mean(seed_predictions, axis=0)
                key = f"{family.value}::{aggregate_config.key}::{phase_config.key}"
                predictions[key] = seed_predictions
                rows.append(
                    {
                        "family": family.value,
                        "aggregate_config": aggregate_config.key,
                        "aggregate_config_json": json.dumps(asdict(aggregate_config), sort_keys=True),
                        "phase_config": phase_config.key,
                        "phase_config_json": json.dumps(asdict(phase_config), sort_keys=True),
                        **metrics(panel.two_phase_target, prediction),
                        "seed_rmse_sd": float(
                            np.std(
                                [
                                    metrics(panel.two_phase_target, seed_prediction)["rmse"]
                                    for seed_prediction in seed_predictions
                                ]
                            )
                        ),
                    }
                )
    grid = pd.DataFrame(rows)
    grid.to_csv(args.output_dir / "production_hyperparameter_grid.csv", index=False)
    selected_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for family, candidates in grid.groupby("family", sort=False):
        best = candidates.sort_values(["rmse", "regret_at_1", "worst_optimism"]).iloc[0]
        key = f"{family}::{best['aggregate_config']}::{best['phase_config']}"
        prediction = np.mean(predictions[key], axis=0)
        selected_rows.append(best.to_dict())
        for index, value in enumerate(prediction):
            prediction_rows.append(
                {
                    "family": family,
                    "row_index": index,
                    "observed": float(panel.two_phase_target[index]),
                    "predicted": float(value),
                    "run_name": str(panel.frame.iloc[index].get("run_name", index)),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(args.output_dir / "selected_configs_and_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "selected_oof_predictions.csv", index=False)
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
