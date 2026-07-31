# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Falsify a reduced-order gradient-flow state with a convex BPB bowl."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.freeze_baseline_gate import (  # noqa: E402
    assert_sealed_absent,
    metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.mechanistic_models import (  # noqa: E402
    Panel,
    family_weight,
    finite_subset_replay,
    group_sum,
    normalized_group_exposure,
    simulated_epochs,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717.screen_portfolio import (  # noqa: E402
    DASHBOARD,
    PANEL_IDS,
    dashboard_fit_rows,
    heldout_data,
    load_panel,
    split_panel_id,
)

RESEARCH_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round22_gradient_flow_bowl"
)
RATES = (0.5, 2.0, 8.0, 16.0)
FLOORS = (0.03, 0.3, 1.0)
EXPONENTS = (0.1, 0.5)
L2_GRID = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class Config:
    adaptation_rate: float
    floor: float
    exponent: float
    l2: float

    @property
    def key(self) -> str:
        return f"rate-{self.adaptation_rate:g}__floor-{self.floor:g}__exponent-{self.exponent:g}__l2-{self.l2:g}"


@dataclass(frozen=True)
class StateDesign:
    deficit: np.ndarray
    replay: np.ndarray
    state: np.ndarray


@dataclass(frozen=True)
class Model:
    config: Config
    intercept: float
    deficit_coefficients: np.ndarray
    replay_coefficients: np.ndarray
    curvature: np.ndarray
    linear_state: np.ndarray

    def predict_design(self, design: StateDesign) -> np.ndarray:
        return (
            self.intercept
            + design.deficit @ self.deficit_coefficients
            + design.replay @ self.replay_coefficients
            + design.state**2 @ self.curvature
            + design.state @ self.linear_state
        )

    @property
    def preferred_state(self) -> np.ndarray:
        return np.divide(
            -self.linear_state,
            2.0 * self.curvature,
            out=np.zeros_like(self.linear_state),
            where=self.curvature > 1e-10,
        )


def representation_state(panel: Panel, weights: np.ndarray, rate: float) -> np.ndarray:
    state = np.zeros((len(weights), len(panel.family_names)), dtype=float)
    for phase_weight, duration in (
        (weights[:, 0], panel.phase_fractions[0]),
        (weights[:, 1], panel.phase_fractions[1]),
    ):
        target = family_weight(phase_weight, panel)
        retention = math.exp(-rate * duration)
        state = target + (state - target) * retention
    return state


def build_design(panel: Panel, weights: np.ndarray, config: Config) -> StateDesign:
    _phase0, _phase1, total = simulated_epochs(panel, weights)
    proportional = np.broadcast_to(panel.proportional, (1, 2, panel.m))
    _ref0, _ref1, reference = simulated_epochs(panel, proportional)
    ratio = normalized_group_exposure(total, reference, panel)
    deficit = np.power(np.maximum(ratio, 0.0) + config.floor, -config.exponent)
    deficit -= math.pow(1.0 + config.floor, -config.exponent)
    replay = group_sum(finite_subset_replay(total), panel)
    state = representation_state(panel, weights, config.adaptation_rate)
    return StateDesign(deficit, replay, state)


def fit_model(panel: Panel, design: StateDesign, indices: np.ndarray, config: Config) -> Model:
    g = design.deficit.shape[1]
    f = design.state.shape[1]
    deficit = cp.Variable(g, nonneg=True)
    replay = cp.Variable(g, nonneg=True)
    curvature = cp.Variable(f, nonneg=True)
    linear = cp.Variable(f)
    intercept = cp.Variable()
    prediction = (
        intercept
        + design.deficit[indices] @ deficit
        + design.replay[indices] @ replay
        + design.state[indices] ** 2 @ curvature
        + design.state[indices] @ linear
    )
    terminal_mass = 1.0 - math.exp(-config.adaptation_rate)
    constraints = [linear <= 0.0, linear >= -2.0 * terminal_mass * curvature]
    penalty = cp.sum_squares(deficit) + cp.sum_squares(replay)
    penalty += cp.sum_squares(curvature) + cp.sum_squares(linear)
    objective = cp.Minimize(cp.sum_squares(prediction - panel.observed[indices]) + config.l2 * penalty)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Gradient-flow fit failed: {problem.status}")
    return Model(
        config=config,
        intercept=float(intercept.value),
        deficit_coefficients=np.asarray(deficit.value, dtype=float),
        replay_coefficients=np.asarray(replay.value, dtype=float),
        curvature=np.asarray(curvature.value, dtype=float),
        linear_state=np.asarray(linear.value, dtype=float),
    )


def oof_prediction(panel: Panel, raw: Any, config: Config, seeds: tuple[int, ...]) -> np.ndarray:
    design = build_design(panel, panel.weights, config)
    predictions = []
    for seed in seeds:
        prediction = np.full(panel.n, np.nan, dtype=float)
        for train, test in observatory.folds(raw, seed):
            model = fit_model(panel, design, np.asarray(train), config)
            prediction[test] = model.predict_design(
                StateDesign(design.deficit[test], design.replay[test], design.state[test])
            )
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete OOF prediction")
        predictions.append(prediction)
    return np.mean(predictions, axis=0)


def region_prediction(panel: Panel, config: Config) -> np.ndarray:
    labels = KMeans(n_clusters=5, random_state=0, n_init=20).fit_predict(panel.weights[:, :, 1])
    design = build_design(panel, panel.weights, config)
    prediction = np.full(panel.n, np.nan, dtype=float)
    for region in sorted(set(labels)):
        test = np.flatnonzero(labels == region)
        train = np.flatnonzero(labels != region)
        model = fit_model(panel, design, train, config)
        prediction[test] = model.predict_design(
            StateDesign(design.deficit[test], design.replay[test], design.state[test])
        )
    return prediction


def configs() -> tuple[Config, ...]:
    return tuple(
        Config(rate, floor, exponent, l2)
        for rate in RATES
        for floor in FLOORS
        for exponent in EXPONENTS
        for l2 in L2_GRID
    )


def screen(panel_id: str, output_dir: Path) -> None:
    assert_sealed_absent(DASHBOARD)
    bundle = json.loads(DASHBOARD.read_text())
    panel, raw = load_panel(bundle, panel_id)
    rows = []
    best: tuple[float, float, Config] | None = None
    for index, config in enumerate(configs(), 1):
        prediction = oof_prediction(panel, raw, config, (0,))
        summary, _bins = metrics(panel.observed, prediction)
        rows.append({"config": config.key, **summary})
        candidate = (float(summary["rmse"]), -float(summary["spearman"]), config)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        print(f"{panel_id}: [{index}/{len(configs())}] {config.key}", flush=True)
    if best is None:
        raise RuntimeError("No gradient-flow config selected")
    selected = best[2]
    metric_rows = []
    prediction_rows = []
    oof = oof_prediction(panel, raw, selected, (0, 1, 2))
    oof_summary, _bins = metrics(panel.observed, oof)
    metric_rows.append({"panel": panel_id, "config": selected.key, "split": "fit_oof", **oof_summary})
    prediction_rows.extend(
        {
            "panel": panel_id,
            "config": selected.key,
            "split": "fit_oof",
            "row_id": row["name"],
            "observed": observed,
            "predicted": predicted,
        }
        for row, observed, predicted in zip(
            dashboard_fit_rows(bundle, split_panel_id(panel_id)[0]), panel.observed, oof, strict=True
        )
    )
    full_design = build_design(panel, panel.weights, selected)
    model = fit_model(panel, full_design, np.arange(panel.n), selected)
    swarm, target = split_panel_id(panel_id)
    heldout = heldout_data(bundle, swarm, target)
    if heldout is not None:
        heldout_weights, heldout_observed, heldout_rows = heldout
        heldout_design = build_design(panel, heldout_weights, selected)
        heldout_prediction = model.predict_design(heldout_design)
        heldout_summary, _bins = metrics(heldout_observed, heldout_prediction)
        metric_rows.append(
            {"panel": panel_id, "config": selected.key, "split": "heldout_policy_matched", **heldout_summary}
        )
        prediction_rows.extend(
            {
                "panel": panel_id,
                "config": selected.key,
                "split": "heldout_policy_matched",
                "row_id": row["name"],
                "observed": observed,
                "predicted": predicted,
            }
            for row, observed, predicted in zip(heldout_rows, heldout_observed, heldout_prediction, strict=True)
        )
    if panel.m == 2:
        lro = region_prediction(panel, selected)
        lro_summary, _bins = metrics(panel.observed, lro)
        metric_rows.append({"panel": panel_id, "config": selected.key, "split": "leave_region_out", **lro_summary})
    parameter_rows = []
    for family, curvature, linear, preferred in zip(
        panel.family_names, model.curvature, model.linear_state, model.preferred_state, strict=True
    ):
        parameter_rows.append(
            {
                "panel": panel_id,
                "config": selected.key,
                "parameter": f"state_bowl:{family}",
                "curvature": curvature,
                "linear": linear,
                "preferred_state": preferred,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "hyperparameter_screen.csv", index=False)
    pd.DataFrame(metric_rows).to_csv(output_dir / "selected_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "selected_predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(output_dir / "selected_parameters.csv", index=False)
    (output_dir / "selection.json").write_text(json.dumps(selected.__dict__, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANEL_IDS, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or DEFAULT_OUTPUT_ROOT / args.panel
    screen(args.panel, output)


if __name__ == "__main__":
    main()
