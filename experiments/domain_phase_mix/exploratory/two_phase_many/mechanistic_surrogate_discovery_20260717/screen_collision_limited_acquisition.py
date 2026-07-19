# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Test whether finite-corpus collisions reduce useful acquisition globally.

For phase ``t``, the dimensionless collision load is
``C_t=sum_i c_i^t w_i^2``. Relative to proportional training, useful exposure
is slowed by ``g_t=(1+chi C_t^prop)/(1+chi C_t)``. This factor changes only
the acquisition state; literal physical replay and the policy's phase shift
remain computed from the actual policy. ``chi=0`` is the exact frozen model.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_deficit_output_link_20260716 as output_link,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_deficit_response_20260716 as deficit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    freeze_baseline_gate as gate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (  # noqa: E402
    screen_nested_support_invariants as support,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_OUTPUT = (
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round13_collision_limited_acquisition"
)
DEFICIT_VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
COLLISION_RATES = (0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3)


@dataclass(frozen=True)
class Model:
    dataset: Any
    deficit_config: deficit.Config
    link_config: output_link.LinkConfig
    collision_rate: float
    floor: float
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        values, names, _ridge = collision_limited_design(candidate, self.deficit_config, self.collision_rate)
        if names != self.names:
            raise ValueError("Prediction design differs from fitted design")
        latent = self.intercept + values @ self.coefficients
        return output_link.inverse_link(latent, self.link_config.link, self.floor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def collision_efficiency(dataset: Any, weights: np.ndarray, rate: float) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    proportional = base.proportional_weights(dataset)
    phase_scales = np.stack([dataset.c0, dataset.c1])
    load = np.sum(phase_scales[None, :, :] * weights**2, axis=2)
    reference_load = np.sum(phase_scales * proportional[None, :] ** 2, axis=1)
    return (1.0 + rate * reference_load[None, :]) / (1.0 + rate * load)


def collision_limited_design(
    dataset: Any,
    deficit_config: deficit.Config,
    collision_rate: float,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    physical_design = deficit.build_design(dataset, deficit_config)
    if collision_rate == 0.0:
        return physical_design.values, physical_design.names, physical_design.ridge_multipliers
    efficiency = collision_efficiency(dataset, dataset.weights, collision_rate)
    effective_dataset = replace(dataset, weights=dataset.weights * efficiency[:, :, None])
    effective_design = deficit.build_design(effective_dataset, deficit_config)
    values = effective_design.values.copy()
    # These channels describe the actual schedule, not useful-state acquisition.
    for name in ("shared_literal_replay", "phase_shift_tv"):
        if name in effective_design.names:
            index = effective_design.names.index(name)
            values[:, index] = physical_design.values[:, index]
    return values, effective_design.names, effective_design.ridge_multipliers


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    collision_rate: float,
    indices: np.ndarray,
) -> Model:
    values, names, ridge_multipliers = collision_limited_design(dataset, deficit_config, collision_rate)
    floor = (
        0.0
        if link_config.link is output_link.Link.IDENTITY
        else link_config.floor_fraction * float(dataset.target.min())
    )
    transformed = output_link.transformed_target(dataset.target[indices], link_config.link, floor)
    x = values[indices]
    x_mean = x.mean(axis=0)
    y_mean = float(transformed.mean())
    centered_x = x - x_mean
    centered_y = transformed - y_mean
    if link_config.l2 > 0.0:
        penalty = np.sqrt(link_config.l2 * ridge_multipliers)
        fit_x = np.vstack([centered_x, np.diag(penalty)])
        fit_y = np.concatenate([centered_y, np.zeros(len(penalty), dtype=float)])
    else:
        fit_x, fit_y = centered_x, centered_y
    coefficients, _residual = nnls(fit_x, fit_y, maxiter=40 * fit_x.shape[1])
    intercept = y_mean - float(x_mean @ coefficients)
    return Model(dataset, deficit_config, link_config, collision_rate, floor, intercept, coefficients, names)


def oof_prediction(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    collision_rate: float,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        model = fit_model(dataset, deficit_config, link_config, collision_rate, train)
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete OOF prediction for collision_rate={collision_rate}")
    return prediction


def benchmark(
    dataset_id: base.DatasetId, source: pd.DataFrame, links: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    deficit_config = output_link.selected_deficit_config(dataset_id, DEFICIT_VARIANT, source)
    link_config = support.selected_link_config(dataset_id, links)
    screen_rows: list[dict[str, Any]] = []
    predictions: dict[float, np.ndarray] = {}
    for rate in COLLISION_RATES:
        prediction = oof_prediction(dataset, deficit_config, link_config, rate, splits)
        summary, _bins = gate.metrics(dataset.target, prediction)
        screen_rows.append({"dataset": dataset_id.value, "collision_rate": rate, **summary})
        predictions[rate] = prediction
    selected_rate = min(screen_rows, key=lambda row: (row["rmse"], row["regret_at_1"]))["collision_rate"]

    heldout_frame, heldout_weights, heldout_target = base.heldout_data(dataset_id, dataset)
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    metric_rows: list[dict[str, Any]] = []
    for rate in (0.0, selected_rate):
        model = fit_model(dataset, deficit_config, link_config, rate, np.arange(dataset.n))
        efficiency = collision_efficiency(dataset, dataset.weights, rate)
        for split, observed, predicted in (
            ("fit_oof", dataset.target, predictions[rate]),
            ("heldout_policy_matched", heldout_target[policy_mask], model.predict(heldout_weights)[policy_mask]),
        ):
            summary, _bins = gate.metrics(observed, predicted)
            metric_rows.append(
                {
                    "dataset": dataset_id.value,
                    "split": split,
                    "collision_rate": rate,
                    "min_fit_efficiency": float(efficiency.min()),
                    "max_fit_efficiency": float(efficiency.max()),
                    "parameter_count": 1 + len(model.coefficients) + int(rate > 0.0),
                    **summary,
                }
            )
    return metric_rows, screen_rows


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.source_metrics)
    gate.assert_sealed_absent(args.link_metrics)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    metrics: list[dict[str, Any]] = []
    screen: list[dict[str, Any]] = []
    for dataset_id in (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9):
        panel_metrics, panel_screen = benchmark(dataset_id, source, links)
        metrics.extend(panel_metrics)
        screen.extend(panel_screen)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Collision-limited acquisition audit\n\n"
        "The collision rate was selected on fit-panel OOF predictions before heldout scoring. "
        "A zero rate is the exact frozen baseline.\n\n" + metric_frame.to_markdown(index=False, floatfmt=".6f") + "\n"
    )
    print(json.dumps(metric_frame.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
