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
"""Test an importance-weight effective-sample-size scaling mechanism.

If ``p`` is the reference target support and ``q`` the support induced by a
candidate's retained exposure, importance sampling gives
``N_eff/N = 1/sum_i p_i^2/q_i``. A scaling-law loss gap is therefore
``(N/N_eff)^alpha - 1``. This is zero at proportional support, grows more
strongly than reverse KL near missing support, and adds one nonnegative BPB
amplitude to the frozen model. The floor and exponent are selected on fit OOF.
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
    RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round14_importance_ess_scaling"
)
DEFICIT_VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
SUPPORT_FLOORS = (0.01, 0.03, 0.1, 0.3)
SCALING_EXPONENTS = (0.25, 0.5, 1.0)


@dataclass(frozen=True)
class Config:
    floor: float
    exponent: float

    @property
    def key(self) -> str:
        return f"ess_scaling__floor-{self.floor:g}__alpha-{self.exponent:g}"


@dataclass(frozen=True)
class Model:
    dataset: Any
    deficit_config: deficit.Config
    link_config: output_link.LinkConfig
    config: Config | None
    floor: float
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(self.dataset, weights=np.asarray(weights, dtype=float), target=np.zeros(len(weights)))
        values, names, _ridge = combined_design(candidate, self.deficit_config, self.config)
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


def ess_feature(dataset: Any, deficit_config: deficit.Config, config: Config) -> np.ndarray:
    ratio = support.retained_ratio(dataset, deficit_config)
    mass = support.proportional_mass(dataset)
    induced = support.supported_distribution(ratio, mass, config.floor)
    inverse_efficiency = np.sum(mass[None, :] ** 2 / np.maximum(induced, 1e-12), axis=1)
    return np.maximum(inverse_efficiency, 1.0) ** config.exponent - 1.0


def combined_design(
    dataset: Any,
    deficit_config: deficit.Config,
    config: Config | None,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    design = deficit.build_design(dataset, deficit_config)
    if config is None:
        return design.values, design.names, design.ridge_multipliers
    feature = ess_feature(dataset, deficit_config, config)
    return (
        np.column_stack([design.values, feature]),
        (*design.names, f"importance_ess:{config.key}"),
        np.concatenate([design.ridge_multipliers, np.ones(1)]),
    )


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    config: Config | None,
    indices: np.ndarray,
) -> Model:
    values, names, ridge_multipliers = combined_design(dataset, deficit_config, config)
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
        fit_y = np.concatenate([centered_y, np.zeros(len(penalty))])
    else:
        fit_x, fit_y = centered_x, centered_y
    coefficients, _residual = nnls(fit_x, fit_y, maxiter=40 * fit_x.shape[1])
    intercept = y_mean - float(x_mean @ coefficients)
    return Model(dataset, deficit_config, link_config, config, floor, intercept, coefficients, names)


def oof_prediction(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    config: Config | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan)
    for train, test in splits:
        prediction[test] = fit_model(dataset, deficit_config, link_config, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError("Incomplete OOF prediction")
    return prediction


def benchmark(
    dataset_id: base.DatasetId, source: pd.DataFrame, links: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    deficit_config = output_link.selected_deficit_config(dataset_id, DEFICIT_VARIANT, source)
    link_config = support.selected_link_config(dataset_id, links)
    candidates: tuple[Config | None, ...] = (
        None,
        *tuple(Config(floor, exponent) for floor in SUPPORT_FLOORS for exponent in SCALING_EXPONENTS),
    )
    predictions: dict[str, np.ndarray] = {}
    screen_rows: list[dict[str, Any]] = []
    for config in candidates:
        key = "baseline" if config is None else config.key
        prediction = oof_prediction(dataset, deficit_config, link_config, config, splits)
        summary, _bins = gate.metrics(dataset.target, prediction)
        screen_rows.append({"dataset": dataset_id.value, "config": key, **summary})
        predictions[key] = prediction
    selected_key = min(screen_rows, key=lambda row: (row["rmse"], row["regret_at_1"]))["config"]
    selected = next(config for config in candidates if ("baseline" if config is None else config.key) == selected_key)

    heldout_frame, heldout_weights, heldout_target = base.heldout_data(dataset_id, dataset)
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    metric_rows: list[dict[str, Any]] = []
    for config in (None, selected):
        key = "baseline" if config is None else config.key
        model = fit_model(dataset, deficit_config, link_config, config, np.arange(dataset.n))
        amplitude = 0.0 if config is None else float(model.coefficients[-1])
        for split, observed, predicted in (
            ("fit_oof", dataset.target, predictions[key]),
            ("heldout_policy_matched", heldout_target[policy_mask], model.predict(heldout_weights)[policy_mask]),
        ):
            summary, _bins = gate.metrics(observed, predicted)
            metric_rows.append(
                {
                    "dataset": dataset_id.value,
                    "split": split,
                    "config": key,
                    "support_floor": 0.0 if config is None else config.floor,
                    "scaling_exponent": 0.0 if config is None else config.exponent,
                    "ess_amplitude": amplitude,
                    "parameter_count": 1 + len(model.coefficients),
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
        "# Importance-weight effective-sample-size scaling audit\n\n"
        "Support floor and scaling exponent were selected on fit-panel OOF before heldout scoring.\n\n"
        + metric_frame.to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )
    print(json.dumps(metric_frame.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
