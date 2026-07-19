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
"""Test a finite-corpus collision/ESS mechanism nested in the frozen model.

For a phase mixture ``w`` over corpora with sizes ``S_i``, the probability
that two independently drawn token identities collide is ``sum_i w_i^2/S_i``.
Multiplying by phase token count gives the dimensionless collision load
``sum_i w_i e_i``. The associated effective-sample-size loss is represented by
``phi_beta(C)=(1+C)^beta`` (or ``log(1+C)`` at beta zero), centered at the
proportional policy. A nonnegative BPB amplitude is the only fitted extension.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
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
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round12_kish_collision"
DEFICIT_VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
BETAS = (0.0, 0.25, 0.5, 1.0)


class Mechanism(StrEnum):
    BASELINE = "baseline"
    AGGREGATE_COLLISION = "aggregate_collision"
    WITHIN_PHASE_COLLISION = "within_phase_collision"


@dataclass(frozen=True)
class Config:
    mechanism: Mechanism
    beta: float = 0.0

    @property
    def key(self) -> str:
        if self.mechanism is Mechanism.BASELINE:
            return self.mechanism.value
        return f"{self.mechanism.value}__beta-{self.beta:g}"


@dataclass(frozen=True)
class Model:
    dataset: Any
    deficit_config: deficit.Config
    link_config: output_link.LinkConfig
    collision_config: Config
    floor: float
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]
    effective_degrees_of_freedom: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = type(self.dataset)(
            **{
                **self.dataset.__dict__,
                "weights": np.asarray(weights, dtype=float),
                "target": np.zeros(len(weights), dtype=float),
            }
        )
        values, names, _ridge = combined_design(candidate, self.deficit_config, self.collision_config)
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


def phase_fraction(dataset: Any) -> float:
    ratio = dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)
    if not np.allclose(ratio, np.median(ratio), atol=1e-10):
        raise ValueError("Phase fraction differs by bucket")
    return float(np.median(ratio))


def collision_load(dataset: Any, weights: np.ndarray, mechanism: Mechanism) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    e0 = weights[:, 0] * dataset.c0
    e1 = weights[:, 1] * dataset.c1
    if mechanism is Mechanism.WITHIN_PHASE_COLLISION:
        return np.sum(weights[:, 0] * e0 + weights[:, 1] * e1, axis=1)
    if mechanism is Mechanism.AGGREGATE_COLLISION:
        gamma = phase_fraction(dataset)
        aggregate = gamma * weights[:, 0] + (1.0 - gamma) * weights[:, 1]
        return np.sum(aggregate * (e0 + e1), axis=1)
    raise ValueError(mechanism)


def ess_response(load: np.ndarray, beta: float) -> np.ndarray:
    if abs(beta) < 1e-12:
        return np.log1p(load)
    return np.power(1.0 + load, beta)


def collision_feature(dataset: Any, config: Config) -> np.ndarray:
    load = collision_load(dataset, dataset.weights, config.mechanism)
    proportional = base.proportional_weights(dataset)
    reference_weights = np.broadcast_to(proportional, (1, 2, len(proportional)))
    reference_load = float(collision_load(dataset, reference_weights, config.mechanism).item())
    return ess_response(load, config.beta) - float(ess_response(np.asarray([reference_load]), config.beta).item())


def combined_design(
    dataset: Any, deficit_config: deficit.Config, config: Config
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    design = deficit.build_design(dataset, deficit_config)
    if config.mechanism is Mechanism.BASELINE:
        return design.values, design.names, design.ridge_multipliers
    feature = collision_feature(dataset, config)
    return (
        np.column_stack([design.values, feature]),
        (*design.names, f"ess_cost:{config.key}"),
        np.concatenate([design.ridge_multipliers, np.ones(1, dtype=float)]),
    )


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    config: Config,
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
        fit_y = np.concatenate([centered_y, np.zeros(len(penalty), dtype=float)])
    else:
        fit_x, fit_y = centered_x, centered_y
    coefficients, _residual = nnls(fit_x, fit_y, maxiter=40 * fit_x.shape[1])
    intercept = y_mean - float(x_mean @ coefficients)
    active = coefficients > max(1e-10, 1e-6 * float(np.max(coefficients, initial=0.0)))
    if active.any():
        gram = centered_x[:, active].T @ centered_x[:, active]
        ridge = link_config.l2 * np.diag(ridge_multipliers[active])
        effective_df = 1.0 + float(np.trace(np.linalg.solve(gram + ridge, gram)))
    else:
        effective_df = 1.0
    return Model(dataset, deficit_config, link_config, config, floor, intercept, coefficients, names, effective_df)


def oof_prediction(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    config: Config,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        prediction[test] = fit_model(dataset, deficit_config, link_config, config, train).predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Incomplete OOF prediction for {config.key}")
    return prediction


def configs() -> tuple[Config, ...]:
    return (
        Config(Mechanism.BASELINE),
        *tuple(
            Config(mechanism, beta)
            for mechanism in (Mechanism.AGGREGATE_COLLISION, Mechanism.WITHIN_PHASE_COLLISION)
            for beta in BETAS
        ),
    )


def model_record(model: Model) -> dict[str, Any]:
    coefficient = 0.0 if model.collision_config.mechanism is Mechanism.BASELINE else float(model.coefficients[-1])
    return {
        "mechanism": model.collision_config.mechanism.value,
        "config": model.collision_config.key,
        "beta": model.collision_config.beta,
        "parameter_count": 1 + len(model.coefficients),
        "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
        "collision_coefficient": coefficient,
        "collision_coefficient_active": coefficient > 1e-10,
    }


def benchmark(
    dataset_id: base.DatasetId, source: pd.DataFrame, links: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    deficit_config = output_link.selected_deficit_config(dataset_id, DEFICIT_VARIANT, source)
    link_config = support.selected_link_config(dataset_id, links)
    screen_rows = []
    predictions: dict[Config, np.ndarray] = {}
    for config in configs():
        prediction = oof_prediction(dataset, deficit_config, link_config, config, splits)
        model = fit_model(dataset, deficit_config, link_config, config, np.arange(dataset.n))
        summary, _bins = gate.metrics(dataset.target, prediction)
        screen_rows.append({"dataset": dataset_id.value, **model_record(model), **summary})
        predictions[config] = prediction
    screen_frame = pd.DataFrame(screen_rows)
    selected = [Config(Mechanism.BASELINE)]
    for mechanism in (Mechanism.AGGREGATE_COLLISION, Mechanism.WITHIN_PHASE_COLLISION):
        row = (
            screen_frame.loc[screen_frame["mechanism"].eq(mechanism.value)].sort_values(["rmse", "regret_at_1"]).iloc[0]
        )
        selected.append(next(config for config in predictions if config.key == row["config"]))

    heldout_frame, heldout_weights, heldout_target = base.heldout_data(dataset_id, dataset)
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    metric_rows = []
    prediction_rows = []
    for config in selected:
        model = fit_model(dataset, deficit_config, link_config, config, np.arange(dataset.n))
        for split, observed, predicted, row_ids in (
            ("fit_oof", dataset.target, predictions[config], dataset.frame["run_name"].astype(str).tolist()),
            (
                "heldout_policy_matched",
                heldout_target[policy_mask],
                model.predict(heldout_weights)[policy_mask],
                heldout_frame.loc[policy_mask, "wandb_run_name"].astype(str).tolist(),
            ),
        ):
            summary, _bins = gate.metrics(observed, predicted)
            metric_rows.append({"dataset": dataset_id.value, "split": split, **model_record(model), **summary})
            prediction_rows.extend(
                {
                    "dataset": dataset_id.value,
                    "split": split,
                    "mechanism": config.mechanism.value,
                    "row_id": row_id,
                    "observed": y,
                    "predicted": prediction,
                }
                for row_id, y, prediction in zip(row_ids, observed, predicted, strict=True)
            )
    return metric_rows, screen_rows, prediction_rows


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.source_metrics)
    gate.assert_sealed_absent(args.link_metrics)
    source = pd.read_csv(args.source_metrics)
    links = pd.read_csv(args.link_metrics)
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9):
        metrics, screen, predictions = benchmark(dataset_id, source, links)
        metric_rows.extend(metrics)
        screen_rows.extend(screen)
        prediction_rows.extend(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "report.md").write_text(
        "# Finite-corpus collision effective-sample-size audit\n\n"
        "Collision shape was selected on fit-panel OOF predictions before heldout scoring.\n\n"
        + metrics.to_markdown(index=False, floatfmt=".6f")
        + "\n"
    )
    print(json.dumps(metrics.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
