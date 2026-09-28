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
"""Screen noncompensatory support invariants as nested model mechanisms.

This script freezes the strongest pre-search retained-state/inverse-deficit
model and adds exactly one nonnegative mechanism amplitude. Nonlinear support
shapes are selected only by fit-panel grouped OOF RMSE. The coordinate-disjoint
Delphi archive is scored only after that choice.

The support distribution is

``q_i(x) = p_i (x_i / x_i^prop + eps) / sum_j p_j (x_j / x_j^prop + eps)``.

Here ``p`` is the proportional bucket distribution and ``x`` is retained
effective exposure. This separates total exposure, already represented by the
base model, from compositional support mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
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

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
DEFAULT_LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round3_nested_support"
DEFICIT_VARIANT = deficit.Variant.POWER_DEFICIT_HYBRID_EARLY_FAMILY_ASYMMETRIC
SUPPORT_FLOORS = (0.01, 0.03, 0.1, 0.3)
RENYI_ORDERS = (0.5, 1.0, 2.0)
CES_ORDERS = (-4.0, -2.0, -1.0, -0.5)
SERIES_RATES = (0.5, 1.0, 2.0, 4.0)


class Mechanism(StrEnum):
    BASELINE = "baseline"
    BUCKET_REVERSE_KL = "bucket_reverse_kl"
    BUCKET_IMPORTANCE_VARIANCE = "bucket_importance_variance"
    BUCKET_RENYI = "bucket_renyi"
    FAMILY_REVERSE_KL = "family_reverse_kl"
    FAMILY_HARMONIC_BOTTLENECK = "family_harmonic_bottleneck"
    FAMILY_SERIES_RELIABILITY = "family_series_reliability"


@dataclass(frozen=True)
class SupportConfig:
    mechanism: Mechanism
    floor: float = 0.0
    shape: float = 0.0

    @property
    def key(self) -> str:
        if self.mechanism is Mechanism.BASELINE:
            return self.mechanism.value
        return f"{self.mechanism.value}__floor-{self.floor:g}__shape-{self.shape:g}"


@dataclass(frozen=True)
class Model:
    dataset: Any
    deficit_config: deficit.Config
    link_config: output_link.LinkConfig
    support_config: SupportConfig
    floor: float
    intercept: float
    coefficients: np.ndarray
    names: tuple[str, ...]
    effective_degrees_of_freedom: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        values, names, _ridge = combined_design(candidate, self.deficit_config, self.support_config)
        if names != self.names:
            raise ValueError("Prediction design differs from the fitted design")
        latent = self.intercept + values @ self.coefficients
        return output_link.inverse_link(latent, self.link_config.link, self.floor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE_METRICS)
    parser.add_argument("--link-metrics", type=Path, default=DEFAULT_LINK_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def proportional_mass(dataset: Any) -> np.ndarray:
    mass = base.proportional_weights(dataset)
    return mass / mass.sum()


def family_mass(dataset: Any) -> np.ndarray:
    bucket_mass = proportional_mass(dataset)
    mass = np.asarray([bucket_mass[members].sum() for members in dataset.family_members], dtype=float)
    return mass / mass.sum()


def retained_ratio(dataset: Any, config: deficit.Config) -> np.ndarray:
    exposure = base.retained_exposure(dataset, config.base.shape)
    reference = base.proportional_bucket_exposure(dataset, config.base.shape)
    return exposure / np.maximum(reference[None, :], 1e-12)


def family_ratio(bucket_ratio: np.ndarray, dataset: Any) -> np.ndarray:
    bucket_mass = proportional_mass(dataset)
    values = np.empty((len(bucket_ratio), len(dataset.family_members)), dtype=float)
    for index, members in enumerate(dataset.family_members):
        local_mass = bucket_mass[members]
        local_mass /= local_mass.sum()
        values[:, index] = bucket_ratio[:, members] @ local_mass
    return values


def supported_distribution(ratio: np.ndarray, mass: np.ndarray, floor: float) -> np.ndarray:
    unnormalized = mass[None, :] * (np.maximum(ratio, 0.0) + floor)
    return unnormalized / np.maximum(unnormalized.sum(axis=1, keepdims=True), 1e-12)


def reverse_kl(ratio: np.ndarray, mass: np.ndarray, floor: float) -> np.ndarray:
    supported = supported_distribution(ratio, mass, floor)
    return np.sum(mass[None, :] * (np.log(mass[None, :]) - np.log(supported)), axis=1)


def importance_variance(ratio: np.ndarray, mass: np.ndarray, floor: float) -> np.ndarray:
    supported = supported_distribution(ratio, mass, floor)
    chi_square = np.sum(mass[None, :] ** 2 / supported, axis=1) - 1.0
    return np.log1p(np.maximum(chi_square, 0.0))


def renyi_divergence(ratio: np.ndarray, mass: np.ndarray, floor: float, order: float) -> np.ndarray:
    supported = supported_distribution(ratio, mass, floor)
    if abs(order - 1.0) < 1e-10:
        return reverse_kl(ratio, mass, floor)
    moment = np.sum(mass[None, :] ** order * supported ** (1.0 - order), axis=1)
    return np.log(np.maximum(moment, 1e-12)) / (order - 1.0)


def harmonic_bottleneck(ratio: np.ndarray, mass: np.ndarray, floor: float, order: float) -> np.ndarray:
    safe = np.maximum(ratio, 0.0) + floor
    mean = np.power(np.sum(mass[None, :] * np.power(safe, order), axis=1), 1.0 / order)
    reference = 1.0 + floor
    return np.maximum(reference / np.maximum(mean, 1e-12) - 1.0, 0.0)


def series_failure(ratio: np.ndarray, mass: np.ndarray, floor: float, rate: float) -> np.ndarray:
    learned = np.maximum(-np.expm1(-rate * (np.maximum(ratio, 0.0) + floor)), 1e-12)
    reference = max(-math.expm1(-rate * (1.0 + floor)), 1e-12)
    log_reliability = np.log(learned) @ mass
    return np.maximum(math.log(reference) - log_reliability, 0.0)


def support_feature(dataset: Any, config: deficit.Config, support: SupportConfig) -> np.ndarray:
    ratio = retained_ratio(dataset, config)
    bucket_mass = proportional_mass(dataset)
    if support.mechanism is Mechanism.BUCKET_REVERSE_KL:
        return reverse_kl(ratio, bucket_mass, support.floor)
    if support.mechanism is Mechanism.BUCKET_IMPORTANCE_VARIANCE:
        return importance_variance(ratio, bucket_mass, support.floor)
    if support.mechanism is Mechanism.BUCKET_RENYI:
        return renyi_divergence(ratio, bucket_mass, support.floor, support.shape)

    grouped_ratio = family_ratio(ratio, dataset)
    grouped_mass = family_mass(dataset)
    if support.mechanism is Mechanism.FAMILY_REVERSE_KL:
        return reverse_kl(grouped_ratio, grouped_mass, support.floor)
    if support.mechanism is Mechanism.FAMILY_HARMONIC_BOTTLENECK:
        return harmonic_bottleneck(grouped_ratio, grouped_mass, support.floor, support.shape)
    if support.mechanism is Mechanism.FAMILY_SERIES_RELIABILITY:
        return series_failure(grouped_ratio, grouped_mass, support.floor, support.shape)
    raise ValueError(f"Unsupported mechanism {support.mechanism}")


def combined_design(
    dataset: Any,
    config: deficit.Config,
    support: SupportConfig,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    design = deficit.build_design(dataset, config)
    if support.mechanism is Mechanism.BASELINE:
        return design.values, design.names, design.ridge_multipliers
    feature = support_feature(dataset, config, support)
    return (
        np.column_stack([design.values, feature]),
        (*design.names, f"support:{support.key}"),
        np.concatenate([design.ridge_multipliers, np.ones(1, dtype=float)]),
    )


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    support_config: SupportConfig,
    indices: np.ndarray,
) -> Model:
    values, names, ridge_multipliers = combined_design(dataset, deficit_config, support_config)
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
        augmented_x = np.vstack([centered_x, np.diag(penalty)])
        augmented_y = np.concatenate([centered_y, np.zeros(len(penalty), dtype=float)])
    else:
        augmented_x = centered_x
        augmented_y = centered_y
    coefficients, _residual = nnls(augmented_x, augmented_y, maxiter=40 * augmented_x.shape[1])
    intercept = y_mean - float(x_mean @ coefficients)
    active = coefficients > max(1e-10, 1e-6 * float(np.max(coefficients, initial=0.0)))
    if active.any():
        active_x = centered_x[:, active]
        gram = active_x.T @ active_x
        ridge = link_config.l2 * np.diag(ridge_multipliers[active])
        effective_degrees = 1.0 + float(np.trace(np.linalg.solve(gram + ridge, gram)))
    else:
        effective_degrees = 1.0
    return Model(
        dataset=dataset,
        deficit_config=deficit_config,
        link_config=link_config,
        support_config=support_config,
        floor=floor,
        intercept=intercept,
        coefficients=coefficients,
        names=names,
        effective_degrees_of_freedom=effective_degrees,
    )


def oof_prediction(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    support_config: SupportConfig,
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    for train, test in splits:
        model = fit_model(dataset, deficit_config, link_config, support_config, train)
        prediction[test] = model.predict(dataset.weights[test])
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {support_config.key}")
    return prediction


def support_configs() -> tuple[SupportConfig, ...]:
    configs = [SupportConfig(Mechanism.BASELINE)]
    configs.extend(
        SupportConfig(mechanism, floor)
        for mechanism in (
            Mechanism.BUCKET_REVERSE_KL,
            Mechanism.BUCKET_IMPORTANCE_VARIANCE,
            Mechanism.FAMILY_REVERSE_KL,
        )
        for floor in SUPPORT_FLOORS
    )
    configs.extend(
        SupportConfig(Mechanism.BUCKET_RENYI, floor, order) for floor in SUPPORT_FLOORS for order in RENYI_ORDERS
    )
    configs.extend(
        SupportConfig(Mechanism.FAMILY_HARMONIC_BOTTLENECK, floor, order)
        for floor in SUPPORT_FLOORS
        for order in CES_ORDERS
    )
    configs.extend(
        SupportConfig(Mechanism.FAMILY_SERIES_RELIABILITY, floor, rate)
        for floor in SUPPORT_FLOORS
        for rate in SERIES_RATES
    )
    return tuple(configs)


def selected_link_config(dataset_id: base.DatasetId, frame: pd.DataFrame) -> output_link.LinkConfig:
    preferred = output_link.Link.IDENTITY if "uncheatable" in dataset_id.value else output_link.Link.LOG_EXCESS
    selected = frame.loc[
        frame["dataset"].eq(dataset_id.value)
        & frame["deficit_variant"].eq(DEFICIT_VARIANT.value)
        & frame["link"].eq(preferred.value)
        & frame["split"].eq("fit_oof")
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected link for {dataset_id.value}; found {len(selected)}")
    row = selected.iloc[0]
    return output_link.LinkConfig(preferred, float(row["floor_fraction"]), float(row["l2"]))


def model_record(model: Model) -> dict[str, Any]:
    active = model.coefficients > max(1e-10, 1e-6 * float(np.max(model.coefficients, initial=0.0)))
    support_coefficient = 0.0 if model.support_config.mechanism is Mechanism.BASELINE else float(model.coefficients[-1])
    return {
        "mechanism": model.support_config.mechanism.value,
        "support_config": model.support_config.key,
        "support_floor": model.support_config.floor,
        "support_shape": model.support_config.shape,
        "link": model.link_config.link.value,
        "link_floor_fraction": model.link_config.floor_fraction,
        "l2": model.link_config.l2,
        "parameter_count": 1 + len(model.coefficients),
        "active_parameter_count": 1 + int(active.sum()),
        "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
        "support_coefficient": support_coefficient,
        "support_coefficient_active": bool(support_coefficient > 1e-10),
    }


def benchmark_dataset(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), base.SCREEN_SEED)
    deficit_config = output_link.selected_deficit_config(dataset_id, DEFICIT_VARIANT, source_metrics)
    link_config = selected_link_config(dataset_id, link_metrics)
    screen_rows: list[dict[str, Any]] = []
    predictions: dict[SupportConfig, np.ndarray] = {}
    for support in support_configs():
        prediction = oof_prediction(dataset, deficit_config, link_config, support, splits)
        predictions[support] = prediction
        fitted = fit_model(dataset, deficit_config, link_config, support, np.arange(dataset.n))
        summary, _bins = gate.metrics(dataset.target, prediction)
        screen_rows.append(
            {
                "dataset": dataset_id.value,
                **model_record(fitted),
                **summary,
            }
        )

    screen = pd.DataFrame(screen_rows)
    selected_configs: list[SupportConfig] = []
    for mechanism in Mechanism:
        local = screen.loc[screen["mechanism"].eq(mechanism.value)].sort_values(
            ["rmse", "spearman"], ascending=[True, False]
        )
        selected_configs.append(next(config for config in predictions if config.key == local.iloc[0]["support_config"]))

    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(f"Expected frozen heldouts for {dataset_id.value}")
    heldout_frame, heldout_weights, heldout_target = heldout
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    for support in selected_configs:
        fit_prediction = predictions[support]
        model = fit_model(dataset, deficit_config, link_config, support, np.arange(dataset.n))
        fit_summary, fit_bins = gate.metrics(dataset.target, fit_prediction)
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "split": "fit_oof",
                **model_record(model),
                **fit_summary,
            }
        )
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "split": "fit_oof",
                "mechanism": support.mechanism.value,
                "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for index, (observed, predicted) in enumerate(zip(dataset.target, fit_prediction, strict=True))
        )

        heldout_prediction = model.predict(heldout_weights)
        policy_summary, policy_bins = gate.metrics(heldout_target[policy_mask], heldout_prediction[policy_mask])
        metric_rows.append(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched",
                **model_record(model),
                **policy_summary,
            }
        )
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched" if matched else "heldout_off_policy",
                "mechanism": support.mechanism.value,
                "row_id": str(row["wandb_run_name"]),
                "training_series": str(row["training_series"]),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for (_, row), observed, predicted, matched in zip(
                heldout_frame.iterrows(), heldout_target, heldout_prediction, policy_mask, strict=True
            )
        )
        for name, coefficient in zip(model.names, model.coefficients, strict=True):
            parameter_rows.append(
                {
                    "dataset": dataset_id.value,
                    "mechanism": support.mechanism.value,
                    "name": name,
                    "coefficient": coefficient,
                }
            )
        for split, bins in (("fit_oof", fit_bins), ("heldout_policy_matched", policy_bins)):
            for row in bins:
                row.update(
                    {
                        "dataset": dataset_id.value,
                        "split": split,
                        "mechanism": support.mechanism.value,
                    }
                )
    return metric_rows, screen_rows, prediction_rows, parameter_rows


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.source_metrics)
    gate.assert_sealed_absent(args.link_metrics)
    source_metrics = pd.read_csv(args.source_metrics)
    link_metrics = pd.read_csv(args.link_metrics)
    metric_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    for dataset_id in (base.DatasetId.DELPHI_3E18_UNCHEATABLE, base.DatasetId.DELPHI_3E18_TABLE9):
        metrics, screen, predictions, parameters = benchmark_dataset(dataset_id, source_metrics, link_metrics)
        metric_rows.extend(metrics)
        screen_rows.extend(screen)
        prediction_rows.extend(predictions)
        parameter_rows.extend(parameters)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "hyperparameter_screen.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)
    manifest = {
        "deficit_variant": DEFICIT_VARIANT.value,
        "source_metrics": str(args.source_metrics),
        "link_metrics": str(args.link_metrics),
        "selection_rule": "minimum grouped fit-panel OOF RMSE within mechanism; heldouts not consulted",
        "support_configs": [asdict(config) for config in support_configs()],
        "sealed_tokens_checked_absent": list(gate.SEALED_TOKENS),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
