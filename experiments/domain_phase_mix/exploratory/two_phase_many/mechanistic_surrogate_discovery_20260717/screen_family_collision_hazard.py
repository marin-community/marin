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
"""Test family-specific physical replay hazards as an exact nested mechanism.

The frozen strongest deficit model contains one shared physical collision term
``sum_i max(epoch_i - 1, 0)^2``. This script replaces that one column with its
per-family summands. Equal family coefficients recover the original model
exactly; the extension asks whether repeated examples from different semantic
families have measurably different harm rates. All selection remains on the
fit panel. The coordinate-disjoint 3e18 archive is scored only afterward.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
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
    screen_nested_support_invariants as frozen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round6_family_collision"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
SHARED_REPLAY_NAME = "shared_literal_replay"


class ReplayMechanism(StrEnum):
    SHARED_COLLISION = "shared_collision"
    FAMILY_COLLISION = "family_collision"
    FAMILY_DUPLICATE_MASS = "family_duplicate_mass"


@dataclass(frozen=True)
class Design:
    values: np.ndarray
    names: tuple[str, ...]
    ridge_multipliers: np.ndarray


@dataclass(frozen=True)
class Model:
    dataset: Any
    deficit_config: deficit.Config
    link_config: output_link.LinkConfig
    mechanism: ReplayMechanism
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
        design = replay_design(candidate, self.deficit_config, self.mechanism)
        if design.names != self.names:
            raise ValueError("Prediction design differs from fitted design")
        latent = self.intercept + design.values @ self.coefficients
        return output_link.inverse_link(latent, self.link_config.link, self.floor)


def physical_epochs(dataset: Any) -> np.ndarray:
    return dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :]


def family_replay_features(dataset: Any, mechanism: ReplayMechanism) -> np.ndarray:
    epochs = np.maximum(physical_epochs(dataset), 0.0)
    if mechanism is ReplayMechanism.FAMILY_COLLISION:
        bucket_harm = np.maximum(epochs - 1.0, 0.0) ** 2
    elif mechanism is ReplayMechanism.FAMILY_DUPLICATE_MASS:
        bucket_harm = epochs + np.expm1(-epochs)
    else:
        raise ValueError(mechanism)
    return np.column_stack([bucket_harm[:, members].sum(axis=1) for members in dataset.family_members])


def replay_design(dataset: Any, config: deficit.Config, mechanism: ReplayMechanism) -> Design:
    original = deficit.build_design(dataset, config)
    if mechanism is ReplayMechanism.SHARED_COLLISION:
        return Design(original.values, original.names, original.ridge_multipliers)
    replay_index = original.names.index(SHARED_REPLAY_NAME)
    replay = family_replay_features(dataset, mechanism)
    values = np.column_stack([original.values[:, :replay_index], replay, original.values[:, replay_index + 1 :]])
    names = (
        *original.names[:replay_index],
        *(f"{mechanism.value}:{name}" for name in dataset.family_names),
        *original.names[replay_index + 1 :],
    )
    ridge = np.concatenate(
        [
            original.ridge_multipliers[:replay_index],
            np.ones(len(dataset.family_names), dtype=float),
            original.ridge_multipliers[replay_index + 1 :],
        ]
    )
    return Design(values, names, ridge)


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    mechanism: ReplayMechanism,
    indices: np.ndarray,
) -> Model:
    design = replay_design(dataset, deficit_config, mechanism)
    floor = (
        0.0
        if link_config.link is output_link.Link.IDENTITY
        else link_config.floor_fraction * float(dataset.target.min())
    )
    target = output_link.transformed_target(dataset.target[indices], link_config.link, floor)
    x = design.values[indices]
    x_mean = x.mean(axis=0)
    y_mean = float(target.mean())
    centered_x = x - x_mean
    centered_y = target - y_mean
    penalty = np.sqrt(link_config.l2 * design.ridge_multipliers)
    augmented_x = np.vstack([centered_x, np.diag(penalty)])
    augmented_y = np.concatenate([centered_y, np.zeros(len(penalty), dtype=float)])
    coefficients, _residual = nnls(augmented_x, augmented_y, maxiter=40 * augmented_x.shape[1])
    intercept = y_mean - float(x_mean @ coefficients)
    active = coefficients > max(1e-10, 1e-6 * float(np.max(coefficients, initial=0.0)))
    if active.any():
        active_x = centered_x[:, active]
        gram = active_x.T @ active_x
        ridge = link_config.l2 * np.diag(design.ridge_multipliers[active])
        effective_degrees = 1.0 + float(np.trace(np.linalg.solve(gram + ridge, gram)))
    else:
        effective_degrees = 1.0
    return Model(
        dataset=dataset,
        deficit_config=deficit_config,
        link_config=link_config,
        mechanism=mechanism,
        floor=floor,
        intercept=intercept,
        coefficients=coefficients,
        names=design.names,
        effective_degrees_of_freedom=effective_degrees,
    )


def oof_prediction(
    dataset: Any,
    dataset_id: base.DatasetId,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    mechanism: ReplayMechanism,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    predictions: list[np.ndarray] = []
    stability: list[dict[str, float | str]] = []
    for seed in seeds:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        splits = base.split_indices(dataset, dataset_id, np.arange(dataset.n), seed)
        for fold, (train, test) in enumerate(splits):
            model = fit_model(dataset, deficit_config, link_config, mechanism, train)
            prediction[test] = model.predict(dataset.weights[test])
            for name, coefficient in zip(model.names, model.coefficients, strict=True):
                if name.startswith(f"{mechanism.value}:") or name == SHARED_REPLAY_NAME:
                    stability.append(
                        {
                            "seed": float(seed),
                            "fold": float(fold),
                            "mechanism": mechanism.value,
                            "feature": name,
                            "coefficient": float(coefficient),
                        }
                    )
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete OOF prediction")
        predictions.append(prediction)
    return np.mean(predictions, axis=0), stability


def model_record(model: Model) -> dict[str, float | int | str]:
    active = model.coefficients > max(1e-10, 1e-6 * float(np.max(model.coefficients, initial=0.0)))
    return {
        "mechanism": model.mechanism.value,
        "link": model.link_config.link.value,
        "l2": model.link_config.l2,
        "parameter_count": 1 + len(model.coefficients),
        "active_parameter_count": 1 + int(active.sum()),
        "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
    }


def evaluate_dataset(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    deficit_config = output_link.selected_deficit_config(dataset_id, frozen.DEFICIT_VARIANT, source_metrics)
    link_config = frozen.selected_link_config(dataset_id, link_metrics)
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError(f"Expected frozen heldouts for {dataset_id.value}")
    heldout_frame, heldout_weights, heldout_target = heldout
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for mechanism in ReplayMechanism:
        oof, stability = oof_prediction(
            dataset,
            dataset_id,
            deficit_config,
            link_config,
            mechanism,
            seeds=(0, 1, 2),
        )
        stability_rows.extend({"dataset": dataset_id.value, **row} for row in stability)
        model = fit_model(
            dataset,
            deficit_config,
            link_config,
            mechanism,
            np.arange(dataset.n),
        )
        summary, _bins = gate.metrics(dataset.target, oof)
        metrics_rows.append({"dataset": dataset_id.value, "split": "fit_oof", **model_record(model), **summary})
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "split": "fit_oof",
                "mechanism": mechanism.value,
                "row_id": str(dataset.frame.iloc[index].get("run_name", index)),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for index, (observed, predicted) in enumerate(zip(dataset.target, oof, strict=True))
        )
        heldout_prediction = model.predict(heldout_weights)
        heldout_summary, _bins = gate.metrics(heldout_target[policy_mask], heldout_prediction[policy_mask])
        metrics_rows.append(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched",
                **model_record(model),
                **heldout_summary,
            }
        )
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched" if matched else "heldout_off_policy",
                "mechanism": mechanism.value,
                "row_id": str(row["wandb_run_name"]),
                "training_series": str(row["training_series"]),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for (_, row), observed, predicted, matched in zip(
                heldout_frame.iterrows(), heldout_target, heldout_prediction, policy_mask, strict=True
            )
        )
        parameter_rows.extend(
            {
                "dataset": dataset_id.value,
                "mechanism": mechanism.value,
                "feature": name,
                "coefficient": float(coefficient),
            }
            for name, coefficient in zip(model.names, model.coefficients, strict=True)
        )
    return metrics_rows, prediction_rows, parameter_rows, stability_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for dataset_id in (
        base.DatasetId.DELPHI_3E18_UNCHEATABLE,
        base.DatasetId.DELPHI_3E18_TABLE9,
    ):
        local = evaluate_dataset(dataset_id, source_metrics, link_metrics)
        metrics_rows.extend(local[0])
        prediction_rows.extend(local[1])
        parameter_rows.extend(local[2])
        stability_rows.extend(local[3])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(args.output_dir / "parameters.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(args.output_dir / "fold_stability.csv", index=False)
    manifest = {
        "deficit_variant": frozen.DEFICIT_VARIANT.value,
        "source_metrics": str(SOURCE_METRICS),
        "link_metrics": str(LINK_METRICS),
        "mechanisms": [item.value for item in ReplayMechanism],
        "selection_boundary": "all nonlinear settings frozen before scoring coordinate-disjoint heldouts",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    columns = [
        "dataset",
        "split",
        "mechanism",
        "rmse",
        "spearman",
        "regret_at_1",
        "calibration_slope_observed_on_predicted",
        "optimism_gt_0p05_count",
        "worst_optimism",
        "effective_degrees_of_freedom",
    ]
    print(metrics_frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
