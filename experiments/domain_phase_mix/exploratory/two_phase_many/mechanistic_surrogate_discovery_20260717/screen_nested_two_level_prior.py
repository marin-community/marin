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
"""Test broad-versus-specialist equivalent prior exposure in the frozen model.

All state-transition, response-head, output-link, and ridge settings are frozen
from the strongest pre-search retained-state deficit model. Only the common
deficit floor is relaxed into a foundation prior and a shared specialist prior,
with the mechanistic ordering ``foundation >= specialist`` imposed before
screening. The equal-prior configuration recovers the frozen design exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
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
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717/round7_nested_prior"
SOURCE_METRICS = RESEARCH_DIR / "reference_outputs/hierarchical_deficit_response_20260716/metrics.csv"
LINK_METRICS = RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/metrics.csv"
FOUNDATION_FLOORS = (0.3, 1.0, 3.0)
SPECIALIST_FLOORS = (0.03, 0.1, 0.3, 1.0)


@dataclass(frozen=True)
class PriorConfig:
    foundation_floor: float
    specialist_floor: float

    @property
    def key(self) -> str:
        return f"foundation-{self.foundation_floor:g}__specialist-{self.specialist_floor:g}"


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
    prior_config: PriorConfig
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
        design = build_design(candidate, self.deficit_config, self.prior_config)
        if design.names != self.names:
            raise ValueError("Prediction design differs from fitted design")
        latent = self.intercept + design.values @ self.coefficients
        return output_link.inverse_link(latent, self.link_config.link, self.floor)


def foundation_family_index(dataset: Any) -> int:
    if deficit.FOUNDATION_FAMILY in dataset.family_names:
        return dataset.family_names.index(deficit.FOUNDATION_FAMILY)
    proportional = base.proportional_weights(dataset)
    mass = np.asarray([proportional[members].sum() for members in dataset.family_members])
    return int(np.argmax(mass))


def bucket_floors(dataset: Any, prior: PriorConfig) -> np.ndarray:
    foundation = foundation_family_index(dataset)
    values = np.full(dataset.m, prior.specialist_floor, dtype=float)
    values[dataset.family_members[foundation]] = prior.foundation_floor
    return values


def family_floors(dataset: Any, prior: PriorConfig) -> np.ndarray:
    values = np.full(len(dataset.family_names), prior.specialist_floor, dtype=float)
    values[foundation_family_index(dataset)] = prior.foundation_floor
    return values


def normalized_deficit(
    ratio: np.ndarray,
    floors: np.ndarray,
    config: deficit.Config,
) -> np.ndarray:
    safe = np.maximum(ratio, 0.0) + floors[None, :]
    reference = 1.0 + floors[None, :]
    response = safe ** (-config.base.shape.exponent) - reference ** (-config.base.shape.exponent)
    if config.variant.asymmetric_surplus:
        return np.maximum(response, 0.0) - config.surplus_credit * np.maximum(-response, 0.0)
    return response


def build_design(dataset: Any, config: deficit.Config, prior: PriorConfig) -> Design:
    if config.variant is not frozen.DEFICIT_VARIANT:
        raise ValueError(f"Unsupported frozen variant {config.variant}")
    exposure = base.retained_exposure(dataset, config.base.shape)
    reference = base.proportional_bucket_exposure(dataset, config.base.shape)
    bucket_ratio = exposure / np.maximum(reference[None, :], 1e-12)
    bucket_deficit = normalized_deficit(bucket_ratio, bucket_floors(dataset, prior), config)
    family_total = np.column_stack([exposure[:, members].sum(axis=1) for members in dataset.family_members])
    family_reference = np.asarray([reference[members].sum() for members in dataset.family_members], dtype=float)
    family_ratio = family_total / np.maximum(family_reference[None, :], 1e-12)
    family_deficit = normalized_deficit(family_ratio, family_floors(dataset, prior), config)

    pieces: list[np.ndarray] = []
    names: list[str] = []
    ridge: list[float] = []
    singleton = [members[0] for members in dataset.family_members if len(members) == 1]
    nonsingleton = [
        (name, members)
        for name, members in zip(dataset.family_names, dataset.family_members, strict=True)
        if len(members) > 1
    ]
    if singleton:
        pieces.append(bucket_deficit[:, singleton])
        names.extend(f"net_singleton:{dataset.domains[index]}" for index in singleton)
        ridge.extend([1.0] * len(singleton))
    for family_name, members in nonsingleton:
        pieces.append(bucket_deficit[:, members].sum(axis=1, keepdims=True))
        names.append(f"net_pooled_family:{family_name}")
        ridge.append(1.0)
    if nonsingleton:
        residual_members = np.concatenate([members for _name, members in nonsingleton])
        pieces.append(bucket_deficit[:, residual_members])
        names.extend(f"net_bucket_excess:{dataset.domains[index]}" for index in residual_members)
        ridge.extend([config.base.residual_shrink] * len(residual_members))
        family_indices = [index for index, members in enumerate(dataset.family_members) if len(members) > 1]
        pieces.append(family_deficit[:, family_indices])
        names.extend(f"net_family_coverage:{dataset.family_names[index]}" for index in family_indices)
        ridge.extend([1.0] * len(family_indices))

    family_member_replay = np.column_stack(
        [
            base.overexposure_harm(exposure, config.base.shape.penalty_threshold)[:, members].mean(axis=1)
            for members in dataset.family_members
        ]
    )
    pieces.append(base.overexposure_harm(family_total, config.base.shape.penalty_threshold))
    names.extend(f"family_total_replay:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))
    pieces.append(family_member_replay)
    names.extend(f"family_member_replay:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))

    physical_epochs = dataset.weights[:, 0, :] * dataset.c0[None, :] + dataset.weights[:, 1, :] * dataset.c1[None, :]
    pieces.append((np.maximum(physical_epochs - 1.0, 0.0) ** 2).sum(axis=1, keepdims=True))
    names.append("shared_literal_replay")
    ridge.append(1.0)

    proportional = base.proportional_weights(dataset)
    phase0_family_ratio = np.column_stack(
        [
            dataset.weights[:, 0, members].sum(axis=1) / max(proportional[members].sum(), 1e-12)
            for members in dataset.family_members
        ]
    )
    phase0_deficit = normalized_deficit(phase0_family_ratio, family_floors(dataset, prior), config)
    pieces.append(phase0_deficit)
    names.extend(f"phase0_net_family:{name}" for name in dataset.family_names)
    ridge.extend([1.0] * len(dataset.family_names))

    phase_tv = 0.5 * np.abs(dataset.weights[:, 0, :] - dataset.weights[:, 1, :]).sum(axis=1)
    pieces.append(phase_tv[:, None])
    names.append("phase_shift_tv")
    ridge.append(1.0)
    return Design(np.hstack(pieces), tuple(names), np.asarray(ridge, dtype=float))


def fit_model(
    dataset: Any,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    prior_config: PriorConfig,
    indices: np.ndarray,
) -> Model:
    design = build_design(dataset, deficit_config, prior_config)
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
    coefficients, _residual = nnls(
        np.vstack([centered_x, np.diag(penalty)]),
        np.concatenate([centered_y, np.zeros(len(penalty), dtype=float)]),
        maxiter=40 * centered_x.shape[1],
    )
    intercept = y_mean - float(x_mean @ coefficients)
    active = coefficients > max(1e-10, 1e-6 * float(np.max(coefficients, initial=0.0)))
    if active.any():
        active_x = centered_x[:, active]
        gram = active_x.T @ active_x
        ridge_matrix = link_config.l2 * np.diag(design.ridge_multipliers[active])
        effective_degrees = 1.0 + float(np.trace(np.linalg.solve(gram + ridge_matrix, gram)))
    else:
        effective_degrees = 1.0
    return Model(
        dataset,
        deficit_config,
        link_config,
        prior_config,
        floor,
        intercept,
        coefficients,
        design.names,
        effective_degrees,
    )


def oof_prediction(
    dataset: Any,
    dataset_id: base.DatasetId,
    deficit_config: deficit.Config,
    link_config: output_link.LinkConfig,
    prior_config: PriorConfig,
    seeds: tuple[int, ...],
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for seed in seeds:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in base.split_indices(dataset, dataset_id, np.arange(dataset.n), seed):
            model = fit_model(dataset, deficit_config, link_config, prior_config, train)
            prediction[test] = model.predict(dataset.weights[test])
        if not np.isfinite(prediction).all():
            raise RuntimeError("Incomplete OOF prediction")
        predictions.append(prediction)
    return np.mean(predictions, axis=0)


def prior_configs() -> tuple[PriorConfig, ...]:
    return tuple(
        PriorConfig(foundation, specialist)
        for foundation in FOUNDATION_FLOORS
        for specialist in SPECIALIST_FLOORS
        if foundation >= specialist
    )


def evaluate_dataset(
    dataset_id: base.DatasetId,
    source_metrics: pd.DataFrame,
    link_metrics: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = base.load_dataset(dataset_id)
    deficit_config = output_link.selected_deficit_config(dataset_id, frozen.DEFICIT_VARIANT, source_metrics)
    link_config = frozen.selected_link_config(dataset_id, link_metrics)
    common = PriorConfig(deficit_config.deficit_floor, deficit_config.deficit_floor)
    original = deficit.build_design(dataset, deficit_config)
    nested = build_design(dataset, deficit_config, common)
    nested_error = float(np.max(np.abs(original.values - nested.values)))
    if original.names != nested.names or nested_error > 1e-12:
        raise AssertionError(f"Common-prior nested design mismatch: {nested_error}")

    screen_rows: list[dict[str, Any]] = []
    selected: tuple[float, float, PriorConfig] | None = None
    for prior in prior_configs():
        prediction = oof_prediction(
            dataset,
            dataset_id,
            deficit_config,
            link_config,
            prior,
            seeds=(base.SCREEN_SEED,),
        )
        summary, _bins = gate.metrics(dataset.target, prediction)
        screen_rows.append(
            {
                "dataset": dataset_id.value,
                **asdict(prior),
                "prior_config": prior.key,
                **summary,
            }
        )
        candidate = (float(summary["rmse"]), -float(summary["spearman"]), prior)
        if selected is None or candidate[:2] < selected[:2]:
            selected = candidate
    if selected is None:
        raise RuntimeError("No prior configuration screened")
    selected_prior = selected[2]
    heldout = base.heldout_data(dataset_id, dataset)
    if heldout is None:
        raise ValueError("Expected heldout archive")
    heldout_frame, heldout_weights, heldout_target = heldout
    policy_mask = heldout_frame["policy_class"].eq("two_phase").to_numpy()
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for label, prior in (("common_prior", common), ("two_level_prior", selected_prior)):
        oof = oof_prediction(
            dataset,
            dataset_id,
            deficit_config,
            link_config,
            prior,
            seeds=(0, 1, 2),
        )
        model = fit_model(dataset, deficit_config, link_config, prior, np.arange(dataset.n))
        fit_summary, _bins = gate.metrics(dataset.target, oof)
        metrics_rows.append(
            {
                "dataset": dataset_id.value,
                "split": "fit_oof",
                "mechanism": label,
                **asdict(prior),
                "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
                **fit_summary,
            }
        )
        heldout_prediction = model.predict(heldout_weights)
        heldout_summary, _bins = gate.metrics(heldout_target[policy_mask], heldout_prediction[policy_mask])
        metrics_rows.append(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched",
                "mechanism": label,
                **asdict(prior),
                "effective_degrees_of_freedom": model.effective_degrees_of_freedom,
                **heldout_summary,
            }
        )
        prediction_rows.extend(
            {
                "dataset": dataset_id.value,
                "split": "heldout_policy_matched" if matched else "heldout_off_policy",
                "mechanism": label,
                "row_id": str(row["wandb_run_name"]),
                "training_series": str(row["training_series"]),
                "observed": float(observed),
                "predicted": float(predicted),
            }
            for (_, row), observed, predicted, matched in zip(
                heldout_frame.iterrows(), heldout_target, heldout_prediction, policy_mask, strict=True
            )
        )
    return metrics_rows, screen_rows, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    for path in (SOURCE_METRICS, LINK_METRICS):
        gate.assert_sealed_absent(path)
    source_metrics = pd.read_csv(SOURCE_METRICS)
    link_metrics = pd.read_csv(LINK_METRICS)
    metrics_rows: list[dict[str, Any]] = []
    screen_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset_id in (
        base.DatasetId.DELPHI_3E18_UNCHEATABLE,
        base.DatasetId.DELPHI_3E18_TABLE9,
    ):
        local = evaluate_dataset(dataset_id, source_metrics, link_metrics)
        metrics_rows.extend(local[0])
        screen_rows.extend(local[1])
        prediction_rows.extend(local[2])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_frame.to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(screen_rows).to_csv(args.output_dir / "fit_only_prior_screen.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    manifest = {
        "foundation_floors": FOUNDATION_FLOORS,
        "specialist_floors": SPECIALIST_FLOORS,
        "ordering": "foundation_floor >= specialist_floor",
        "selection_boundary": "prior scales selected on fit-panel grouped OOF only",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    columns = [
        "dataset",
        "split",
        "mechanism",
        "foundation_floor",
        "specialist_floor",
        "rmse",
        "spearman",
        "regret_at_1",
        "calibration_slope_observed_on_predicted",
        "optimism_gt_0p05_count",
        "worst_optimism",
    ]
    print(metrics_frame[columns].to_string(index=False))


if __name__ == "__main__":
    main()
