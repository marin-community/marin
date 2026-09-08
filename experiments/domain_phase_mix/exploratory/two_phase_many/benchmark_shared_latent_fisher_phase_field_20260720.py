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
"""Test a shared latent Fisher-tangent phase field across two targets."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_family_state_phase_surrogate_20260720 as family_state,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_heterogeneous_design_aware_hpr_20260719 as heterogeneous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_fisher_phase_field_20260720 as fisher,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_pair,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/shared_latent_fisher_phase_field_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
FAMILY_RIDGES = (0.01, 0.1, 1.0, 10.0)
LATENT_PRECISION_RATIOS = (1.0, 10.0, 100.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
PAIR_RATIO_GATE = 0.9
FIBER_RATIO_GATE = 1.05
STABILITY_GATE = 0.5
MAX_ALTERNATIONS = 100
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class MultiTargetRows:
    features: np.ndarray
    feature_blocks: np.ndarray
    targets: np.ndarray
    target_names: tuple[str, ...]
    source: np.ndarray
    groups: np.ndarray
    blocks: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class LatentFit:
    family_coefficients: np.ndarray
    latent_direction: np.ndarray
    target_loadings: np.ndarray
    feature_scale: np.ndarray
    target_scale: np.ndarray
    family_ridge: float
    latent_precision_ratio: float
    objective: float

    def predict_features(self, features: np.ndarray, feature_blocks: np.ndarray) -> np.ndarray:
        standardized = features / self.feature_scale[None, :]
        family = standardized[:, feature_blocks != "bucket"] @ self.family_coefficients
        latent = standardized[:, feature_blocks == "bucket"] @ self.latent_direction
        normalized = family + latent[:, None] * self.target_loadings[None, :]
        return normalized * self.target_scale[None, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def multi_target_rows(
    matched: matched_pair.MatchedSources,
    dataset: object,
    basis: fisher.FisherBasis,
) -> MultiTargetRows:
    target_names = tuple(heterogeneous.TARGETS)
    target_rows = [fisher.phase_rows(matched, dataset, basis, target) for target in target_names]
    reference = target_rows[0]
    for rows in target_rows[1:]:
        if not np.allclose(rows.features, reference.features):
            raise ValueError("Target feature matrices differ")
        for field in ("source", "groups", "blocks", "row_ids"):
            if not np.array_equal(getattr(rows, field), getattr(reference, field)):
                raise ValueError(f"Target row field differs: {field}")
    return MultiTargetRows(
        features=reference.features,
        feature_blocks=reference.feature_blocks,
        targets=np.column_stack([rows.target for rows in target_rows]),
        target_names=target_names,
        source=reference.source,
        groups=reference.groups,
        blocks=reference.blocks,
        row_ids=reference.row_ids,
    )


def whitened_system(rows: MultiTargetRows, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selected = np.asarray(indices, dtype=int)
    for source in ("pair", "fiber"):
        source_indices = selected[rows.source[selected] == source]
        design = rows.features[source_indices]
        target = rows.targets[source_indices]
        if source == "fiber":
            block_designs: list[np.ndarray] = []
            block_targets: list[np.ndarray] = []
            for block in np.unique(rows.blocks[source_indices]):
                local = source_indices[rows.blocks[source_indices] == block]
                whitening = heterogeneous.inverse_sqrt_shared_center_covariance(len(local))
                block_designs.append(whitening @ rows.features[local])
                block_targets.append(whitening @ rows.targets[local])
            design = np.vstack(block_designs)
            target = np.vstack(block_targets)
        source_scale = 1.0 / math.sqrt(len(target))
        designs.append(source_scale * design)
        targets.append(source_scale * target)
    return np.vstack(designs), np.vstack(targets)


def fit_latent(
    rows: MultiTargetRows,
    indices: np.ndarray,
    family_ridge: float,
    latent_precision_ratio: float,
) -> LatentFit:
    design, target = whitened_system(rows, indices)
    feature_scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-12)
    target_scale = np.maximum(np.sqrt(np.mean(target**2, axis=0)), 1e-12)
    standardized = design / feature_scale[None, :]
    normalized_target = target / target_scale[None, :]
    family_design = standardized[:, rows.feature_blocks != "bucket"]
    bucket_design = standardized[:, rows.feature_blocks == "bucket"]
    latent_ridge = family_ridge * latent_precision_ratio

    full_penalty = np.concatenate(
        [
            np.full(family_design.shape[1], family_ridge),
            np.full(bucket_design.shape[1], latent_ridge),
        ]
    )
    full_design = np.column_stack([family_design, bucket_design])
    information = full_design.T @ full_design + np.diag(full_penalty)
    full_coefficients = np.linalg.solve(information, full_design.T @ normalized_target)
    bucket_coefficients = full_coefficients[family_design.shape[1] :]
    left, singular, right = np.linalg.svd(bucket_coefficients, full_matrices=False)
    if singular[0] <= 1e-12:
        latent_direction = np.zeros(bucket_design.shape[1])
        latent_direction[0] = 1.0
        target_loadings = np.zeros(normalized_target.shape[1])
    else:
        latent_direction = left[:, 0] * math.sqrt(singular[0])
        target_loadings = right[0] * math.sqrt(singular[0])
    family_coefficients = full_coefficients[: family_design.shape[1]].copy()

    previous: float | None = None
    for _ in range(MAX_ALTERNATIONS):
        latent_feature = bucket_design @ latent_direction
        target_coefficients = []
        for target_index in range(normalized_target.shape[1]):
            local_design = np.column_stack([family_design, latent_feature])
            penalties = np.concatenate([np.full(family_design.shape[1], family_ridge), np.asarray([latent_ridge])])
            augmented = np.vstack([local_design, np.diag(np.sqrt(penalties))])
            augmented_target = np.concatenate([normalized_target[:, target_index], np.zeros(local_design.shape[1])])
            lower = np.full(local_design.shape[1], -np.inf)
            information_positions = np.flatnonzero(rows.feature_blocks != "bucket")[
                rows.feature_blocks[rows.feature_blocks != "bucket"] == "information"
            ]
            lower[information_positions] = 0.0
            result = lsq_linear(
                augmented,
                augmented_target,
                bounds=(lower, np.full(local_design.shape[1], np.inf)),
                max_iter=5_000,
                lsmr_tol="auto",
            )
            if not result.success:
                raise RuntimeError(f"Target-head fit failed: {result.message}")
            target_coefficients.append(result.x)
        target_coefficients_array = np.column_stack(target_coefficients)
        family_coefficients = target_coefficients_array[:-1]
        target_loadings = target_coefficients_array[-1]

        residual = normalized_target - family_design @ family_coefficients
        loading_power = float(target_loadings @ target_loadings)
        latent_information = loading_power * (bucket_design.T @ bucket_design) + latent_ridge * np.eye(
            bucket_design.shape[1]
        )
        latent_direction = np.linalg.solve(latent_information, bucket_design.T @ (residual @ target_loadings))
        prediction = family_design @ family_coefficients + (bucket_design @ latent_direction)[:, None] * target_loadings
        objective = float(
            np.sum((prediction - normalized_target) ** 2)
            + family_ridge * np.sum(family_coefficients**2)
            + latent_ridge * (np.sum(latent_direction**2) + np.sum(target_loadings**2))
        )
        if previous is not None and abs(previous - objective) <= 1e-10 * max(1.0, previous):
            break
        previous = objective

    return LatentFit(
        family_coefficients=family_coefficients,
        latent_direction=latent_direction,
        target_loadings=target_loadings,
        feature_scale=feature_scale,
        target_scale=target_scale,
        family_ridge=family_ridge,
        latent_precision_ratio=latent_precision_ratio,
        objective=objective,
    )


def source_ratios(rows: MultiTargetRows, prediction: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    selected = np.asarray(indices, dtype=int)
    result: dict[str, float] = {}
    for source in ("pair", "fiber"):
        local = selected[rows.source[selected] == source]
        for target_index, target_name in enumerate(rows.target_names):
            observed = rows.targets[local, target_index]
            predicted = prediction[local, target_index]
            rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            zero = float(np.sqrt(np.mean(observed**2)))
            result[f"{target_name}_{source}_rmse"] = rmse
            result[f"{target_name}_{source}_zero_rmse"] = zero
            result[f"{target_name}_{source}_rmse_ratio"] = rmse / zero
    return result


def inner_hyperparameters(rows: MultiTargetRows, indices: np.ndarray, salt: str) -> tuple[float, float]:
    selected = np.asarray(indices, dtype=int)
    folds = family_state.grouped_balanced_folds(rows, selected, INNER_FOLDS, salt)
    scores = []
    for family_ridge in FAMILY_RIDGES:
        for ratio in LATENT_PRECISION_RATIOS:
            ratios = []
            for fold in range(INNER_FOLDS):
                train = selected[folds != fold]
                test = selected[folds == fold]
                model = fit_latent(rows, train, family_ridge, ratio)
                prediction = np.full_like(rows.targets, np.nan)
                prediction[test] = model.predict_features(rows.features[test], rows.feature_blocks)
                metrics = source_ratios(rows, prediction, test)
                ratios.extend(value for name, value in metrics.items() if name.endswith("_rmse_ratio"))
            scores.append((float(np.mean(ratios)), family_ridge, ratio))
    _, family_ridge, ratio = min(scores)
    return family_ridge, ratio


def nested_oof(rows: MultiTargetRows) -> tuple[np.ndarray, tuple[LatentFit, ...], list[dict[str, Any]]]:
    indices = np.arange(len(rows.targets))
    folds = family_state.grouped_balanced_folds(rows, indices, OUTER_FOLDS, "shared_latent_outer")
    prediction = np.full_like(rows.targets, np.nan)
    models: list[LatentFit] = []
    selections: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        train = indices[folds != fold]
        test = indices[folds == fold]
        family_ridge, ratio = inner_hyperparameters(rows, train, f"shared_latent_inner::{fold}")
        model = fit_latent(rows, train, family_ridge, ratio)
        prediction[test] = model.predict_features(rows.features[test], rows.feature_blocks)
        models.append(model)
        selections.append({"fold": fold, "family_ridge": family_ridge, "latent_precision_ratio": ratio})
    return prediction, tuple(models), selections


def direction_stability(models: tuple[LatentFit, ...]) -> dict[str, float]:
    directions = np.stack(
        [model.latent_direction / np.maximum(np.linalg.norm(model.latent_direction), 1e-12) for model in models]
    )
    cosine = np.abs(directions @ directions.T)
    upper = cosine[np.triu_indices(len(models), k=1)]
    return {
        "shared_direction_absolute_cosine_mean": float(np.mean(upper)),
        "shared_direction_absolute_cosine_min": float(np.min(upper)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = output_dir / "preregistered_candidates.json"
    if not preregistration.exists():
        if output_dir == DEFAULT_OUTPUT_DIR:
            raise FileNotFoundError(f"Missing preregistration {PREREGISTRATION_PATH}")
        preregistration.write_text(PREREGISTRATION_PATH.read_text())

    matched = matched_pair.matched_sources()
    dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.broad.frame,
        matched.sources.broad.weights,
        "uncheatable",
        "shared_latent_fisher_reference",
    )
    basis = fisher.fisher_basis(dataset)
    rows = multi_target_rows(matched, dataset, basis)
    prediction, models, selections = nested_oof(rows)
    metrics = source_ratios(rows, prediction, np.arange(len(rows.targets)))
    stability = direction_stability(models)
    metric_rows = []
    for target in rows.target_names:
        metric_rows.append(
            {
                "target": target,
                **{
                    name.removeprefix(f"{target}_"): value
                    for name, value in metrics.items()
                    if name.startswith(f"{target}_")
                },
                **stability,
            }
        )
    metrics_frame = pd.DataFrame(metric_rows)
    selections_frame = pd.DataFrame(selections)
    prediction_rows = []
    for row_index, row_id in enumerate(rows.row_ids):
        for target_index, target in enumerate(rows.target_names):
            prediction_rows.append(
                {
                    "row_id": row_id,
                    "source": rows.source[row_index],
                    "target": target,
                    "observed": rows.targets[row_index, target_index],
                    "predicted": prediction[row_index, target_index],
                    "residual": prediction[row_index, target_index] - rows.targets[row_index, target_index],
                }
            )
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(output_dir / "stage1_metrics.csv", index=False)
    selections_frame.to_csv(output_dir / "hyperparameter_selections.csv", index=False)
    predictions_frame.to_csv(output_dir / "stage1_predictions.csv", index=False)

    stage1_pass = bool(
        (metrics_frame["pair_rmse_ratio"] <= PAIR_RATIO_GATE).all()
        and (metrics_frame["fiber_rmse_ratio"] <= FIBER_RATIO_GATE).all()
        and stability["shared_direction_absolute_cosine_mean"] >= STABILITY_GATE
    )
    ratio_frame = metrics_frame.melt(
        id_vars=["target"],
        value_vars=["pair_rmse_ratio", "fiber_rmse_ratio"],
        var_name="source",
        value_name="rmse_ratio",
    )
    figure = px.bar(
        ratio_frame,
        x="source",
        y="rmse_ratio",
        color="target",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Shared latent Fisher phase field: nested OOF gain over zero phase correction",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "stage1_rmse_ratios.html", include_plotlyjs=True, config=PLOT_CONFIG)

    registry = pd.DataFrame(
        [
            {
                "id": "SLF-PF",
                "family": "Shared latent Fisher phase field",
                "materially_new_mechanism": (
                    "One jointly learned within-family curriculum state with target-specific BPB readouts."
                ),
                "relationship_to_prior": (
                    "A Fisher-orthogonal special case of prior JLPT; retained only as a stricter "
                    "local falsification, not a new headline family."
                ),
                "nominal_additional_dof": dataset.m - len(dataset.family_members) + len(rows.target_names) - 1,
                "status": "promoted_stage1" if stage1_pass else "rejected_stage1_overlaps_prior_jlpt",
                "evidence": "Frozen joint exact-pair/fiber nested OOF gate.",
            }
        ]
    )
    registry.to_csv(output_dir / "approach_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "round": "SLF-PF-stage1",
                "candidate": "shared_latent_fisher_phase_field",
                "prior_stage1_outcome_inspected": "hierarchical_fisher_phase_field",
                "new_mechanism": "cross-target shared latent phase state",
                "historical_archive_opened": False,
                "adversarial_archive_opened": False,
                "optimization_run": False,
                "stage1_pass": stage1_pass,
            }
        ]
    ).to_csv(output_dir / "data_use_ledger.csv", index=False)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "stage1_pass": stage1_pass,
                "historical_archive_opened": False,
                "adversarial_archive_opened": False,
                "optimization_run": False,
                "preregistration": json.loads(preregistration.read_text()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    lines = [
        "# Shared latent Fisher phase field",
        "",
        "The within-family phase response is constrained to one latent curriculum-state displacement shared by "
        "Uncheatable and Table-9, with target-specific scalar readouts. Family/state/information effects remain "
        "target-specific. This is a reduced-rank mechanistic state, not an ensemble or output calibrator.",
        "",
        "## Frozen Stage-1 metrics",
        "",
        metrics_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Hyperparameter selections",
        "",
        selections_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"**Stage-1 gate:** `{'PASS' if stage1_pass else 'FAIL'}`.",
        "",
        (
            "Historical and adversarial absolute heldouts remained unopened; no optimization was run. The shared "
            "direction is fold-stable and improves exact pairs, but Uncheatable fiber RMSE worsens to 1.149 times "
            "the zero-phase baseline. A registry audit also shows that this is a Fisher-orthogonal special case of "
            "the previously rejected JLPT route, so it is not a materially new surviving family."
            if not stage1_pass
            else "The candidate may proceed to frozen absolute heldouts and optimization audit."
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(metrics_frame.to_string(index=False))
    print(selections_frame.to_string(index=False))
    print(f"Stage-1 gate: {'PASS' if stage1_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
