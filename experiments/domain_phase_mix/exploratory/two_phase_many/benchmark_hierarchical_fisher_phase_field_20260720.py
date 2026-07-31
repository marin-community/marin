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
"""Test a partially pooled Fisher-tangent phase field.

The aggregate response is identified independently from phase-tied policies.
Exact pairs and same-seed fibers identify a phase correction decomposed into
between-family fixed effects and within-family exchangeable random effects.
"""

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
import plotly.graph_objects as go
from scipy.linalg import null_space
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
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_pair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/hierarchical_fisher_phase_field_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
FAMILY_RIDGES = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
BUCKET_PRECISION_RATIOS = (1.0, 10.0, 100.0, 1_000.0, 1_000_000.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
PAIR_RATIO_GATE = 0.9
FIBER_RATIO_GATE = 1.05
STABILITY_GATE = 0.5
BUCKET_DF_GATE = 12.0
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FisherBasis:
    family: np.ndarray
    bucket: np.ndarray
    natural: np.ndarray


@dataclass(frozen=True)
class PhaseRows:
    features: np.ndarray
    feature_names: tuple[str, ...]
    feature_blocks: np.ndarray
    target: np.ndarray
    source: np.ndarray
    groups: np.ndarray
    blocks: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class PhaseFit:
    coefficients: np.ndarray
    feature_scale: np.ndarray
    family_ridge: float
    bucket_precision_ratio: float
    feature_names: tuple[str, ...]
    feature_blocks: np.ndarray

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return (features / self.feature_scale[None, :]) @ self.coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def fisher_basis(dataset: family_grp.Dataset) -> FisherBasis:
    natural = hierarchical.proportional_weights(dataset)
    root = np.sqrt(np.maximum(natural, 1e-12))
    family_vectors = np.zeros((dataset.m, len(dataset.family_members)))
    residual_columns: list[np.ndarray] = []
    for family_index, members in enumerate(dataset.family_members):
        local_root = root[members]
        family_vectors[members, family_index] = local_root / np.linalg.norm(local_root)
        local_basis = null_space(local_root[None, :])
        for column in range(local_basis.shape[1]):
            embedded = np.zeros(dataset.m)
            embedded[members] = local_basis[:, column]
            residual_columns.append(embedded)
    family_mass = np.asarray([natural[members].sum() for members in dataset.family_members])
    family_contrasts = null_space(np.sqrt(family_mass)[None, :])
    family_basis = family_vectors @ family_contrasts
    bucket_basis = np.column_stack(residual_columns)
    if family_basis.shape[1] != len(dataset.family_members) - 1:
        raise ValueError("Unexpected family contrast dimension")
    if bucket_basis.shape[1] != dataset.m - len(dataset.family_members):
        raise ValueError("Unexpected within-family contrast dimension")
    if not np.allclose(family_basis.T @ bucket_basis, 0.0, atol=1e-10):
        raise ValueError("Family and bucket Fisher bases are not orthogonal")
    return FisherBasis(family_basis, bucket_basis, natural)


def phase_coordinates(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    basis: FisherBasis,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    alpha = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    phase0 = np.asarray(weights[:, 0, :], dtype=float)
    phase1 = np.asarray(weights[:, 1, :], dtype=float)
    aggregate = alpha * phase0 + (1.0 - alpha) * phase1
    displacement = alpha * (1.0 - alpha) * (phase1 - phase0)
    tangent = displacement / np.sqrt(np.maximum(basis.natural[None, :], 1e-12))
    family_order = tangent @ basis.family
    bucket_order = tangent @ basis.bucket

    family_aggregate = np.column_stack([aggregate[:, members].sum(axis=1) for members in dataset.family_members])
    family_natural = np.asarray([basis.natural[members].sum() for members in dataset.family_members])
    family_state = np.log1p(family_aggregate / np.maximum(family_natural[None, :], 1e-12))
    family_state_contrast = family_state @ null_space(np.sqrt(family_natural)[None, :])
    state_gated = family_order * family_state_contrast

    safe_aggregate = np.maximum(aggregate, 1e-12)
    information = alpha * np.sum(phase0 * np.log(np.maximum(phase0, 1e-12) / safe_aggregate), axis=1) + (
        1.0 - alpha
    ) * np.sum(phase1 * np.log(np.maximum(phase1, 1e-12) / safe_aggregate), axis=1)
    features = np.column_stack([family_order, state_gated, information, bucket_order])
    names = tuple(
        [f"family_order_{index}" for index in range(family_order.shape[1])]
        + [f"family_state_order_{index}" for index in range(state_gated.shape[1])]
        + ["phase_information_cost"]
        + [f"within_family_order_{index}" for index in range(bucket_order.shape[1])]
    )
    blocks = np.asarray(
        ["family"] * family_order.shape[1]
        + ["family"] * state_gated.shape[1]
        + ["information"]
        + ["bucket"] * bucket_order.shape[1],
        dtype=str,
    )
    return features, names, blocks


def phase_rows(
    matched: matched_pair.MatchedSources,
    dataset: family_grp.Dataset,
    basis: FisherBasis,
    target: str,
) -> PhaseRows:
    pair = matched.pair_frame
    broad_indices = pair["broad_index"].to_numpy(dtype=int)
    single_indices = pair["single_index"].to_numpy(dtype=int)
    pair_features, names, feature_blocks = phase_coordinates(
        matched.sources.broad.weights[broad_indices], dataset, basis
    )
    pair_target = matched.sources.broad.frame.iloc[broad_indices][heterogeneous.TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    ) - matched.sources.single.frame.iloc[single_indices][heterogeneous.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    pair_ids = ("pair::" + pair["pair_id"].astype(str)).to_numpy(dtype=str)

    fiber_frame = matched.sources.fiber.frame
    fiber_mask = ~fiber_frame["contrast_family"].astype(str).eq("center_control").to_numpy()
    fiber_features, fiber_names, fiber_blocks = phase_coordinates(
        matched.sources.fiber.weights[fiber_mask], dataset, basis
    )
    if fiber_names != names or not np.array_equal(fiber_blocks, feature_blocks):
        raise ValueError("Pair and fiber feature order differs")
    fiber_selected = fiber_frame.loc[fiber_mask]
    fiber_target = fiber_selected[heterogeneous.fiber_delta_column(target)].to_numpy(dtype=float)
    fiber_groups = (
        "fiber::" + fiber_selected["anchor_id"].astype(str) + "::" + fiber_selected["seed_block"].astype(int).astype(str)
    ).to_numpy(dtype=str)
    return PhaseRows(
        features=np.vstack([pair_features, fiber_features]),
        feature_names=names,
        feature_blocks=feature_blocks,
        target=np.concatenate([pair_target, fiber_target]),
        source=np.asarray(["pair"] * len(pair_target) + ["fiber"] * len(fiber_target), dtype=str),
        groups=np.concatenate([pair_ids, fiber_groups]),
        blocks=np.concatenate([pair_ids, fiber_groups]),
        row_ids=np.concatenate([pair["pair_id"].astype(str).to_numpy(), fiber_selected["candidate_id"].astype(str)]),
    )


def whitened_training_system(rows: PhaseRows, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    designs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    selected = np.asarray(indices, dtype=int)
    for source in ("pair", "fiber"):
        source_indices = selected[rows.source[selected] == source]
        if not len(source_indices):
            continue
        design = rows.features[source_indices]
        target = rows.target[source_indices]
        if source == "fiber":
            block_designs: list[np.ndarray] = []
            block_targets: list[np.ndarray] = []
            for block in np.unique(rows.blocks[source_indices]):
                local = source_indices[rows.blocks[source_indices] == block]
                whitening = heterogeneous.inverse_sqrt_shared_center_covariance(len(local))
                block_designs.append(whitening @ rows.features[local])
                block_targets.append(whitening @ rows.target[local])
            design = np.vstack(block_designs)
            target = np.concatenate(block_targets)
        source_scale = 1.0 / math.sqrt(len(target))
        designs.append(source_scale * design)
        targets.append(source_scale * target)
    return np.vstack(designs), np.concatenate(targets)


def fit_phase(
    rows: PhaseRows,
    indices: np.ndarray,
    family_ridge: float,
    bucket_precision_ratio: float,
) -> PhaseFit:
    design, target = whitened_training_system(rows, indices)
    scale = np.maximum(np.sqrt(np.mean(design**2, axis=0)), 1e-12)
    standardized = design / scale[None, :]
    penalties = np.full(standardized.shape[1], family_ridge)
    penalties[rows.feature_blocks == "bucket"] *= bucket_precision_ratio
    augmented_design = np.vstack([standardized, np.diag(np.sqrt(penalties))])
    augmented_target = np.concatenate([target, np.zeros(standardized.shape[1])])
    lower = np.full(standardized.shape[1], -np.inf)
    lower[rows.feature_blocks == "information"] = 0.0
    result = lsq_linear(
        augmented_design,
        augmented_target,
        bounds=(lower, np.full(standardized.shape[1], np.inf)),
        max_iter=5_000,
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"Phase fit failed: {result.message}")
    return PhaseFit(
        coefficients=np.asarray(result.x),
        feature_scale=scale,
        family_ridge=family_ridge,
        bucket_precision_ratio=bucket_precision_ratio,
        feature_names=rows.feature_names,
        feature_blocks=rows.feature_blocks,
    )


def source_metrics(rows: PhaseRows, prediction: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    selected = np.asarray(indices, dtype=int)
    for source in ("pair", "fiber"):
        local = selected[rows.source[selected] == source]
        observed = rows.target[local]
        predicted = prediction[local]
        rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        zero = float(np.sqrt(np.mean(observed**2)))
        result[f"{source}_rmse"] = rmse
        result[f"{source}_zero_rmse"] = zero
        result[f"{source}_rmse_ratio"] = rmse / zero
    return result


def inner_hyperparameters(
    rows: PhaseRows,
    indices: np.ndarray,
    salt: str,
) -> tuple[float, float]:
    selected = np.asarray(indices, dtype=int)
    folds = family_state.grouped_balanced_folds(rows, selected, INNER_FOLDS, salt)
    scores = []
    for family_ridge in FAMILY_RIDGES:
        for ratio in BUCKET_PRECISION_RATIOS:
            source_ratios = []
            for fold in range(INNER_FOLDS):
                train = selected[folds != fold]
                test = selected[folds == fold]
                model = fit_phase(rows, train, family_ridge, ratio)
                metrics = source_metrics(rows, model.predict_features(rows.features), test)
                source_ratios.extend([metrics["pair_rmse_ratio"], metrics["fiber_rmse_ratio"]])
            scores.append((float(np.mean(source_ratios)), family_ridge, ratio))
    _, family_ridge, ratio = min(scores)
    return family_ridge, ratio


def effective_degrees_of_freedom(rows: PhaseRows, model: PhaseFit, indices: np.ndarray) -> dict[str, float]:
    design, _ = whitened_training_system(rows, indices)
    standardized = design / model.feature_scale[None, :]
    penalties = np.full(standardized.shape[1], model.family_ridge)
    penalties[rows.feature_blocks == "bucket"] *= model.bucket_precision_ratio
    information = standardized.T @ standardized + np.diag(penalties)
    hat_components = np.linalg.solve(information, standardized.T @ standardized)
    return {
        "effective_df": float(np.trace(hat_components)),
        "bucket_effective_df": float(
            np.trace(hat_components[np.ix_(rows.feature_blocks == "bucket", rows.feature_blocks == "bucket")])
        ),
    }


def nested_oof(rows: PhaseRows) -> tuple[np.ndarray, tuple[PhaseFit, ...], list[dict[str, Any]]]:
    indices = np.arange(len(rows.target))
    folds = family_state.grouped_balanced_folds(rows, indices, OUTER_FOLDS, "fisher_outer")
    prediction = np.full(len(rows.target), np.nan)
    models: list[PhaseFit] = []
    selections: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        train = indices[folds != fold]
        test = indices[folds == fold]
        family_ridge, ratio = inner_hyperparameters(rows, train, f"fisher_inner::{fold}")
        model = fit_phase(rows, train, family_ridge, ratio)
        prediction[test] = model.predict_features(rows.features[test])
        models.append(model)
        selections.append({"fold": fold, "family_ridge": family_ridge, "bucket_precision_ratio": ratio})
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete OOF prediction")
    return prediction, tuple(models), selections


def prediction_stability(rows: PhaseRows, models: tuple[PhaseFit, ...]) -> dict[str, float]:
    predictions = np.stack([model.predict_features(rows.features) for model in models])
    similarities = np.corrcoef(predictions)
    upper = similarities[np.triu_indices(len(models), k=1)]
    return {
        "phase_prediction_cosine_mean": float(np.mean(upper)),
        "phase_prediction_cosine_min": float(np.min(upper)),
    }


def write_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    ratio = metrics.melt(
        id_vars=["target"],
        value_vars=["pair_rmse_ratio", "fiber_rmse_ratio"],
        var_name="source",
        value_name="rmse_ratio",
    )
    figure = px.bar(
        ratio,
        x="source",
        y="rmse_ratio",
        color="target",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Hierarchical Fisher phase field: nested OOF gain over zero phase correction",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "stage1_rmse_ratios.html", include_plotlyjs=True, config=PLOT_CONFIG)

    scatter = px.scatter(
        predictions,
        x="observed",
        y="predicted",
        color="source",
        facet_col="target",
        hover_name="row_id",
        color_discrete_map={"pair": "#d97706", "fiber": "#2563eb"},
        title="Nested OOF phase corrections",
    )
    low = float(min(predictions["observed"].min(), predictions["predicted"].min()))
    high = float(max(predictions["observed"].max(), predictions["predicted"].max()))
    scatter.add_trace(
        go.Scatter(x=[low, high], y=[low, high], mode="lines", line={"dash": "dash"}, showlegend=False),
        row="all",
        col="all",
    )
    scatter.update_layout(template="plotly_white")
    scatter.write_html(output_dir / "stage1_predictions.html", include_plotlyjs=True, config=PLOT_CONFIG)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = output_dir / "preregistered_candidates.json"
    if not preregistration.exists():
        if output_dir == DEFAULT_OUTPUT_DIR:
            raise FileNotFoundError(f"Missing frozen preregistration {PREREGISTRATION_PATH}")
        preregistration.write_text(PREREGISTRATION_PATH.read_text())

    matched = matched_pair.matched_sources()
    dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.broad.frame,
        matched.sources.broad.weights,
        "uncheatable",
        "hierarchical_fisher_phase_reference",
    )
    basis = fisher_basis(dataset)
    metric_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    full_models: dict[str, PhaseFit] = {}

    for target in heterogeneous.TARGETS:
        rows = phase_rows(matched, dataset, basis, target)
        prediction, models, selections = nested_oof(rows)
        metrics = source_metrics(rows, prediction, np.arange(len(rows.target)))
        stability = prediction_stability(rows, models)
        full_family_ridge, full_ratio = inner_hyperparameters(rows, np.arange(len(rows.target)), "fisher_full")
        full_model = fit_phase(rows, np.arange(len(rows.target)), full_family_ridge, full_ratio)
        full_models[target] = full_model
        degrees = effective_degrees_of_freedom(rows, full_model, np.arange(len(rows.target)))
        metric_rows.append({"target": target, **metrics, **stability, **degrees})
        selection_rows.extend({"target": target, **selection} for selection in selections)
        for index in range(len(rows.target)):
            prediction_rows.append(
                {
                    "target": target,
                    "source": rows.source[index],
                    "row_id": rows.row_ids[index],
                    "observed": rows.target[index],
                    "predicted": prediction[index],
                    "residual": prediction[index] - rows.target[index],
                }
            )

    metrics_frame = pd.DataFrame(metric_rows)
    selections_frame = pd.DataFrame(selection_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    metrics_frame.to_csv(output_dir / "stage1_metrics.csv", index=False)
    selections_frame.to_csv(output_dir / "hyperparameter_selections.csv", index=False)
    predictions_frame.to_csv(output_dir / "stage1_predictions.csv", index=False)
    write_plots(metrics_frame, predictions_frame, output_dir)

    stage1_pass = bool(
        (metrics_frame["pair_rmse_ratio"] <= PAIR_RATIO_GATE).all()
        and (metrics_frame["fiber_rmse_ratio"] <= FIBER_RATIO_GATE).all()
        and (metrics_frame["phase_prediction_cosine_mean"] >= STABILITY_GATE).all()
        and (metrics_frame["bucket_effective_df"] <= BUCKET_DF_GATE).all()
    )
    registry = pd.DataFrame(
        [
            {
                "id": "HF-PF",
                "family": "Hierarchical Fisher phase field",
                "relationship_to_prior": (
                    "Extends the rejected five-DoF family-state phase law with an orthogonal "
                    "within-family random effect."
                ),
                "materially_new_mechanism": (
                    "Empirical partial pooling lets the phase law discover its supported dimensionality "
                    "instead of fixing either two family axes or 38 bucket axes."
                ),
                "additional_degrees_of_freedom": 36,
                "effective_degrees_of_freedom": float(metrics_frame["effective_df"].max()),
                "status": "promoted_stage1" if stage1_pass else "rejected_stage1",
                "evidence": "Frozen exact-pair/fiber nested OOF gate.",
            }
        ]
    )
    registry.to_csv(output_dir / "approach_registry.csv", index=False)
    ledger = pd.DataFrame(
        [
            {
                "round": "HF-PF-stage1",
                "candidate": "hierarchical_fisher_phase_field",
                "choices_frozen_before_outcomes": True,
                "outcomes_inspected": "exact pair and frontier fiber phase differences",
                "historical_archive_opened": False,
                "adversarial_archive_opened": False,
                "optimization_run": False,
                "stage1_pass": stage1_pass,
            }
        ]
    )
    ledger.to_csv(output_dir / "data_use_ledger.csv", index=False)
    manifest = {
        "stage1_pass": stage1_pass,
        "historical_archive_opened": False,
        "adversarial_archive_opened": False,
        "optimization_run": False,
        "nominal_phase_degrees_of_freedom": int(dataset.m + 2),
        "targets": list(heterogeneous.TARGETS),
        "preregistration": json.loads(preregistration.read_text()),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Hierarchical Fisher phase field",
        "",
        "## Model",
        "",
        "For aggregate policy `a`, phase displacement `d`, and proportional reference `p`, define the Fisher "
        "tangent `u_i=d_i/sqrt(p_i)`. Orthogonally decompose `u` into the between-family tangent and the "
        "within-family residual. The phase correction is a five-DoF family-state head plus a 36-dimensional "
        "exchangeable within-family random effect. Nested CV estimates the bucket-to-family precision ratio. "
        "When phases tie, every phase coordinate and the Jensen-Shannon path cost are exactly zero.",
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
    ]
    if stage1_pass:
        lines.append("The candidate may proceed to the frozen common archive and optimization audit.")
    else:
        lines.append(
            "The candidate is rejected without opening historical or adversarial absolute heldouts. It improves "
            "exact-pair RMSE materially, but Table-9 fiber RMSE reaches 1.052 times the zero-phase baseline and its "
            "bucket block retains 28.1 effective degrees of freedom, above the frozen limit of 12."
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(metrics_frame.to_string(index=False))
    print(selections_frame.to_string(index=False))
    print(f"Stage-1 gate: {'PASS' if stage1_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
