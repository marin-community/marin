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
"""Test a low-dimensional family-state model for one- and two-phase policies.

The aggregate response is identified from independently trained one-phase and
phase-tied policies. Exact aggregate-matched pairs and same-seed frontier
fibers identify a five-degree-of-freedom phase correction. Data-source labels
affect only the estimating equations and never enter the prediction function.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import lsq_linear

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/family_state_phase_surrogate_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
PAIR_EFFECTS_PATH = (
    SCRIPT_DIR / "reference_outputs/delphi_3e18_frontier_phase_fiber_results_20260719/paired_phase_effects.csv"
)
CENTER_SUMMARY_PATH = (
    SCRIPT_DIR / "reference_outputs/delphi_3e18_frontier_phase_fiber_results_20260719/center_control_summary.csv"
)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
PAIR_RATIO_GATE = 0.95
FIBER_RATIO_GATE = 1.05
STABILITY_GATE = 0.5
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Candidate:
    name: str
    state_interaction: bool
    information_cost: bool


CANDIDATES = (
    Candidate("family_order", False, False),
    Candidate("family_state_order", True, False),
    Candidate("family_state_order_information", True, True),
)


@dataclass(frozen=True)
class PhaseRows:
    features: np.ndarray
    feature_names: tuple[str, ...]
    target: np.ndarray
    source: np.ndarray
    groups: np.ndarray
    blocks: np.ndarray
    row_ids: np.ndarray


@dataclass(frozen=True)
class PhaseFit:
    candidate: Candidate
    coefficients: np.ndarray
    feature_scale: np.ndarray
    ridge: float
    feature_names: tuple[str, ...]

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return (features / self.feature_scale[None, :]) @ self.coefficients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def grouped_balanced_folds(rows: PhaseRows, indices: np.ndarray, folds: int, salt: str) -> np.ndarray:
    """Assign complete acquisition groups while balancing each source across folds."""
    selected = np.asarray(indices, dtype=int)
    result = np.full(len(selected), -1, dtype=int)
    for source in ("pair", "fiber"):
        source_mask = rows.source[selected] == source
        groups = np.unique(rows.groups[selected[source_mask]])
        ordered = sorted(
            groups,
            key=lambda value: hashlib.sha256(f"{salt}::{source}::{value}".encode()).digest(),
        )
        assignment = {group: position % folds for position, group in enumerate(ordered)}
        result[source_mask] = np.asarray([assignment[group] for group in rows.groups[selected[source_mask]]], dtype=int)
    if np.any(result < 0):
        raise ValueError("Incomplete grouped fold assignment")
    for fold in range(folds):
        for source in ("pair", "fiber"):
            if not np.any((result == fold) & (rows.source[selected] == source)):
                raise ValueError(f"Fold {fold} has no {source} rows")
    return result


def phase_coordinates(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    candidate: Candidate,
) -> tuple[np.ndarray, tuple[str, ...]]:
    phase_fraction = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    phase0 = np.asarray(weights[:, 0, :], dtype=float)
    phase1 = np.asarray(weights[:, 1, :], dtype=float)
    aggregate = phase_fraction * phase0 + (1.0 - phase_fraction) * phase1
    contrast = phase_fraction * (1.0 - phase_fraction) * (phase1 - phase0)
    natural = hierarchical.proportional_weights(dataset)

    family_aggregate = np.column_stack([aggregate[:, members].sum(axis=1) for members in dataset.family_members])
    family_contrast = np.column_stack([contrast[:, members].sum(axis=1) for members in dataset.family_members])
    family_natural = np.asarray([natural[members].sum() for members in dataset.family_members])
    broad_index = dataset.family_names.index("broad_text")
    specialist_indices = tuple(index for index, name in enumerate(dataset.family_names) if name != "broad_text")

    relative_contrast = family_contrast / np.maximum(family_natural[None, :], 1e-12)
    relative_state = np.log1p(family_aggregate / np.maximum(family_natural[None, :], 1e-12))
    base = np.column_stack(
        [relative_contrast[:, index] - relative_contrast[:, broad_index] for index in specialist_indices]
    )
    names = [f"family_order:{dataset.family_names[index]}_vs_broad" for index in specialist_indices]
    pieces = [base]

    if candidate.state_interaction:
        state_gap = np.column_stack(
            [relative_state[:, index] - relative_state[:, broad_index] for index in specialist_indices]
        )
        pieces.append(base * state_gap)
        names.extend(f"state_gated_order:{dataset.family_names[index]}_vs_broad" for index in specialist_indices)

    if candidate.information_cost:
        safe_aggregate = np.maximum(aggregate, 1e-12)
        phase_information = phase_fraction * np.sum(
            phase0 * np.log(np.maximum(phase0, 1e-12) / safe_aggregate), axis=1
        ) + (1.0 - phase_fraction) * np.sum(phase1 * np.log(np.maximum(phase1, 1e-12) / safe_aggregate), axis=1)
        pieces.append(phase_information[:, None])
        names.append("phase_information_cost")

    return np.column_stack(pieces), tuple(names)


def phase_rows(
    matched: matched_pair.MatchedSources,
    dataset: family_grp.Dataset,
    candidate: Candidate,
    target: str,
) -> PhaseRows:
    pair = matched.pair_frame
    broad_indices = pair["broad_index"].to_numpy(dtype=int)
    single_indices = pair["single_index"].to_numpy(dtype=int)
    pair_features, names = phase_coordinates(matched.sources.broad.weights[broad_indices], dataset, candidate)
    pair_target = matched.sources.broad.frame.iloc[broad_indices][heterogeneous.TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    ) - matched.sources.single.frame.iloc[single_indices][heterogeneous.TARGET_COLUMNS[target]].to_numpy(dtype=float)
    pair_ids = ("pair::" + pair["pair_id"].astype(str)).to_numpy(dtype=str)

    fiber_frame = matched.sources.fiber.frame
    fiber_mask = ~fiber_frame["contrast_family"].astype(str).eq("center_control").to_numpy()
    fiber_features, fiber_names = phase_coordinates(matched.sources.fiber.weights[fiber_mask], dataset, candidate)
    if fiber_names != names:
        raise ValueError("Pair and fiber phase feature order differs")
    fiber_selected = fiber_frame.loc[fiber_mask]
    fiber_target = fiber_selected[heterogeneous.fiber_delta_column(target)].to_numpy(dtype=float)
    fiber_groups = (
        "fiber::" + fiber_selected["anchor_id"].astype(str) + "::" + fiber_selected["seed_block"].astype(int).astype(str)
    ).to_numpy(dtype=str)
    fiber_ids = fiber_selected["candidate_id"].astype(str).to_numpy()

    return PhaseRows(
        features=np.vstack([pair_features, fiber_features]),
        feature_names=names,
        target=np.concatenate([pair_target, fiber_target]),
        source=np.asarray(["pair"] * len(pair_target) + ["fiber"] * len(fiber_target), dtype=str),
        groups=np.concatenate([pair_ids, fiber_groups]),
        blocks=np.concatenate([pair_ids, fiber_groups]),
        row_ids=np.concatenate([pair["pair_id"].astype(str).to_numpy(), fiber_ids]),
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
        # Each acquisition source contributes one unit of information before
        # ridge regularization, preventing the larger source from dominating.
        source_scale = 1.0 / math.sqrt(len(target))
        designs.append(source_scale * design)
        targets.append(source_scale * target)
    return np.vstack(designs), np.concatenate(targets)


def fit_phase(rows: PhaseRows, indices: np.ndarray, candidate: Candidate, ridge: float) -> PhaseFit:
    design, target = whitened_training_system(rows, indices)
    scale = np.sqrt(np.mean(design**2, axis=0))
    scale = np.maximum(scale, 1e-12)
    standardized = design / scale[None, :]
    if ridge > 0.0:
        standardized = np.vstack([standardized, np.sqrt(ridge) * np.eye(standardized.shape[1])])
        target = np.concatenate([target, np.zeros(standardized.shape[1])])
    lower = np.full(standardized.shape[1], -np.inf)
    if candidate.information_cost:
        lower[-1] = 0.0
    result = lsq_linear(
        standardized,
        target,
        bounds=(lower, np.full(standardized.shape[1], np.inf)),
        max_iter=5_000,
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(f"Phase fit failed: {result.message}")
    return PhaseFit(candidate, np.asarray(result.x), scale, ridge, rows.feature_names)


def source_rmse(rows: PhaseRows, prediction: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    selected = np.asarray(indices, dtype=int)
    for source in ("pair", "fiber"):
        local = selected[rows.source[selected] == source]
        observed = rows.target[local]
        predicted = prediction[local]
        result[f"{source}_rmse"] = float(np.sqrt(np.mean((predicted - observed) ** 2)))
        result[f"{source}_zero_rmse"] = float(np.sqrt(np.mean(observed**2)))
    return result


def inner_ridge(rows: PhaseRows, indices: np.ndarray, candidate: Candidate, salt: str) -> float:
    selected = np.asarray(indices, dtype=int)
    folds = grouped_balanced_folds(rows, selected, INNER_FOLDS, salt)
    scores = []
    for ridge in RIDGE_GRID:
        ratios = []
        for fold in range(INNER_FOLDS):
            train = selected[folds != fold]
            test = selected[folds == fold]
            if not len(train) or not len(test):
                continue
            model = fit_phase(rows, train, candidate, ridge)
            metrics = source_rmse(rows, model.predict_features(rows.features), test)
            for source in ("pair", "fiber"):
                if metrics[f"{source}_zero_rmse"] > 0.0:
                    ratios.append(metrics[f"{source}_rmse"] / metrics[f"{source}_zero_rmse"])
        scores.append((float(np.mean(ratios)), ridge))
    return min(scores)[1]


def nested_oof(rows: PhaseRows, candidate: Candidate) -> tuple[np.ndarray, tuple[PhaseFit, ...], list[dict[str, Any]]]:
    all_indices = np.arange(len(rows.target))
    folds = grouped_balanced_folds(rows, all_indices, OUTER_FOLDS, "outer")
    prediction = np.full(len(rows.target), np.nan)
    models: list[PhaseFit] = []
    selections: list[dict[str, Any]] = []
    for fold in range(OUTER_FOLDS):
        train = all_indices[folds != fold]
        test = all_indices[folds == fold]
        ridge = inner_ridge(rows, train, candidate, f"inner::{fold}")
        model = fit_phase(rows, train, candidate, ridge)
        prediction[test] = model.predict_features(rows.features[test])
        models.append(model)
        selections.append({"fold": fold, "ridge": ridge})
    if not np.isfinite(prediction).all():
        raise RuntimeError("Nested OOF prediction is incomplete")
    return prediction, tuple(models), selections


def coefficient_stability(models: tuple[PhaseFit, ...]) -> dict[str, float]:
    coefficients = np.stack([model.coefficients / model.feature_scale for model in models])
    norms = np.linalg.norm(coefficients, axis=1, keepdims=True)
    normalized = coefficients / np.maximum(norms, 1e-12)
    similarities = normalized @ normalized.T
    upper = similarities[np.triu_indices(len(models), k=1)]
    return {
        "coefficient_cosine_mean": float(np.mean(upper)),
        "coefficient_cosine_min": float(np.min(upper)),
    }


def phase_information_budget(output_dir: Path) -> pd.DataFrame:
    centers = pd.read_csv(CENTER_SUMMARY_PATH)
    effects = pd.read_csv(PAIR_EFFECTS_PATH)
    summaries = (
        effects.groupby(["anchor_id", "target", "contrast_family"], sort=True)
        .agg(odd_effect_rms=("odd_effect_plus_minus_over_2", lambda x: float(np.sqrt(np.mean(x**2)))))
        .reset_index()
    )
    centers = centers[["anchor_id", "target", "fresh_center_sd_bpb"]]
    table = summaries.merge(centers, on=["anchor_id", "target"], validate="many_to_one")
    table["independent_odd_noise_sd"] = table["fresh_center_sd_bpb"] / math.sqrt(2.0)
    table["debiased_odd_signal_rms"] = np.sqrt(
        np.maximum(table["odd_effect_rms"] ** 2 - table["independent_odd_noise_sd"] ** 2, 0.0)
    )
    table["odd_signal_to_noise"] = table["odd_effect_rms"] / table["independent_odd_noise_sd"]
    table.to_csv(output_dir / "phase_information_budget.csv", index=False)

    figure = px.bar(
        table,
        x="anchor_id",
        y="odd_signal_to_noise",
        color="target",
        facet_col="contrast_family",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Frontier phase-fiber odd effect relative to independent-run noise",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    figure.update_layout(template="plotly_white", yaxis_title="observed odd RMS / estimated noise SD")
    figure.write_html(output_dir / "phase_information_budget.html", include_plotlyjs=True, config=PLOT_CONFIG)
    return table


def aggregate_spine(
    matched: matched_pair.MatchedSources,
    target: str,
) -> tuple[family_grp.Dataset, hierarchical.Model]:
    tied_indices = matched.tied_broad_indices
    frame = pd.concat(
        [
            matched.sources.single.frame.copy(),
            matched.sources.broad.frame.iloc[tied_indices].copy(),
        ],
        ignore_index=True,
        sort=False,
    )
    weights = np.concatenate([matched.sources.single.weights, matched.sources.broad.weights[tied_indices]], axis=0)
    dataset = composition.custom_dataset(
        matched.sources.reference,
        frame,
        weights,
        target,
        f"family_state_aggregate_{target}",
    )
    config = composition.hpr_config(target)
    return dataset, hierarchical.fit_model(dataset, config, np.arange(dataset.n))


def write_stage1_plots(metrics: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    ratios = metrics.melt(
        id_vars=["target", "candidate"],
        value_vars=["pair_rmse_ratio", "fiber_rmse_ratio"],
        var_name="source",
        value_name="rmse_ratio",
    )
    figure = px.bar(
        ratios,
        x="candidate",
        y="rmse_ratio",
        color="target",
        facet_col="source",
        barmode="group",
        color_discrete_map={"uncheatable": "#2f855a", "table9": "#c53030"},
        title="Nested OOF phase-delta RMSE relative to a zero-phase correction",
    )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#243746")
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "stage1_rmse_ratios.html", include_plotlyjs=True, config=PLOT_CONFIG)

    scatter = px.scatter(
        predictions,
        x="observed",
        y="predicted",
        color="source",
        facet_row="target",
        facet_col="candidate",
        hover_name="row_id",
        color_discrete_map={"pair": "#d97706", "fiber": "#2563eb"},
        title="Nested OOF phase corrections",
    )
    minimum = float(min(predictions["observed"].min(), predictions["predicted"].min()))
    maximum = float(max(predictions["observed"].max(), predictions["predicted"].max()))
    scatter.add_trace(
        go.Scatter(x=[minimum, maximum], y=[minimum, maximum], mode="lines", line={"dash": "dash"}, showlegend=False),
        row="all",
        col="all",
    )
    scatter.update_layout(template="plotly_white")
    scatter.write_html(output_dir / "stage1_phase_predictions.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    information: pd.DataFrame,
    metrics: pd.DataFrame,
    selections: pd.DataFrame,
    stage1_pass: bool,
    output_dir: Path,
) -> None:
    lines = [
        "# Family-state orthogonal phase surrogate",
        "",
        "## Model",
        "",
        "The aggregate response `F(a)` is fit only from 238 independently trained one-phase rows plus 42 tied controls. "
        "For `a = alpha*w0 + (1-alpha)*w1` and `d = alpha*(1-alpha)*(w1-w0)`, the phase head uses two "
        "predeclared family contrasts (tech/code and reasoning versus broad text), their interaction with the "
        "corresponding relative aggregate-saturation gaps, and optionally one nonnegative Jensen-Shannon phase "
        "information cost. The full candidate has five effective phase degrees of freedom. All phase features vanish "
        "when `w0=w1`, so the independently fitted one-phase model is the exact restriction.",
        "",
        "Exact aggregate-matched pair differences and same-seed fiber differences estimate the phase head. Source "
        "labels determine only source-balanced GLS equations and never enter prediction.",
        "",
        "## Phase information budget",
        "",
        information.to_markdown(index=False, floatfmt=".6f"),
        "",
        "The independent-run noise comparison is conservative because same-seed plus/minus contrasts may cancel a "
        "shared seed effect. It nevertheless shows why a free 38-dimensional phase gradient is not identified by one "
        "observation per direction.",
        "",
        "## Frozen Stage-1 results",
        "",
        metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Selected ridge values by outer fold:",
        "",
        selections.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"**Stage-1 gate:** `{'PASS' if stage1_pass else 'FAIL'}`.",
    ]
    if stage1_pass:
        lines.extend(
            [
                "",
                "The common archive may now be opened and an optimization audit may be run.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Per the preregistration, the common archive was not inspected and no surrogate optimum was computed. "
                "The low-dimensional family state is insufficient to predict phase effects under the frozen gate.",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


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
    reference_dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.broad.frame,
        matched.sources.broad.weights,
        "uncheatable",
        "family_state_reference",
    )
    information = phase_information_budget(output_dir)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    # Fit the aggregate spines now to validate the clean one-phase restriction,
    # but do not inspect the common archive unless the phase gate passes.
    aggregate_models = {target: aggregate_spine(matched, target) for target in heterogeneous.TARGETS}

    for target in heterogeneous.TARGETS:
        for candidate in CANDIDATES:
            rows = phase_rows(matched, reference_dataset, candidate, target)
            prediction, models, selections = nested_oof(rows, candidate)
            metrics = source_rmse(rows, prediction, np.arange(len(rows.target)))
            stability = coefficient_stability(models)
            pair_ratio = metrics["pair_rmse"] / metrics["pair_zero_rmse"]
            fiber_ratio = metrics["fiber_rmse"] / metrics["fiber_zero_rmse"]
            metric_rows.append(
                {
                    "target": target,
                    "candidate": candidate.name,
                    **metrics,
                    "pair_rmse_ratio": pair_ratio,
                    "fiber_rmse_ratio": fiber_ratio,
                    **stability,
                }
            )
            for row in selections:
                selection_rows.append({"target": target, "candidate": candidate.name, **row})
            for index in range(len(rows.target)):
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "source": rows.source[index],
                        "row_id": rows.row_ids[index],
                        "observed": rows.target[index],
                        "predicted": prediction[index],
                        "residual": prediction[index] - rows.target[index],
                    }
                )

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    selections_frame = pd.DataFrame(selection_rows)
    metrics_frame.to_csv(output_dir / "stage1_metrics.csv", index=False)
    predictions_frame.to_csv(output_dir / "stage1_predictions.csv", index=False)
    selections_frame.to_csv(output_dir / "ridge_selections.csv", index=False)
    write_stage1_plots(metrics_frame, predictions_frame, output_dir)

    headline = metrics_frame.loc[metrics_frame["candidate"].eq("family_state_order_information")]
    stage1_pass = bool(
        len(headline) == len(heterogeneous.TARGETS)
        and (headline["pair_rmse_ratio"] <= PAIR_RATIO_GATE).all()
        and (headline["fiber_rmse_ratio"] <= FIBER_RATIO_GATE).all()
        and (headline["coefficient_cosine_mean"] >= STABILITY_GATE).all()
    )

    manifest: dict[str, Any] = {
        "stage1_pass": stage1_pass,
        "common_archive_opened": False,
        "optimization_run": False,
        "aggregate_spines_fit": sorted(aggregate_models),
        "candidate_count": len(CANDIDATES),
        "targets": list(heterogeneous.TARGETS),
        "preregistration": json.loads(preregistration.read_text()),
    }
    if stage1_pass:
        common_rows = []
        full_models: dict[tuple[str, str], PhaseFit] = {}
        for target in heterogeneous.TARGETS:
            _, aggregate_model = aggregate_models[target]
            common_observed = matched.sources.common.frame[heterogeneous.TARGET_COLUMNS[target]].to_numpy(float)
            for candidate in CANDIDATES:
                rows = phase_rows(matched, reference_dataset, candidate, target)
                ridge = inner_ridge(rows, np.arange(len(rows.target)), candidate, "full")
                phase_model = fit_phase(rows, np.arange(len(rows.target)), candidate, ridge)
                full_models[(target, candidate.name)] = phase_model
                tied_common = heterogeneous.tied_weights(
                    replace(
                        reference_dataset, weights=matched.sources.common.weights, target=np.zeros(len(common_observed))
                    )
                )
                aggregate_prediction = aggregate_model.predict(tied_common)
                phase_features, names = phase_coordinates(matched.sources.common.weights, reference_dataset, candidate)
                if names != phase_model.feature_names:
                    raise ValueError("Common archive phase feature order changed")
                prediction = aggregate_prediction + phase_model.predict_features(phase_features)
                common_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        **composition.prediction_metrics(common_observed, prediction),
                    }
                )
        pd.DataFrame(common_rows).to_csv(output_dir / "common_archive_metrics.csv", index=False)
        manifest["common_archive_opened"] = True
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(information, metrics_frame, selections_frame, stage1_pass, output_dir)
    print(metrics_frame.to_string(index=False))
    print(f"Stage-1 gate: {'PASS' if stage1_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
