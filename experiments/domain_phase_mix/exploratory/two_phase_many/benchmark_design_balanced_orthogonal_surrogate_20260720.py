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
"""Benchmark design-balanced orthogonal learning of aggregate and phase effects."""

from __future__ import annotations

import argparse
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
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_3e18_fixed_budget_frontier_composition as composition,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_matched_pair_heterogeneous_hpr_20260720 as matched_pair,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_plasticity_potential_transport_20260720 as ppt,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/design_balanced_orthogonal_surrogate_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
TARGETS = composition.TARGETS
TARGET_COLUMNS = composition.TARGET_COLUMNS
PHASE_CANDIDATE_NAME = "symmetric_potential_even_cost"
N_SPLITS = 5
DELETION_IMPROVEMENT = 0.05
OOF_REGRESSION_TOLERANCE = 0.05
REGRET_TOLERANCE = 0.002
FIBER_TOLERANCE = 0.01
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Candidate:
    name: str
    equation_mode: str
    balance: str


CANDIDATES = (
    Candidate("uniform_absolute", "absolute", "uniform"),
    Candidate("sqrt_balanced_absolute", "absolute", "sqrt"),
    Candidate("equal_block_absolute", "absolute", "equal"),
    Candidate("sqrt_balanced_contrast", "contrast", "sqrt"),
    Candidate("equal_block_contrast", "contrast", "equal"),
)


@dataclass(frozen=True)
class AggregateModel:
    dataset: family_grp.Dataset
    config: hierarchical.Config
    intercept: float
    coefficients: np.ndarray

    def predict(self, weights: np.ndarray) -> np.ndarray:
        candidate = replace(
            self.dataset,
            weights=np.asarray(weights, dtype=float),
            target=np.zeros(len(weights), dtype=float),
        )
        design = hierarchical.build_design(candidate, self.config)
        return np.asarray(self.intercept + design.values @ self.coefficients, dtype=float)


@dataclass(frozen=True)
class CombinedModel:
    aggregate: AggregateModel
    phase: ppt.PhaseFit
    phase_dataset: family_grp.Dataset

    def predict_phase(self, weights: np.ndarray) -> np.ndarray:
        design, names = ppt.phase_design(weights, self.phase_dataset, self.phase.candidate, self.phase.tau)
        if names != self.phase.feature_names:
            raise ValueError("Phase feature order changed")
        return design @ self.phase.coefficients

    def predict(self, weights: np.ndarray) -> np.ndarray:
        tied = ppt.tied_policy(weights, self.phase_dataset)
        return self.aggregate.predict(tied) + self.predict_phase(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-only", action="store_true")
    return parser.parse_args()


def phase_candidate() -> ppt.Candidate:
    selected = [candidate for candidate in ppt.CANDIDATES if candidate.name == PHASE_CANDIDATE_NAME]
    if len(selected) != 1:
        raise ValueError(f"Expected one phase candidate named {PHASE_CANDIDATE_NAME}")
    return selected[0]


def aggregate_dataset(matched: matched_pair.MatchedSources, target: str) -> family_grp.Dataset:
    single_indices = matched.pair_frame["single_index"].to_numpy(dtype=int)
    single = matched.sources.single.frame.iloc[single_indices].copy().reset_index(drop=True)
    single["panel_source"] = matched.pair_frame["panel_source"].to_numpy(dtype=str)
    tied = matched.sources.broad.frame.iloc[matched.tied_broad_indices].copy().reset_index(drop=True)
    frame = pd.concat([single, tied], ignore_index=True, sort=False)
    weights = np.concatenate(
        [
            matched.sources.single.weights[single_indices],
            matched.sources.broad.weights[matched.tied_broad_indices],
        ],
        axis=0,
    )
    counts = frame["panel_source"].value_counts().to_dict()
    if counts != {"qsplit_signal": 241, "domain_deletion": 39}:
        raise ValueError(f"Unexpected aggregate source counts: {counts}")
    return composition.custom_dataset(matched.sources.reference, frame, weights, target, f"design_balanced_{target}")


def proportional_index(dataset: family_grp.Dataset) -> int:
    matches = np.flatnonzero(dataset.frame["run_name"].eq("baseline_proportional").to_numpy())
    if len(matches) != 1:
        raise ValueError(f"Expected one proportional anchor, found {len(matches)}")
    return int(matches[0])


def block_weights(labels: np.ndarray, balance: str) -> np.ndarray:
    counts = pd.Series(labels).value_counts().to_dict()
    if balance == "uniform":
        weights = np.ones(len(labels), dtype=float)
    elif balance == "sqrt":
        weights = np.asarray([1.0 / math.sqrt(counts[label]) for label in labels], dtype=float)
    elif balance == "equal":
        weights = np.asarray([1.0 / counts[label] for label in labels], dtype=float)
    else:
        raise ValueError(f"Unknown balance rule {balance}")
    return weights / np.mean(weights)


def equation_system(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    candidate: Candidate,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = hierarchical.build_design(dataset, config)
    source = dataset.frame["panel_source"].to_numpy(dtype=str)
    anchor = proportional_index(dataset)
    if anchor not in indices:
        raise ValueError("The proportional control must remain in every training fold")

    if candidate.equation_mode == "absolute":
        selected_design = design.values[indices]
        matrix = np.column_stack([np.ones(len(indices), dtype=float), selected_design])
        target = dataset.target[indices]
        labels = source[indices]
    elif candidate.equation_mode == "contrast":
        qsplit = indices[source[indices] == "qsplit_signal"]
        deletion = indices[source[indices] == "domain_deletion"]
        absolute = np.column_stack([np.ones(len(qsplit), dtype=float), design.values[qsplit]])
        contrasts = np.column_stack(
            [
                np.zeros(len(deletion), dtype=float),
                design.values[deletion] - design.values[anchor][None, :],
            ]
        )
        matrix = np.vstack([absolute, contrasts])
        target = np.concatenate(
            [
                dataset.target[qsplit],
                dataset.target[deletion] - dataset.target[anchor],
            ]
        )
        labels = np.concatenate(
            [
                np.full(len(qsplit), "qsplit_signal", dtype=object),
                np.full(len(deletion), "domain_deletion", dtype=object),
            ]
        )
    else:
        raise ValueError(f"Unknown equation mode {candidate.equation_mode}")
    return matrix, target, block_weights(labels, candidate.balance)


def fit_aggregate(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    candidate: Candidate,
    indices: np.ndarray,
) -> AggregateModel:
    matrix, target, weights = equation_system(dataset, config, candidate, indices)
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    design = hierarchical.build_design(dataset, config)
    if config.l2 > 0.0:
        ridge = np.sqrt(config.l2 * design.ridge_multipliers)
        ridge_matrix = np.column_stack([np.zeros(len(ridge), dtype=float), np.diag(ridge)])
        weighted_matrix = np.vstack([weighted_matrix, ridge_matrix])
        weighted_target = np.concatenate([weighted_target, np.zeros(len(ridge), dtype=float)])
    lower = np.concatenate([[-np.inf], np.zeros(design.values.shape[1], dtype=float)])
    upper = np.full(design.values.shape[1] + 1, np.inf, dtype=float)
    result = lsq_linear(weighted_matrix, weighted_target, bounds=(lower, upper), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(f"Aggregate solve failed: {result.message}")
    return AggregateModel(dataset, config, float(result.x[0]), result.x[1:])


def oof_splits(dataset: family_grp.Dataset) -> list[tuple[np.ndarray, np.ndarray]]:
    anchor = proportional_index(dataset)
    eligible = np.asarray([index for index in range(dataset.n) if index != anchor], dtype=int)
    source = dataset.frame.iloc[eligible]["panel_source"].to_numpy(dtype=str)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=20_720)
    result = []
    for _local_train, local_test in splitter.split(eligible, source):
        test = eligible[local_test]
        train = np.setdiff1d(np.arange(dataset.n), test, assume_unique=True)
        result.append((train, test))
    return result


def aggregate_oof(
    dataset: family_grp.Dataset,
    config: hierarchical.Config,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    evaluated = np.zeros(dataset.n, dtype=bool)
    for train, test in oof_splits(dataset):
        model = fit_aggregate(dataset, config, candidate, train)
        prediction[test] = model.predict(dataset.weights[test])
        evaluated[test] = True
    return prediction, evaluated


def scoped_aggregate_metrics(
    dataset: family_grp.Dataset,
    prediction: np.ndarray,
    evaluated: np.ndarray,
) -> list[dict[str, Any]]:
    source = dataset.frame["panel_source"].to_numpy(dtype=str)
    scopes = {
        "aggregate_all": evaluated,
        "aggregate_qsplit": evaluated & (source == "qsplit_signal"),
        "aggregate_deletion": evaluated & (source == "domain_deletion"),
    }
    return [
        {"scope": scope, **composition.prediction_metrics(dataset.target[mask], prediction[mask])}
        for scope, mask in scopes.items()
    ]


def stage1_gate(metrics: pd.DataFrame) -> dict[str, bool]:
    result = {CANDIDATES[0].name: True}
    for candidate in CANDIDATES[1:]:
        passed = True
        for target in TARGETS:
            for scope in ("aggregate_all", "aggregate_qsplit"):
                baseline = metrics.loc[
                    (metrics["target"] == target)
                    & (metrics["candidate"] == CANDIDATES[0].name)
                    & (metrics["scope"] == scope),
                    "rmse",
                ].iloc[0]
                value = metrics.loc[
                    (metrics["target"] == target)
                    & (metrics["candidate"] == candidate.name)
                    & (metrics["scope"] == scope),
                    "rmse",
                ].iloc[0]
                passed &= float(value) <= (1.0 + OOF_REGRESSION_TOLERANCE) * float(baseline)
            baseline_deletion = metrics.loc[
                (metrics["target"] == target)
                & (metrics["candidate"] == CANDIDATES[0].name)
                & (metrics["scope"] == "aggregate_deletion"),
                "rmse",
            ].iloc[0]
            value_deletion = metrics.loc[
                (metrics["target"] == target)
                & (metrics["candidate"] == candidate.name)
                & (metrics["scope"] == "aggregate_deletion"),
                "rmse",
            ].iloc[0]
            passed &= float(value_deletion) <= (1.0 - DELETION_IMPROVEMENT) * float(baseline_deletion)
        result[candidate.name] = bool(passed)
    return result


def fit_combined(
    matched: matched_pair.MatchedSources,
    target: str,
    target_index: int,
    candidate: Candidate,
) -> tuple[CombinedModel, float]:
    aggregate = aggregate_dataset(matched, target)
    aggregate_model = fit_aggregate(
        aggregate,
        composition.hpr_config(target),
        candidate,
        np.arange(aggregate.n),
    )
    phase_dataset, observed_delta, _frame = ppt.phase_pair_dataset(matched, target)
    phase_fit, pair_cv_rmse = ppt.full_phase_fit(
        phase_dataset,
        observed_delta,
        phase_candidate(),
        target_index,
    )
    return CombinedModel(aggregate_model, phase_fit, phase_dataset), pair_cv_rmse


def evaluate_development(
    matched: matched_pair.MatchedSources,
    candidates: list[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fiber_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        observed = matched.sources.common.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float)
        fiber_observed = matched.sources.fiber.frame[ppt.heterogeneous.fiber_delta_column(target)].to_numpy(dtype=float)
        for candidate in candidates:
            model, pair_cv_rmse = fit_combined(matched, target, target_index, candidate)
            predicted = model.predict(matched.sources.common.weights)
            for scope, mask in composition.scope_masks(matched.sources.common.frame, target).items():
                if int(mask.sum()) < 3:
                    continue
                metric_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "scope": scope,
                        "pair_cv_rmse": pair_cv_rmse,
                        **composition.prediction_metrics(observed[mask], predicted[mask]),
                    }
                )
            for index, row in matched.sources.common.frame.iterrows():
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "row_id": row["row_id"],
                        "policy_class": row["policy_class"],
                        "observed": observed[index],
                        "predicted": predicted[index],
                    }
                )
            fiber_prediction = ppt.fiber_delta_prediction(
                model,
                matched.sources.fiber.frame,
                matched.sources.fiber.weights,
            )
            for anchor in ["all", *sorted(matched.sources.fiber.frame["anchor_id"].unique())]:
                mask = np.ones(len(fiber_observed), dtype=bool)
                if anchor != "all":
                    mask = matched.sources.fiber.frame["anchor_id"].eq(anchor).to_numpy()
                fiber_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "anchor": anchor,
                        **composition.prediction_metrics(fiber_observed[mask], fiber_prediction[mask]),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows), pd.DataFrame(fiber_rows)


def stage2_gate(development: pd.DataFrame, fiber: pd.DataFrame, candidates: list[Candidate]) -> dict[str, bool]:
    result = {CANDIDATES[0].name: True}
    for candidate in candidates:
        if candidate == CANDIDATES[0]:
            continue
        passed = True
        for target in TARGETS:
            baseline = development.loc[
                (development["target"] == target)
                & (development["candidate"] == CANDIDATES[0].name)
                & (development["scope"] == "common_all")
            ].iloc[0]
            value = development.loc[
                (development["target"] == target)
                & (development["candidate"] == candidate.name)
                & (development["scope"] == "common_all")
            ].iloc[0]
            passed &= float(value["regret_at_1"]) <= float(baseline["regret_at_1"]) + REGRET_TOLERANCE
            passed &= abs(float(value["calibration_slope"]) - 1.0) <= abs(float(baseline["calibration_slope"]) - 1.0)
            passed &= int(value["optimism_gt_0p05"]) <= int(baseline["optimism_gt_0p05"])
            for anchor in fiber.loc[fiber["target"] == target, "anchor"].unique():
                base_rmse = fiber.loc[
                    (fiber["target"] == target)
                    & (fiber["candidate"] == CANDIDATES[0].name)
                    & (fiber["anchor"] == anchor),
                    "rmse",
                ].iloc[0]
                value_rmse = fiber.loc[
                    (fiber["target"] == target) & (fiber["candidate"] == candidate.name) & (fiber["anchor"] == anchor),
                    "rmse",
                ].iloc[0]
                passed &= float(value_rmse) <= (1.0 + FIBER_TOLERANCE) * float(base_rmse)
        result[candidate.name] = bool(passed)
    return result


def render_stage1(metrics: pd.DataFrame, output_dir: Path) -> None:
    plot = metrics.loc[metrics["scope"].isin(["aggregate_qsplit", "aggregate_deletion"])].copy()
    plot["source"] = plot["scope"].str.removeprefix("aggregate_")
    figure = px.bar(
        plot,
        x="candidate",
        y="rmse",
        color="source",
        facet_col="target",
        barmode="group",
        title="Design-balanced aggregate OOF by acquisition source",
        color_discrete_map={"qsplit": "#2f6b50", "deletion": "#d57832"},
    )
    figure.update_layout(template="plotly_white", xaxis_title="", yaxis_title="OOF RMSE")
    figure.write_html(output_dir / "aggregate_oof_by_source.html", include_plotlyjs=True, config=PLOT_CONFIG)


def render_development(predictions: pd.DataFrame, output_dir: Path) -> None:
    if predictions.empty:
        return
    targets = list(TARGETS)
    figure = make_subplots(rows=1, cols=len(targets), subplot_titles=targets)
    palette = px.colors.qualitative.Safe
    for column, target in enumerate(targets, start=1):
        local = predictions.loc[predictions["target"] == target]
        for index, (candidate, group) in enumerate(local.groupby("candidate", sort=False)):
            figure.add_trace(
                go.Scatter(
                    x=group["observed"],
                    y=group["predicted"],
                    mode="markers",
                    name=candidate,
                    legendgroup=candidate,
                    showlegend=column == 1,
                    marker={"size": 6, "opacity": 0.65, "color": palette[index % len(palette)]},
                    customdata=group[["row_id", "policy_class"]],
                    hovertemplate="%{customdata[0]}<br>%{customdata[1]}<br>obs=%{x:.5f}<br>pred=%{y:.5f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        lower = float(min(local["observed"].min(), local["predicted"].min()))
        upper = float(max(local["observed"].max(), local["predicted"].max()))
        figure.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                line={"dash": "dash", "color": "#718096"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
    figure.update_layout(template="plotly_white", title="Frozen development calibration", height=560)
    figure.update_xaxes(title_text="Observed BPB")
    figure.update_yaxes(title_text="Predicted BPB")
    figure.write_html(output_dir / "development_calibration.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_registry(stage1: dict[str, bool], stage2: dict[str, bool], output_dir: Path) -> None:
    rows = []
    for candidate in CANDIDATES:
        if candidate == CANDIDATES[0]:
            status = "reference"
        elif not stage1.get(candidate.name, False):
            status = "rejected_stage1"
        elif stage2.get(candidate.name, False):
            status = "promoted"
        else:
            status = "rejected_stage2"
        rows.append(
            {
                "family": candidate.name,
                "mechanism": "design-balanced orthogonal estimating equations",
                "equation_mode": candidate.equation_mode,
                "balance": candidate.balance,
                "additional_prediction_dof": 0,
                "status": status,
                "stage1_pass": stage1.get(candidate.name, False),
                "stage2_pass": stage2.get(candidate.name, False),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "approach_registry.csv", index=False)


def write_report(
    output_dir: Path,
    stage1_metrics: pd.DataFrame,
    phase_metrics: pd.DataFrame,
    stage1: dict[str, bool],
    development: pd.DataFrame,
    fiber: pd.DataFrame,
    stage2: dict[str, bool],
    stage1_only: bool,
) -> None:
    aggregate = stage1_metrics.loc[
        stage1_metrics["scope"].isin(["aggregate_all", "aggregate_qsplit", "aggregate_deletion"])
    ]
    lines = [
        "# Design-balanced orthogonal surrogate",
        "",
        (
            "The estimator uses the heterogeneous acquisition design as three orthogonal equation blocks rather than "
            "treating all checkpoints as exchangeable IID rows. Qsplit absolute levels estimate the aggregate response, "
            "phase-tied domain deletions test aggregate intervention transfer, and exact same-seed one/two-phase "
            "differences estimate the phase field. Source labels never enter prediction."
        ),
        "",
        "## Stage 1: fit-panel falsification",
        "",
        aggregate[
            ["target", "candidate", "scope", "n", "rmse", "spearman", "calibration_slope", "regret_at_1"]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Fixed phase field",
        "",
        phase_metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Gate",
        "",
        pd.DataFrame([{"candidate": key, "stage1_pass": value} for key, value in stage1.items()]).to_markdown(
            index=False
        ),
    ]
    if stage1_only:
        lines.extend(["", "Development outcomes were intentionally not read in this stage-1-only run."])
    elif development.empty:
        lines.extend(["", "No non-reference candidate cleared the frozen stage-1 gate on both targets."])
    else:
        common = development.loc[development["scope"] == "common_all"]
        lines.extend(
            [
                "",
                "## Stage 2: frozen development evidence",
                "",
                common[
                    [
                        "target",
                        "candidate",
                        "n",
                        "rmse",
                        "spearman",
                        "calibration_slope",
                        "regret_at_1",
                        "optimism_gt_0p05",
                        "worst_optimism",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
                "### Frontier phase fibers",
                "",
                fiber[["target", "candidate", "anchor", "rmse", "spearman", "calibration_slope"]].to_markdown(
                    index=False, floatfmt=".6f"
                ),
                "",
                "### Gate",
                "",
                pd.DataFrame([{"candidate": key, "stage2_pass": value} for key, value in stage2.items()]).to_markdown(
                    index=False
                ),
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_dir == DEFAULT_OUTPUT_DIR and not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing preregistration {PREREGISTRATION_PATH}")
    matched = matched_pair.matched_sources()
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        dataset = aggregate_dataset(matched, target)
        config = composition.hpr_config(target)
        for candidate in CANDIDATES:
            prediction, evaluated = aggregate_oof(dataset, config, candidate)
            for row in scoped_aggregate_metrics(dataset, prediction, evaluated):
                metric_rows.append({"target": target, "candidate": candidate.name, **row})
            for index in np.flatnonzero(evaluated):
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "row_id": dataset.frame.iloc[index]["run_name"],
                        "panel_source": dataset.frame.iloc[index]["panel_source"],
                        "observed": dataset.target[index],
                        "predicted": prediction[index],
                    }
                )
        phase_dataset, observed_delta, _frame = ppt.phase_pair_dataset(matched, target)
        phase_prediction, _selections, _coefficients, _names = ppt.nested_pair_prediction(
            phase_dataset,
            observed_delta,
            phase_candidate(),
            target_index,
        )
        phase_rows.append(
            {
                "target": target,
                "candidate": PHASE_CANDIDATE_NAME,
                **composition.prediction_metrics(observed_delta, phase_prediction),
            }
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    phase_metrics = pd.DataFrame(phase_rows)
    stage1 = stage1_gate(metrics)
    promoted = [candidate for candidate in CANDIDATES[1:] if stage1[candidate.name]]
    metrics.to_csv(args.output_dir / "aggregate_oof_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "aggregate_oof_predictions.csv", index=False)
    phase_metrics.to_csv(args.output_dir / "phase_pair_oof_metrics.csv", index=False)
    render_stage1(metrics, args.output_dir)

    development = pd.DataFrame()
    development_predictions = pd.DataFrame()
    fiber = pd.DataFrame()
    stage2: dict[str, bool] = {}
    if promoted and not args.stage1_only:
        evaluated = [CANDIDATES[0], *promoted]
        development, development_predictions, fiber = evaluate_development(matched, evaluated)
        stage2 = stage2_gate(development, fiber, evaluated)
        development.to_csv(args.output_dir / "development_metrics.csv", index=False)
        development_predictions.to_csv(args.output_dir / "development_predictions.csv", index=False)
        fiber.to_csv(args.output_dir / "fiber_metrics.csv", index=False)
        render_development(development_predictions, args.output_dir)

    ledger_rows = [
        {
            "candidate": candidate.name,
            "stage": "fit_panel",
            "outcomes_inspected_before_freeze": "prior exposed development evidence only; no outcomes from this round",
            "prediction_uses_source_label": False,
            "stage1_pass": stage1[candidate.name],
        }
        for candidate in CANDIDATES
    ]
    if not args.stage1_only:
        ledger_rows.extend(
            {
                "candidate": candidate.name,
                "stage": "frozen_development",
                "outcomes_inspected_before_freeze": (
                    "candidate equations and gates frozen in preregistered_candidates.json"
                ),
                "prediction_uses_source_label": False,
                "stage1_pass": True,
            }
            for candidate in promoted
        )
    pd.DataFrame(ledger_rows).to_csv(args.output_dir / "data_use_ledger.csv", index=False)
    write_registry(stage1, stage2, args.output_dir)
    write_report(args.output_dir, metrics, phase_metrics, stage1, development, fiber, stage2, args.stage1_only)
    (args.output_dir / "gate_results.json").write_text(
        json.dumps({"stage1": stage1, "stage2": stage2, "stage1_only": args.stage1_only}, indent=2) + "\n"
    )
    print(json.dumps({"stage1": stage1, "stage2": stage2}, indent=2))


if __name__ == "__main__":
    main()
