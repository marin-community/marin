# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "kaleido==0.2.1",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Materialize UniMax-8-anchored aggregate and low-phase-information paths.

This is a prior-sensitivity analysis of the decoupled policy optimizer. The
fitted one- and two-phase surrogates, aggregate-KL coefficients, and
phase-information budgets are held fixed. Only the aggregate reference
distribution changes from proportional to UniMax-8.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_decoupled_phase_information_constraints_300m as decoupled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_decoupled_phase_information_model_family_panel_300m as family_panel,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as matched,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_unimax_anchor_20260712"
DEFAULT_GCS_OUTPUT_DIR = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_decoupled_phase_information_unimax_anchor_20260712/mixtures"
)
PROPORTIONAL_PATH_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_low_epsilon_paths_20260712"
AGGREGATE_KL_VALUES = (0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
PHASE_INFORMATION_BUDGETS = (0.0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.025)
REQUESTED_FAMILIES = ("separate_heads", "effective_exposure")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
COLORS = {
    "proportional prior": "#5B7282",
    "UniMax-8 prior": "#E36F2C",
    "separate_heads": "#1B9E77",
    "effective_exposure": "#D95F02",
}


@dataclass(frozen=True)
class Anchor:
    tag: str
    objective: str
    aggregate_kl: float
    label: str


ANCHORS = (
    Anchor("unch05", "uncheatable", 0.05, "Uncheatable aggregate KL coefficient 0.05"),
    Anchor("t9s05", "table9", 0.05, "Table-9 stable aggregate KL coefficient 0.05"),
    Anchor("t9b075", "table9", 0.075, "Table-9 observed-best aggregate KL coefficient 0.075"),
)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def parse_str_tuple(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def unimax8_weights(domains: list[str], token_counts: np.ndarray, target_budget: int) -> np.ndarray:
    if len(domains) != len(token_counts):
        raise ValueError("UniMax-8 domain and token-count lengths differ")
    max_allocations = 8.0 * token_counts.astype(float)
    remaining_budget = float(target_budget)
    remaining_domains = len(domains)
    allocations = np.zeros(len(domains), dtype=float)
    allocated = np.zeros(len(domains), dtype=bool)
    for index in np.argsort(token_counts):
        uniform_share = remaining_budget / remaining_domains
        if uniform_share <= max_allocations[index]:
            allocations[~allocated] = uniform_share
            break
        allocations[index] = max_allocations[index]
        allocated[index] = True
        remaining_budget -= max_allocations[index]
        remaining_domains -= 1
    weights = allocations / allocations.sum()
    if np.min(weights) < 0.0 or not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("Invalid UniMax-8 reference distribution")
    epochs = float(target_budget) * weights / token_counts
    if float(epochs.max()) > 8.0 + 1e-10:
        raise ValueError(f"UniMax-8 reference exceeds its epoch cap: {epochs.max()}")
    return weights


def policy_tv(left: np.ndarray, right: np.ndarray) -> float:
    phase_fractions = np.asarray(matched.PHASE_FRACTIONS, dtype=float)
    return float(phase_fractions @ (0.5 * np.abs(left - right).sum(axis=1)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return float("nan")
    return float(np.dot(left, right) / (left_norm * right_norm))


def write_candidate(
    output_dir: Path,
    gcs_output_dir: str,
    candidate: str,
    frame: pd.DataFrame,
    *,
    upload: bool,
) -> None:
    path = output_dir / "mixtures" / f"{candidate}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    if upload:
        with fsspec.open(f"{gcs_output_dir.rstrip('/')}/{candidate}.csv", "wt") as handle:
            frame.to_csv(handle, index=False)


def load_one_phase_context(
    source_frame: pd.DataFrame,
) -> tuple[
    dict[str, matched.pooled.Dataset],
    dict[str, matched.SeparateHeadsModel],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, int],
]:
    selected_l2 = json.loads((decoupled.PANEL_DIR / "selected_models.json").read_text())
    datasets: dict[str, matched.pooled.Dataset] = {}
    models: dict[str, matched.SeparateHeadsModel] = {}
    proportional: dict[str, np.ndarray] = {}
    token_counts: dict[str, np.ndarray] = {}
    target_budgets: dict[str, int] = {}
    for objective in matched.OBJECTIVES:
        one_phase, _two_phase = matched.matched_datasets(source_frame, objective)
        datasets[objective] = one_phase
        models[objective] = matched.fit_model(
            one_phase,
            np.arange(one_phase.n),
            "1p",
            float(selected_l2[objective]["1p"]),
        )
        _packet, domains, natural, counts, target_budget, _folds = matched.bowl.load_objective(objective)
        if domains != one_phase.domain_names:
            raise ValueError(f"{objective}: domain ordering differs between fit and deployment data")
        proportional[objective] = natural
        token_counts[objective] = counts
        target_budgets[objective] = target_budget
    return datasets, models, proportional, token_counts, target_budgets


def materialize_aggregate_paths(
    output_dir: Path,
    gcs_output_dir: str,
    kl_values: tuple[float, ...],
    datasets: dict[str, matched.pooled.Dataset],
    models: dict[str, matched.SeparateHeadsModel],
    proportional: dict[str, np.ndarray],
    token_counts: dict[str, np.ndarray],
    target_budgets: dict[str, int],
    *,
    upload: bool,
) -> tuple[pd.DataFrame, dict[tuple[str, float], np.ndarray], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    aggregate_weights: dict[tuple[str, float], np.ndarray] = {}
    for objective in matched.OBJECTIVES:
        dataset = datasets[objective]
        model = models[objective]
        predict = matched.predictor(model, dataset)
        natural = proportional[objective]
        unimax = unimax8_weights(dataset.domain_names, token_counts[objective], target_budgets[objective])
        for kl_reg in kl_values:
            result = matched.optimize(predict, dataset, unimax, kl_reg, "1p")
            weights = result.weights
            aggregate = weights[0]
            aggregate_weights[(objective, kl_reg)] = aggregate
            epochs = matched.olmix.simulated_epochs(
                weights,
                token_counts[objective],
                target_budget=target_budgets[objective],
            )
            objective_tag = matched.TARGET_ABBR[objective]
            candidate = f"um8_sep_{objective_tag}_1p_kl{value_tag(kl_reg)}"
            frame = matched.per_component.mixture_frame(
                domains=dataset.domain_names,
                natural=natural,
                weights=weights,
                token_counts=token_counts[objective],
                target_budget=target_budgets[objective],
            )
            write_candidate(output_dir, gcs_output_dir, candidate, frame, upload=upload)
            rows.append(
                {
                    "candidate": candidate,
                    "objective": objective,
                    "aggregate_reference": "unimax8",
                    "aggregate_kl_coefficient": kl_reg,
                    "selected_l2": model.l2,
                    "predicted_bpb": predict(weights),
                    "regularized_objective": result.regularized_objective,
                    "aggregate_kl_to_unimax8": decoupled.categorical_kl(aggregate, unimax),
                    "aggregate_tv_to_unimax8": float(0.5 * np.abs(aggregate - unimax).sum()),
                    "aggregate_kl_to_proportional": decoupled.categorical_kl(aggregate, natural),
                    "aggregate_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
                    "max_weight": float(aggregate.max()),
                    "max_simulated_epoch": float(epochs.max()),
                    "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
                    "optimizer_successful_starts": result.successful_starts,
                }
            )

            proportional_candidate = f"origstyle_sep_{objective_tag}_1p_kl{value_tag(kl_reg)}"
            proportional_path = decoupled.PANEL_DIR / "mixtures" / f"{proportional_candidate}.csv"
            proportional_weights = decoupled.weights_from_frame(proportional_path, dataset.domain_names)
            proportional_epochs = matched.olmix.simulated_epochs(
                proportional_weights,
                token_counts[objective],
                target_budget=target_budgets[objective],
            )
            comparisons.append(
                {
                    "objective": objective,
                    "aggregate_kl_coefficient": kl_reg,
                    "unimax_candidate": candidate,
                    "proportional_candidate": proportional_candidate,
                    "aggregate_tv_between_priors": float(0.5 * np.abs(aggregate - proportional_weights[0]).sum()),
                    "unimax_predicted_bpb": predict(weights),
                    "proportional_predicted_bpb": predict(proportional_weights),
                    "unimax_minus_proportional_predicted_bpb": predict(weights) - predict(proportional_weights),
                    "unimax_max_simulated_epoch": float(epochs.max()),
                    "proportional_max_simulated_epoch": float(proportional_epochs.max()),
                    "unimax_tv_to_proportional": float(0.5 * np.abs(aggregate - natural).sum()),
                    "proportional_anchor_tv_to_proportional": float(
                        0.5 * np.abs(proportional_weights[0] - natural).sum()
                    ),
                    "unimax_tv_to_unimax8": float(0.5 * np.abs(aggregate - unimax).sum()),
                    "proportional_anchor_tv_to_unimax8": float(0.5 * np.abs(proportional_weights[0] - unimax).sum()),
                }
            )
    return pd.DataFrame(rows), aggregate_weights, pd.DataFrame(comparisons)


def proportional_path_candidate(anchor: Anchor, family: str, epsilon: float) -> str:
    if epsilon == 0.0:
        return f"dphase_{anchor.tag}_tied"
    return f"dphase_{anchor.tag}_{family_panel.FAMILY_TAGS[family]}_e{value_tag(epsilon)}"


def materialize_phase_paths(
    output_dir: Path,
    gcs_output_dir: str,
    phase_information_budgets: tuple[float, ...],
    requested_families: tuple[str, ...],
    datasets: dict[str, matched.pooled.Dataset],
    predictors: dict[str, dict[str, family_panel.Predictor]],
    proportional: dict[str, np.ndarray],
    token_counts: dict[str, np.ndarray],
    target_budgets: dict[str, int],
    aggregate_weights: dict[tuple[str, float], np.ndarray],
    *,
    upload: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unknown_families = sorted(set(requested_families).difference(REQUESTED_FAMILIES))
    if unknown_families:
        raise ValueError(f"Unsupported model families: {unknown_families}")
    rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    emitted: set[str] = set()
    for anchor in ANCHORS:
        dataset = datasets[anchor.objective]
        natural = proportional[anchor.objective]
        unimax = unimax8_weights(
            dataset.domain_names,
            token_counts[anchor.objective],
            target_budgets[anchor.objective],
        )
        aggregate = aggregate_weights[(anchor.objective, anchor.aggregate_kl)]
        tied = np.stack([aggregate, aggregate])
        for model_family in requested_families:
            predictor = predictors[anchor.objective][model_family]
            tied_prediction = predictor.predict(tied)
            for epsilon in phase_information_budgets:
                family_tag = family_panel.FAMILY_TAGS[model_family]
                candidate = (
                    f"dphase_um8_{anchor.tag}_tied"
                    if epsilon == 0.0
                    else f"dphase_um8_{anchor.tag}_{family_tag}_e{value_tag(epsilon)}"
                )
                result = family_panel.generic_optimize_fixed_aggregate(predictor.predict, aggregate, epsilon)
                aggregate_check = (
                    matched.PHASE_FRACTIONS[0] * result.weights[0] + matched.PHASE_FRACTIONS[1] * result.weights[1]
                )
                max_aggregate_error = float(np.max(np.abs(aggregate_check - aggregate)))
                if max_aggregate_error > 1e-9:
                    raise ValueError(f"{candidate} changed its aggregate by {max_aggregate_error}")
                geometry = decoupled.policy_geometry(result.weights, natural)
                aggregate_kl_to_unimax = decoupled.categorical_kl(aggregate, unimax)
                epochs = matched.olmix.simulated_epochs(
                    result.weights,
                    token_counts[anchor.objective],
                    target_budget=target_budgets[anchor.objective],
                )
                nearest_tv, nearest_bpb = decoupled.fixed_aggregate.nearest_observed_tv(dataset, result.weights)
                emit_manifest_row = candidate not in emitted
                if emit_manifest_row:
                    frame = matched.per_component.mixture_frame(
                        domains=dataset.domain_names,
                        natural=natural,
                        weights=result.weights,
                        token_counts=token_counts[anchor.objective],
                        target_budget=target_budgets[anchor.objective],
                    )
                    write_candidate(output_dir, gcs_output_dir, candidate, frame, upload=upload)
                    emitted.add(candidate)
                if emit_manifest_row:
                    rows.append(
                        {
                            "candidate": candidate,
                            "objective": anchor.objective,
                            "anchor_tag": anchor.tag,
                            "anchor_label": anchor.label,
                            "aggregate_reference": "unimax8",
                            "aggregate_kl_coefficient": anchor.aggregate_kl,
                            "family": "control" if epsilon == 0.0 else model_family,
                            "predictor_family": model_family,
                            "phase_information_budget": epsilon,
                            "phase_information": geometry["phase_information"],
                            "phase_tv": geometry["phase_tv"],
                            "aggregate_kl_to_unimax8": aggregate_kl_to_unimax,
                            "joint_policy_kl_to_unimax8": aggregate_kl_to_unimax + geometry["phase_information"],
                            "aggregate_tv_to_unimax8": float(0.5 * np.abs(aggregate - unimax).sum()),
                            "aggregate_kl_to_proportional": geometry["aggregate_kl"],
                            "aggregate_tv_to_proportional": geometry["aggregate_tv"],
                            "predicted_bpb": result.prediction,
                            "tied_prediction": tied_prediction,
                            "predicted_gain_vs_tied": tied_prediction - result.prediction,
                            "max_weight": float(result.weights.max()),
                            "max_simulated_epoch": float(epochs.max()),
                            "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
                            "nearest_observed_tv": nearest_tv,
                            "nearest_observed_bpb": nearest_bpb,
                            "successful_starts": result.successful_starts,
                            "max_aggregate_error": max_aggregate_error,
                        }
                    )

                proportional_candidate = proportional_path_candidate(anchor, model_family, epsilon)
                proportional_path = PROPORTIONAL_PATH_DIR / "mixtures" / f"{proportional_candidate}.csv"
                proportional_weights = decoupled.weights_from_frame(proportional_path, dataset.domain_names)
                proportional_aggregate = (
                    matched.PHASE_FRACTIONS[0] * proportional_weights[0]
                    + matched.PHASE_FRACTIONS[1] * proportional_weights[1]
                )
                proportional_epochs = matched.olmix.simulated_epochs(
                    proportional_weights,
                    token_counts[anchor.objective],
                    target_budget=target_budgets[anchor.objective],
                )
                comparisons.append(
                    {
                        "objective": anchor.objective,
                        "anchor_tag": anchor.tag,
                        "family": model_family,
                        "phase_information_budget": epsilon,
                        "unimax_candidate": candidate,
                        "proportional_candidate": proportional_candidate,
                        "weighted_policy_tv": policy_tv(result.weights, proportional_weights),
                        "aggregate_tv": float(0.5 * np.abs(aggregate - proportional_aggregate).sum()),
                        "phase0_tv": float(0.5 * np.abs(result.weights[0] - proportional_weights[0]).sum()),
                        "phase1_tv": float(0.5 * np.abs(result.weights[1] - proportional_weights[1]).sum()),
                        "phase_direction_cosine": cosine_similarity(
                            result.weights[1] - result.weights[0],
                            proportional_weights[1] - proportional_weights[0],
                        ),
                        "unimax_phase_tv": geometry["phase_tv"],
                        "proportional_phase_tv": float(
                            0.5 * np.abs(proportional_weights[0] - proportional_weights[1]).sum()
                        ),
                        "unimax_predicted_bpb": result.prediction,
                        "proportional_predicted_bpb": predictor.predict(proportional_weights),
                        "unimax_minus_proportional_predicted_bpb": (
                            result.prediction - predictor.predict(proportional_weights)
                        ),
                        "unimax_max_simulated_epoch": float(epochs.max()),
                        "proportional_max_simulated_epoch": float(proportional_epochs.max()),
                    }
                )
    manifest = pd.DataFrame(rows).sort_values(
        ["objective", "anchor_tag", "predictor_family", "phase_information_budget"]
    )
    comparison = pd.DataFrame(comparisons).sort_values(["objective", "anchor_tag", "family", "phase_information_budget"])
    return manifest, comparison


def aggregate_path_plot(aggregate_paths: pd.DataFrame, comparison: pd.DataFrame, output: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable predicted BPB",
            "Table-9 predicted BPB",
            "Uncheatable maximum epochs",
            "Table-9 maximum epochs",
        ),
    )
    for column, objective in enumerate(matched.OBJECTIVES, start=1):
        data = comparison.loc[comparison["objective"].eq(objective)].sort_values("aggregate_kl_coefficient")
        for reference, prediction_column, epoch_column in (
            ("proportional prior", "proportional_predicted_bpb", "proportional_max_simulated_epoch"),
            ("UniMax-8 prior", "unimax_predicted_bpb", "unimax_max_simulated_epoch"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=data["aggregate_kl_coefficient"],
                    y=data[prediction_column],
                    mode="lines+markers",
                    name=reference,
                    legendgroup=reference,
                    showlegend=column == 1,
                    line={"color": COLORS[reference]},
                ),
                row=1,
                col=column,
            )
            figure.add_trace(
                go.Scatter(
                    x=data["aggregate_kl_coefficient"],
                    y=data[epoch_column],
                    mode="lines+markers",
                    name=reference,
                    legendgroup=reference,
                    showlegend=False,
                    line={"color": COLORS[reference]},
                ),
                row=2,
                col=column,
            )
    figure.update_xaxes(title_text="aggregate KL coefficient", row=2, col=1)
    figure.update_xaxes(title_text="aggregate KL coefficient", row=2, col=2)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=1)
    figure.update_yaxes(title_text="predicted BPB", row=1, col=2)
    figure.update_yaxes(title_text="maximum simulated epochs", row=2, col=1)
    figure.update_yaxes(title_text="maximum simulated epochs", row=2, col=2)
    figure.update_layout(
        title="Tied aggregate path: proportional versus UniMax-8 regularization",
        template="plotly_white",
        width=1300,
        height=850,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.1},
        margin={"l": 70, "r": 30, "t": 90, "b": 110},
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def aggregate_weight_plot(
    aggregate_weights: dict[tuple[str, float], np.ndarray],
    datasets: dict[str, matched.pooled.Dataset],
    proportional: dict[str, np.ndarray],
    token_counts: dict[str, np.ndarray],
    target_budgets: dict[str, int],
    output: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=len(ANCHORS),
        subplot_titles=[anchor.label for anchor in ANCHORS],
        shared_yaxes=True,
    )
    for column, anchor in enumerate(ANCHORS, start=1):
        dataset = datasets[anchor.objective]
        natural = proportional[anchor.objective]
        unimax = unimax8_weights(
            dataset.domain_names,
            token_counts[anchor.objective],
            target_budgets[anchor.objective],
        )
        unimax_anchor = aggregate_weights[(anchor.objective, anchor.aggregate_kl)]
        proportional_candidate = (
            f"origstyle_sep_{matched.TARGET_ABBR[anchor.objective]}_1p_kl{value_tag(anchor.aggregate_kl)}"
        )
        proportional_anchor = decoupled.weights_from_frame(
            decoupled.PANEL_DIR / "mixtures" / f"{proportional_candidate}.csv",
            dataset.domain_names,
        )[0]
        ordering = np.argsort(np.maximum(unimax_anchor, proportional_anchor))
        for label, weights, color, symbol in (
            ("proportional", natural, "#AAB6BF", "circle-open"),
            ("UniMax-8", unimax, "#E9B949", "diamond-open"),
            ("proportional-anchored optimum", proportional_anchor, "#5B7282", "circle"),
            ("UniMax-anchored optimum", unimax_anchor, "#E36F2C", "diamond"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=weights[ordering],
                    y=np.asarray(dataset.domain_names)[ordering],
                    mode="markers",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    marker={"color": color, "symbol": symbol, "size": 8},
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="aggregate weight", row=1, col=column)
    figure.update_layout(
        title="Aggregate anchors induced by proportional and UniMax-8 priors",
        template="plotly_white",
        width=1750,
        height=1050,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.08},
        margin={"l": 230, "r": 40, "t": 110, "b": 120},
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def phase_path_plot(comparison: pd.DataFrame, output: Path) -> None:
    nonzero = comparison.loc[comparison["phase_information_budget"].gt(0)].copy()
    metrics = (
        ("unimax_phase_tv", "UniMax-anchored phase TV"),
        ("unimax_minus_proportional_predicted_bpb", "UniMax minus proportional predicted BPB"),
        ("weighted_policy_tv", "Policy TV between aggregate priors"),
    )
    figure = make_subplots(
        rows=len(metrics),
        cols=len(ANCHORS),
        subplot_titles=[anchor.label for _metric in metrics for anchor in ANCHORS],
        shared_xaxes=True,
    )
    for column, anchor in enumerate(ANCHORS, start=1):
        for model_family in REQUESTED_FAMILIES:
            data = nonzero.loc[nonzero["anchor_tag"].eq(anchor.tag) & nonzero["family"].eq(model_family)].sort_values(
                "phase_information_budget"
            )
            for row, (metric, _label) in enumerate(metrics, start=1):
                figure.add_trace(
                    go.Scatter(
                        x=data["phase_information_budget"],
                        y=data[metric],
                        mode="lines+markers",
                        name=model_family,
                        legendgroup=model_family,
                        showlegend=column == 1 and row == 1,
                        line={"color": COLORS[model_family]},
                    ),
                    row=row,
                    col=column,
                )
    for row, (_metric, label) in enumerate(metrics, start=1):
        figure.update_yaxes(title_text=label, row=row, col=1)
    for column in range(1, len(ANCHORS) + 1):
        figure.update_xaxes(title_text="phase-information budget", row=len(metrics), col=column)
    figure.update_layout(
        title="Low-phase-information paths: UniMax-8 versus proportional aggregate anchors",
        template="plotly_white",
        width=1700,
        height=1150,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.06},
        margin={"l": 110, "r": 40, "t": 120, "b": 100},
    )
    figure.write_html(output, include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(
    aggregate_paths: pd.DataFrame,
    aggregate_comparison: pd.DataFrame,
    phase_manifest: pd.DataFrame,
    phase_comparison: pd.DataFrame,
    output: Path,
) -> None:
    selected_aggregate = aggregate_paths.loc[
        aggregate_paths.apply(
            lambda row: any(
                row["objective"] == anchor.objective and np.isclose(row["aggregate_kl_coefficient"], anchor.aggregate_kl)
                for anchor in ANCHORS
            ),
            axis=1,
        )
    ]
    low_epsilon_summary = (
        phase_manifest.loc[phase_manifest["phase_information_budget"].gt(0)]
        .groupby(["objective", "anchor_tag", "predictor_family"], as_index=False)
        .agg(
            maximum_phase_tv=("phase_tv", "max"),
            maximum_predicted_gain=("predicted_gain_vs_tied", "max"),
            maximum_simulated_epoch=("max_simulated_epoch", "max"),
            minimum_successful_starts=("successful_starts", "min"),
        )
    )
    paired_summary = (
        phase_comparison.loc[phase_comparison["phase_information_budget"].gt(0)]
        .groupby(["objective", "anchor_tag", "family"], as_index=False)
        .agg(
            maximum_policy_tv=("weighted_policy_tv", "max"),
            mean_phase_direction_cosine=("phase_direction_cosine", "mean"),
            minimum_predicted_bpb_delta=("unimax_minus_proportional_predicted_bpb", "min"),
            maximum_predicted_bpb_delta=("unimax_minus_proportional_predicted_bpb", "max"),
        )
    )
    lines = [
        "# UniMax-8 aggregate-anchor sensitivity",
        "",
        "The fitted surrogates and phase-information constraints are unchanged. The only intervention is "
        "replacing proportional with UniMax-8 as the reference distribution in the tied aggregate optimization. "
        "Mixture CSVs still report epoch multipliers relative to the true proportional distribution.",
        "",
        "Equal aggregate-KL coefficients are a controlled sensitivity comparison, not an assertion that they impose "
        "equal effective shrinkage under the two priors. The full tied aggregate paths should be inspected before "
        "choosing candidates for training validation.",
        "",
        "## Selected same-coefficient UniMax aggregates",
        "",
        selected_aggregate.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Same-coefficient aggregate comparison",
        "",
        aggregate_comparison.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Low-epsilon path summary",
        "",
        low_epsilon_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired prior-sensitivity summary",
        "",
        paired_summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation boundary",
        "",
        "These are surrogate optima. A better predicted value under UniMax anchoring is not evidence of better "
        "trained-model performance. The first validation gate should compare tied UniMax aggregates with their "
        "proportional-anchored counterparts; phase-asymmetric UniMax candidates are justified only if the tied "
        "aggregate is competitive.",
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gcs-output-dir", default=DEFAULT_GCS_OUTPUT_DIR)
    parser.add_argument(
        "--aggregate-kl-values",
        default=",".join(str(value) for value in AGGREGATE_KL_VALUES),
    )
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument("--families", default=",".join(REQUESTED_FAMILIES))
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_kl_values = parse_float_tuple(args.aggregate_kl_values)
    phase_information_budgets = parse_float_tuple(args.phase_information_budgets)
    requested_families = parse_str_tuple(args.families)
    required_kl_values = {anchor.aggregate_kl for anchor in ANCHORS}
    if not required_kl_values.issubset(aggregate_kl_values):
        raise ValueError(f"Aggregate KL grid must include anchor coefficients {sorted(required_kl_values)}")
    if 0.0 not in phase_information_budgets:
        raise ValueError("Phase-information grid must include the tied epsilon=0 control")

    reference = matched.pooled.load_300m_dataset("table9")
    source_frame = matched.joint.attach_single_phase_weights(
        pd.read_csv(matched.joint.PACKET),
        matched.joint.ONE_PHASE_SOURCE,
        reference.domain_names,
    )
    one_datasets, one_models, proportional, token_counts, target_budgets = load_one_phase_context(source_frame)
    aggregate_paths, aggregate_weights, aggregate_comparison = materialize_aggregate_paths(
        args.output_dir,
        args.gcs_output_dir,
        aggregate_kl_values,
        one_datasets,
        one_models,
        proportional,
        token_counts,
        target_budgets,
        upload=args.upload,
    )

    two_datasets, separate_models, two_proportional, two_token_counts, two_target_budgets = decoupled.load_context()
    for objective in matched.OBJECTIVES:
        if not np.allclose(proportional[objective], two_proportional[objective], atol=1e-12):
            raise ValueError(f"{objective}: one- and two-phase proportional references differ")
        if not np.array_equal(token_counts[objective], two_token_counts[objective]):
            raise ValueError(f"{objective}: one- and two-phase token counts differ")
        if target_budgets[objective] != two_target_budgets[objective]:
            raise ValueError(f"{objective}: one- and two-phase target budgets differ")
    predictors = family_panel.fit_predictors(two_datasets, separate_models)
    phase_manifest, phase_comparison = materialize_phase_paths(
        args.output_dir,
        args.gcs_output_dir,
        phase_information_budgets,
        requested_families,
        two_datasets,
        predictors,
        proportional,
        token_counts,
        target_budgets,
        aggregate_weights,
        upload=args.upload,
    )

    if phase_manifest["candidate"].isna().any() or phase_manifest["candidate"].duplicated().any():
        raise AssertionError("Candidate names must be present and unique")
    if float(phase_manifest["max_aggregate_error"].max()) > 1e-9:
        raise AssertionError("A fixed-aggregate phase path changed aggregate weights")
    if (
        float((phase_manifest["phase_information"] - phase_manifest["phase_information_budget"]).max())
        > family_panel.FEASIBILITY_TOLERANCE
    ):
        raise AssertionError("A phase-information solve exceeded its budget")

    aggregate_paths.to_csv(args.output_dir / "aggregate_path_manifest.csv", index=False)
    aggregate_comparison.to_csv(args.output_dir / "aggregate_prior_comparison.csv", index=False)
    phase_manifest.to_csv(args.output_dir / "candidate_manifest.csv", index=False)
    phase_comparison.to_csv(args.output_dir / "proportional_prior_comparison.csv", index=False)
    aggregate_path_plot(aggregate_paths, aggregate_comparison, args.output_dir / "tied_aggregate_kl_paths.html")
    aggregate_weight_plot(
        aggregate_weights,
        one_datasets,
        proportional,
        token_counts,
        target_budgets,
        args.output_dir / "aggregate_weights.html",
    )
    phase_path_plot(phase_comparison, args.output_dir / "low_epsilon_prior_comparison.html")
    write_report(
        aggregate_paths,
        aggregate_comparison,
        phase_manifest,
        phase_comparison,
        args.output_dir / "report.md",
    )
    (args.output_dir / "panel_config.json").write_text(
        json.dumps(
            {
                "aggregate_reference": "unimax8",
                "aggregate_kl_values": aggregate_kl_values,
                "phase_information_budgets": phase_information_budgets,
                "families": requested_families,
                "anchors": [anchor.__dict__ for anchor in ANCHORS],
                "gcs_output_dir": args.gcs_output_dir,
                "upload": args.upload,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(aggregate_comparison.to_string(index=False))
    print(phase_comparison.to_string(index=False))
    print(f"Wrote {len(aggregate_paths)} aggregate candidates and {len(phase_manifest)} path rows to {args.output_dir}")


if __name__ == "__main__":
    main()
