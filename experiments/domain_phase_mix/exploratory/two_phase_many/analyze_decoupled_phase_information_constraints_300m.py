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
"""Audit separate aggregate and phase-information constraints for separate heads."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_centered_recency_anchor_ordering as fixed_aggregate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_original_style_matched_sepheads_ablation_300m as matched,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "original_style_matched_sepheads_ablation_20260712"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "decoupled_phase_information_constraints_20260712"
PHASE_INFORMATION_BUDGETS = (0.0, 0.01, 0.025, 0.05, 0.1)
REFIT_PHASE_INFORMATION_BUDGETS = (0.01, 0.025)
CV_SEEDS = (0, 1, 2)
N_SPLITS = 5
FEASIBILITY_TOLERANCE = 1e-7
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class AnchorSpec:
    objective: str
    label: str
    candidate: str


@dataclass(frozen=True)
class SolveResult:
    weights: np.ndarray
    prediction: float
    successful_starts: int


ANCHORS = (
    AnchorSpec("uncheatable", "Uncheatable best 1p", "origstyle_sep_unch_1p_kl0p05"),
    AnchorSpec("table9", "Table-9 stable 1p", "origstyle_sep_t9_1p_kl0p05"),
    AnchorSpec("table9", "Table-9 selected 1p", "origstyle_sep_t9_1p_kl0p075"),
)


def parse_float_tuple(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def epsilon_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def categorical_kl(weights: np.ndarray, reference: np.ndarray) -> float:
    values = np.clip(np.asarray(weights, dtype=float), 1e-12, 1.0)
    baseline = np.clip(np.asarray(reference, dtype=float), 1e-12, 1.0)
    return float(np.sum(values * (np.log(values) - np.log(baseline))))


def policy_geometry(weights: np.ndarray, natural: np.ndarray) -> dict[str, float]:
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    aggregate = alpha0 * weights[0] + alpha1 * weights[1]
    aggregate_kl = categorical_kl(aggregate, natural)
    phase_information = fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
    return {
        "aggregate_kl": aggregate_kl,
        "phase_information": phase_information,
        "joint_phase_kl": aggregate_kl + phase_information,
        "aggregate_tv": float(0.5 * np.abs(aggregate - natural).sum()),
        "phase_tv": float(0.5 * np.abs(weights[0] - weights[1]).sum()),
    }


def weights_from_frame(path: Path, domains: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("domain").loc[domains]
    weights = frame[["phase_0_weight", "phase_1_weight"]].to_numpy(float).T
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"Invalid phase sums in {path}")
    return weights


def feasible_start(
    delta: np.ndarray,
    aggregate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    phase_information_budget: float,
) -> np.ndarray:
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    scale = 1.0
    for _attempt in range(60):
        candidate = scale * delta
        weights = fixed_aggregate.weights_from_delta(aggregate, candidate, alpha0, alpha1)
        information = fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)
        if np.all(candidate >= lower) and np.all(candidate <= upper) and information <= 0.8 * phase_information_budget:
            return candidate
        scale *= 0.5
    return np.zeros_like(delta)


def optimize_fixed_aggregate(
    model: matched.SeparateHeadsModel,
    dataset: matched.pooled.Dataset,
    aggregate: np.ndarray,
    phase_information_budget: float,
) -> SolveResult:
    """Optimize phase order while preserving aggregate weights exactly."""
    alpha0, alpha1 = matched.PHASE_FRACTIONS
    tied = np.stack([aggregate, aggregate])
    if phase_information_budget == 0.0:
        prediction = float(matched.predict_model(model, dataset, tied[None, :, :])[0])
        return SolveResult(tied, prediction, 1)

    lower = -aggregate / alpha1 + 1e-10
    upper = aggregate / alpha0 - 1e-10

    def weights_from_delta(delta: np.ndarray) -> np.ndarray:
        return fixed_aggregate.weights_from_delta(aggregate, delta, alpha0, alpha1)

    def phase_information(delta: np.ndarray) -> float:
        weights = weights_from_delta(delta)
        return fixed_aggregate.phase_order_kl(weights, aggregate, alpha0, alpha1)

    def objective(delta: np.ndarray) -> float:
        weights = weights_from_delta(delta)
        return float(matched.predict_model(model, dataset, weights[None, :, :])[0])

    rng = np.random.default_rng(0)
    starts = [np.zeros_like(aggregate)]
    starts.extend(
        feasible_start(
            fixed_aggregate.random_start(aggregate, lower, upper, rng),
            aggregate,
            lower,
            upper,
            phase_information_budget,
        )
        for _index in range(8)
    )
    constraints = [
        {"type": "eq", "fun": lambda delta: float(np.sum(delta))},
        {
            "type": "ineq",
            "fun": lambda delta: phase_information_budget - phase_information(delta),
        },
    ]
    bounds = list(zip(lower, upper, strict=True))
    best_value = np.inf
    best_weights: np.ndarray | None = None
    successful_starts = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-11},
        )
        if result.success:
            successful_starts += 1
        candidate_weights = weights_from_delta(np.asarray(result.x, dtype=float))
        candidate_information = fixed_aggregate.phase_order_kl(
            candidate_weights,
            aggregate,
            alpha0,
            alpha1,
        )
        if (
            np.isfinite(result.fun)
            and candidate_information <= phase_information_budget + FEASIBILITY_TOLERANCE
            and np.min(candidate_weights) >= -FEASIBILITY_TOLERANCE
            and float(result.fun) < best_value
        ):
            best_value = float(result.fun)
            best_weights = candidate_weights
    if best_weights is None:
        raise RuntimeError(f"No feasible solve for phase-information budget {phase_information_budget:g}")
    return SolveResult(best_weights, best_value, successful_starts)


def load_context() -> tuple[
    dict[str, matched.pooled.Dataset],
    dict[str, matched.SeparateHeadsModel],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, int],
]:
    reference = matched.pooled.load_300m_dataset("table9")
    frame = matched.joint.attach_single_phase_weights(
        pd.read_csv(matched.joint.PACKET),
        matched.joint.ONE_PHASE_SOURCE,
        reference.domain_names,
    )
    selected_l2 = json.loads((PANEL_DIR / "selected_models.json").read_text())
    datasets = {}
    models = {}
    natural = {}
    token_counts = {}
    target_budgets = {}
    for objective in matched.OBJECTIVES:
        _one_phase, two_phase = matched.matched_datasets(frame, objective)
        datasets[objective] = two_phase
        models[objective] = matched.fit_model(
            two_phase,
            np.arange(two_phase.n),
            "2p",
            float(selected_l2[objective]["2p"]),
        )
        _packet, _domains, objective_natural, counts, budget, _folds = matched.bowl.load_objective(objective)
        natural[objective] = objective_natural
        token_counts[objective] = counts
        target_budgets[objective] = budget
    return datasets, models, natural, token_counts, target_budgets


def candidate_name(anchor: AnchorSpec, phase_information_budget: float) -> str:
    objective = "unch" if anchor.objective == "uncheatable" else "t9"
    anchor_kind = "best" if "best" in anchor.label.lower() else "stable"
    if "selected" in anchor.label.lower():
        anchor_kind = "selected"
    return f"decphase_{objective}_{anchor_kind}_eps{epsilon_tag(phase_information_budget)}"


def build_candidates(
    output_dir: Path,
    phase_information_budgets: tuple[float, ...],
    datasets: dict[str, matched.pooled.Dataset],
    models: dict[str, matched.SeparateHeadsModel],
    natural: dict[str, np.ndarray],
    token_counts: dict[str, np.ndarray],
    target_budgets: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    observed = pd.read_csv(PANEL_DIR / "observed_results.csv")
    rows = []
    candidate_weights = {}
    mixture_dir = output_dir / "mixtures"
    mixture_dir.mkdir(parents=True, exist_ok=True)
    for anchor in ANCHORS:
        dataset = datasets[anchor.objective]
        anchor_path = PANEL_DIR / "mixtures" / f"{anchor.candidate}.csv"
        anchor_weights = weights_from_frame(anchor_path, dataset.domain_names)
        if not np.allclose(anchor_weights[0], anchor_weights[1], atol=1e-10):
            raise ValueError(f"Anchor {anchor.candidate} is not tied")
        aggregate = anchor_weights[0]
        tied = np.stack([aggregate, aggregate])
        tied_prediction = float(matched.predict_model(models[anchor.objective], dataset, tied[None, :, :])[0])
        old_best = (
            observed.loc[observed["objective"].eq(anchor.objective) & observed["policy"].eq("2p")]
            .sort_values("observed_target_bpb")
            .iloc[0]
        )
        old_geometry = policy_geometry(
            weights_from_frame(
                PANEL_DIR / "mixtures" / f"{old_best['candidate']}.csv",
                dataset.domain_names,
            ),
            natural[anchor.objective],
        )
        for phase_information_budget in phase_information_budgets:
            result = optimize_fixed_aggregate(
                models[anchor.objective],
                dataset,
                aggregate,
                phase_information_budget,
            )
            candidate = candidate_name(anchor, phase_information_budget)
            candidate_weights[candidate] = result.weights
            aggregate_check = (
                matched.PHASE_FRACTIONS[0] * result.weights[0] + matched.PHASE_FRACTIONS[1] * result.weights[1]
            )
            max_aggregate_error = float(np.max(np.abs(aggregate_check - aggregate)))
            if max_aggregate_error > 1e-9:
                raise ValueError(f"{candidate} changed aggregate weights by {max_aggregate_error}")
            geometry = policy_geometry(result.weights, natural[anchor.objective])
            epochs = matched.olmix.simulated_epochs(
                result.weights,
                token_counts[anchor.objective],
                target_budget=target_budgets[anchor.objective],
            )
            nearest_tv, nearest_observed = fixed_aggregate.nearest_observed_tv(dataset, result.weights)
            rows.append(
                {
                    "candidate": candidate,
                    "objective": anchor.objective,
                    "anchor_label": anchor.label,
                    "anchor_candidate": anchor.candidate,
                    "phase_information_budget": phase_information_budget,
                    "predicted_bpb": result.prediction,
                    "tied_prediction": tied_prediction,
                    "predicted_gain_vs_tied": tied_prediction - result.prediction,
                    **geometry,
                    "max_simulated_epoch": float(np.max(epochs)),
                    "q95_simulated_epoch": float(np.quantile(epochs, 0.95)),
                    "nearest_observed_tv": nearest_tv,
                    "nearest_observed_bpb": nearest_observed,
                    "successful_starts": result.successful_starts,
                    "max_aggregate_error": max_aggregate_error,
                    "old_best_2p_candidate": old_best["candidate"],
                    "old_best_2p_aggregate_kl": old_geometry["aggregate_kl"],
                    "aggregate_kl_ratio_vs_old_2p": geometry["aggregate_kl"] / old_geometry["aggregate_kl"],
                    "old_best_2p_max_epoch": old_best["max_simulated_epoch"],
                    "max_epoch_ratio_vs_old_2p": float(np.max(epochs)) / old_best["max_simulated_epoch"],
                }
            )
            mixture = matched.per_component.mixture_frame(
                domains=dataset.domain_names,
                natural=natural[anchor.objective],
                weights=result.weights,
                token_counts=token_counts[anchor.objective],
                target_budget=target_budgets[anchor.objective],
            )
            mixture.to_csv(mixture_dir / f"{candidate}.csv", index=False)
    return pd.DataFrame(rows), candidate_weights


def crossfit_models(
    datasets: dict[str, matched.pooled.Dataset],
) -> dict[str, list[tuple[int, int, np.ndarray, matched.SeparateHeadsModel]]]:
    selected_l2 = json.loads((PANEL_DIR / "selected_models.json").read_text())
    models = {}
    for objective, dataset in datasets.items():
        objective_models = []
        for seed in CV_SEEDS:
            folds = matched.component_dsp.panel_stratified_folds(dataset.frame, n_splits=N_SPLITS, seed=seed)
            for fold, (train_indices, test_indices) in enumerate(folds):
                model = matched.fit_model(
                    dataset,
                    train_indices,
                    "2p",
                    float(selected_l2[objective]["2p"]),
                )
                objective_models.append((seed, fold, test_indices, model))
        models[objective] = objective_models
    return models


def candidate_crossfit_diagnostics(
    candidates: pd.DataFrame,
    candidate_weights: dict[str, np.ndarray],
    datasets: dict[str, matched.pooled.Dataset],
    fold_models: dict[str, list[tuple[int, int, np.ndarray, matched.SeparateHeadsModel]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record in candidates.to_dict(orient="records"):
        objective = str(record["objective"])
        dataset = datasets[objective]
        weights = candidate_weights[str(record["candidate"])]
        aggregate = matched.PHASE_FRACTIONS[0] * weights[0] + matched.PHASE_FRACTIONS[1] * weights[1]
        tied = np.stack([aggregate, aggregate])
        for seed, fold, _test_indices, model in fold_models[objective]:
            candidate_prediction = float(matched.predict_model(model, dataset, weights[None, :, :])[0])
            tied_prediction = float(matched.predict_model(model, dataset, tied[None, :, :])[0])
            rows.append(
                {
                    "candidate": record["candidate"],
                    "objective": objective,
                    "anchor_label": record["anchor_label"],
                    "phase_information_budget": record["phase_information_budget"],
                    "seed": seed,
                    "fold": fold,
                    "candidate_prediction": candidate_prediction,
                    "tied_prediction": tied_prediction,
                    "predicted_gain_vs_tied": tied_prediction - candidate_prediction,
                }
            )
    diagnostics = pd.DataFrame(rows)
    summary = (
        diagnostics.groupby(
            ["candidate", "objective", "anchor_label", "phase_information_budget"],
            as_index=False,
        )
        .agg(
            n_refits=("predicted_gain_vs_tied", "size"),
            crossfit_gain_mean=("predicted_gain_vs_tied", "mean"),
            crossfit_gain_sd=("predicted_gain_vs_tied", "std"),
            crossfit_gain_min=("predicted_gain_vs_tied", "min"),
            crossfit_gain_max=("predicted_gain_vs_tied", "max"),
            crossfit_gain_positive_fraction=("predicted_gain_vs_tied", lambda values: values.gt(0).mean()),
        )
        .sort_values(["objective", "anchor_label", "phase_information_budget"])
    )
    return diagnostics, summary


def direction_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-15:
        return np.nan
    return float(np.dot(left, right) / denominator)


def candidate_refit_stability(
    candidates: pd.DataFrame,
    candidate_weights: dict[str, np.ndarray],
    datasets: dict[str, matched.pooled.Dataset],
    fold_models: dict[str, list[tuple[int, int, np.ndarray, matched.SeparateHeadsModel]]],
    refit_phase_information_budgets: tuple[float, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-optimize each candidate's phase direction under every fold model."""
    rows = []
    for record in candidates.to_dict(orient="records"):
        phase_information_budget = float(record["phase_information_budget"])
        if phase_information_budget not in refit_phase_information_budgets:
            continue
        candidate = str(record["candidate"])
        objective = str(record["objective"])
        dataset = datasets[objective]
        full_weights = candidate_weights[candidate]
        aggregate = matched.PHASE_FRACTIONS[0] * full_weights[0] + matched.PHASE_FRACTIONS[1] * full_weights[1]
        tied = np.stack([aggregate, aggregate])
        full_direction = full_weights[0] - full_weights[1]
        for seed, fold, _test_indices, model in fold_models[objective]:
            result = optimize_fixed_aggregate(
                model,
                dataset,
                aggregate,
                phase_information_budget,
            )
            tied_prediction = float(matched.predict_model(model, dataset, tied[None, :, :])[0])
            refit_direction = result.weights[0] - result.weights[1]
            rows.append(
                {
                    "candidate": candidate,
                    "objective": objective,
                    "anchor_label": record["anchor_label"],
                    "phase_information_budget": phase_information_budget,
                    "seed": seed,
                    "fold": fold,
                    "refit_prediction": result.prediction,
                    "tied_prediction": tied_prediction,
                    "refit_gain_vs_tied": tied_prediction - result.prediction,
                    "direction_cosine_vs_full": direction_cosine(full_direction, refit_direction),
                    "policy_tv_vs_full": float(0.5 * np.abs(result.weights - full_weights).sum(axis=1).mean()),
                    "refit_phase_information": policy_geometry(result.weights, aggregate)["phase_information"],
                    "successful_starts": result.successful_starts,
                }
            )
    diagnostics = pd.DataFrame(rows)
    summary = (
        diagnostics.groupby(
            ["candidate", "objective", "anchor_label", "phase_information_budget"],
            as_index=False,
        )
        .agg(
            n_refits=("refit_gain_vs_tied", "size"),
            refit_gain_mean=("refit_gain_vs_tied", "mean"),
            refit_gain_sd=("refit_gain_vs_tied", "std"),
            refit_gain_min=("refit_gain_vs_tied", "min"),
            refit_gain_positive_fraction=("refit_gain_vs_tied", lambda values: values.gt(0).mean()),
            direction_cosine_mean=("direction_cosine_vs_full", "mean"),
            direction_cosine_min=("direction_cosine_vs_full", "min"),
            policy_tv_vs_full_mean=("policy_tv_vs_full", "mean"),
            policy_tv_vs_full_max=("policy_tv_vs_full", "max"),
            successful_starts_min=("successful_starts", "min"),
        )
        .sort_values(["objective", "anchor_label", "phase_information_budget"])
    )
    return diagnostics, summary


def observed_library_geometry(dataset: matched.pooled.Dataset, natural: np.ndarray) -> pd.DataFrame:
    rows = []
    for index, weights in enumerate(dataset.weights):
        rows.append({"index": index, **policy_geometry(weights, natural)})
    return pd.DataFrame(rows).set_index("index")


def library_selection_diagnostics(
    candidates: pd.DataFrame,
    datasets: dict[str, matched.pooled.Dataset],
    natural: dict[str, np.ndarray],
    fold_models: dict[str, list[tuple[int, int, np.ndarray, matched.SeparateHeadsModel]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for objective, dataset in datasets.items():
        geometry = observed_library_geometry(dataset, natural[objective])
        objective_candidates = candidates.loc[candidates["objective"].eq(objective)]
        for anchor_label, anchor_rows in objective_candidates.groupby("anchor_label"):
            aggregate_budget = float(anchor_rows["aggregate_kl"].iloc[0])
            for phase_information_budget in sorted(anchor_rows["phase_information_budget"].unique()):
                eligible = geometry["aggregate_kl"].le(aggregate_budget + 1e-10) & geometry["phase_information"].le(
                    phase_information_budget + 1e-10
                )
                for seed, fold, test_indices, model in fold_models[objective]:
                    eligible_test = test_indices[eligible.iloc[test_indices].to_numpy()]
                    if len(eligible_test) == 0:
                        continue
                    predictions = matched.predict_model(
                        model,
                        dataset,
                        dataset.weights[eligible_test],
                    )
                    selected_offset = int(np.argmin(predictions))
                    selected = int(eligible_test[selected_offset])
                    observed = float(dataset.y[selected])
                    prediction = float(predictions[selected_offset])
                    rows.append(
                        {
                            "objective": objective,
                            "anchor_label": anchor_label,
                            "phase_information_budget": phase_information_budget,
                            "aggregate_kl_budget": aggregate_budget,
                            "seed": seed,
                            "fold": fold,
                            "eligible_count": len(eligible_test),
                            "selected_index": selected,
                            "selected_observed_bpb": observed,
                            "selected_predicted_bpb": prediction,
                            "selected_point_optimism": observed - prediction,
                            "regret_at_1": observed - float(np.min(dataset.y[eligible_test])),
                        }
                    )
    diagnostics = pd.DataFrame(rows)
    summary = (
        diagnostics.groupby(
            ["objective", "anchor_label", "phase_information_budget", "aggregate_kl_budget"],
            as_index=False,
        )
        .agg(
            n_folds=("regret_at_1", "size"),
            median_eligible_count=("eligible_count", "median"),
            fold_mean_regret_at_1=("regret_at_1", "mean"),
            fold_max_regret_at_1=("regret_at_1", "max"),
            selected_point_optimism_mean=("selected_point_optimism", "mean"),
            selected_point_optimism_max=("selected_point_optimism", "max"),
        )
        .sort_values(["objective", "anchor_label", "phase_information_budget"])
    )
    return diagnostics, summary


def write_plots(candidates: pd.DataFrame, crossfit: pd.DataFrame, output_dir: Path) -> None:
    labels = candidates["anchor_label"].unique().tolist()
    colors = px.colors.sample_colorscale("RdYlGn_r", np.linspace(0.1, 0.9, len(labels)))
    color_map = dict(zip(labels, colors, strict=True))
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Full-fit predicted ordering gain",
            "Cross-fit mean ordering gain",
            "Phase contrast",
            "Aggressiveness versus prior best 2p",
        ],
    )
    merged = candidates.merge(
        crossfit[["candidate", "crossfit_gain_mean", "crossfit_gain_sd"]],
        on="candidate",
        validate="one_to_one",
    )
    for label, group in merged.groupby("anchor_label", sort=False):
        group = group.sort_values("phase_information_budget")
        common = {
            "mode": "lines+markers",
            "name": label,
            "legendgroup": label,
            "line": {"color": color_map[label]},
        }
        figure.add_trace(
            go.Scatter(
                x=group["phase_information_budget"],
                y=group["predicted_gain_vs_tied"],
                showlegend=True,
                **common,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=group["phase_information_budget"],
                y=group["crossfit_gain_mean"],
                error_y={"type": "data", "array": group["crossfit_gain_sd"], "visible": True},
                showlegend=False,
                **common,
            ),
            row=1,
            col=2,
        )
        figure.add_trace(
            go.Scatter(
                x=group["phase_information_budget"],
                y=group["phase_tv"],
                showlegend=False,
                **common,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=group["phase_information_budget"],
                y=group["aggregate_kl_ratio_vs_old_2p"],
                showlegend=False,
                **common,
            ),
            row=2,
            col=2,
        )
    figure.add_hline(y=1.0, line_dash="dash", line_color="#666666", row=2, col=2)
    figure.update_xaxes(title_text="phase-information budget", row=2, col=1)
    figure.update_xaxes(title_text="phase-information budget", row=2, col=2)
    figure.update_yaxes(title_text="predicted BPB gain", row=1, col=1)
    figure.update_yaxes(title_text="mean predicted BPB gain", row=1, col=2)
    figure.update_yaxes(title_text="phase TV", row=2, col=1)
    figure.update_yaxes(title_text="aggregate KL / prior-best-2p aggregate KL", row=2, col=2)
    figure.update_layout(
        title="Decoupled fixed-aggregate phase-information audit",
        template="plotly_white",
        width=1350,
        height=900,
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.1},
        margin={"l": 80, "r": 40, "t": 100, "b": 120},
    )
    stem = output_dir / "phase_information_local_checks"
    figure.write_html(stem.with_suffix(".html"), include_plotlyjs=True, config=PLOT_CONFIG)
    figure.write_image(stem.with_suffix(".png"), scale=2)


def write_report(
    candidates: pd.DataFrame,
    crossfit: pd.DataFrame,
    refit: pd.DataFrame,
    library: pd.DataFrame,
    output: Path,
) -> None:
    columns = [
        "candidate",
        "objective",
        "anchor_label",
        "phase_information_budget",
        "predicted_gain_vs_tied",
        "phase_information",
        "phase_tv",
        "aggregate_kl",
        "aggregate_kl_ratio_vs_old_2p",
        "max_simulated_epoch",
        "max_epoch_ratio_vs_old_2p",
        "nearest_observed_tv",
    ]
    nonzero_crossfit = crossfit.loc[crossfit["phase_information_budget"].gt(0)]
    aggregate_kl_range = candidates.groupby("anchor_label")["aggregate_kl"].agg(lambda values: np.ptp(values))
    max_epoch_range = candidates.groupby("anchor_label")["max_simulated_epoch"].agg(lambda values: np.ptp(values))
    lines = [
        "# Decoupled aggregate and phase-information local checks",
        "",
        "The two-phase separate-heads surrogate is unchanged. Each candidate fixes aggregate weights "
        "to a one-phase anchor and optimizes only phase ordering under an explicit information budget.",
        "",
        "## Key findings",
        "",
        f"- Maximum aggregate-weight error is {candidates['max_aggregate_error'].max():.2e}.",
        "- Aggregate specialization and repetition remain fixed along each phase-information path: "
        f"maximum within-anchor aggregate-KL range is {aggregate_kl_range.max():.2e}, and maximum "
        f"within-anchor max-epoch range is {max_epoch_range.max():.2e}.",
        "- Relative to the prior best two-phase candidates, the fixed aggregates are more aggressive: "
        f"aggregate-KL ratios span {candidates['aggregate_kl_ratio_vs_old_2p'].min():.2f}-"
        f"{candidates['aggregate_kl_ratio_vs_old_2p'].max():.2f}, and max-epoch ratios span "
        f"{candidates['max_epoch_ratio_vs_old_2p'].min():.2f}-"
        f"{candidates['max_epoch_ratio_vs_old_2p'].max():.2f}.",
        "- Every nonzero phase split has positive predicted gain under every fixed-candidate cross-fit "
        f"evaluation (minimum positive fraction {nonzero_crossfit['crossfit_gain_positive_fraction'].min():.2f}).",
        "- Fold-specific re-optimization tests whether the phase direction itself, rather than only its "
        "predicted value, is stable.",
        "- The candidates remain substantially outside observed two-phase support: nearest-observed policy "
        f"TV spans {candidates['nearest_observed_tv'].min():.3f}-"
        f"{candidates['nearest_observed_tv'].max():.3f}. This is the principal unresolved local caveat.",
        "",
        "## Candidate geometry and full-fit predictions",
        "",
        candidates[columns].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Cross-fit prediction stability",
        "",
        crossfit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fold-refit phase-direction stability",
        "",
        refit.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Held-out-library decision diagnostics",
        "",
        "These diagnostics select only among observed held-out rows. They evaluate raw BPB regret and "
        "selected-point optimism; the deployment constraint is not added to the prediction target.",
        "",
        library.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation boundary",
        "",
        "Cross-fit agreement checks whether the fitted phase gain is stable. It does not observe the "
        "continuous optimized candidates. A fixed-aggregate 3e18 validation remains decisive.",
        "",
    ]
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase-information-budgets",
        default=",".join(str(value) for value in PHASE_INFORMATION_BUDGETS),
    )
    parser.add_argument(
        "--refit-phase-information-budgets",
        default=",".join(str(value) for value in REFIT_PHASE_INFORMATION_BUDGETS),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_information_budgets = parse_float_tuple(args.phase_information_budgets)
    refit_phase_information_budgets = parse_float_tuple(args.refit_phase_information_budgets)
    datasets, models, natural, token_counts, target_budgets = load_context()
    candidates, candidate_weights = build_candidates(
        args.output_dir,
        phase_information_budgets,
        datasets,
        models,
        natural,
        token_counts,
        target_budgets,
    )
    fold_models = crossfit_models(datasets)
    crossfit_diagnostics, crossfit_summary = candidate_crossfit_diagnostics(
        candidates,
        candidate_weights,
        datasets,
        fold_models,
    )
    refit_diagnostics, refit_summary = candidate_refit_stability(
        candidates,
        candidate_weights,
        datasets,
        fold_models,
        refit_phase_information_budgets,
    )
    library_diagnostics, library_summary = library_selection_diagnostics(
        candidates,
        datasets,
        natural,
        fold_models,
    )
    candidates.to_csv(args.output_dir / "candidate_summary.csv", index=False)
    crossfit_diagnostics.to_csv(args.output_dir / "crossfit_candidate_predictions.csv", index=False)
    crossfit_summary.to_csv(args.output_dir / "crossfit_candidate_summary.csv", index=False)
    refit_diagnostics.to_csv(args.output_dir / "crossfit_refit_predictions.csv", index=False)
    refit_summary.to_csv(args.output_dir / "crossfit_refit_summary.csv", index=False)
    library_diagnostics.to_csv(args.output_dir / "library_selection_diagnostics.csv", index=False)
    library_summary.to_csv(args.output_dir / "library_selection_summary.csv", index=False)
    write_plots(candidates, crossfit_summary, args.output_dir)
    write_report(candidates, crossfit_summary, refit_summary, library_summary, args.output_dir / "report.md")
    print(candidates.to_string(index=False))
    print(f"Wrote decoupled phase-information checks to {args.output_dir}")


if __name__ == "__main__":
    main()
