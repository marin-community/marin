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
"""Test a global one-phase spine plus a local phase-tangent response model."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px

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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/local_phase_tangent_surrogate_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
PAIR_EFFECTS_PATH = (
    SCRIPT_DIR / "reference_outputs/delphi_3e18_frontier_phase_fiber_results_20260719/paired_phase_effects.csv"
)
TAU_GRID = (0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
RESIDUAL_SHRINK_GRID = (1.0, 3.0, 10.0, 30.0, 100.0)
OUTER_FOLDS = 4
INNER_FOLDS = 3
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Coordinate(StrEnum):
    RAW = "raw"
    LEARNABILITY = "learnability"


@dataclass(frozen=True)
class Candidate:
    name: str
    coordinate: Coordinate
    hierarchical_gradient: bool
    family_curvature: bool


CANDIDATES = (
    Candidate("raw_bucket_gradient", Coordinate.RAW, False, False),
    Candidate("learnability_bucket_gradient", Coordinate.LEARNABILITY, True, False),
    Candidate("learnability_gradient_family_curvature", Coordinate.LEARNABILITY, True, True),
)


@dataclass(frozen=True)
class LocalPanel:
    anchor: str
    target: str
    frame: pd.DataFrame
    odd_design_by_tau: dict[float, np.ndarray]
    even_design_by_tau: dict[float, np.ndarray]
    odd_target: np.ndarray
    even_target: np.ndarray
    plus_target: np.ndarray
    minus_target: np.ndarray
    blocks: np.ndarray
    families: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Fit:
    tau: float
    ridge: float
    residual_shrink: float
    gradient: np.ndarray
    curvature: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def relative_phase_coordinate(
    weights: np.ndarray,
    dataset: object,
    coordinate: Coordinate,
    tau: float,
) -> np.ndarray:
    c0 = np.asarray(dataset.c0, dtype=float)
    c1 = np.asarray(dataset.c1, dtype=float)
    alpha = float(np.median(c0 / np.maximum(c0 + c1, 1e-12)))
    natural = hierarchical.proportional_weights(dataset)
    aggregate = alpha * weights[:, 0, :] + (1.0 - alpha) * weights[:, 1, :]
    displacement = alpha * (1.0 - alpha) * (weights[:, 1, :] - weights[:, 0, :]) / natural[None, :]
    if coordinate is Coordinate.RAW:
        return displacement
    relative_exposure = aggregate / natural[None, :]
    return displacement / (tau + relative_exposure)


def family_curvature_design(coordinate: np.ndarray, dataset: object) -> np.ndarray:
    natural = hierarchical.proportional_weights(dataset)
    members = dataset.family_members
    return 0.5 * np.column_stack(
        [np.sum(natural[family][None, :] * coordinate[:, family] ** 2, axis=1) for family in members]
    )


def local_panels() -> list[LocalPanel]:
    matched = matched_pair.matched_sources()
    dataset = composition.custom_dataset(
        matched.sources.reference,
        matched.sources.fiber.frame,
        matched.sources.fiber.weights,
        "uncheatable",
        "local_phase_tangent_coordinates",
    )
    effects = pd.read_csv(PAIR_EFFECTS_PATH)
    candidate_to_index = {
        str(candidate): index for index, candidate in enumerate(matched.sources.fiber.frame["candidate_id"].astype(str))
    }
    panels = []
    for (anchor, target), frame in effects.groupby(["anchor_id", "target"], sort=True):
        frame = frame.copy().reset_index(drop=True)
        plus_indices = np.asarray([candidate_to_index[value] for value in frame["plus_candidate_id"].astype(str)])
        minus_indices = np.asarray([candidate_to_index[value] for value in frame["minus_candidate_id"].astype(str)])
        odd_design_by_tau = {}
        even_design_by_tau = {}
        for tau in TAU_GRID:
            plus = relative_phase_coordinate(
                matched.sources.fiber.weights[plus_indices], dataset, Coordinate.LEARNABILITY, tau
            )
            minus = relative_phase_coordinate(
                matched.sources.fiber.weights[minus_indices], dataset, Coordinate.LEARNABILITY, tau
            )
            coordinate = 0.5 * (plus - minus)
            odd_design_by_tau[tau] = coordinate
            even_design_by_tau[tau] = family_curvature_design(coordinate, dataset)
        plus_raw = relative_phase_coordinate(matched.sources.fiber.weights[plus_indices], dataset, Coordinate.RAW, 1.0)
        minus_raw = relative_phase_coordinate(matched.sources.fiber.weights[minus_indices], dataset, Coordinate.RAW, 1.0)
        odd_design_by_tau[math.inf] = 0.5 * (plus_raw - minus_raw)
        even_design_by_tau[math.inf] = family_curvature_design(odd_design_by_tau[math.inf], dataset)
        panels.append(
            LocalPanel(
                anchor=str(anchor),
                target=str(target),
                frame=frame,
                odd_design_by_tau=odd_design_by_tau,
                even_design_by_tau=even_design_by_tau,
                odd_target=frame["odd_effect_plus_minus_over_2"].to_numpy(dtype=float),
                even_target=frame["mean_contrast_minus_center"].to_numpy(dtype=float),
                plus_target=frame["plus_delta_vs_center"].to_numpy(dtype=float),
                minus_target=frame["minus_delta_vs_center"].to_numpy(dtype=float),
                blocks=frame["seed_block"].to_numpy(dtype=int),
                families=tuple(np.asarray(value, dtype=int) for value in dataset.family_members),
            )
        )
    return panels


def family_residual_matrix(families: tuple[np.ndarray, ...], width: int) -> np.ndarray:
    result = np.zeros((width, width), dtype=float)
    for members in families:
        local = np.eye(len(members)) - np.full((len(members), len(members)), 1.0 / len(members))
        result[np.ix_(members, members)] = local
    return result


def fit_model(
    panel: LocalPanel,
    candidate: Candidate,
    indices: np.ndarray,
    tau: float,
    ridge: float,
    residual_shrink: float,
) -> Fit:
    odd_design = panel.odd_design_by_tau[tau][indices]
    odd_target = panel.odd_target[indices]
    gradient = cp.Variable(odd_design.shape[1])
    penalty = cp.sum_squares(gradient)
    if candidate.hierarchical_gradient:
        residual = family_residual_matrix(panel.families, odd_design.shape[1])
        penalty += residual_shrink * cp.sum_squares(residual @ gradient)
    odd_problem = cp.Problem(
        cp.Minimize(cp.sum_squares(odd_design @ gradient - odd_target) / len(indices) + ridge * penalty)
    )
    odd_problem.solve(solver=cp.CLARABEL)
    if odd_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or gradient.value is None:
        raise RuntimeError(f"Odd local solve failed: {odd_problem.status}")

    curvature_value = np.zeros(len(panel.families), dtype=float)
    if candidate.family_curvature:
        even_design = panel.even_design_by_tau[tau][indices]
        curvature = cp.Variable(even_design.shape[1], nonneg=True)
        even_problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(even_design @ curvature - panel.even_target[indices]) / len(indices)
                + ridge * cp.sum_squares(curvature)
            )
        )
        even_problem.solve(solver=cp.CLARABEL)
        if even_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or curvature.value is None:
            raise RuntimeError(f"Even local solve failed: {even_problem.status}")
        curvature_value = np.asarray(curvature.value, dtype=float)
    return Fit(
        tau=tau,
        ridge=ridge,
        residual_shrink=residual_shrink,
        gradient=np.asarray(gradient.value, dtype=float),
        curvature=curvature_value,
    )


def predict(panel: LocalPanel, fit: Fit, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    odd = panel.odd_design_by_tau[fit.tau][indices] @ fit.gradient
    even = panel.even_design_by_tau[fit.tau][indices] @ fit.curvature
    return odd, even, odd + even, -odd + even


def relative_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    rmse = float(np.sqrt(np.mean((predicted - observed) ** 2)))
    baseline = float(np.sqrt(np.mean(observed**2)))
    return rmse / max(baseline, 1e-12)


def configurations(candidate: Candidate) -> list[tuple[float, float, float]]:
    taus = (math.inf,) if candidate.coordinate is Coordinate.RAW else TAU_GRID
    shrink = RESIDUAL_SHRINK_GRID if candidate.hierarchical_gradient else (0.0,)
    return list(itertools.product(taus, RIDGE_GRID, shrink))


def select_config(
    panel: LocalPanel,
    candidate: Candidate,
    indices: np.ndarray,
) -> tuple[float, float, float, float]:
    blocks = np.unique(panel.blocks[indices])
    if len(blocks) < INNER_FOLDS:
        raise ValueError("Too few center blocks for inner CV")
    records = []
    for tau, ridge, shrink in configurations(candidate):
        odd_prediction = np.full(len(indices), np.nan, dtype=float)
        plus_prediction = np.full(len(indices), np.nan, dtype=float)
        minus_prediction = np.full(len(indices), np.nan, dtype=float)
        for fold, heldout_block in enumerate(blocks[:INNER_FOLDS]):
            del fold
            train = indices[panel.blocks[indices] != heldout_block]
            local_test = np.flatnonzero(panel.blocks[indices] == heldout_block)
            test = indices[local_test]
            fitted = fit_model(panel, candidate, train, tau, ridge, shrink)
            odd, _even, plus, minus = predict(panel, fitted, test)
            odd_prediction[local_test] = odd
            plus_prediction[local_test] = plus
            minus_prediction[local_test] = minus
        # A fourth block may remain; rotate it into validation by using the first
        # block as training-only only when exactly four blocks are available.
        missing = ~np.isfinite(odd_prediction)
        for local_index in np.flatnonzero(missing):
            test = indices[[local_index]]
            train = indices[panel.blocks[indices] != panel.blocks[test[0]]]
            fitted = fit_model(panel, candidate, train, tau, ridge, shrink)
            odd, _even, plus, minus = predict(panel, fitted, test)
            odd_prediction[local_index] = odd[0]
            plus_prediction[local_index] = plus[0]
            minus_prediction[local_index] = minus[0]
        full_observed = np.concatenate([panel.plus_target[indices], panel.minus_target[indices]])
        full_prediction = np.concatenate([plus_prediction, minus_prediction])
        score = max(
            relative_rmse(panel.odd_target[indices], odd_prediction),
            relative_rmse(full_observed, full_prediction),
        )
        records.append((score, tau, ridge, shrink))
    best = min(row[0] for row in records)
    eligible = [row for row in records if row[0] <= 1.01 * best]
    score, tau, ridge, shrink = sorted(
        eligible,
        key=lambda row: (-row[2], -row[3], -row[1], row[0]),
    )[0]
    return tau, ridge, shrink, score


def coefficient_stability(gradients: list[np.ndarray]) -> dict[str, float]:
    matrix = np.vstack(gradients)
    cosines = []
    for left, right in itertools.combinations(range(len(matrix)), 2):
        denominator = float(np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right]))
        if denominator > 1e-12:
            cosines.append(float(matrix[left] @ matrix[right] / denominator))
    signs = np.sign(np.where(np.abs(matrix) < 1e-8, 0.0, matrix))
    agreements = [max(float(np.mean(column == value)) for value in (-1.0, 0.0, 1.0)) for column in signs.T]
    return {
        "median_gradient_cosine": float(np.median(cosines)),
        "median_sign_agreement": float(np.median(agreements)),
    }


def evaluate_candidate(panel: LocalPanel, candidate: Candidate) -> tuple[dict[str, float | str], pd.DataFrame]:
    train = np.flatnonzero(panel.frame["contrast_family"].eq("domain_vs_rest").to_numpy())
    test = np.flatnonzero(panel.frame["contrast_family"].eq("high_mass_pair").to_numpy())
    gradients = []
    selected = []
    for heldout_block in np.unique(panel.blocks[train]):
        outer_train = train[panel.blocks[train] != heldout_block]
        tau, ridge, shrink, inner_score = select_config(panel, candidate, outer_train)
        fitted = fit_model(panel, candidate, outer_train, tau, ridge, shrink)
        gradients.append(fitted.gradient)
        selected.append((tau, ridge, shrink, inner_score))

    tau, ridge, shrink, inner_score = select_config(panel, candidate, train)
    fitted = fit_model(panel, candidate, train, tau, ridge, shrink)
    odd, even, plus, minus = predict(panel, fitted, test)
    odd_metrics = composition.prediction_metrics(panel.odd_target[test], odd)
    full_observed = np.concatenate([panel.plus_target[test], panel.minus_target[test]])
    full_prediction = np.concatenate([plus, minus])
    full_metrics = composition.prediction_metrics(full_observed, full_prediction)
    stability = coefficient_stability(gradients)
    record: dict[str, float | str] = {
        "anchor": panel.anchor,
        "target": panel.target,
        "candidate": candidate.name,
        "tau": tau,
        "ridge": ridge,
        "residual_shrink": shrink,
        "inner_score": inner_score,
        "odd_zero_rmse": float(np.sqrt(np.mean(panel.odd_target[test] ** 2))),
        "full_zero_rmse": float(np.sqrt(np.mean(full_observed**2))),
        **{f"odd_{key}": value for key, value in odd_metrics.items()},
        **{f"full_{key}": value for key, value in full_metrics.items()},
        **stability,
        "outer_tau_iqr": float(np.subtract(*np.percentile([row[0] for row in selected], [75, 25]))),
        "outer_log10_ridge_iqr": float(np.subtract(*np.percentile(np.log10([row[1] for row in selected]), [75, 25]))),
    }
    predictions = []
    for local, index in enumerate(test):
        base = {
            "anchor": panel.anchor,
            "target": panel.target,
            "candidate": candidate.name,
            "direction_id": panel.frame.iloc[index]["direction_id"],
        }
        predictions.extend(
            [
                {**base, "response": "odd", "observed": panel.odd_target[index], "predicted": odd[local]},
                {**base, "response": "plus", "observed": panel.plus_target[index], "predicted": plus[local]},
                {**base, "response": "minus", "observed": panel.minus_target[index], "predicted": minus[local]},
                {**base, "response": "even", "observed": panel.even_target[index], "predicted": even[local]},
            ]
        )
    return record, pd.DataFrame(predictions)


def acceptance_gate(metrics: pd.DataFrame) -> dict[str, bool]:
    primary = {
        ("uncheatable_frontier", "uncheatable"),
        ("table9_frontier", "table9"),
    }
    result = {}
    for candidate in CANDIDATES:
        rows = metrics.loc[metrics["candidate"].eq(candidate.name)]
        passed = True
        for _, row in rows.iterrows():
            is_primary = (str(row["anchor"]), str(row["target"])) in primary
            odd_ratio = float(row["odd_rmse"] / row["odd_zero_rmse"])
            full_ratio = float(row["full_rmse"] / row["full_zero_rmse"])
            if is_primary:
                passed &= odd_ratio <= 0.9
                passed &= full_ratio <= 1.0
            else:
                passed &= odd_ratio <= 1.1
                passed &= full_ratio <= 1.1
            passed &= float(row["median_gradient_cosine"]) >= 0.5
            passed &= float(row["median_sign_agreement"]) >= 0.6
        result[candidate.name] = bool(passed)
    return result


def render_predictions(predictions: pd.DataFrame, output_dir: Path) -> None:
    local = predictions.loc[predictions["response"].isin(["plus", "minus"])]
    figure = px.scatter(
        local,
        x="observed",
        y="predicted",
        color="candidate",
        symbol="response",
        facet_row="target",
        facet_col="anchor",
        hover_data=["direction_id"],
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Frozen high-mass-pair phase-fiber predictions",
    )
    bound = float(max(local["observed"].abs().max(), local["predicted"].abs().max()))
    figure.add_shape(type="line", x0=-bound, y0=-bound, x1=bound, y1=bound, line={"dash": "dash"})
    figure.update_layout(template="plotly_white", height=850, legend={"orientation": "h", "y": -0.1})
    figure.write_html(output_dir / "high_mass_pair_predictions.html", include_plotlyjs=True, config=PLOT_CONFIG)


def write_report(metrics: pd.DataFrame, gate: dict[str, bool], output_dir: Path) -> None:
    table = metrics.copy()
    table["odd_rmse_ratio"] = table["odd_rmse"] / table["odd_zero_rmse"]
    table["full_rmse_ratio"] = table["full_rmse"] / table["full_zero_rmse"]
    table["gate_pass"] = table["candidate"].map(gate)
    survivors = [name for name, passed in gate.items() if passed]
    lines = [
        "# Local phase-tangent surrogate",
        "",
        "## Model",
        "",
        (
            "The global aggregate surface is learned from phase-tied policies. Around a frozen frontier aggregate, "
            "symmetric same-seed phase probes identify a local odd recency gradient and an even nonnegative family "
            "fatigue cost:"
        ),
        "",
        (
            "$$Y(a_\\star,\\pm d)-F(a_\\star)=\\pm g^Tx_\\tau(a_\\star,d)"
            "+\\tfrac12\\sum_f h_f\\lVert x_{\\tau,f}\\rVert_p^2,\\qquad h_f\\ge0.$ $"
        ).replace("$ $", "$$"),
        "",
        (
            "Hyperparameters and coefficients use only the 39 domain-vs-rest directions. The nine high-mass-pair "
            "directions are a frozen compositional generalization test."
        ),
        "",
        "## Frozen high-mass-pair test",
        "",
        table[
            [
                "anchor",
                "target",
                "candidate",
                "tau",
                "ridge",
                "residual_shrink",
                "odd_rmse_ratio",
                "full_rmse_ratio",
                "odd_spearman",
                "median_gradient_cosine",
                "median_sign_agreement",
                "gate_pass",
            ]
        ].to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Verdict",
        "",
        (
            "Provisional local-method survivors: " + ", ".join(survivors) + ". Optimization audit is now permitted."
            if survivors
            else "No local tangent candidate passes the frozen compositional-generalization gate."
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration: {PREREGISTRATION_PATH}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text())
    if not preregistration.get("frozen_before_high_mass_pair_evaluation"):
        raise ValueError("Preregistration is not frozen")

    metric_rows = []
    prediction_frames = []
    for panel in local_panels():
        for candidate in CANDIDATES:
            print(f"Fitting {panel.anchor}/{panel.target}/{candidate.name}", flush=True)
            metrics, predictions = evaluate_candidate(panel, candidate)
            metric_rows.append(metrics)
            prediction_frames.append(predictions)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    gate = acceptance_gate(metrics)
    metrics.to_csv(output_dir / "high_mass_pair_metrics.csv", index=False)
    predictions.to_csv(output_dir / "high_mass_pair_predictions.csv", index=False)
    render_predictions(predictions, output_dir)
    write_report(metrics, gate, output_dir)
    (output_dir / "manifest.json").write_text(
        json.dumps({"preregistration": str(PREREGISTRATION_PATH), "gate": gate}, indent=2, sort_keys=True) + "\n"
    )
    print((output_dir / "report.md").resolve())


if __name__ == "__main__":
    main()
