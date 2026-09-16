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
"""Benchmark an identifiable one-phase spine plus plasticity-potential transport."""

from __future__ import annotations

import argparse
import itertools
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
from plotly.subplots import make_subplots
from scipy.optimize import lsq_linear
from sklearn.model_selection import KFold

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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/plasticity_potential_transport_20260720"
PREREGISTRATION_PATH = DEFAULT_OUTPUT_DIR / "preregistered_candidates.json"
TARGETS = composition.TARGETS
TARGET_COLUMNS = composition.TARGET_COLUMNS
TAU_GRID = (0.1, 0.3, 1.0, 3.0)
RIDGE_GRID = (1e-4, 1e-3, 1e-2, 0.1, 1.0)
OUTER_FOLDS = 5
INNER_FOLDS = 4
STAGE1_RMSE_IMPROVEMENT = 0.01
STAGE1_COSINE = 0.5
STAGE1_SIGN_AGREEMENT = 0.6
FIBER_PRESERVATION_RATIO = 1.01
REGRET_TOLERANCE = 0.002
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Candidate:
    name: str
    interaction: str
    even_cost: bool = False


CANDIDATES = (
    Candidate("zero_phase", "zero"),
    Candidate("constant_transport", "constant"),
    Candidate("diagonal_potential", "diagonal"),
    Candidate("symmetric_potential", "symmetric"),
    Candidate("symmetric_potential_even_cost", "symmetric", even_cost=True),
)


@dataclass(frozen=True)
class PhaseFit:
    candidate: Candidate
    tau: float
    ridge: float
    coefficients: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class TargetModels:
    aggregate_spine: hierarchical.Model
    phase_fit: PhaseFit
    phase_dataset: family_grp.Dataset

    def predict_phase(self, weights: np.ndarray) -> np.ndarray:
        design, names = phase_design(weights, self.phase_dataset, self.phase_fit.candidate, self.phase_fit.tau)
        if names != self.phase_fit.feature_names:
            raise ValueError("Phase feature order changed between fit and prediction")
        return design @ self.phase_fit.coefficients

    def predict(self, weights: np.ndarray) -> np.ndarray:
        tied = tied_policy(weights, self.phase_dataset)
        return self.aggregate_spine.predict(tied) + self.predict_phase(weights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stage1-only", action="store_true")
    return parser.parse_args()


def tied_policy(weights: np.ndarray, dataset: family_grp.Dataset) -> np.ndarray:
    alpha = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    aggregate = alpha * weights[:, 0, :] + (1.0 - alpha) * weights[:, 1, :]
    return np.stack([aggregate, aggregate], axis=1)


def aggregate_spine_dataset(matched: matched_pair.MatchedSources, target: str) -> family_grp.Dataset:
    pair_single = matched.pair_frame["single_index"].to_numpy(dtype=int)
    tied_broad = matched.tied_broad_indices
    frame = pd.concat(
        [
            matched.sources.single.frame.iloc[pair_single],
            matched.sources.broad.frame.iloc[tied_broad],
        ],
        ignore_index=True,
        sort=False,
    )
    weights = np.concatenate(
        [
            matched.sources.single.weights[pair_single],
            matched.sources.broad.weights[tied_broad],
        ],
        axis=0,
    )
    if len(frame) != 280:
        raise ValueError(f"Expected 280 aggregate-spine observations, found {len(frame)}")
    return composition.custom_dataset(matched.sources.reference, frame, weights, target, f"ppt_spine_{target}")


def phase_pair_dataset(
    matched: matched_pair.MatchedSources,
    target: str,
) -> tuple[family_grp.Dataset, np.ndarray, pd.DataFrame]:
    broad_indices = matched.pair_frame["broad_index"].to_numpy(dtype=int)
    single_indices = matched.pair_frame["single_index"].to_numpy(dtype=int)
    frame = matched.sources.broad.frame.iloc[broad_indices].copy().reset_index(drop=True)
    weights = matched.sources.broad.weights[broad_indices]
    dataset = composition.custom_dataset(matched.sources.reference, frame, weights, target, f"ppt_pairs_{target}")
    observed = matched.sources.broad.frame.iloc[broad_indices][TARGET_COLUMNS[target]].to_numpy(
        dtype=float
    ) - matched.sources.single.frame.iloc[single_indices][TARGET_COLUMNS[target]].to_numpy(dtype=float)
    frame["pair_id"] = matched.pair_frame["pair_id"].to_numpy(dtype=str)
    frame["observed_delta"] = observed
    return dataset, observed, frame


def phase_coordinates(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    tau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = float(np.median(dataset.c0 / np.maximum(dataset.c0 + dataset.c1, 1e-12)))
    natural = hierarchical.proportional_weights(dataset)
    aggregate = alpha * weights[:, 0, :] + (1.0 - alpha) * weights[:, 1, :]
    relative_exposure = aggregate / np.maximum(natural[None, :], 1e-12)
    displacement = (
        alpha
        * (1.0 - alpha)
        * (weights[:, 1, :] - weights[:, 0, :])
        / np.maximum(
            natural[None, :],
            1e-12,
        )
    )

    family_state = []
    family_transport = []
    for members in dataset.family_members:
        conditional = natural[members] / natural[members].sum()
        state = np.sum(
            conditional[None, :] * (np.log(tau + relative_exposure[:, members]) - math.log(tau + 1.0)),
            axis=1,
        )
        transport = np.sum(
            conditional[None, :] * displacement[:, members] / (tau + relative_exposure[:, members]),
            axis=1,
        )
        family_state.append(state)
        family_transport.append(transport)
    state_array = np.column_stack(family_state)
    transport_array = np.column_stack(family_transport)
    even_cost = np.sum(
        natural[None, :] * displacement**2 / (tau + relative_exposure),
        axis=1,
    )
    return state_array, transport_array, even_cost


def phase_design(
    weights: np.ndarray,
    dataset: family_grp.Dataset,
    candidate: Candidate,
    tau: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if candidate.interaction == "zero":
        return np.zeros((len(weights), 0), dtype=float), ()

    state, transport, even_cost = phase_coordinates(weights, dataset, tau)
    pieces = [transport]
    names = [f"marginal_phase_value:{name}" for name in dataset.family_names]
    if candidate.interaction == "diagonal":
        pieces.append(transport * state)
        names.extend(f"plasticity:{name}:{name}" for name in dataset.family_names)
    elif candidate.interaction == "symmetric":
        interactions = []
        for left, right in itertools.combinations_with_replacement(range(len(dataset.family_names)), 2):
            if left == right:
                feature = transport[:, left] * state[:, left]
            else:
                feature = transport[:, left] * state[:, right] + transport[:, right] * state[:, left]
            interactions.append(feature)
            names.append(f"plasticity:{dataset.family_names[left]}:{dataset.family_names[right]}")
        pieces.append(np.column_stack(interactions))
    elif candidate.interaction != "constant":
        raise ValueError(f"Unknown interaction {candidate.interaction}")

    if candidate.even_cost:
        pieces.append(even_cost[:, None])
        names.append("finite_phase_variation_cost")
    return np.hstack(pieces), tuple(names)


def fit_linear_phase(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    ridge: float,
    constrain_last_nonnegative: bool,
) -> np.ndarray:
    if design.shape[1] == 0:
        return np.zeros(0, dtype=float)
    train_design = design[train]
    scale = np.maximum(np.sqrt(np.mean(train_design**2, axis=0)), 1e-10)
    scaled = train_design / scale[None, :]
    augmented_design = np.vstack([scaled, math.sqrt(ridge) * np.eye(design.shape[1])])
    augmented_target = np.concatenate([target[train], np.zeros(design.shape[1], dtype=float)])
    lower = np.full(design.shape[1], -np.inf, dtype=float)
    upper = np.full(design.shape[1], np.inf, dtype=float)
    if constrain_last_nonnegative:
        lower[-1] = 0.0
    result = lsq_linear(augmented_design, augmented_target, bounds=(lower, upper), lsmr_tol="auto")
    if not result.success:
        raise RuntimeError(f"Phase solve failed: {result.message}")
    return result.x / scale


def folds(indices: np.ndarray, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = KFold(n_splits=count, shuffle=True, random_state=seed)
    result = []
    for local_train, local_test in splitter.split(indices):
        result.append((indices[local_train], indices[local_test]))
    return result


def candidate_grid(candidate: Candidate) -> tuple[tuple[float, float], ...]:
    if candidate.interaction == "zero":
        return ((1.0, 0.0),)
    return tuple(itertools.product(TAU_GRID, RIDGE_GRID))


def select_hyperparameters(
    dataset: family_grp.Dataset,
    observed: np.ndarray,
    candidate: Candidate,
    train: np.ndarray,
    split_count: int,
    seed: int,
) -> tuple[float, float, float]:
    if candidate.interaction == "zero":
        return 1.0, 0.0, float(np.sqrt(np.mean(observed[train] ** 2)))
    records = []
    local_folds = folds(train, split_count, seed)
    for tau, ridge in candidate_grid(candidate):
        design, _names = phase_design(dataset.weights, dataset, candidate, tau)
        prediction = np.full(len(observed), np.nan, dtype=float)
        for inner_train, inner_test in local_folds:
            coefficient = fit_linear_phase(design, observed, inner_train, ridge, candidate.even_cost)
            prediction[inner_test] = design[inner_test] @ coefficient
        score = float(np.sqrt(np.mean((prediction[train] - observed[train]) ** 2)))
        records.append((score, tau, ridge))
    best = min(score for score, _tau, _ridge in records)
    eligible = [record for record in records if record[0] <= 1.01 * best]
    score, tau, ridge = sorted(eligible, key=lambda record: (-record[2], -record[1], record[0]))[0]
    return tau, ridge, score


def nested_pair_prediction(
    dataset: family_grp.Dataset,
    observed: np.ndarray,
    candidate: Candidate,
    target_index: int,
) -> tuple[np.ndarray, list[dict[str, Any]], tuple[np.ndarray, ...], tuple[str, ...]]:
    prediction = np.full(len(observed), np.nan, dtype=float)
    coefficients = []
    selections = []
    names: tuple[str, ...] = ()
    outer = folds(np.arange(len(observed)), OUTER_FOLDS, 7_100 + target_index)
    for fold_index, (train, test) in enumerate(outer):
        tau, ridge, inner_rmse = select_hyperparameters(
            dataset,
            observed,
            candidate,
            train,
            INNER_FOLDS,
            8_100 + 100 * target_index + fold_index,
        )
        design, names = phase_design(dataset.weights, dataset, candidate, tau)
        coefficient = fit_linear_phase(design, observed, train, ridge, candidate.even_cost)
        prediction[test] = design[test] @ coefficient
        coefficients.append(coefficient)
        selections.append(
            {
                "outer_fold": fold_index,
                "tau": tau,
                "ridge": ridge,
                "inner_rmse": inner_rmse,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete pair OOF prediction for {candidate.name}")
    return prediction, selections, tuple(coefficients), names


def coefficient_stability(coefficients: tuple[np.ndarray, ...]) -> dict[str, float]:
    if not coefficients or coefficients[0].size == 0:
        return {"median_pairwise_cosine": float("nan"), "median_sign_agreement": float("nan")}
    matrix = np.vstack(coefficients)
    cosines = []
    for left, right in itertools.combinations(range(len(matrix)), 2):
        denominator = float(np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right]))
        if denominator > 1e-12:
            cosines.append(float(matrix[left] @ matrix[right] / denominator))
    signs = np.sign(np.where(np.abs(matrix) < 1e-8, 0.0, matrix))
    agreements = []
    for column in signs.T:
        agreements.append(max(float(np.mean(column == sign)) for sign in (-1.0, 0.0, 1.0)))
    return {
        "median_pairwise_cosine": float(np.median(cosines)) if cosines else float("nan"),
        "median_sign_agreement": float(np.median(agreements)),
    }


def full_phase_fit(
    dataset: family_grp.Dataset,
    observed: np.ndarray,
    candidate: Candidate,
    target_index: int,
) -> tuple[PhaseFit, float]:
    indices = np.arange(len(observed))
    tau, ridge, cv_rmse = select_hyperparameters(
        dataset,
        observed,
        candidate,
        indices,
        OUTER_FOLDS,
        9_100 + target_index,
    )
    design, names = phase_design(dataset.weights, dataset, candidate, tau)
    coefficients = fit_linear_phase(design, observed, indices, ridge, candidate.even_cost)
    return PhaseFit(candidate, tau, ridge, coefficients, names), cv_rmse


def fiber_delta_prediction(
    model: TargetModels,
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> np.ndarray:
    """Predict each fiber policy relative to its same-seed center."""
    absolute = model.predict_phase(weights)
    prediction = np.full(len(frame), np.nan, dtype=float)
    for (anchor, block), indices in frame.groupby(["anchor_id", "seed_block"], sort=True).indices.items():
        local = np.asarray(indices, dtype=int)
        centers = local[frame.iloc[local]["contrast_family"].eq("center_control").to_numpy()]
        if len(centers) != 1:
            raise ValueError(f"Expected one center for fiber {anchor}/{block}, found {len(centers)}")
        prediction[local] = absolute[local] - absolute[int(centers[0])]
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete frontier-fiber delta prediction")
    return prediction


def stage1_gate(metrics: pd.DataFrame, stability: pd.DataFrame) -> dict[str, bool]:
    result = {}
    for candidate in CANDIDATES:
        if candidate.interaction == "zero":
            result[candidate.name] = False
            continue
        passed = True
        for target in TARGETS:
            row = metrics.loc[(metrics["candidate"] == candidate.name) & (metrics["target"] == target)].iloc[0]
            baseline = metrics.loc[
                (metrics["candidate"] == "zero_phase") & (metrics["target"] == target),
                "rmse",
            ].iloc[0]
            stable = stability.loc[(stability["candidate"] == candidate.name) & (stability["target"] == target)].iloc[0]
            passed &= float(row["rmse"]) <= (1.0 - STAGE1_RMSE_IMPROVEMENT) * float(baseline)
            passed &= float(stable["median_pairwise_cosine"]) >= STAGE1_COSINE
            passed &= float(stable["median_sign_agreement"]) >= STAGE1_SIGN_AGREEMENT
        result[candidate.name] = bool(passed)
    return result


def evaluate_development(
    matched: matched_pair.MatchedSources,
    promoted: list[Candidate],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], TargetModels]]:
    metric_rows = []
    fiber_rows = []
    prediction_rows = []
    models: dict[tuple[str, str], TargetModels] = {}
    evaluated = [CANDIDATES[0], *promoted]
    for target_index, target in enumerate(TARGETS):
        spine_dataset = aggregate_spine_dataset(matched, target)
        spine = hierarchical.fit_model(spine_dataset, composition.hpr_config(target), np.arange(spine_dataset.n))
        phase_dataset, observed_delta, _pair_frame = phase_pair_dataset(matched, target)
        for candidate in evaluated:
            phase_fit, pair_cv_rmse = full_phase_fit(phase_dataset, observed_delta, candidate, target_index)
            model = TargetModels(spine, phase_fit, phase_dataset)
            models[(target, candidate.name)] = model

            common_observed = matched.sources.common.frame[TARGET_COLUMNS[target]].to_numpy(dtype=float)
            common_prediction = model.predict(matched.sources.common.weights)
            for scope, mask in composition.scope_masks(matched.sources.common.frame, target).items():
                if int(mask.sum()) < 3:
                    continue
                metric_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "scope": scope,
                        "pair_cv_rmse": pair_cv_rmse,
                        **composition.prediction_metrics(common_observed[mask], common_prediction[mask]),
                    }
                )
            for index, row in matched.sources.common.frame.iterrows():
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "source": "common_archive",
                        "row_id": row["row_id"],
                        "policy_class": row["policy_class"],
                        "observed": common_observed[index],
                        "predicted": common_prediction[index],
                    }
                )

            fiber_observed = matched.sources.fiber.frame[heterogeneous.fiber_delta_column(target)].to_numpy(dtype=float)
            fiber_prediction = fiber_delta_prediction(
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
    return pd.DataFrame(metric_rows), pd.DataFrame(fiber_rows), pd.DataFrame(prediction_rows), models


def stage2_gate(development: pd.DataFrame, fiber: pd.DataFrame, candidates: list[Candidate]) -> dict[str, bool]:
    result = {}
    for candidate in candidates:
        passed = True
        for target in TARGETS:
            base_common = development.loc[
                (development["target"] == target)
                & (development["candidate"] == "zero_phase")
                & (development["scope"] == "common_all")
            ].iloc[0]
            candidate_common = development.loc[
                (development["target"] == target)
                & (development["candidate"] == candidate.name)
                & (development["scope"] == "common_all")
            ].iloc[0]
            passed &= float(candidate_common["regret_at_1"]) <= float(base_common["regret_at_1"]) + REGRET_TOLERANCE
            passed &= abs(float(candidate_common["calibration_slope"]) - 1.0) <= abs(
                float(base_common["calibration_slope"]) - 1.0
            )
            anchors = fiber.loc[(fiber["target"] == target) & (fiber["candidate"] == candidate.name), "anchor"]
            for anchor in anchors:
                if anchor == "all":
                    continue
                candidate_rmse = fiber.loc[
                    (fiber["target"] == target) & (fiber["candidate"] == candidate.name) & (fiber["anchor"] == anchor),
                    "rmse",
                ].iloc[0]
                baseline_rmse = fiber.loc[
                    (fiber["target"] == target) & (fiber["candidate"] == "zero_phase") & (fiber["anchor"] == anchor),
                    "rmse",
                ].iloc[0]
                passed &= float(candidate_rmse) <= FIBER_PRESERVATION_RATIO * float(baseline_rmse)
        result[candidate.name] = bool(passed)
    return result


def render_pair_plot(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure = px.scatter(
        predictions,
        x="observed_delta",
        y="predicted_delta",
        color="candidate",
        facet_col="target",
        facet_col_wrap=2,
        hover_data=["pair_id", "residual"],
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Plasticity-potential transport: nested OOF same-seed phase contrasts",
    )
    minimum = min(predictions["observed_delta"].min(), predictions["predicted_delta"].min())
    maximum = max(predictions["observed_delta"].max(), predictions["predicted_delta"].max())
    figure.add_shape(type="line", x0=minimum, y0=minimum, x1=maximum, y1=maximum, line={"dash": "dash"})
    figure.update_layout(template="plotly_white", height=620, legend={"orientation": "h", "y": -0.15})
    figure.write_html(output_dir / "pair_delta_oof_scatter.html", include_plotlyjs=True, config=PLOT_CONFIG)


def render_development_plot(predictions: pd.DataFrame, output_dir: Path) -> None:
    figure = px.scatter(
        predictions,
        x="observed",
        y="predicted",
        color="candidate",
        symbol="policy_class",
        facet_col="target",
        hover_data=["row_id"],
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Frozen common-archive predictions after pair-contrast promotion",
    )
    minimum = min(predictions["observed"].min(), predictions["predicted"].min())
    maximum = max(predictions["observed"].max(), predictions["predicted"].max())
    figure.add_shape(type="line", x0=minimum, y0=minimum, x1=maximum, y1=maximum, line={"dash": "dash"})
    figure.update_layout(template="plotly_white", height=650, legend={"orientation": "h", "y": -0.18})
    figure.write_html(output_dir / "development_calibration_scatter.html", include_plotlyjs=True, config=PLOT_CONFIG)


def render_phase_coefficients(models: dict[tuple[str, str], TargetModels], output_dir: Path) -> None:
    records = []
    for (target, candidate), model in models.items():
        for name, value in zip(
            model.phase_fit.feature_names,
            model.phase_fit.coefficients,
            strict=True,
        ):
            records.append({"target": target, "candidate": candidate, "feature": name, "coefficient": value})
    if not records:
        return
    frame = pd.DataFrame(records)
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9"])
    for column, target in enumerate(TARGETS, start=1):
        local = frame.loc[frame["target"] == target]
        for candidate, rows in local.groupby("candidate", sort=False):
            figure.add_trace(
                go.Bar(x=rows["coefficient"], y=rows["feature"], orientation="h", name=candidate, legendgroup=candidate),
                row=1,
                col=column,
            )
    figure.update_layout(
        barmode="group",
        template="plotly_white",
        height=max(650, 24 * frame["feature"].nunique()),
        title="Full-data phase-response coefficients",
    )
    figure.write_html(output_dir / "phase_coefficients.html", include_plotlyjs=True, config=PLOT_CONFIG)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_Not evaluated._"
    return frame[columns].to_markdown(index=False, floatfmt=".5f")


def write_report(
    output_dir: Path,
    pair_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    stage1: dict[str, bool],
    development: pd.DataFrame,
    fiber: pd.DataFrame,
    stage2: dict[str, bool],
) -> None:
    stage1_table = pair_metrics.merge(stability, on=["target", "candidate"], how="left")
    stage1_table["stage1_pass"] = stage1_table["candidate"].map(stage1).fillna(False)
    common = development.loc[development["scope"] == "common_all"] if not development.empty else development
    lines = [
        "# Plasticity-potential transport",
        "",
        "## Model",
        "",
        "The model separates the independently identified tied response from the value of phase placement:",
        "",
        "$$Y(w^{(0)},w^{(1)})=F_{1p}(a)+r^T(b+Mz)+\\chi q,$$",
        "",
        "where $a=\\alpha_0w^{(0)}+\\alpha_1w^{(1)}$, $z$ is log aggregate competence by family, "
        "$r$ is the first-order family-state displacement from moving fixed exposure between phases, $M=M^T$ is "
        "the Hessian of a scalar plasticity potential, and $q$ is a nonnegative finite-variation cost. The symmetry "
        "constraint makes the phase field integrable rather than an arbitrary interaction layer. When phases are tied, "
        "$r=q=0$ exactly and the model reduces to the independently fitted $F_{1p}$.",
        "",
        "The aggregate spine uses 238 independently trained one-phase policies plus 42 tied two-phase controls. The "
        "phase field uses only 238 exact same-seed two-minus-one contrasts, preventing aggregate quality from being "
        "misattributed to phase order.",
        "",
        "## Stage 1: pair-contrast falsification",
        "",
        markdown_table(
            stage1_table,
            [
                "target",
                "candidate",
                "rmse",
                "spearman",
                "calibration_slope",
                "median_pairwise_cosine",
                "median_sign_agreement",
                "stage1_pass",
            ],
        ),
        "",
    ]
    if development.empty:
        lines.extend(
            [
                "No candidate cleared the preregistered pair-contrast gate on both targets, so the development "
                "heldouts were not inspected for this batch.",
                "",
            ]
        )
    else:
        common = common.copy()
        common["stage2_pass"] = common["candidate"].map(stage2).fillna(False)
        lines.extend(
            [
                "## Stage 2: frozen development evidence",
                "",
                markdown_table(
                    common,
                    [
                        "target",
                        "candidate",
                        "rmse",
                        "spearman",
                        "calibration_slope",
                        "regret_at_1",
                        "optimism_gt_0p05",
                        "worst_optimism",
                        "stage2_pass",
                    ],
                ),
                "",
                "### Frontier-fiber phase deltas",
                "",
                markdown_table(
                    fiber.loc[fiber["anchor"] != "all"],
                    ["target", "candidate", "anchor", "rmse", "spearman", "calibration_slope"],
                ),
                "",
            ]
        )
    survivors = [name for name, passed in stage2.items() if passed]
    if survivors:
        verdict = (
            "Stage-2 survivors: " + ", ".join(survivors) + ". They require StarCoder, raw-optimum, bootstrap, and "
            "nested-ablation audits before promotion."
        )
    else:
        verdict = (
            "No candidate is promoted. The aggregate/phase decomposition remains useful, but this transport law is "
            "blocked."
        )
    lines.extend(["## Verdict", "", verdict, ""])
    (output_dir / "report.md").write_text("\n".join(lines))


def write_registry(
    output_dir: Path,
    stage1: dict[str, bool],
    stage2: dict[str, bool],
    pair_metrics: pd.DataFrame,
) -> None:
    rows = []
    for candidate in CANDIDATES:
        if candidate.interaction == "zero":
            status = "baseline"
        elif not stage1.get(candidate.name, False):
            status = "rejected_stage1"
        elif not stage2.get(candidate.name, False):
            status = "rejected_stage2"
        else:
            status = "promoted_stage3"
        evidence = pair_metrics.loc[pair_metrics["candidate"] == candidate.name, ["target", "rmse"]]
        evidence_text = "; ".join(f"{row.target} RMSE={row.rmse:.6f}" for row in evidence.itertuples())
        rows.append(
            {
                "family": candidate.name,
                "relationship_to_prior": "Extends PMVT/FCT with an integrable anchor-dependent family Jacobian",
                "new_mechanism": "Gradient transport through a scalar family plasticity potential",
                "additional_dof": 0 if candidate.interaction == "zero" else "3-10 target-specific phase coefficients",
                "single_phase_restriction": "Exact: phase displacement and finite-variation cost are zero",
                "status": status,
                "evidence": evidence_text,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "approach_registry.csv", index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not PREREGISTRATION_PATH.exists():
        raise FileNotFoundError(f"Missing frozen preregistration: {PREREGISTRATION_PATH}")
    preregistration = json.loads(PREREGISTRATION_PATH.read_text())
    if preregistration["mechanism"]["name"] != "Plasticity-potential transport":
        raise ValueError("Unexpected preregistration contents")

    matched = matched_pair.matched_sources()
    pair_metric_rows = []
    prediction_rows = []
    selection_rows = []
    stability_rows = []
    for target_index, target in enumerate(TARGETS):
        dataset, observed, frame = phase_pair_dataset(matched, target)
        for candidate in CANDIDATES:
            prediction, selections, coefficients, names = nested_pair_prediction(
                dataset,
                observed,
                candidate,
                target_index,
            )
            metrics = composition.prediction_metrics(observed, prediction)
            pair_metric_rows.append({"target": target, "candidate": candidate.name, **metrics})
            stability_rows.append(
                {
                    "target": target,
                    "candidate": candidate.name,
                    "parameter_count": len(names),
                    **coefficient_stability(coefficients),
                }
            )
            for selection in selections:
                selection_rows.append({"target": target, "candidate": candidate.name, **selection})
            for index, row in frame.iterrows():
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate": candidate.name,
                        "pair_id": row["pair_id"],
                        "observed_delta": observed[index],
                        "predicted_delta": prediction[index],
                        "residual": prediction[index] - observed[index],
                    }
                )

    pair_metrics = pd.DataFrame(pair_metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    stability = pd.DataFrame(stability_rows)
    stage1 = stage1_gate(pair_metrics, stability)
    pair_metrics.to_csv(args.output_dir / "pair_oof_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "pair_oof_predictions.csv", index=False)
    selections.to_csv(args.output_dir / "selected_hyperparameters.csv", index=False)
    stability.to_csv(args.output_dir / "coefficient_stability.csv", index=False)
    render_pair_plot(predictions, args.output_dir)

    promoted = [candidate for candidate in CANDIDATES if stage1.get(candidate.name, False)]
    development = pd.DataFrame()
    fiber = pd.DataFrame()
    development_predictions = pd.DataFrame()
    models: dict[tuple[str, str], TargetModels] = {}
    stage2: dict[str, bool] = {}
    if promoted and not args.stage1_only:
        development, fiber, development_predictions, models = evaluate_development(matched, promoted)
        stage2 = stage2_gate(development, fiber, promoted)
        development.to_csv(args.output_dir / "development_metrics.csv", index=False)
        fiber.to_csv(args.output_dir / "fiber_metrics.csv", index=False)
        development_predictions.to_csv(args.output_dir / "development_predictions.csv", index=False)
        render_development_plot(development_predictions, args.output_dir)
        render_phase_coefficients(models, args.output_dir)

    ledger = [
        {
            "stage": "preregistration",
            "evidence_inspected": "algebra, prior registry, matched-pair identities",
            "decision": "froze five nested candidates and numerical stage-1 gate",
            "adversarial_outcomes_inspected": False,
        },
        {
            "stage": "pair_contrast",
            "evidence_inspected": "238 exact same-seed two-minus-one contrasts per target",
            "decision": json.dumps(stage1, sort_keys=True),
            "adversarial_outcomes_inspected": False,
        },
    ]
    if promoted and not args.stage1_only:
        ledger.append(
            {
                "stage": "frozen_development",
                "evidence_inspected": "frontier fibers and append-only common archive after stage-1 freeze",
                "decision": json.dumps(stage2, sort_keys=True),
                "adversarial_outcomes_inspected": True,
            }
        )
    pd.DataFrame(ledger).to_csv(args.output_dir / "data_use_ledger.csv", index=False)
    write_registry(args.output_dir, stage1, stage2, pair_metrics)
    write_report(args.output_dir, pair_metrics, stability, stage1, development, fiber, stage2)


if __name__ == "__main__":
    main()
