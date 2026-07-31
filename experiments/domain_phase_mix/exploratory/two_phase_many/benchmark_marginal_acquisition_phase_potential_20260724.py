# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "gcsfs",
#   "joblib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Benchmark an order-only phase potential derived from aggregate marginal value.

The aggregate model is fit only to phase-tied rows. Fixed-aggregate antithetic
pairs then identify the odd phase-order response:

    O(a, d) = -alpha_0 alpha_1 sum_i gamma_i m_i(a) d_i,

where ``m_i(a)`` is the marginal BPB benefit implied by the frozen physical
pooled-acquisition aggregate model. The phase coefficients are nonnegative and
are either global, family-pooled, or bucket-resolved with family shrinkage.

The even response is modeled separately as a nonnegative family switching cost.
No sealed targeted-pairwise outcomes are loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_aggregate_phase_pairs_20260723 as pair_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_fixed_budget_aggregate_comparators_20260724 as comparators,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_frontier_control_aggregate_identification_20260724 as aggregate_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_orthogonal_aggregate_phase_identification_20260724 as orthogonal,
)

REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "marginal_acquisition_phase_potential_20260724"
DEFAULT_SEEDS = (20260724, 20260725, 20260726)
RIDGE_GRID = (0.0, 0.1, 1.0, 10.0, 100.0)
HIERARCHY_GRID = (0.1, 1.0, 10.0, 100.0)
N_FOLDS = 5
HUBER_THRESHOLD = 1.345
NUMERICAL_FLOOR = 1e-10
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class PairDataset:
    """Target-specific fixed-aggregate antithetic pair observations."""

    target: str
    frame: pd.DataFrame
    aggregate: np.ndarray
    contrast: np.ndarray
    odd: np.ndarray
    even: np.ndarray
    noise: np.ndarray
    domain_names: tuple[str, ...]

    @property
    def n(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class Candidate:
    """One frozen odd-response feature family."""

    name: str
    level: str
    mechanism: str


@dataclass(frozen=True)
class FitResult:
    """A nonnegative robust phase head."""

    coefficients: np.ndarray
    family_means: np.ndarray
    ridge: float
    hierarchy: float

    def predict(self, design: np.ndarray) -> np.ndarray:
        return np.asarray(design @ self.coefficients, dtype=float)


CANDIDATES = (
    Candidate(
        name="raw_family_phase_potential",
        level="raw_family",
        mechanism="family phase contrast without aggregate marginal-value coupling",
    ),
    Candidate(
        name="marginal_global_phase_potential",
        level="marginal_global",
        mechanism="one shared recency coefficient multiplying aggregate marginal value",
    ),
    Candidate(
        name="marginal_family_phase_potential",
        level="marginal_family",
        mechanism="family-specific recency coefficients multiplying aggregate marginal value",
    ),
    Candidate(
        name="marginal_bucket_phase_potential",
        level="marginal_bucket",
        mechanism="bucket recency coefficients with explicit shrinkage to family means",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    return parser.parse_args()


def pair_datasets() -> dict[str, PairDataset]:
    semantics = pair_audit.bucket_semantics()
    domains = tuple(semantics["domain"].tolist())
    controls = pd.read_csv(pair_audit.DEFAULT_OUTPUT_DIR / "control_noise_summary.csv").set_index(
        ["panel", "anchor_id", "target"]
    )["control_sd_bpb"]
    rows_by_target: dict[str, list[dict[str, Any]]] = {target: [] for target in orthogonal.TARGETS}
    aggregates_by_target: dict[str, list[np.ndarray]] = {target: [] for target in orthogonal.TARGETS}
    contrasts_by_target: dict[str, list[np.ndarray]] = {target: [] for target in orthogonal.TARGETS}
    odd_by_target: dict[str, list[float]] = {target: [] for target in orthogonal.TARGETS}
    even_by_target: dict[str, list[float]] = {target: [] for target in orthogonal.TARGETS}
    noise_by_target: dict[str, list[float]] = {target: [] for target in orthogonal.TARGETS}

    for panel in pair_audit.load_panel_inputs():
        grouped = panel.results.groupby(list(panel.pair_columns), sort=True, dropna=False)
        for pair_index, (_key, group) in enumerate(grouped):
            if set(group["sign"]) != {"plus", "minus"} or len(group) != 2:
                raise ValueError(f"Malformed antithetic pair in {panel.name}")
            plus_row = group[group["sign"].eq("plus")].iloc[0]
            minus_row = group[group["sign"].eq("minus")].iloc[0]
            plus = pair_audit.phase_matrix(panel.weights, str(plus_row["candidate_id"]), list(domains))
            minus = pair_audit.phase_matrix(panel.weights, str(minus_row["candidate_id"]), list(domains))
            aggregate_plus = pair_audit.PHASE_FRACTIONS @ plus
            aggregate_minus = pair_audit.PHASE_FRACTIONS @ minus
            if not np.allclose(aggregate_plus, aggregate_minus, atol=1e-9):
                raise ValueError("Phase pair does not preserve aggregate weights")
            contrast = plus[1] - plus[0]
            if not np.allclose(contrast, -(minus[1] - minus[0]), atol=1e-8):
                raise ValueError("Phase pair is not antithetic")
            pair_id = f"{panel.name}:{plus_row['anchor_id']}:{plus_row['direction_id']}:{pair_index:03d}"
            for target, metric_column in pair_audit.TARGET_COLUMNS.items():
                center_column, _delta_column = pair_audit.control_columns(panel.name, target)
                plus_bpb = float(plus_row[metric_column])
                minus_bpb = float(minus_row[metric_column])
                center_bpb = float(plus_row[center_column])
                rows_by_target[target].append(
                    {
                        "pair_id": pair_id,
                        "panel": panel.name,
                        "anchor_id": str(plus_row["anchor_id"]),
                        "contrast_family": str(plus_row["contrast_family"]),
                        "direction_id": str(plus_row["direction_id"]),
                        "direction_group": f"{panel.name}::{plus_row['direction_id']}",
                        "phase_tv": 0.5 * float(np.abs(contrast).sum()),
                        "seed_block": int(plus_row["seed_block"]),
                        "plus_candidate_id": str(plus_row["candidate_id"]),
                        "minus_candidate_id": str(minus_row["candidate_id"]),
                        "plus_bpb": plus_bpb,
                        "minus_bpb": minus_bpb,
                        "center_bpb": center_bpb,
                    }
                )
                aggregates_by_target[target].append(aggregate_plus)
                contrasts_by_target[target].append(contrast)
                odd_by_target[target].append(0.5 * (plus_bpb - minus_bpb))
                even_by_target[target].append(0.5 * (plus_bpb + minus_bpb) - center_bpb)
                control_sd = float(controls.loc[(panel.name, plus_row["anchor_id"], target)])
                noise_by_target[target].append(max(control_sd / np.sqrt(2.0), 2e-4))

    result = {}
    for target in orthogonal.TARGETS:
        frame = pd.DataFrame(rows_by_target[target])
        result[target] = PairDataset(
            target=target,
            frame=frame,
            aggregate=np.asarray(aggregates_by_target[target], dtype=float),
            contrast=np.asarray(contrasts_by_target[target], dtype=float),
            odd=np.asarray(odd_by_target[target], dtype=float),
            even=np.asarray(even_by_target[target], dtype=float),
            noise=np.asarray(noise_by_target[target], dtype=float),
            domain_names=domains,
        )
        if result[target].n != 192:
            raise ValueError(f"Expected 192 {target} pairs, found {result[target].n}")
    return result


def response_derivative(exposure: np.ndarray, model: orthogonal.AggregateModel) -> np.ndarray:
    exposure = np.maximum(np.asarray(exposure, dtype=float), NUMERICAL_FLOOR)
    rho = model.shape.rho
    power = model.shape.power
    return power * rho**power * exposure ** (power - 1.0) * np.exp(-((rho * exposure) ** power))


def marginal_bucket_value(
    model: orthogonal.AggregateModel,
    aggregate: np.ndarray,
) -> np.ndarray:
    """Return positive marginal BPB benefit per unit aggregate mixture weight."""

    exposure = aggregate * model.c_total[None, :]
    marginal = model.bucket_coef[None, :] * model.c_total[None, :] * response_derivative(exposure, model)
    for family_index, members in enumerate(model.families.members):
        token_fraction = float(np.sum(1.0 / model.c_total[members]))
        family_epochs = aggregate[:, members].sum(axis=1) / token_fraction
        family_marginal = model.family_coef[family_index] * response_derivative(family_epochs, model) / token_fraction
        marginal[:, members] += family_marginal[:, None]
    return marginal


def aligned_pair_arrays(dataset: PairDataset) -> tuple[np.ndarray, np.ndarray]:
    aggregate_domains = tuple(comparators.target_data(dataset.target)[3].domain_names)
    if set(dataset.domain_names) != set(aggregate_domains):
        raise ValueError("Pair and aggregate domain sets do not agree")
    source_index = {domain: index for index, domain in enumerate(dataset.domain_names)}
    permutation = np.asarray([source_index[domain] for domain in aggregate_domains], dtype=int)
    return dataset.aggregate[:, permutation], dataset.contrast[:, permutation]


def candidate_design(
    dataset: PairDataset,
    model: orthogonal.AggregateModel,
    candidate: Candidate,
) -> tuple[np.ndarray, np.ndarray]:
    alpha0 = model.phase_fraction
    alpha1 = 1.0 - alpha0
    family_group = model.families.bucket_group
    aggregate, contrast = aligned_pair_arrays(dataset)
    if candidate.level == "raw_family":
        bucket = -alpha0 * alpha1 * contrast
        design = np.column_stack([bucket[:, members].sum(axis=1) for members in model.families.members])
        return design, np.arange(len(model.families.members), dtype=int)

    marginal = marginal_bucket_value(model, aggregate)
    bucket = -alpha0 * alpha1 * marginal * contrast
    if candidate.level == "marginal_global":
        return bucket.sum(axis=1, keepdims=True), np.zeros(1, dtype=int)
    if candidate.level == "marginal_family":
        design = np.column_stack([bucket[:, members].sum(axis=1) for members in model.families.members])
        return design, np.arange(len(model.families.members), dtype=int)
    if candidate.level == "marginal_bucket":
        return bucket, family_group
    raise ValueError(f"Unknown candidate level {candidate.level}")


def even_design(
    dataset: PairDataset,
    model: orthogonal.AggregateModel,
) -> np.ndarray:
    alpha0 = model.phase_fraction
    alpha1 = 1.0 - alpha0
    _aggregate, contrast = aligned_pair_arrays(dataset)
    moved = alpha0 * alpha1 * np.abs(contrast)
    return np.column_stack([moved[:, members].sum(axis=1) ** 2 for members in model.families.members])


def fit_nonnegative_head(
    design: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
    family_group: np.ndarray,
    ridge: float,
    hierarchy: float,
) -> FitResult:
    width = design.shape[1]
    coefficients = cp.Variable(width, nonneg=True)
    standardized_residual = (design @ coefficients - target) / noise
    objective = cp.sum(cp.huber(standardized_residual, HUBER_THRESHOLD))
    if ridge > 0:
        objective += ridge * cp.sum_squares(coefficients)
    family_count = int(np.max(family_group)) + 1
    family_means = cp.Variable(family_count, nonneg=True)
    if width > family_count and hierarchy > 0:
        objective += hierarchy * cp.sum_squares(coefficients - family_means[family_group])
    else:
        objective += 1e-8 * cp.sum_squares(family_means)
    problem = cp.Problem(cp.Minimize(objective))
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Phase-potential fit failed: {problem.status}")
    if coefficients.value is None or family_means.value is None:
        raise RuntimeError("Phase-potential fit did not return coefficients")
    return FitResult(
        coefficients=np.asarray(coefficients.value, dtype=float),
        family_means=np.asarray(family_means.value, dtype=float),
        ridge=ridge,
        hierarchy=hierarchy,
    )


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = predicted - observed
    rmse = float(np.sqrt(np.mean(residual**2)))
    zero_rmse = float(np.sqrt(np.mean(observed**2)))
    correlation = spearmanr(observed, predicted).statistic
    if np.std(predicted) < 1e-12:
        slope = 0.0
    else:
        slope = float(np.polyfit(predicted, observed, 1)[0])
    resolved = np.abs(observed) >= np.median(np.abs(observed))
    sign_accuracy = float(np.mean(np.sign(predicted[resolved]) == np.sign(observed[resolved])))
    return {
        "rmse": rmse,
        "zero_rmse": zero_rmse,
        "rmse_ratio": rmse / max(zero_rmse, NUMERICAL_FLOOR),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
        "calibration_slope": slope,
        "bias": float(np.mean(residual)),
        "resolved_sign_accuracy": sign_accuracy,
    }


def grouped_splits(dataset: PairDataset) -> list[tuple[np.ndarray, np.ndarray]]:
    return local_grouped_splits(dataset.frame["direction_group"].to_numpy())


def local_grouped_splits(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    fold_count = min(N_FOLDS, len(unique_groups))
    if fold_count < 2:
        raise ValueError("Grouped validation requires at least two direction groups")
    splitter = GroupKFold(n_splits=fold_count)
    return list(splitter.split(np.arange(len(groups)), groups=groups))


def candidate_grid(candidate: Candidate) -> tuple[tuple[float, float], ...]:
    if candidate.level != "marginal_bucket":
        return tuple((ridge, 0.0) for ridge in RIDGE_GRID)
    return tuple((ridge, hierarchy) for ridge in RIDGE_GRID for hierarchy in HIERARCHY_GRID)


def cross_validated_predictions(
    design: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
    family_group: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    ridge: float,
    hierarchy: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    prediction = np.full(len(target), np.nan, dtype=float)
    fold_coefficients = []
    for train, test in splits:
        fitted = fit_nonnegative_head(
            design[train],
            target[train],
            noise[train],
            family_group,
            ridge,
            hierarchy,
        )
        prediction[test] = fitted.predict(design[test])
        fold_coefficients.append(fitted.coefficients)
    if not np.isfinite(prediction).all():
        raise ValueError("Cross-validated prediction is incomplete")
    return prediction, fold_coefficients


def selected_config(
    design: np.ndarray,
    target: np.ndarray,
    noise: np.ndarray,
    family_group: np.ndarray,
    groups: np.ndarray,
    grid: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    splits = local_grouped_splits(groups)
    candidates = []
    for ridge, hierarchy in grid:
        prediction, _coefficients = cross_validated_predictions(
            design,
            target,
            noise,
            family_group,
            splits,
            ridge,
            hierarchy,
        )
        candidates.append((metric_row(target, prediction)["rmse"], ridge, hierarchy))
    _score, ridge, hierarchy = min(candidates)
    return ridge, hierarchy


def nested_grouped_predictions(
    dataset: PairDataset,
    design: np.ndarray,
    target: np.ndarray,
    family_group: np.ndarray,
    grid: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selected = []
    groups = dataset.frame["direction_group"].to_numpy()
    for train, test in grouped_splits(dataset):
        ridge, hierarchy = selected_config(
            design[train],
            target[train],
            dataset.noise[train],
            family_group,
            groups[train],
            grid,
        )
        fitted = fit_nonnegative_head(
            design[train],
            target[train],
            dataset.noise[train],
            family_group,
            ridge,
            hierarchy,
        )
        prediction[test] = fitted.predict(design[test])
        selected.append((ridge, hierarchy))
    if not np.isfinite(prediction).all():
        raise ValueError("Nested grouped prediction is incomplete")
    return prediction, selected


def leave_anchor_predictions(
    dataset: PairDataset,
    design: np.ndarray,
    target: np.ndarray,
    family_group: np.ndarray,
    grid: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    selected = []
    groups = dataset.frame["direction_group"].to_numpy()
    for anchor in sorted(dataset.frame["anchor_id"].unique()):
        test = np.flatnonzero(dataset.frame["anchor_id"].eq(anchor).to_numpy())
        train = np.flatnonzero(~dataset.frame["anchor_id"].eq(anchor).to_numpy())
        ridge, hierarchy = selected_config(
            design[train],
            target[train],
            dataset.noise[train],
            family_group,
            groups[train],
            grid,
        )
        fitted = fit_nonnegative_head(
            design[train],
            target[train],
            dataset.noise[train],
            family_group,
            ridge,
            hierarchy,
        )
        prediction[test] = fitted.predict(design[test])
        selected.append((ridge, hierarchy))
    if not np.isfinite(prediction).all():
        raise ValueError("Leave-anchor-out prediction is incomplete")
    return prediction, selected


def fit_target_seed(
    dataset: PairDataset,
    seed: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    (
        _reference,
        _heldout_frame,
        _heldout_weights,
        single,
        controls,
        _evaluation_frame,
        _evaluation_weights,
        _observed,
        _clusters,
    ) = comparators.target_data(dataset.target)
    training = aggregate_audit.training_dataset(
        dataset.target,
        single,
        controls,
        "tied_272_plus_controls",
        seed,
    )
    fold = comparators.strict_protocol.grouped_stratified_folds(training, seed)
    aggregate_fit = aggregate_audit.frozen_pooled_fit(training, fold)
    model = aggregate_fit.model
    splits = grouped_splits(dataset)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames = []
    selection_rows = []

    for candidate in CANDIDATES:
        design, family_group = candidate_design(dataset, model, candidate)
        grid = candidate_grid(candidate)
        candidates = []
        for ridge, hierarchy in grid:
            prediction, fold_coefficients = cross_validated_predictions(
                design,
                dataset.odd,
                dataset.noise,
                family_group,
                splits,
                ridge,
                hierarchy,
            )
            metrics = metric_row(dataset.odd, prediction)
            stability = float(
                np.median(
                    [
                        np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), NUMERICAL_FLOOR)
                        for left in fold_coefficients
                        for right in fold_coefficients
                        if left is not right
                    ]
                )
            )
            candidates.append((metrics["rmse"], ridge, hierarchy, prediction, metrics, stability))
            selection_rows.append(
                {
                    "target": dataset.target,
                    "seed": seed,
                    "candidate": candidate.name,
                    "ridge": ridge,
                    "hierarchy": hierarchy,
                    "coefficient_stability_cosine": stability,
                    **metrics,
                }
            )
        _score, ridge, hierarchy, _prediction, _metrics, stability = min(
            candidates,
            key=lambda row: (row[0], row[1], row[2]),
        )
        nested_prediction, nested_configs = nested_grouped_predictions(
            dataset,
            design,
            dataset.odd,
            family_group,
            grid,
        )
        nested_metrics = metric_row(dataset.odd, nested_prediction)
        leave_anchor, leave_anchor_configs = leave_anchor_predictions(
            dataset,
            design,
            dataset.odd,
            family_group,
            grid,
        )
        leave_anchor_metrics = metric_row(dataset.odd, leave_anchor)
        fitted = fit_nonnegative_head(
            design,
            dataset.odd,
            dataset.noise,
            family_group,
            ridge,
            hierarchy,
        )
        metric_rows.append(
            {
                "target": dataset.target,
                "seed": seed,
                "candidate": candidate.name,
                "mechanism": candidate.mechanism,
                "parameter_count": len(fitted.coefficients),
                "ridge": ridge,
                "hierarchy": hierarchy,
                "coefficient_stability_cosine": stability,
                "nested_config_count": len(set(nested_configs)),
                "leave_anchor_config_count": len(set(leave_anchor_configs)),
                **{f"grouped_{key}": value for key, value in nested_metrics.items()},
                **{f"leave_anchor_{key}": value for key, value in leave_anchor_metrics.items()},
            }
        )
        local = dataset.frame.copy()
        local["target"] = dataset.target
        local["seed"] = seed
        local["candidate"] = candidate.name
        local["observed_odd"] = dataset.odd
        local["predicted_odd"] = nested_prediction
        local["leave_anchor_predicted_odd"] = leave_anchor
        local["residual"] = nested_prediction - dataset.odd
        prediction_frames.append(local)

    even = even_design(dataset, model)
    even_family_group = np.arange(even.shape[1], dtype=int)
    even_candidates = []
    even_grid = tuple((ridge, 0.0) for ridge in RIDGE_GRID)
    for ridge in RIDGE_GRID:
        prediction, _coefficients = cross_validated_predictions(
            even,
            dataset.even,
            dataset.noise,
            even_family_group,
            splits,
            ridge,
            0.0,
        )
        metrics = metric_row(dataset.even, prediction)
        even_candidates.append((metrics["rmse"], ridge, prediction, metrics))
    _score, even_ridge, _even_prediction, _even_metrics = min(
        even_candidates,
        key=lambda row: (row[0], row[1]),
    )
    even_nested_prediction, even_nested_configs = nested_grouped_predictions(
        dataset,
        even,
        dataset.even,
        even_family_group,
        even_grid,
    )
    even_nested_metrics = metric_row(dataset.even, even_nested_prediction)
    even_leave_anchor, even_leave_anchor_configs = leave_anchor_predictions(
        dataset,
        even,
        dataset.even,
        even_family_group,
        even_grid,
    )
    even_leave_anchor_metrics = metric_row(dataset.even, even_leave_anchor)
    metric_rows.append(
        {
            "target": dataset.target,
            "seed": seed,
            "candidate": "nonnegative_family_switching_cost",
            "mechanism": "quadratic family phase-mass switching cost",
            "parameter_count": even.shape[1],
            "ridge": even_ridge,
            "hierarchy": 0.0,
            "nested_config_count": len(set(even_nested_configs)),
            "leave_anchor_config_count": len(set(even_leave_anchor_configs)),
            **{f"grouped_{key}": value for key, value in even_nested_metrics.items()},
            **{f"leave_anchor_{key}": value for key, value in even_leave_anchor_metrics.items()},
        }
    )
    even_frame = dataset.frame.copy()
    even_frame["target"] = dataset.target
    even_frame["seed"] = seed
    even_frame["candidate"] = "nonnegative_family_switching_cost"
    even_frame["observed_even"] = dataset.even
    even_frame["predicted_even"] = even_nested_prediction
    even_frame["leave_anchor_predicted_even"] = even_leave_anchor
    prediction_frames.append(even_frame)
    return metric_rows, pd.DataFrame(selection_rows), pd.concat(prediction_frames, ignore_index=True)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
) -> None:
    columns = [
        "target",
        "candidate",
        "seed",
        "parameter_count",
        "ridge",
        "hierarchy",
        "grouped_rmse",
        "grouped_rmse_ratio",
        "grouped_spearman",
        "grouped_calibration_slope",
        "grouped_resolved_sign_accuracy",
        "leave_anchor_rmse_ratio",
        "leave_anchor_spearman",
        "leave_anchor_resolved_sign_accuracy",
        "coefficient_stability_cosine",
    ]
    available = [column for column in columns if column in metrics]
    lines = [
        "# Marginal-acquisition phase potential",
        "",
        "## Equation",
        "",
        (
            r"The phase-odd response is \(O(a,d)=-\alpha_0\alpha_1\sum_i"
            r"\gamma_i m_i(a)d_i\), where \(m_i(a)=-\partial A(a)/\partial a_i\) "
            "comes from a phase-invariant pooled-acquisition aggregate fit. "
            "The tied limit is exactly zero."
        ),
        "",
        "## Results",
        "",
        metrics[available]
        .sort_values(["target", "grouped_rmse_ratio", "leave_anchor_rmse_ratio"])
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "Grouped CV holds out complete direction families across all radii. "
        "Leave-anchor-out fits one frontier aggregate and predicts the other.",
        "",
        "The sealed targeted pairwise panel was not accessed.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"commit": commit, "dirty": dirty}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    datasets = pair_datasets()
    results = [fit_target_seed(datasets[target], seed) for target in orthogonal.TARGETS for seed in seeds]
    metrics = pd.DataFrame([row for result in results for row in result[0]])
    selections = pd.concat([result[1] for result in results], ignore_index=True)
    predictions = pd.concat([result[2] for result in results], ignore_index=True)
    metrics.to_csv(args.output_dir / "metrics.csv", index=False)
    selections.to_csv(args.output_dir / "hyperparameter_selections.csv", index=False)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)

    odd_metrics = metrics[metrics["candidate"].ne("nonnegative_family_switching_cost")].copy()
    figure = px.scatter(
        odd_metrics,
        x="grouped_rmse_ratio",
        y="leave_anchor_rmse_ratio",
        color="candidate",
        symbol="target",
        hover_data=["seed", "grouped_spearman", "leave_anchor_spearman"],
        title="Phase-order prediction: direction CV versus anchor transfer",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
    )
    figure.add_vline(x=1.0, line_dash="dash", line_color="#334155")
    figure.add_hline(y=1.0, line_dash="dash", line_color="#334155")
    figure.update_layout(template="plotly_white", width=1100, height=750)
    figure.write_html(
        args.output_dir / "phase_potential_falsification.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    write_report(args.output_dir, metrics)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "aggregate_training_design": {
                    "tied_rows": 272,
                    "frontier_control_rows": 8,
                    "phase_rows": 0,
                },
                "candidates": [asdict(candidate) for candidate in CANDIDATES],
                "ridge_grid": RIDGE_GRID,
                "hierarchy_grid": HIERARCHY_GRID,
                "grouped_cv": "five folds grouped by panel and phase direction",
                "anchor_transfer": "leave one frontier aggregate out",
                "sealed_targeted_pairwise_panel_accessed": False,
                "script_sha256": script_hash,
                "git": git_metadata(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
