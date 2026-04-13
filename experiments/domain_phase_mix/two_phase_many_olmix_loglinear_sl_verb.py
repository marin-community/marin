# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit an Olmix log-linear baseline on two-phase-many MMLU SL-Verb results."""

from __future__ import annotations

import json
from dataclasses import dataclass

import fsspec
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.nextgen.utils import normalize_phase_weights
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import (
    OLMIX_LOGLINEAR_HUBER_DELTA,
    OLMIX_LOGLINEAR_KL_LAMBDA,
    OLMIX_LOGLINEAR_PHASE_WEIGHTS,
)

RUN_ID = 243
RUN_NAME = "baseline_olmix_loglinear_sl_verb_choice_logprob_norm"
SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_sl_verb_choice_logprob_norm"
SOURCE_RESULTS_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_mmlu_sl_verb_rerun"
OBJECTIVE_METRIC = "lm_eval/mmlu_sl_verb_5shot/choice_logprob_norm"
NEGATED_OBJECTIVE_METRIC = f"negated/{OBJECTIVE_METRIC}"
DEFAULT_RESULTS_ROOT = "gs://marin-us-east5"
FIT_SUMMARY_JSON = "olmix_sl_verb_fit_summary.json"
FIT_START_SEED = 0
FIT_N_STARTS = 48
SOLVE_SEED = 0
SOLVE_RANDOM_STARTS = 8


@dataclass(frozen=True)
class OlmixLoglinearFit:
    """Fitted Olmix log-linear surrogate in flattened phase-weight space."""

    log_c: float
    coefficients: tuple[float, ...]
    huber_loss: float

    def predict(self, weights: np.ndarray) -> np.ndarray:
        matrix = np.asarray(weights, dtype=float).reshape(len(weights), -1)
        logits = np.clip(matrix @ np.asarray(self.coefficients, dtype=float), -50.0, 50.0)
        return np.exp(self.log_c) + np.exp(logits)


@dataclass(frozen=True)
class OlmixSlVerbFitResult:
    """Summary of the fitted SL-Verb Olmix baseline."""

    objective_metric: str
    negated_objective_metric: str
    source_results_path: str
    fit_log_c: float
    fit_coefficients: list[float]
    fit_huber_loss: float
    kl_lambda: float
    predicted_negated_objective: float
    predicted_choice_logprob_norm: float
    regularized_objective: float
    phase_weights: dict[str, dict[str, float]]


def resolve_unique_results_csv(
    *,
    source_experiment: str = SOURCE_RESULTS_EXPERIMENT,
    results_root: str = DEFAULT_RESULTS_ROOT,
) -> str:
    """Resolve the unique collected-results CSV for a completed rerun experiment."""
    pattern = f"{results_root}/{source_experiment}/collect_results-*/results.csv"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    matches = sorted(fs.glob(pattern))
    if not matches:
        raise ValueError(f"No collect_results CSV matched {pattern}")
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one collect_results CSV for {source_experiment}, found {len(matches)}")
    match = matches[0]
    return match if match.startswith("gs://") else f"gs://{match}"


def load_results_frame(results_path: str) -> pd.DataFrame:
    """Load the completed SL-Verb rerun results into a dataframe."""
    with fsspec.open(results_path, "r") as f:
        frame = pd.read_csv(f)
    if OBJECTIVE_METRIC not in frame.columns:
        raise ValueError(f"Missing {OBJECTIVE_METRIC!r} in {results_path}")
    return frame


def aggregate_candidate_mean_frame(
    frame: pd.DataFrame,
    *,
    candidate_group_column: str = "candidate_run_name",
) -> pd.DataFrame:
    """Collapse replicated candidate rows to one mean row per candidate."""
    if candidate_group_column not in frame.columns:
        raise ValueError(f"Missing candidate group column {candidate_group_column!r}")

    group_columns = [
        column
        for column in ("candidate_run_id", "candidate_run_name", "candidate_source_experiment")
        if column in frame.columns
    ]
    if not group_columns:
        raise ValueError("No candidate metadata columns available for candidate-mean aggregation")

    numeric_columns = [
        column
        for column in frame.columns
        if column not in group_columns and pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = frame[[candidate_group_column, *numeric_columns]].groupby(candidate_group_column, as_index=False).mean()
    metadata = frame[group_columns].drop_duplicates(subset=[candidate_group_column])
    aggregated = metadata.merge(numeric, on=candidate_group_column, how="inner", validate="one_to_one")
    return aggregated.sort_values(group_columns).reset_index(drop=True)


def _negated_objective_frame(frame: pd.DataFrame, *, objective_metric: str = OBJECTIVE_METRIC) -> pd.DataFrame:
    result = frame.copy()
    result = result[result[objective_metric].notna()].reset_index(drop=True)
    result[NEGATED_OBJECTIVE_METRIC] = -result[objective_metric].astype(float)
    return result


def _huber_sum(residuals: np.ndarray, *, delta: float) -> float:
    abs_residuals = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (abs_residuals - 0.5 * delta)
    return float(np.where(abs_residuals <= delta, quadratic, linear).sum())


def fit_olmix_loglinear_model(
    weights: np.ndarray,
    targets: np.ndarray,
    *,
    delta: float = OLMIX_LOGLINEAR_HUBER_DELTA,
    seed: int = FIT_START_SEED,
    n_starts: int = FIT_N_STARTS,
) -> OlmixLoglinearFit:
    """Fit the Olmix log-linear surrogate on positive targets."""
    x = np.asarray(weights, dtype=float).reshape(len(weights), -1)
    y = np.asarray(targets, dtype=float)
    rng = np.random.default_rng(seed)

    def objective(params: np.ndarray) -> float:
        log_c = float(params[0])
        coefficients = params[1:]
        logits = np.clip(x @ coefficients, -50.0, 50.0)
        predictions = np.exp(log_c) + np.exp(logits)
        return _huber_sum(predictions - y, delta=delta)

    best_params = None
    best_loss = float("inf")
    log_c_candidates = np.linspace(np.log(max(np.min(y) * 0.25, 1e-3)), np.log(max(np.median(y), 1e-3)), 6)
    starts: list[np.ndarray] = []
    for log_c in log_c_candidates:
        starts.append(np.concatenate([[log_c], np.zeros(x.shape[1], dtype=float)]))
        for _ in range(max(n_starts // len(log_c_candidates) - 1, 0)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 1.0, size=x.shape[1])]))

    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B")
        if not result.success and best_params is not None:
            continue
        if float(result.fun) < best_loss:
            best_loss = float(result.fun)
            best_params = np.asarray(result.x, dtype=float)

    if best_params is None:
        raise RuntimeError("Olmix loglinear fit failed")

    return OlmixLoglinearFit(
        log_c=float(best_params[0]),
        coefficients=tuple(float(value) for value in best_params[1:]),
        huber_loss=best_loss,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp / exp.sum(axis=1, keepdims=True)


def _phase_weights_from_matrix(
    weights: np.ndarray,
    *,
    phase_names: list[str],
    domain_names: list[str],
) -> dict[str, dict[str, float]]:
    return normalize_phase_weights(
        {
            phase_names[phase_idx]: {
                domain_names[domain_idx]: float(weights[phase_idx, domain_idx])
                for domain_idx in range(len(domain_names))
            }
            for phase_idx in range(len(phase_names))
        }
    )


def _weighted_multiclass_kl(weights: np.ndarray, natural_proportions: np.ndarray, phase_fractions: np.ndarray) -> float:
    eps = 1e-9
    q = np.clip(np.asarray(weights, dtype=float), eps, 1.0)
    p = np.clip(np.asarray(natural_proportions, dtype=float), eps, 1.0)
    return float(phase_fractions @ np.sum(q * (np.log(q) - np.log(p[None, :])), axis=1))


def _old_olmix_logits(domain_names: list[str], phase_names: list[str]) -> np.ndarray:
    if any(phase_name not in OLMIX_LOGLINEAR_PHASE_WEIGHTS for phase_name in phase_names):
        raise KeyError("phase mismatch")
    for phase_name in phase_names:
        if any(domain_name not in OLMIX_LOGLINEAR_PHASE_WEIGHTS[phase_name] for domain_name in domain_names):
            raise KeyError("domain mismatch")
    return np.array(
        [
            [np.log(max(OLMIX_LOGLINEAR_PHASE_WEIGHTS[phase_name][domain_name], 1e-9)) for domain_name in domain_names]
            for phase_name in phase_names
        ],
        dtype=float,
    )


def solve_olmix_loglinear_schedule(
    fit: OlmixLoglinearFit,
    *,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
    phase_names: list[str],
    domain_names: list[str],
    lambda_kl: float = OLMIX_LOGLINEAR_KL_LAMBDA,
    seed: int = SOLVE_SEED,
    random_starts: int = SOLVE_RANDOM_STARTS,
) -> tuple[dict[str, dict[str, float]], float, float]:
    """Solve the KL-regularized Olmix schedule under a multiclass proportional prior."""
    rng = np.random.default_rng(seed)
    n_phases = len(phase_names)
    n_domains = len(domain_names)
    natural_logits = np.log(np.clip(natural_proportions, 1e-9, 1.0))
    starts = [
        np.tile(natural_logits, n_phases),
        np.zeros(n_phases * n_domains, dtype=float),
    ]
    try:
        starts.append(_old_olmix_logits(domain_names, phase_names).reshape(-1))
    except KeyError:
        pass
    starts.extend(
        natural_logits.repeat(n_phases).reshape(n_phases, n_domains).ravel()
        + rng.normal(0.0, 1.0, size=n_phases * n_domains)
        for _ in range(random_starts)
    )

    def objective(params: np.ndarray) -> float:
        phase_logits = params.reshape(n_phases, n_domains)
        phase_weights = _softmax(phase_logits)
        predicted_negated_objective = float(fit.predict(phase_weights[None, :, :])[0])
        return predicted_negated_objective + lambda_kl * _weighted_multiclass_kl(
            phase_weights,
            natural_proportions,
            phase_fractions,
        )

    best_result = None
    for start in starts:
        result = minimize(objective, np.asarray(start, dtype=float), method="L-BFGS-B")
        if not result.success and best_result is not None:
            continue
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Olmix schedule solve failed")

    best_weights = _softmax(np.asarray(best_result.x, dtype=float).reshape(n_phases, n_domains))
    predicted_negated_objective = float(fit.predict(best_weights[None, :, :])[0])
    return (
        _phase_weights_from_matrix(best_weights, phase_names=phase_names, domain_names=domain_names),
        predicted_negated_objective,
        float(best_result.fun),
    )


def fit_olmix_sl_verb_from_frame(
    frame: pd.DataFrame,
    *,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
    objective_metric: str = OBJECTIVE_METRIC,
    source_results_path: str = "<in-memory>",
) -> OlmixSlVerbFitResult:
    """Fit the SL-Verb Olmix baseline from a long-form results frame."""
    fit_frame = _negated_objective_frame(frame, objective_metric=objective_metric)
    spec = build_dataset_spec_from_frame(
        fit_frame,
        objective_metric=NEGATED_OBJECTIVE_METRIC,
        name="two_phase_many_mmlu_sl_verb",
    )
    fit = fit_olmix_loglinear_model(spec.weights, spec.y)
    phase_weights, predicted_negated_objective, regularized_objective = solve_olmix_loglinear_schedule(
        fit,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    return OlmixSlVerbFitResult(
        objective_metric=objective_metric,
        negated_objective_metric=NEGATED_OBJECTIVE_METRIC,
        source_results_path=source_results_path,
        fit_log_c=fit.log_c,
        fit_coefficients=list(fit.coefficients),
        fit_huber_loss=fit.huber_loss,
        kl_lambda=OLMIX_LOGLINEAR_KL_LAMBDA,
        predicted_negated_objective=predicted_negated_objective,
        predicted_choice_logprob_norm=-predicted_negated_objective,
        regularized_objective=regularized_objective,
        phase_weights=phase_weights,
    )


def load_fit_from_results(
    *,
    source_experiment: str = SOURCE_RESULTS_EXPERIMENT,
    results_root: str = DEFAULT_RESULTS_ROOT,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
    candidate_group_column: str | None = None,
) -> OlmixSlVerbFitResult:
    """Resolve the swarm SL-Verb results CSV and fit the new Olmix baseline from it."""
    results_path = resolve_unique_results_csv(source_experiment=source_experiment, results_root=results_root)
    frame = load_results_frame(results_path)
    source_results_path = results_path
    if candidate_group_column is not None:
        frame = aggregate_candidate_mean_frame(frame, candidate_group_column=candidate_group_column)
        source_results_path = f"{results_path}::{candidate_group_column}_means"
    return fit_olmix_sl_verb_from_frame(
        frame,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        source_results_path=source_results_path,
    )


def fit_summary_json(fit: OlmixSlVerbFitResult) -> str:
    """Serialize one fit summary as stable JSON."""
    return json.dumps(fit.__dict__, indent=2, sort_keys=True)
