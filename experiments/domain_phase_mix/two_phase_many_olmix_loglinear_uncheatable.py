# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit an Olmix log-linear baseline on two-phase-many Uncheatable BPB results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.static_batch_selection import (
    average_phase_tv_distance,
    build_dataset_spec_from_frame,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import TWO_PHASE_MANY_CSV_PATH
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear import OLMIX_LOGLINEAR_KL_LAMBDA
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    fit_olmix_loglinear_model,
    solve_olmix_loglinear_schedule,
)

RUN_ID = 248
RUN_NAME = "baseline_olmix_loglinear_uncheatable_bpb"
SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_uncheatable_bpb"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
FIT_SUMMARY_JSON = "olmix_uncheatable_fit_summary.json"
SOURCE_RESULTS_PATH = TWO_PHASE_MANY_CSV_PATH


@dataclass(frozen=True)
class OlmixUncheatableFitResult:
    """Summary of the fitted Uncheatable-BPB Olmix baseline."""

    objective_metric: str
    source_results_path: str
    fit_log_c: float
    fit_coefficients: list[float]
    fit_huber_loss: float
    kl_lambda: float
    predicted_objective: float
    regularized_objective: float
    best_observed_run_name: str
    best_observed_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    phase_weights: dict[str, dict[str, float]]


def load_results_frame(results_path: str | Path = SOURCE_RESULTS_PATH) -> pd.DataFrame:
    """Load the historical two-phase-many swarm table for Olmix fitting."""
    frame = pd.read_csv(results_path)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    if OBJECTIVE_METRIC not in frame.columns:
        raise ValueError(f"Missing {OBJECTIVE_METRIC!r} in {results_path}")
    return frame


def _run_name_column(frame: pd.DataFrame) -> str:
    if "candidate_run_name" in frame.columns:
        return "candidate_run_name"
    if "run_name" in frame.columns:
        return "run_name"
    raise ValueError("No run-name column present in frame")


def _phase_weight_matrix(
    phase_weights: dict[str, dict[str, float]],
    *,
    phase_names: list[str],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [[float(phase_weights[phase_name][domain_name]) for domain_name in domain_names] for phase_name in phase_names],
        dtype=float,
    )


def fit_olmix_uncheatable_from_frame(
    frame: pd.DataFrame,
    *,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
    objective_metric: str = OBJECTIVE_METRIC,
    source_results_path: str = "<in-memory>",
) -> OlmixUncheatableFitResult:
    """Fit the Uncheatable-BPB Olmix baseline from a flat run table."""
    fit_frame = frame[frame[objective_metric].notna()].reset_index(drop=True)
    spec = build_dataset_spec_from_frame(
        fit_frame,
        objective_metric=objective_metric,
        name="two_phase_many_uncheatable_bpb",
    )
    fit = fit_olmix_loglinear_model(spec.weights, spec.y)
    phase_weights, predicted_objective, regularized_objective = solve_olmix_loglinear_schedule(
        fit,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
        lambda_kl=OLMIX_LOGLINEAR_KL_LAMBDA,
    )
    optimum_weights = _phase_weight_matrix(
        phase_weights,
        phase_names=spec.phase_names,
        domain_names=spec.domain_names,
    )
    distances = average_phase_tv_distance(spec.weights, optimum_weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    best_idx = int(np.argmin(spec.y))
    name_column = _run_name_column(fit_frame)
    return OlmixUncheatableFitResult(
        objective_metric=objective_metric,
        source_results_path=str(source_results_path),
        fit_log_c=fit.log_c,
        fit_coefficients=list(fit.coefficients),
        fit_huber_loss=fit.huber_loss,
        kl_lambda=OLMIX_LOGLINEAR_KL_LAMBDA,
        predicted_objective=predicted_objective,
        regularized_objective=regularized_objective,
        best_observed_run_name=str(fit_frame.iloc[best_idx][name_column]),
        best_observed_value=float(spec.y[best_idx]),
        nearest_observed_run_name=str(fit_frame.iloc[nearest_idx][name_column]),
        nearest_observed_value=float(spec.y[nearest_idx]),
        nearest_observed_tv_distance=float(distances[nearest_idx]),
        phase_weights=phase_weights,
    )


def load_fit_from_local_results(
    *,
    results_path: str | Path = SOURCE_RESULTS_PATH,
    natural_proportions: np.ndarray,
    phase_fractions: np.ndarray,
) -> OlmixUncheatableFitResult:
    """Fit the new Olmix baseline from the local original-swarm CSV."""
    frame = load_results_frame(results_path)
    return fit_olmix_uncheatable_from_frame(
        frame,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        source_results_path=str(results_path),
    )


def fit_summary_json(fit: OlmixUncheatableFitResult) -> str:
    """Serialize one fit summary as stable JSON."""
    return json.dumps(fit.__dict__, indent=2, sort_keys=True)
