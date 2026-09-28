# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additional baseline generators for the StarCoder phase-mix experiments.

This module adds:
- token-proportional schedules based on corpus sizes, and
- Olmix-style log-linear schedules with the paper's recommended KL-regularized
  optimization objective.

The Olmix adaptation here operates on the per-phase StarCoder proportion for
the fixed two-domain setting. Each phase is therefore parameterized by a single
scalar in ``[0, 1]`` because the Nemotron weight is determined by
``1 - p_starcoder``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.domains import NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN
from experiments.domain_phase_mix.three_phase_starcoder_experiment import PHASE_BOUNDARIES as THREE_PHASE_BOUNDARIES
from experiments.domain_phase_mix.three_phase_starcoder_experiment import TARGET_BUDGET as THREE_PHASE_TARGET_BUDGET
from experiments.domain_phase_mix.two_phase_starcoder_experiment import PHASE_BOUNDARIES as TWO_PHASE_BOUNDARIES
from experiments.domain_phase_mix.two_phase_starcoder_experiment import TARGET_BUDGET as TWO_PHASE_TARGET_BUDGET

TopologyName = Literal["two_phase_starcoder", "three_phase_starcoder"]

SCRIPT_DIR = Path(__file__).resolve().parent
OBJECTIVE_COLUMN = "eval/paloma/dolma_100_programing_languages/bpb"
DEFAULT_OLMIX_KL_LAMBDA = 0.05
DEFAULT_HUBER_DELTA = 0.02
DEFAULT_REPETITION_FACTOR = 4.0

NATURAL_STARCODER_PROPORTION = STARCODER_DOMAIN.total_weight / (
    NEMOTRON_FULL_DOMAIN.total_weight + STARCODER_DOMAIN.total_weight
)


@dataclass(frozen=True)
class StarcoderBaseline:
    """One launch-ready additional baseline."""

    topology: TopologyName
    label: str
    run_id: int
    run_name: str
    phase_starcoder_weights: tuple[float, ...]
    predicted_objective: float | None = None

    def to_weight_config(self) -> WeightConfig:
        phase_weights = {
            f"phase_{phase_index}": {
                "nemotron_full": 1.0 - starcoder_weight,
                "starcoder": starcoder_weight,
            }
            for phase_index, starcoder_weight in enumerate(self.phase_starcoder_weights)
        }
        return WeightConfig(run_id=self.run_id, phase_weights=phase_weights)


@dataclass(frozen=True)
class OlmixFit:
    """Fitted single-task Olmix surrogate ``exp(log_c) + exp(a^T x)``."""

    log_c: float
    coefficients: tuple[float, ...]
    huber_loss: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        features = np.asarray(x, dtype=float)
        logits = np.clip(features @ np.asarray(self.coefficients, dtype=float), -50.0, 50.0)
        return np.exp(self.log_c) + np.exp(logits)


@dataclass(frozen=True)
class TopologyConfig:
    topology: TopologyName
    csv_path: Path
    phase_fractions: tuple[float, ...]
    target_budget: int
    baseline_run_ids: dict[str, int]

    @property
    def n_phases(self) -> int:
        return len(self.phase_fractions)

    @property
    def max_starcoder_share_k4(self) -> float:
        return min(1.0, DEFAULT_REPETITION_FACTOR * STARCODER_DOMAIN.total_weight / self.target_budget)


TOPOLOGY_CONFIGS: dict[TopologyName, TopologyConfig] = {
    "two_phase_starcoder": TopologyConfig(
        topology="two_phase_starcoder",
        csv_path=SCRIPT_DIR / "exploratory" / "two_phase_starcoder_combined.csv",
        phase_fractions=(TWO_PHASE_BOUNDARIES[0], 1.0 - TWO_PHASE_BOUNDARIES[0]),
        target_budget=TWO_PHASE_TARGET_BUDGET,
        baseline_run_ids={
            "proportional": 97001,
            "olmix_unconstrained": 97002,
            "olmix_k4": 97003,
        },
    ),
    "three_phase_starcoder": TopologyConfig(
        topology="three_phase_starcoder",
        csv_path=SCRIPT_DIR / "exploratory" / "three_phase_starcoder.csv",
        phase_fractions=(
            THREE_PHASE_BOUNDARIES[0],
            THREE_PHASE_BOUNDARIES[1] - THREE_PHASE_BOUNDARIES[0],
            1.0 - THREE_PHASE_BOUNDARIES[1],
        ),
        target_budget=THREE_PHASE_TARGET_BUDGET,
        baseline_run_ids={
            "proportional": 98001,
            "olmix_unconstrained": 98002,
            "olmix_k4": 98003,
        },
    ),
}


def get_topology_config(topology: TopologyName) -> TopologyConfig:
    return TOPOLOGY_CONFIGS[topology]


def load_starcoder_dataset(topology: TopologyName) -> pd.DataFrame:
    """Load completed runs for the requested StarCoder topology."""

    config = get_topology_config(topology)
    frame = pd.read_csv(config.csv_path)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].copy()
    if OBJECTIVE_COLUMN not in frame.columns:
        raise ValueError(f"Missing objective column {OBJECTIVE_COLUMN!r} in {config.csv_path}")
    return frame


def _phase_feature_columns(n_phases: int) -> list[str]:
    return [f"phase_{phase_index}_starcoder" for phase_index in range(n_phases)]


def _huber_sum(residuals: np.ndarray, delta: float) -> float:
    abs_residuals = np.abs(residuals)
    quadratic = 0.5 * residuals * residuals
    linear = delta * (abs_residuals - 0.5 * delta)
    return float(np.where(abs_residuals <= delta, quadratic, linear).sum())


def fit_olmix_loglinear(
    topology: TopologyName,
    *,
    objective_column: str = OBJECTIVE_COLUMN,
    delta: float = DEFAULT_HUBER_DELTA,
    seed: int = 0,
    n_starts: int = 360,
) -> OlmixFit:
    """Fit the single-task Olmix log-linear surrogate for one topology."""

    frame = load_starcoder_dataset(topology)
    config = get_topology_config(topology)
    x = frame[_phase_feature_columns(config.n_phases)].to_numpy(dtype=float)
    y = frame[objective_column].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    best_result = None
    starts: list[np.ndarray] = []
    for log_c in np.linspace(-3.0, 0.0, 12):
        for _ in range(max(n_starts // 12, 1)):
            starts.append(np.concatenate([[log_c], rng.normal(0.0, 2.0, size=config.n_phases)]))

    def loss(params: np.ndarray) -> float:
        log_c = float(params[0])
        coeffs = params[1:]
        predictions = np.exp(log_c) + np.exp(np.clip(x @ coeffs, -50.0, 50.0))
        return _huber_sum(predictions - y, delta=delta)

    for start in starts:
        result = minimize(loss, start, method="L-BFGS-B")
        if not result.success and best_result is not None:
            continue
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError(f"Olmix fit failed for topology={topology}")

    return OlmixFit(
        log_c=float(best_result.x[0]),
        coefficients=tuple(float(value) for value in best_result.x[1:]),
        huber_loss=float(best_result.fun),
    )


def _weighted_binary_kl(z: np.ndarray, p0: float, phase_fractions: np.ndarray) -> float:
    eps = 1e-9
    z = np.clip(z, eps, 1.0 - eps)
    p0 = float(np.clip(p0, eps, 1.0 - eps))
    term = z * np.log(z / p0) + (1.0 - z) * np.log((1.0 - z) / (1.0 - p0))
    return float(phase_fractions @ term)


def solve_olmix_schedule(
    topology: TopologyName,
    fit: OlmixFit,
    *,
    lambda_kl: float = DEFAULT_OLMIX_KL_LAMBDA,
    repetition_factor: float | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Solve the KL-regularized Olmix objective for a multi-phase schedule."""

    config = get_topology_config(topology)
    phase_fractions = np.asarray(config.phase_fractions, dtype=float)
    natural = NATURAL_STARCODER_PROPORTION
    bounds = [(1e-6, 1.0 - 1e-6)] * config.n_phases
    rng = np.random.default_rng(seed)

    def objective(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=float)
        prediction = float(fit.predict(z[None, :])[0])
        return prediction + lambda_kl * _weighted_binary_kl(z, natural, phase_fractions)

    starts = [
        np.full(config.n_phases, natural),
        np.full(config.n_phases, 1e-6),
        np.full(config.n_phases, min(0.1, natural * 2.0 + 1e-3)),
        np.linspace(natural, natural * 2.0, config.n_phases),
    ]
    starts.extend(rng.uniform(1e-6, 0.25, size=(8, config.n_phases)))

    best_result = None
    if repetition_factor is None:
        for start in starts:
            result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
            if not result.success and best_result is not None:
                continue
            if best_result is None or result.fun < best_result.fun:
                best_result = result
    else:
        max_average_share = min(1.0, repetition_factor * STARCODER_DOMAIN.total_weight / config.target_budget)
        constraints = [
            {
                "type": "ineq",
                "fun": lambda z, ma=max_average_share, fractions=phase_fractions: ma - float(fractions @ z),
            }
        ]
        for start in starts:
            if float(phase_fractions @ start) > max_average_share:
                start = np.full(config.n_phases, min(natural, max_average_share))
            result = minimize(objective, start, method="SLSQP", bounds=bounds, constraints=constraints)
            if not result.success and best_result is not None:
                continue
            if best_result is None or result.fun < best_result.fun:
                best_result = result

    if best_result is None:
        raise RuntimeError(f"Olmix optimization failed for topology={topology}")

    schedule = np.clip(best_result.x.astype(float), 1e-6, 1.0 - 1e-6)
    return schedule, float(fit.predict(schedule[None, :])[0])


def compute_additional_baselines(
    topology: TopologyName,
    *,
    lambda_kl: float = DEFAULT_OLMIX_KL_LAMBDA,
    repetition_factor: float = DEFAULT_REPETITION_FACTOR,
) -> list[StarcoderBaseline]:
    """Return proportional + Olmix additional baselines for one topology."""

    config = get_topology_config(topology)
    fit = fit_olmix_loglinear(topology)
    unconstrained, unconstrained_prediction = solve_olmix_schedule(
        topology,
        fit,
        lambda_kl=lambda_kl,
        repetition_factor=None,
    )
    k4_schedule, k4_prediction = solve_olmix_schedule(
        topology,
        fit,
        lambda_kl=lambda_kl,
        repetition_factor=repetition_factor,
    )

    proportional_schedule = tuple(float(NATURAL_STARCODER_PROPORTION) for _ in range(config.n_phases))
    baselines = [
        StarcoderBaseline(
            topology=topology,
            label="proportional",
            run_id=config.baseline_run_ids["proportional"],
            run_name="proportional",
            phase_starcoder_weights=proportional_schedule,
        ),
        StarcoderBaseline(
            topology=topology,
            label="olmix_unconstrained",
            run_id=config.baseline_run_ids["olmix_unconstrained"],
            run_name="olmix_unconstrained",
            phase_starcoder_weights=tuple(float(value) for value in unconstrained),
            predicted_objective=unconstrained_prediction,
        ),
        StarcoderBaseline(
            topology=topology,
            label="olmix_k4",
            run_id=config.baseline_run_ids["olmix_k4"],
            run_name="olmix_k4",
            phase_starcoder_weights=tuple(float(value) for value in k4_schedule),
            predicted_objective=k4_prediction,
        ),
    ]
    return baselines
