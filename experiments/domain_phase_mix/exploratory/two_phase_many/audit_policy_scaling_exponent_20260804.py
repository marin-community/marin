# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "scipy>=1.14",
#   "tabulate>=0.9",
# ]
# ///

"""Audit policy-conditioned token-scaling exponents on the WSD80 increase-D track."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE
    / "reference_outputs"
    / "starcoder_wsd80_matched_nd_stage1_20260731"
    / "stage3_dense_surface_results_20260802"
    / "combined_discovery_observations.csv"
)
DEFAULT_OUTPUT = HERE / "reference_outputs" / "policy_scaling_exponent_audit_20260804"
FLOOR_GRID = (0.0, 0.50, 0.60, 0.65, 0.70, 0.74)
PARAMETER_BOOTSTRAP_DRAWS = 1000
PAIRED_BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260804
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
FLOOR_MARGIN = 1e-4
MODEL_LADDER = ("shared", "aggregate", "recency", "policy_floor", "free_floor")
FULL_FIT_MODELS = (*MODEL_LADDER, "free_policy")


@dataclass(frozen=True)
class ScalingFit:
    """A fitted cross-rung scaling model."""

    model: str
    floors: np.ndarray
    floor_cap: float
    floor_coefficients: np.ndarray
    gammas: np.ndarray
    gamma_0: float
    gamma_1: float
    gamma_1_se: float
    phi: float
    amplitudes: np.ndarray
    rmse: float
    objective: float
    floor_bound_fraction: float


def coordinate_key(frame: pd.DataFrame) -> pd.Series:
    return frame["phase_0_starcoder"].round(8).astype(str) + ":" + frame["phase_1_starcoder"].round(8).astype(str)


def load_common_panel(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["track_memberships"].str.contains("increase_d", regex=False)].copy()
    frame["coordinate"] = coordinate_key(frame)
    complete = frame.groupby("coordinate")["cell_id"].nunique()
    frame = frame[frame["coordinate"].isin(complete[complete == 4].index)].copy()
    frame["aggregate"] = PHASE_0_FRACTION * frame["phase_0_starcoder"] + PHASE_1_FRACTION * frame["phase_1_starcoder"]
    frame = frame.sort_values(["coordinate", "materialized_tokens"]).reset_index(drop=True)
    counts = frame.groupby("coordinate").size()
    if not (counts == 4).all():
        raise ValueError("Every retained policy must have exactly four token rungs")
    return frame


def panel_subset(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "stage1":
        return frame[frame["source_stage"] == "stage1"].copy()
    if name == "stage3":
        return frame[frame["source_stage"] == "stage3"].copy()
    if name == "all_common":
        return frame.copy()
    raise ValueError(f"Unknown panel {name}")


def policy_arrays(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    policies = (
        frame[["coordinate", "phase_0_starcoder", "phase_1_starcoder", "aggregate", "source_stage"]]
        .drop_duplicates("coordinate")
        .sort_values("coordinate")
        .reset_index(drop=True)
    )
    rung_tokens = np.sort(frame["materialized_tokens"].unique())
    if len(rung_tokens) != 4:
        raise ValueError(f"Expected four token rungs, found {len(rung_tokens)}")
    pivot = frame.pivot(index="coordinate", columns="materialized_tokens", values="starcoder_bpb")
    pivot = pivot.reindex(index=policies["coordinate"], columns=rung_tokens)
    if pivot.isna().any().any():
        raise ValueError("Common-coordinate panel has missing outcomes")
    ratios = rung_tokens / rung_tokens.min()
    return policies, ratios.astype(float), pivot.to_numpy(dtype=float)


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def logit(value: float) -> float:
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    return float(np.log(clipped / (1.0 - clipped)))


def initial_floor(outcomes: np.ndarray) -> float:
    return max(0.0, float(outcomes.min()) - 0.08)


def recency_gammas(
    gamma_0: float,
    early_coefficient: float,
    late_coefficient: float,
    policies: pd.DataFrame,
) -> np.ndarray:
    return (
        gamma_0
        + early_coefficient * policies["phase_0_starcoder"].to_numpy(dtype=float)
        + late_coefficient * policies["phase_1_starcoder"].to_numpy(dtype=float)
    )


def gamma_for_policies(fit: ScalingFit, policies: pd.DataFrame) -> np.ndarray:
    if fit.model in {"shared", "policy_floor", "free_floor"}:
        return np.full(len(policies), fit.gamma_0)
    if fit.model == "aggregate":
        return fit.gamma_0 + fit.gamma_1 * policies["aggregate"].to_numpy(dtype=float)
    if fit.model == "recency":
        early, late = fit.floor_coefficients[-2:]
        return recency_gammas(fit.gamma_0, early, late, policies)
    raise ValueError(f"Model {fit.model} does not transfer gamma to unseen policies")


def floor_for_policies(fit: ScalingFit, policies: pd.DataFrame) -> np.ndarray:
    if fit.model in {"shared", "aggregate", "recency"}:
        return np.full(len(policies), fit.floors[0])
    if fit.model == "policy_floor":
        intercept, early, late = fit.floor_coefficients[:3]
        linear = (
            intercept
            + early * policies["phase_0_starcoder"].to_numpy(dtype=float)
            + late * policies["phase_1_starcoder"].to_numpy(dtype=float)
        )
        return fit.floor_cap * sigmoid(linear)
    raise ValueError(f"Model {fit.model} does not transfer floor to unseen policies")


def fit_scaling_model(
    policies: pd.DataFrame,
    token_ratios: np.ndarray,
    outcomes: np.ndarray,
    model: str,
) -> ScalingFit:
    """Fit one frozen scaling-law member with deterministic multistart."""

    policy_count = len(policies)
    floor_cap = float(outcomes.min()) - FLOOR_MARGIN
    floor_start = initial_floor(outcomes)
    floor_fraction = floor_start / floor_cap
    log_amplitude_start = np.log(np.maximum(outcomes[:, 0] - floor_start, 1e-3))
    metadata: dict[str, slice | int] = {}

    if model in {"shared", "aggregate", "recency"}:
        values = [floor_start, *log_amplitude_start, 0.1]
        lower = [0.0, *([-12.0] * policy_count), 0.0]
        upper = [floor_cap, *([3.0] * policy_count), 1.0]
        metadata["shared_floor"] = 0
        metadata["amplitude"] = slice(1, 1 + policy_count)
        metadata["gamma_0"] = 1 + policy_count
        if model == "aggregate":
            metadata["gamma_1"] = len(values)
            values.append(0.05)
            lower.append(-1.0)
            upper.append(1.0)
        elif model == "recency":
            metadata["early_gamma"] = len(values)
            metadata["late_gamma"] = len(values) + 1
            values.extend([0.025, 0.025])
            lower.extend([0.0, 0.0])
            upper.extend([1.0, 1.0])
    elif model == "policy_floor":
        values = [logit(floor_fraction), 0.0, 0.0, *log_amplitude_start, 0.1]
        lower = [-10.0, -10.0, -10.0, *([-12.0] * policy_count), 0.0]
        upper = [10.0, 10.0, 10.0, *([3.0] * policy_count), 1.0]
        metadata["floor_coefficients"] = slice(0, 3)
        metadata["amplitude"] = slice(3, 3 + policy_count)
        metadata["gamma_0"] = 3 + policy_count
    elif model == "free_floor":
        values = [*([logit(floor_fraction)] * policy_count), *log_amplitude_start, 0.1]
        lower = [*([-10.0] * policy_count), *([-12.0] * policy_count), 0.0]
        upper = [*([10.0] * policy_count), *([3.0] * policy_count), 1.0]
        metadata["floor_logits"] = slice(0, policy_count)
        metadata["amplitude"] = slice(policy_count, 2 * policy_count)
        metadata["gamma_0"] = 2 * policy_count
    elif model == "free_policy":
        values = [floor_start, *log_amplitude_start, *([0.1] * policy_count)]
        lower = [0.0, *([-12.0] * policy_count), *([0.0] * policy_count)]
        upper = [floor_cap, *([3.0] * policy_count), *([1.0] * policy_count)]
        metadata["shared_floor"] = 0
        metadata["amplitude"] = slice(1, 1 + policy_count)
        metadata["free_gamma"] = slice(1 + policy_count, 1 + 2 * policy_count)
    else:
        raise ValueError(f"Unknown model {model}")

    x0 = np.asarray(values, dtype=float)
    bounds = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, np.ndarray]:
        amplitudes = np.exp(params[metadata["amplitude"]])
        coefficients = np.zeros(3, dtype=float)
        gamma_1 = 0.0
        phi = float("nan")
        if "shared_floor" in metadata:
            floors = np.full(policy_count, params[metadata["shared_floor"]])
        elif "floor_coefficients" in metadata:
            coefficients = params[metadata["floor_coefficients"]]
            linear = (
                coefficients[0]
                + coefficients[1] * policies["phase_0_starcoder"].to_numpy(dtype=float)
                + coefficients[2] * policies["phase_1_starcoder"].to_numpy(dtype=float)
            )
            floors = floor_cap * sigmoid(linear)
        else:
            floors = floor_cap * sigmoid(params[metadata["floor_logits"]])

        if "free_gamma" in metadata:
            gammas = params[metadata["free_gamma"]]
            gamma_0 = float("nan")
        else:
            gamma_0 = float(params[metadata["gamma_0"]])
            if model == "aggregate":
                gamma_1 = float(params[metadata["gamma_1"]])
                gammas = gamma_0 + gamma_1 * policies["aggregate"].to_numpy(dtype=float)
            elif model == "recency":
                early = float(params[metadata["early_gamma"]])
                late = float(params[metadata["late_gamma"]])
                coefficients[-2:] = (early, late)
                gamma_1 = early + late
                phi = late / gamma_1 if gamma_1 > 1e-12 else float("nan")
                gammas = recency_gammas(gamma_0, early, late, policies)
            else:
                gammas = np.full(policy_count, gamma_0)
        return floors, amplitudes, gammas, gamma_0, gamma_1, phi, coefficients

    def residual(params: np.ndarray) -> np.ndarray:
        floors, amplitudes, gammas, _, _, _, _ = unpack(params)
        prediction = floors[:, None] + amplitudes[:, None] * token_ratios[None, :] ** (-gammas[:, None])
        return (prediction - outcomes).ravel()

    starts = [x0]
    if model == "recency":
        for early, late in ((0.0, 0.08), (0.08, 0.0), (0.04, 0.04)):
            start = x0.copy()
            start[metadata["early_gamma"]] = early
            start[metadata["late_gamma"]] = late
            starts.append(start)
    elif model == "policy_floor":
        for early, late in ((-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)):
            start = x0.copy()
            start[1:3] = (early, late)
            starts.append(start)
    elif model == "free_floor":
        for fraction in (0.2, 0.5, 0.8):
            start = x0.copy()
            start[metadata["floor_logits"]] = logit(fraction)
            starts.append(start)

    best_result = None
    best_objective = float("inf")
    for start in starts:
        result = least_squares(residual, x0=start, bounds=bounds, max_nfev=50_000)
        objective = float(np.sum(residual(result.x) ** 2))
        if np.isfinite(objective) and objective < best_objective:
            best_result = result
            best_objective = objective
    if best_result is None:
        raise RuntimeError(f"{model} produced no finite scaling fit")

    floors, amplitudes, gammas, gamma_0, gamma_1, phi, coefficients = unpack(best_result.x)
    gamma_1_se = float("inf")
    if model == "recency":
        early_index = int(metadata["early_gamma"])
        late_index = int(metadata["late_gamma"])
        degrees = max(len(residual(best_result.x)) - len(best_result.x), 1)
        covariance = np.linalg.pinv(best_result.jac.T @ best_result.jac) * (best_objective / degrees)
        variance = (
            covariance[early_index, early_index]
            + covariance[late_index, late_index]
            + 2.0 * covariance[early_index, late_index]
        )
        gamma_1_se = float(np.sqrt(max(variance, 0.0)))

    bound_fraction = float(np.mean((floors <= FLOOR_MARGIN) | (floors >= floor_cap - 10 * FLOOR_MARGIN)))
    return ScalingFit(
        model=model,
        floors=floors,
        floor_cap=floor_cap,
        floor_coefficients=coefficients,
        gammas=gammas,
        gamma_0=gamma_0,
        gamma_1=gamma_1,
        gamma_1_se=gamma_1_se,
        phi=phi,
        amplitudes=amplitudes,
        rmse=float(np.sqrt(best_objective / outcomes.size)),
        objective=best_objective,
        floor_bound_fraction=bound_fraction,
    )


def fit_amplitudes(floors: np.ndarray, gammas: np.ndarray, ratios: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    basis = ratios[None, :] ** (-gammas[:, None])
    numerator = np.sum(basis * (outcomes - floors[:, None]), axis=1)
    denominator = np.sum(basis**2, axis=1)
    return np.maximum(numerator / denominator, 0.0)


def fit_held_free_floor(gamma: float, ratios: np.ndarray, outcomes: np.ndarray) -> tuple[float, float]:
    basis = ratios ** (-gamma)
    design = np.column_stack([np.ones(len(ratios)), basis])
    upper_floor = float(outcomes.min()) - FLOOR_MARGIN
    result = lsq_linear(design, outcomes, bounds=([0.0, 0.0], [upper_floor, np.inf]))
    if not result.success:
        raise RuntimeError(f"Held-policy floor fit failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def leave_rung_out(policies: pd.DataFrame, ratios: np.ndarray, outcomes: np.ndarray, model: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for held_rung in range(len(ratios)):
        keep = np.arange(len(ratios)) != held_rung
        fit = fit_scaling_model(policies, ratios[keep], outcomes[:, keep], model)
        prediction = fit.floors + fit.amplitudes * ratios[held_rung] ** (-fit.gammas)
        for policy_index, policy in policies.iterrows():
            rows.append(
                {
                    "model": model,
                    "held_rung": held_rung,
                    "coordinate": policy["coordinate"],
                    "observed": outcomes[policy_index, held_rung],
                    "predicted": prediction[policy_index],
                    "residual": prediction[policy_index] - outcomes[policy_index, held_rung],
                    "floor": fit.floors[policy_index],
                    "floor_bound_fraction": fit.floor_bound_fraction,
                    "gamma_0": fit.gamma_0,
                    "gamma_1": fit.gamma_1,
                    "phi": fit.phi,
                }
            )
    return pd.DataFrame(rows)


def leave_policy_out_highest_rung(
    policies: pd.DataFrame,
    ratios: np.ndarray,
    outcomes: np.ndarray,
    model: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for held_policy in range(len(policies)):
        keep = np.arange(len(policies)) != held_policy
        fit = fit_scaling_model(policies.iloc[keep].reset_index(drop=True), ratios, outcomes[keep], model)
        held = policies.iloc[[held_policy]].reset_index(drop=True)
        gamma = gamma_for_policies(fit, held)
        if model == "free_floor":
            floor, amplitude = fit_held_free_floor(gamma[0], ratios[:3], outcomes[held_policy, :3])
        else:
            floor = float(floor_for_policies(fit, held)[0])
            amplitude = float(
                fit_amplitudes(
                    np.array([floor]),
                    gamma,
                    ratios[:3],
                    outcomes[held_policy : held_policy + 1, :3],
                )[0]
            )
        prediction = floor + amplitude * ratios[3] ** (-gamma[0])
        rows.append(
            {
                "model": model,
                "coordinate": held.iloc[0]["coordinate"],
                "observed": outcomes[held_policy, 3],
                "predicted": prediction,
                "residual": prediction - outcomes[held_policy, 3],
                "floor": floor,
                "floor_bound_fraction": fit.floor_bound_fraction,
                "gamma_0": fit.gamma_0,
                "gamma_1": fit.gamma_1,
                "phi": fit.phi,
            }
        )
    return pd.DataFrame(rows)


def transformed_policy_exponents(
    policies: pd.DataFrame,
    ratios: np.ndarray,
    outcomes: np.ndarray,
    floor: float,
) -> pd.DataFrame:
    if np.any(outcomes <= floor):
        raise ValueError(f"Floor {floor} is not below all outcomes")
    x = np.log(ratios)
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    records: list[dict[str, float | str | bool]] = []
    for index, policy in policies.iterrows():
        y = np.log(outcomes[index] - floor)
        slope = float(np.sum(x_centered * (y - y.mean())) / denominator)
        fitted = y.mean() + slope * x_centered
        residual = y - fitted
        variance = float(np.sum(residual**2) / 2.0)
        records.append(
            {
                "coordinate": policy["coordinate"],
                "phase_0_starcoder": policy["phase_0_starcoder"],
                "phase_1_starcoder": policy["phase_1_starcoder"],
                "aggregate": policy["aggregate"],
                "source_stage": policy["source_stage"],
                "floor": floor,
                "gamma": -slope,
                "gamma_se": float(np.sqrt(variance / denominator)),
                "tied": bool(abs(policy["phase_0_starcoder"] - policy["phase_1_starcoder"]) <= 1e-10),
            }
        )
    return pd.DataFrame(records)


def best_phi(exponents: pd.DataFrame) -> dict[str, float]:
    gamma = exponents["gamma"].to_numpy(dtype=float)
    best: dict[str, float] | None = None
    for phi in np.linspace(0.0, 1.0, 101):
        state = (1.0 - phi) * exponents["phase_0_starcoder"].to_numpy(dtype=float) + phi * exponents[
            "phase_1_starcoder"
        ].to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(state)), state])
        coefficients, _, _, _ = np.linalg.lstsq(design, gamma, rcond=None)
        prediction = design @ coefficients
        residual = gamma - prediction
        total = gamma - gamma.mean()
        total_sum = float(np.sum(total**2))
        r_squared = float("nan") if total_sum == 0 else 1.0 - float(np.sum(residual**2) / total_sum)
        candidate = {
            "phi": float(phi),
            "gamma_0": float(coefficients[0]),
            "gamma_1": float(coefficients[1]),
            "r_squared": r_squared,
            "spearman": float(spearmanr(state, gamma).statistic),
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
        }
        if best is None or candidate["r_squared"] > best["r_squared"]:
            best = candidate
    if best is None:
        raise AssertionError("Phi scan produced no candidate")
    return best


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def fixed_floor_audit(panel: str, policies: pd.DataFrame, ratios: np.ndarray, outcomes: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for floor in FLOOR_GRID:
        exponents = transformed_policy_exponents(policies, ratios, outcomes, floor)
        phi = best_phi(exponents)
        tied = exponents[exponents["tied"]]
        untied = exponents[~exponents["tied"]]
        rows.append(
            {
                "panel": panel,
                "floor": floor,
                "floor_margin": float(outcomes.min() - floor),
                "aggregate_spearman_all": safe_spearman(exponents["aggregate"], exponents["gamma"]),
                "aggregate_spearman_tied": safe_spearman(tied["aggregate"], tied["gamma"]),
                "aggregate_spearman_untied": safe_spearman(untied["aggregate"], untied["gamma"]),
                **phi,
            }
        )
    return pd.DataFrame(rows)


def parameter_bootstrap(
    policies: pd.DataFrame,
    ratios: np.ndarray,
    outcomes: np.ndarray,
    draws: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    records: list[dict[str, float | int | bool]] = []
    for draw in range(draws):
        indices = rng.integers(0, len(policies), size=len(policies))
        sampled_policies = policies.iloc[indices].reset_index(drop=True)
        sampled_outcomes = outcomes[indices]
        fit = fit_scaling_model(sampled_policies, ratios, sampled_outcomes, "recency")
        identified = bool(fit.gamma_1 > 2.0 * fit.gamma_1_se)
        records.append(
            {
                "draw": draw,
                "gamma_1": fit.gamma_1,
                "gamma_1_se": fit.gamma_1_se,
                "identified": identified,
                "phi": fit.phi if identified else float("nan"),
                "untied_count": int(
                    np.sum(
                        np.abs(
                            sampled_policies["phase_0_starcoder"].to_numpy()
                            - sampled_policies["phase_1_starcoder"].to_numpy()
                        )
                        > 1e-10
                    )
                ),
                "train_rmse": fit.rmse,
            }
        )
    return pd.DataFrame(records)


def paired_cv_bootstrap(predictions: pd.DataFrame, draws: int) -> pd.DataFrame:
    pivot = predictions.pivot(index="coordinate", columns="model", values="residual")
    required = {"recency", "shared", "aggregate", "policy_floor", "free_floor"}
    if not required.issubset(pivot.columns):
        raise ValueError(f"Missing paired models: {sorted(required - set(pivot.columns))}")
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    models = list(sorted(required))
    values = pivot[models].to_numpy(dtype=float)
    recency_index = models.index("recency")
    records: list[dict[str, float | int]] = []
    for draw in range(draws):
        indices = rng.integers(0, len(values), size=len(values))
        rmses = np.sqrt(np.mean(values[indices] ** 2, axis=0))
        record: dict[str, float | int] = {"draw": draw}
        for model_index, model in enumerate(models):
            record[f"rmse_{model}"] = float(rmses[model_index])
            if model != "recency":
                record[f"contrast_recency_minus_{model}"] = float(rmses[recency_index] - rmses[model_index])
        records.append(record)
    return pd.DataFrame(records)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.abs(values) > 1e-18]
    if len(values) > 22:
        raise ValueError("Exact sign-flip enumeration is limited to 22 pairs")
    magnitudes = np.abs(values)
    observed = float(np.mean(values))
    assignments = np.arange(1 << len(values), dtype=np.uint32)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint32)) & 1
    signs = 2.0 * bits - 1.0
    statistics = signs @ magnitudes / len(values)
    return float(np.mean(statistics <= observed + 1e-15))


def paired_tests(predictions: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot(index="coordinate", columns="model", values="residual")
    records: list[dict[str, float | str]] = []
    for comparator in ("shared", "aggregate", "policy_floor", "free_floor"):
        difference = pivot["recency"].to_numpy() ** 2 - pivot[comparator].to_numpy() ** 2
        column = f"contrast_recency_minus_{comparator}"
        records.append(
            {
                "comparator": comparator,
                "mean_squared_error_difference": float(np.mean(difference)),
                "recency_win_fraction": float(np.mean(difference < 0)),
                "exact_one_sided_sign_flip_p": exact_sign_flip_p(difference),
                "rmse_contrast_q025": float(bootstrap[column].quantile(0.025)),
                "rmse_contrast_median": float(bootstrap[column].quantile(0.5)),
                "rmse_contrast_q975": float(bootstrap[column].quantile(0.975)),
            }
        )
    return pd.DataFrame(records)


def summarize_cv(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    summary = (
        frame.groupby("model")
        .agg(
            rows=("residual", "size"),
            rmse=("residual", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            bias=("residual", "mean"),
            mae=("residual", lambda x: float(np.mean(np.abs(np.asarray(x))))),
            floor_bound_fraction=("floor_bound_fraction", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "evaluation", group)
    return summary


def model_rmse(frame: pd.DataFrame, model: str) -> float:
    residual = frame.loc[frame["model"] == model, "residual"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def promotion_decision(
    stage1_floor: pd.DataFrame,
    stage1_lpo: pd.DataFrame,
    stage3_lpo: pd.DataFrame,
    stage1_lro: pd.DataFrame,
    parameter_draws: pd.DataFrame,
    paired_test_frame: pd.DataFrame,
) -> tuple[dict[str, bool], bool]:
    stage1_recency = model_rmse(stage1_lpo, "recency")
    stage3_recency = model_rmse(stage3_lpo, "recency")
    floor_window = stage1_floor[stage1_floor["floor"].between(0.50, 0.74)]
    identified = parameter_draws[parameter_draws["identified"] & parameter_draws["phi"].notna()]
    phi_q10, phi_median, phi_q90 = (
        np.quantile(identified["phi"], [0.1, 0.5, 0.9]) if len(identified) else (np.nan, np.nan, np.nan)
    )
    policy_test = paired_test_frame.set_index("comparator").loc["policy_floor"]
    gates = {
        "stage1_vs_shared_10pct": stage1_recency <= 0.90 * model_rmse(stage1_lpo, "shared"),
        "stage1_vs_aggregate_5pct": stage1_recency <= 0.95 * model_rmse(stage1_lpo, "aggregate"),
        "stage1_vs_policy_floor_5pct": stage1_recency <= 0.95 * model_rmse(stage1_lpo, "policy_floor"),
        "stage1_preserves_free_floor": stage1_recency <= model_rmse(stage1_lpo, "free_floor"),
        "paired_policy_floor_uncertainty": (
            policy_test["rmse_contrast_q975"] < 0 and policy_test["exact_one_sided_sign_flip_p"] <= 0.05
        ),
        "stage3_disjoint_nonnegative": (
            stage3_recency
            <= min(
                model_rmse(stage3_lpo, "shared"),
                model_rmse(stage3_lpo, "aggregate"),
                model_rmse(stage3_lpo, "policy_floor"),
                model_rmse(stage3_lpo, "free_floor"),
            )
        ),
        "positive_gamma1_all_common_floor_sensitivities": bool((floor_window["gamma_1"] > 0).all()),
        "identified_gamma1_80pct_bootstrap": float(np.mean(parameter_draws["identified"])) >= 0.80,
        "phi_stable_and_independently_supported": (
            len(identified) > 0 and phi_q90 - phi_q10 <= 0.40 and 0.64 <= phi_median <= 0.84
        ),
        "leave_rung_out_preserved": model_rmse(stage1_lro, "recency") <= model_rmse(stage1_lro, "aggregate"),
    }
    normalized = {name: bool(value) for name, value in gates.items()}
    return normalized, all(normalized.values())


def write_report(
    output_dir: Path,
    protocol_hash: str,
    input_hash: str,
    fixed_floor: pd.DataFrame,
    fit_summary: pd.DataFrame,
    cv_summary: pd.DataFrame,
    parameter_draws: pd.DataFrame,
    paired_test_frame: pd.DataFrame,
    gates: dict[str, bool],
    survived: bool,
) -> None:
    identified = parameter_draws[parameter_draws["identified"] & parameter_draws["phi"].notna()]
    parameter_quantiles = (
        parameter_draws[["gamma_1", "gamma_1_se"]].quantile([0.1, 0.5, 0.9]).reset_index(names="quantile")
    )
    phi_quantiles = (
        identified[["phi"]].quantile([0.1, 0.5, 0.9]).reset_index(names="quantile")
        if len(identified)
        else pd.DataFrame({"quantile": [], "phi": []})
    )
    lines = [
        "# Policy-Conditioned Scaling Exponent Audit",
        "",
        f"Protocol SHA256: {protocol_hash}",
        "",
        f"Input SHA256: {input_hash}",
        "",
        "This is a development-data falsification audit. The motivating exponent and "
        "recency-weight claims were exposed before the protocol was frozen.",
        "",
        "## Decision",
        "",
        (
            "The recency-conditioned exponent route passes its development gate."
            if survived
            else "The recency-conditioned exponent route does not pass its frozen development gate."
        ),
        "",
        "Even a pass would not promote a single-scale surrogate: at fixed token horizon, "
        "a policy-dependent exponent is exactly absorbed into a policy-dependent response amplitude.",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: {name}" for name, passed in gates.items())
    lines.extend(["", "## Full-fit models", "", fit_summary.to_markdown(index=False)])
    lines.extend(["", "## Cross-validation", "", cv_summary.to_markdown(index=False)])
    lines.extend(["", "## Paired Stage-1 highest-rung tests", "", paired_test_frame.to_markdown(index=False)])
    lines.extend(["", "## Fixed-common-floor sensitivity", "", fixed_floor.to_markdown(index=False)])
    lines.extend(["", "## Coordinate parameter bootstrap", "", parameter_quantiles.to_markdown(index=False)])
    lines.extend(
        [
            "",
            f"Identified draws: {len(identified)}/{len(parameter_draws)}.",
            "",
            phi_quantiles.to_markdown(index=False) if len(phi_quantiles) else "No bootstrap draw identified phi.",
            "",
            "## Interpretation boundary",
            "",
            "The increase-D track also scales optimizer steps and both phase lengths, "
            "changes stream identity by rung, and holds model size fixed. Any surviving "
            "association is a rung moderator, not a causal token-horizon exponent or a "
            "capacity-normalized repetition mechanism.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parameter-bootstrap-draws", type=int, default=PARAMETER_BOOTSTRAP_DRAWS)
    parser.add_argument("--paired-bootstrap-draws", type=int, default=PAIRED_BOOTSTRAP_DRAWS)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = args.output_dir / "preregistration.md"
    if not preregistration.exists():
        raise FileNotFoundError(f"Missing frozen preregistration: {preregistration}")
    protocol_hash = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    common = load_common_panel(args.input)

    fixed_floor_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, float | int | str]] = []
    lro_frames: dict[str, pd.DataFrame] = {}
    lpo_frames: dict[str, pd.DataFrame] = {}
    for panel in ("stage1", "stage3", "all_common"):
        policies, ratios, outcomes = policy_arrays(panel_subset(common, panel))
        fixed_floor_frames.append(fixed_floor_audit(panel, policies, ratios, outcomes))
        for model in FULL_FIT_MODELS:
            fit = fit_scaling_model(policies, ratios, outcomes, model)
            fit_records.append(
                {
                    "panel": panel,
                    "model": model,
                    "policies": len(policies),
                    "floor_min": float(fit.floors.min()),
                    "floor_median": float(np.median(fit.floors)),
                    "floor_max": float(fit.floors.max()),
                    "floor_bound_fraction": fit.floor_bound_fraction,
                    "gamma_0": fit.gamma_0,
                    "gamma_1": fit.gamma_1,
                    "gamma_1_se": fit.gamma_1_se,
                    "phi": fit.phi,
                    "train_rmse": fit.rmse,
                }
            )
        if panel != "all_common":
            lro_frames[panel] = pd.concat(
                [leave_rung_out(policies, ratios, outcomes, model) for model in MODEL_LADDER],
                ignore_index=True,
            )
            lpo_frames[panel] = pd.concat(
                [leave_policy_out_highest_rung(policies, ratios, outcomes, model) for model in MODEL_LADDER],
                ignore_index=True,
            )

    stage1_policies, stage1_ratios, stage1_outcomes = policy_arrays(panel_subset(common, "stage1"))
    parameter_draws = parameter_bootstrap(
        stage1_policies,
        stage1_ratios,
        stage1_outcomes,
        args.parameter_bootstrap_draws,
    )
    paired_draws = paired_cv_bootstrap(lpo_frames["stage1"], args.paired_bootstrap_draws)
    paired_test_frame = paired_tests(lpo_frames["stage1"], paired_draws)
    fixed_floor = pd.concat(fixed_floor_frames, ignore_index=True)
    fit_summary = pd.DataFrame(fit_records)
    cv_summary = pd.concat(
        [summarize_cv(lro_frames[panel], f"{panel}_leave_rung_out") for panel in ("stage1", "stage3")]
        + [summarize_cv(lpo_frames[panel], f"{panel}_leave_policy_out_highest") for panel in ("stage1", "stage3")],
        ignore_index=True,
    )
    gates, survived = promotion_decision(
        fixed_floor[fixed_floor["panel"] == "stage1"],
        lpo_frames["stage1"],
        lpo_frames["stage3"],
        lro_frames["stage1"],
        parameter_draws,
        paired_test_frame,
    )

    fixed_floor.to_csv(args.output_dir / "fixed_floor_sensitivity.csv", index=False)
    fit_summary.to_csv(args.output_dir / "full_fit_models.csv", index=False)
    cv_summary.to_csv(args.output_dir / "cv_summary.csv", index=False)
    parameter_draws.to_csv(args.output_dir / "coordinate_parameter_bootstrap.csv", index=False)
    paired_draws.to_csv(args.output_dir / "paired_cv_bootstrap.csv", index=False)
    paired_test_frame.to_csv(args.output_dir / "paired_cv_tests.csv", index=False)
    for panel, frame in lro_frames.items():
        frame.to_csv(args.output_dir / f"{panel}_leave_rung_out_predictions.csv", index=False)
    for panel, frame in lpo_frames.items():
        frame.to_csv(args.output_dir / f"{panel}_leave_policy_out_highest_predictions.csv", index=False)
    summary = {
        "protocol_sha256": protocol_hash,
        "input_sha256": input_hash,
        "stage1_policy_count": len(stage1_policies),
        "stage3_policy_count": int(panel_subset(common, "stage3")["coordinate"].nunique()),
        "parameter_bootstrap_draws": args.parameter_bootstrap_draws,
        "paired_bootstrap_draws": args.paired_bootstrap_draws,
        "gates": gates,
        "development_route_survived": survived,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.output_dir,
        protocol_hash,
        input_hash,
        fixed_floor,
        fit_summary,
        cv_summary,
        parameter_draws,
        paired_test_frame,
        gates,
        survived,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
