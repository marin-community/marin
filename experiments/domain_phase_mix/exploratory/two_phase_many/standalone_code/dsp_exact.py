# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3

"""Self-contained DSP domain saturation-penalty fitting code.

This is a standalone implementation of the current DSP model family. It does
not import Marin modules; it only depends on numpy, pandas, scipy, and
scikit-learn. The default command fits canonical DSP on the packet-local
300M/6B panel:

    python standalone_code/dsp_exact.py fit --output-dir outputs/dsp_canonical_300m

DSP uses at most four M-dependent parameters per domain:

    a_i      nonnegative benefit amplitude
    p_i      nonnegative overexposure penalty amplitude
    rho_i    saturation rate
    tau_i    overexposure threshold

The canonical two-phase form is:

    e0_i = c0_i w0_i
    e1_i = c1_i w1_i
    z_i  = e0_i + e1_i
    r_i  = e1_i / (z_i + eps)

    L(w) = b0
         - sum_i a_i (1 + gamma r_i) (1 - exp(-rho_i z_i))
         + sum_i p_i softplus(log(1 + z_i) - tau_i)^2

For fixed nonlinear parameters (rho_i, tau_i, gamma), the linear head
(b0, a_i, p_i) is solved by NNLS variable projection. Nonlinear parameters are
tuned with deterministic starts and bounded L-BFGS-B over the same profile
objective used in the repository's DSP sprint:

    train RMSE + 0.5 * lower-tail optimism

The script also includes the empirical effective-exposure comparator:

    z_i = e0_i + gamma e1_i

where gamma enters both benefit saturation and penalty exposure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import basinhopping, minimize, nnls
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

PACKET_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PACKET_ROOT / "data"
DEFAULT_TARGET = "eval/uncheatable_eval/bpb"
DEFAULT_SCALE = "300m_6b"
DEFAULT_RUN_SET = "swarm_like_300m"
CV_SEED = 0
N_SPLITS = 5
LOWER_TAIL_FRAC = 0.15
LINEAR_REG = 1e-6
PHASE_EPS = 1e-9
FIT_MAXITER = 36
START_TOP_K = 3
THREE_HUNDRED_OLMIX_RUN_NAME = "baseline_olmix_loglinear_uncheatable_bpb"


class PhaseMode(StrEnum):
    """Two-phase treatment for a DSP variant."""

    BENEFIT_GAIN = "benefit_gain"
    EFFECTIVE_EXPOSURE = "effective_exposure"
    NONE = "none"


class PenaltyMode(StrEnum):
    """Penalty feature for a DSP variant."""

    LOG_SOFTPLUS_SQUARED = "log_softplus_squared"
    NONE = "none"


class LinearMode(StrEnum):
    """Linear head constraint."""

    NNLS = "nnls"
    SIGNED_RIDGE = "signed_ridge"


@dataclass(frozen=True)
class PacketData:
    """Feature-ready two-phase mixture data."""

    frame: pd.DataFrame
    name_col: str
    y: np.ndarray
    w: np.ndarray
    m: int
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]


@dataclass(frozen=True)
class DSPVariant:
    """One DSP functional form."""

    name: str
    phase_mode: PhaseMode
    penalty_mode: PenaltyMode
    linear_mode: LinearMode
    description: str


@dataclass(frozen=True)
class FittedDSPModel:
    """Fitted DSP model."""

    variant: DSPVariant
    params: dict[str, float | np.ndarray]
    intercept: float
    benefit_coef: np.ndarray
    penalty_coef: np.ndarray
    domain_names: list[str]
    c0: np.ndarray
    c1: np.ndarray

    @property
    def m_dependent_params_per_domain(self) -> int:
        return 4 if self.variant.penalty_mode != PenaltyMode.NONE else 2

    @property
    def total_param_count(self) -> int:
        per_domain = len(self.domain_names)  # rho
        per_domain += len(self.domain_names)  # benefit coefficient
        if self.variant.penalty_mode != PenaltyMode.NONE:
            per_domain += len(self.domain_names)  # tau
            per_domain += len(self.domain_names)  # penalty coefficient
        global_params = 1  # intercept
        if self.variant.phase_mode != PhaseMode.NONE:
            global_params += 1
        return int(per_domain + global_params)


VARIANTS: dict[str, DSPVariant] = {
    "canonical": DSPVariant(
        name="dsp_phase_benefit_penalty_nnls",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Canonical DSP: phase-1 premium only multiplies benefit; penalties use raw exposure.",
    ),
    "effective_exposure": DSPVariant(
        name="dsp_effective_exposure_penalty_nnls",
        phase_mode=PhaseMode.EFFECTIVE_EXPOSURE,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Empirical comparator: phase-1 multiplier enters benefit and penalty exposure.",
    ),
    "no_phase": DSPVariant(
        name="dsp_no_phase_penalty_nnls",
        phase_mode=PhaseMode.NONE,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.NNLS,
        description="Control: no phase term.",
    ),
    "no_penalty": DSPVariant(
        name="dsp_phase_benefit_no_penalty_nnls",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.NONE,
        linear_mode=LinearMode.NNLS,
        description="Control: phase benefit with no explicit overexposure penalty.",
    ),
    "signed": DSPVariant(
        name="dsp_phase_benefit_penalty_signed",
        phase_mode=PhaseMode.BENEFIT_GAIN,
        penalty_mode=PenaltyMode.LOG_SOFTPLUS_SQUARED,
        linear_mode=LinearMode.SIGNED_RIDGE,
        description="Control: signed ridge linear head instead of NNLS.",
    ),
}


def softplus(x: np.ndarray | float) -> np.ndarray:
    """Stable softplus."""
    arr = np.asarray(x, dtype=float)
    return np.where(arr > 20.0, arr, np.log1p(np.exp(np.minimum(arr, 20.0))))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Stable logistic sigmoid."""
    arr = np.asarray(x, dtype=float)
    positive = arr >= 0.0
    out = np.empty_like(arr, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-arr[positive]))
    exp_x = np.exp(arr[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def average_phase_tv_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average total-variation distance across phases."""
    return np.abs(left - right).sum(axis=(1, 2)) / (2.0 * left.shape[1])


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    """Normalize phase rows and fail on invalid mass."""
    sums = weights.sum(axis=-1, keepdims=True)
    if np.any(sums <= 0.0):
        raise ValueError("Encountered a phase row with non-positive mass")
    return weights / sums


def load_domain_metadata(data_dir: Path) -> pd.DataFrame:
    """Load canonical DSP domain order and epoch multipliers."""
    metadata_path = data_dir / "grp_no_l2" / "two_phase_many_epoch_metadata.csv"
    metadata = pd.read_csv(metadata_path)
    required = {"domain_name", "phase_0_epoch_multiplier", "phase_1_epoch_multiplier"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Domain metadata missing columns: {sorted(missing)}")
    return metadata


def load_metric_frame(data_dir: Path) -> pd.DataFrame:
    """Load packet-local metric registry wide table."""
    path = data_dir / "metric_registry" / "metrics_wide.csv"
    return pd.read_csv(path, low_memory=False)


def build_fit_frame(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    target: str,
    scale: str,
    run_set: str,
) -> pd.DataFrame:
    """Build the DSP fit frame from packet-local metrics."""
    domain_names = metadata["domain_name"].tolist()
    weight_columns = [f"{phase}_{domain}" for phase in ("phase_0", "phase_1") for domain in domain_names]
    missing_weight_columns = [column for column in weight_columns if column not in frame.columns]
    if missing_weight_columns:
        raise ValueError(f"Missing canonical weight columns: {missing_weight_columns[:10]}")

    id_columns = [
        column
        for column in (
            "registry_run_key",
            "run_id",
            "run_name",
            "scale",
            "cohort",
            "source_experiment",
            "checkpoint_root",
            "status",
            "is_qsplit240_core",
            "is_baseline_olmix",
            "is_baseline_stratified",
            "is_fit_swarm_60m_default",
        )
        if column in frame.columns
    ]
    subset = frame.loc[
        frame["scale"].eq(scale) & frame["cohort"].eq("signal") & frame[target].notna(),
        id_columns + weight_columns + [target],
    ].copy()
    if run_set == "fit_swarm_60m_default":
        subset = subset.loc[subset["is_fit_swarm_60m_default"].fillna(False)].copy()
    elif run_set == "swarm_like_300m":
        mask = (
            subset["is_qsplit240_core"].fillna(False)
            | subset["is_baseline_olmix"].fillna(False)
            | subset["is_baseline_stratified"].fillna(False)
            | subset["run_name"].eq(THREE_HUNDRED_OLMIX_RUN_NAME)
        )
        subset = subset.loc[mask].copy()
    elif run_set == "all_signal":
        pass
    else:
        raise ValueError(f"Unknown run_set={run_set!r}")

    if subset.empty:
        raise ValueError(f"No rows for scale={scale!r}, run_set={run_set!r}, target={target!r}")
    if subset["run_name"].duplicated().any():
        dupes = subset.loc[subset["run_name"].duplicated(), "run_name"].tolist()
        raise ValueError(f"Duplicate run_name rows in fit frame: {dupes[:10]}")
    subset[weight_columns] = subset[weight_columns].fillna(0.0)
    # Some packet versions include baseline_stratified only with aggregate
    # domain columns. The intended canonical split-domain interpretation is
    # uniform over the 39 metadata domains, matching the repo baseline.
    for phase_name in ("phase_0", "phase_1"):
        phase_columns = [f"{phase_name}_{domain}" for domain in domain_names]
        phase_sums = subset[phase_columns].sum(axis=1)
        stratified_zero_mask = subset["run_name"].eq("baseline_stratified") & phase_sums.le(0.0)
        if stratified_zero_mask.any():
            subset.loc[stratified_zero_mask, phase_columns] = 1.0 / len(domain_names)
    return subset.rename(columns={target: "objective_metric"}).reset_index(drop=True)


def packet_from_frame(frame: pd.DataFrame, metadata: pd.DataFrame) -> PacketData:
    """Convert a fit frame into numerical DSP arrays."""
    domain_names = metadata["domain_name"].tolist()
    weights = np.zeros((len(frame), 2, len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain_name in enumerate(domain_names):
            weights[:, phase_idx, domain_idx] = frame[f"{phase_name}_{domain_name}"].to_numpy(dtype=float)
    weights = normalize_weights(weights)
    return PacketData(
        frame=frame.reset_index(drop=True),
        name_col="run_name",
        y=frame["objective_metric"].to_numpy(dtype=float),
        w=weights,
        m=len(domain_names),
        c0=metadata["phase_0_epoch_multiplier"].to_numpy(dtype=float),
        c1=metadata["phase_1_epoch_multiplier"].to_numpy(dtype=float),
        domain_names=domain_names,
    )


def load_packet(data_dir: Path, *, target: str, scale: str, run_set: str) -> PacketData:
    """Load a DSP fit packet."""
    metadata = load_domain_metadata(data_dir)
    fit_frame = build_fit_frame(load_metric_frame(data_dir), metadata, target=target, scale=scale, run_set=run_set)
    return packet_from_frame(fit_frame, metadata)


def unpack_theta(theta: np.ndarray, variant: DSPVariant, num_domains: int) -> dict[str, float | np.ndarray]:
    """Decode bounded nonlinear parameters from optimizer coordinates."""
    cursor = 0
    log_rho = np.clip(theta[cursor : cursor + num_domains], np.log(1e-4), np.log(2.0))
    cursor += num_domains
    params: dict[str, float | np.ndarray] = {"rho": np.exp(log_rho)}
    if variant.penalty_mode != PenaltyMode.NONE:
        params["tau"] = np.clip(theta[cursor : cursor + num_domains], -2.0, 8.0)
        cursor += num_domains
    if variant.phase_mode != PhaseMode.NONE:
        params["gamma"] = float(np.exp(np.clip(theta[cursor], np.log(1e-4), np.log(100.0))))
        cursor += 1
    if cursor != len(theta):
        raise ValueError(f"Unused theta values for {variant.name}: cursor={cursor}, len={len(theta)}")
    return params


def pack_params(params: dict[str, float | np.ndarray], variant: DSPVariant) -> np.ndarray:
    """Pack nonlinear parameters into optimizer coordinates."""
    values: list[np.ndarray] = [np.log(np.asarray(params["rho"], dtype=float))]
    if variant.penalty_mode != PenaltyMode.NONE:
        values.append(np.asarray(params["tau"], dtype=float))
    if variant.phase_mode != PhaseMode.NONE:
        values.append(np.asarray([np.log(float(params["gamma"]))], dtype=float))
    return np.concatenate(values)


def bounds(variant: DSPVariant, num_domains: int) -> list[tuple[float, float]]:
    """Return bounded optimizer coordinates."""
    out: list[tuple[float, float]] = [(np.log(1e-4), np.log(2.0)) for _ in range(num_domains)]
    if variant.penalty_mode != PenaltyMode.NONE:
        out.extend([(-2.0, 8.0) for _ in range(num_domains)])
    if variant.phase_mode != PhaseMode.NONE:
        out.append((np.log(1e-4), np.log(100.0)))
    return out


def features(
    weights: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build DSP benefit and penalty features."""
    p0 = weights[:, 0, :]
    p1 = weights[:, 1, :]
    e0 = p0 * c0[None, :]
    e1 = p1 * c1[None, :]
    rho = np.asarray(params["rho"], dtype=float)[None, :]
    if variant.phase_mode == PhaseMode.EFFECTIVE_EXPOSURE:
        exposure = e0 + float(params["gamma"]) * e1
        signal = 1.0 - np.exp(-rho * exposure)
    else:
        exposure = e0 + e1
        signal = 1.0 - np.exp(-rho * exposure)
        if variant.phase_mode == PhaseMode.BENEFIT_GAIN:
            phase1_share = e1 / (exposure + PHASE_EPS)
            signal = (1.0 + float(params["gamma"]) * phase1_share) * signal

    if variant.penalty_mode == PenaltyMode.NONE:
        penalty = np.zeros_like(signal)
    else:
        tau = np.asarray(params["tau"], dtype=float)[None, :]
        penalty = softplus(np.log1p(exposure) - tau) ** 2
    return signal, penalty


def fit_linear_head(
    weights: np.ndarray,
    targets: np.ndarray,
    packet: PacketData,
    variant: DSPVariant,
    params: dict[str, float | np.ndarray],
) -> FittedDSPModel:
    """Fit linear DSP head for fixed nonlinear parameters."""
    signal, penalty = features(weights, packet.c0, packet.c1, variant, params)
    if variant.penalty_mode == PenaltyMode.NONE:
        design = -signal
    else:
        design = np.hstack([-signal, penalty])
    design_mean = design.mean(axis=0, keepdims=True)
    target_mean = float(targets.mean())
    centered_design = design - design_mean
    centered_targets = targets - target_mean
    if variant.linear_mode == LinearMode.NNLS:
        if LINEAR_REG > 0.0:
            centered_design = np.vstack([centered_design, np.sqrt(LINEAR_REG) * np.eye(centered_design.shape[1])])
            centered_targets = np.concatenate([centered_targets, np.zeros(centered_design.shape[1], dtype=float)])
        coef, _ = nnls(centered_design, centered_targets)
    else:
        lhs = centered_design.T @ centered_design + LINEAR_REG * np.eye(centered_design.shape[1])
        rhs = centered_design.T @ centered_targets
        coef = np.linalg.solve(lhs, rhs)

    intercept = float(target_mean - (design_mean @ coef).item())
    benefit_coef = np.asarray(coef[: packet.m], dtype=float)
    if variant.penalty_mode == PenaltyMode.NONE:
        penalty_coef = np.zeros(packet.m, dtype=float)
    else:
        penalty_coef = np.asarray(coef[packet.m :], dtype=float)
    return FittedDSPModel(
        variant=variant,
        params=params,
        intercept=intercept,
        benefit_coef=benefit_coef,
        penalty_coef=penalty_coef,
        domain_names=list(packet.domain_names),
        c0=np.asarray(packet.c0, dtype=float),
        c1=np.asarray(packet.c1, dtype=float),
    )


def predict(model: FittedDSPModel, weights: np.ndarray) -> np.ndarray:
    """Predict target value for phase weights."""
    signal, penalty = features(weights, model.c0, model.c1, model.variant, model.params)
    return np.asarray(model.intercept - signal @ model.benefit_coef + penalty @ model.penalty_coef, dtype=float)


def profile_objective(packet: PacketData, variant: DSPVariant, theta: np.ndarray) -> float:
    """Nonlinear profile objective used for DSP tuning."""
    params = unpack_theta(theta, variant, packet.m)
    model = fit_linear_head(packet.w, packet.y, packet, variant, params)
    pred = predict(model, packet.w)
    residual = pred - packet.y
    rmse = float(np.sqrt(np.mean(residual**2)))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(pred)[:tail_count]
    optimism = float(np.mean(np.maximum(packet.y[tail_idx] - pred[tail_idx], 0.0)))
    return rmse + 0.5 * optimism


def start_bank(packet: PacketData, variant: DSPVariant) -> tuple[np.ndarray, ...]:
    """Build deterministic nonlinear starts from observed exposure statistics."""
    z = packet.w[:, 0, :] * packet.c0[None, :] + packet.w[:, 1, :] * packet.c1[None, :]
    positive = np.where(z > 1e-8, z, np.nan)
    median_exposure = np.nanmedian(positive, axis=0)
    median_exposure = np.where(np.isfinite(median_exposure), median_exposure, np.nanmedian(positive))
    base_rho = np.clip(1.0 / np.maximum(median_exposure, 1e-3), 1e-4, 0.5)
    base_tau = np.clip(np.log1p(np.nanpercentile(positive, 85, axis=0)), -2.0, 8.0)
    base_tau = np.where(np.isfinite(base_tau), base_tau, 3.0)

    rng = np.random.default_rng(CV_SEED)
    starts: list[np.ndarray] = []
    for rho_scale, tau_shift, gamma in (
        (0.25, -1.0, 0.25),
        (0.5, -0.5, 0.5),
        (1.0, 0.0, 1.0),
        (2.0, 0.5, 2.0),
        (4.0, 1.0, 8.0),
    ):
        params: dict[str, float | np.ndarray] = {"rho": np.clip(base_rho * rho_scale, 1e-4, 2.0)}
        if variant.penalty_mode != PenaltyMode.NONE:
            params["tau"] = np.clip(base_tau + tau_shift, -2.0, 8.0)
        if variant.phase_mode != PhaseMode.NONE:
            params["gamma"] = gamma
        starts.append(pack_params(params, variant))

    for _ in range(3):
        params = {"rho": np.clip(base_rho * np.exp(rng.normal(scale=0.7, size=packet.m)), 1e-4, 2.0)}
        if variant.penalty_mode != PenaltyMode.NONE:
            params["tau"] = np.clip(base_tau + rng.normal(scale=0.8, size=packet.m), -2.0, 8.0)
        if variant.phase_mode != PhaseMode.NONE:
            params["gamma"] = float(np.exp(rng.normal(loc=np.log(2.0), scale=0.9)))
        starts.append(pack_params(params, variant))
    return tuple(starts)


def fit_variant(
    packet: PacketData,
    variant: DSPVariant,
    *,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[FittedDSPModel, pd.DataFrame]:
    """Tune nonlinear parameters and fit the final DSP model."""
    starts = start_bank(packet, variant)
    coord_bounds = bounds(variant, packet.m)
    coarse_rows = [
        {"stage": "coarse", "start_id": start_id, "objective": profile_objective(packet, variant, start)}
        for start_id, start in enumerate(starts)
    ]
    ranked = sorted(coarse_rows, key=lambda row: float(row["objective"]))
    rows = list(coarse_rows)
    best_objective = float("inf")
    best_theta: np.ndarray | None = None

    for rank, row in enumerate(ranked[:coarse_top_k]):
        start = starts[int(row["start_id"])]
        result = minimize(
            lambda theta: profile_objective(packet, variant, np.asarray(theta, dtype=float)),
            start,
            method="L-BFGS-B",
            bounds=coord_bounds,
            options={"maxiter": maxiter, "ftol": 1e-7, "maxls": 20},
        )
        rows.append(
            {
                "stage": "refine",
                "chosen_rank": rank,
                "start_id": int(row["start_id"]),
                "objective": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
            }
        )
        if float(result.fun) < best_objective:
            best_objective = float(result.fun)
            best_theta = np.asarray(result.x, dtype=float)

    if best_theta is None:
        raise RuntimeError(f"No fit result for {variant.name}")

    if basin_hopping_iters > 0:
        hop_result = basinhopping(
            lambda theta: profile_objective(packet, variant, np.asarray(theta, dtype=float)),
            best_theta,
            niter=basin_hopping_iters,
            stepsize=0.15,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": coord_bounds,
                "options": {"maxiter": max(8, maxiter // 4), "ftol": 1e-7},
            },
            seed=CV_SEED,
        )
        rows.append(
            {
                "stage": "basin_hopping_diagnostic",
                "chosen_rank": -1,
                "start_id": -1,
                "objective": float(hop_result.fun),
                "success": bool(hop_result.lowest_optimization_result.success),
                "message": str(hop_result.message),
            }
        )
        if float(hop_result.fun) < best_objective:
            best_theta = np.asarray(hop_result.x, dtype=float)

    params = unpack_theta(best_theta, variant, packet.m)
    return fit_linear_head(packet.w, packet.y, packet, variant, params), pd.DataFrame.from_records(rows)


def oof_predictions(packet: PacketData, model: FittedDSPModel) -> np.ndarray:
    """Compute row-wise out-of-fold predictions with fixed nonlinear params."""
    oof = np.zeros_like(packet.y)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for train_idx, test_idx in kf.split(packet.w):
        fold_model = fit_linear_head(
            packet.w[train_idx],
            packet.y[train_idx],
            packet,
            model.variant,
            model.params,
        )
        oof[test_idx] = predict(fold_model, packet.w[test_idx])
    return oof


def value_grad_logits(model: FittedDSPModel, logits: np.ndarray) -> tuple[float, np.ndarray]:
    """Return raw optimum value and analytic gradient with softmax logits."""
    num_domains = len(model.domain_names)
    logits0 = logits[:num_domains]
    logits1 = logits[num_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)

    e0 = model.c0 * p0
    e1 = model.c1 * p1
    rho = np.asarray(model.params["rho"], dtype=float)
    benefit_coef = model.benefit_coef
    penalty_coef = model.penalty_coef
    has_penalty = model.variant.penalty_mode != PenaltyMode.NONE

    if model.variant.phase_mode == PhaseMode.EFFECTIVE_EXPOSURE:
        gamma = float(model.params["gamma"])
        exposure = e0 + gamma * e1
        signal = 1.0 - np.exp(-rho * exposure)
        dsignal = rho * np.exp(-rho * exposure)
        if has_penalty:
            tau = np.asarray(model.params["tau"], dtype=float)
            u = np.log1p(exposure) - tau
            sp = softplus(u)
            penalty = sp**2
            dpenalty = 2.0 * sp * sigmoid(u) / (1.0 + exposure)
        else:
            penalty = np.zeros_like(signal)
            dpenalty = np.zeros_like(signal)
        common = -benefit_coef * dsignal + penalty_coef * dpenalty
        grad_e0 = common
        grad_e1 = gamma * common
        value = float(model.intercept - benefit_coef @ signal + penalty_coef @ penalty)
    else:
        exposure = e0 + e1
        denom = exposure + PHASE_EPS
        r = e1 / denom
        base_signal = 1.0 - np.exp(-rho * exposure)
        dbase_signal = rho * np.exp(-rho * exposure)
        signal_amp = np.ones(num_domains, dtype=float)
        dsignal_amp_de0 = np.zeros(num_domains, dtype=float)
        dsignal_amp_de1 = np.zeros(num_domains, dtype=float)
        if model.variant.phase_mode == PhaseMode.BENEFIT_GAIN:
            gamma = float(model.params["gamma"])
            signal_amp = 1.0 + gamma * r
            dsignal_amp_de0 = gamma * (-e1 / (denom * denom))
            dsignal_amp_de1 = gamma * ((e0 + PHASE_EPS) / (denom * denom))
        signal = signal_amp * base_signal
        dsignal_de0 = signal_amp * dbase_signal + base_signal * dsignal_amp_de0
        dsignal_de1 = signal_amp * dbase_signal + base_signal * dsignal_amp_de1
        if has_penalty:
            tau = np.asarray(model.params["tau"], dtype=float)
            u = np.log1p(exposure) - tau
            sp = softplus(u)
            penalty = sp**2
            dpenalty = 2.0 * sp * sigmoid(u) / (1.0 + exposure)
        else:
            penalty = np.zeros_like(signal)
            dpenalty = np.zeros_like(signal)
        grad_e0 = -benefit_coef * dsignal_de0 + penalty_coef * dpenalty
        grad_e1 = -benefit_coef * dsignal_de1 + penalty_coef * dpenalty
        value = float(model.intercept - benefit_coef @ signal + penalty_coef @ penalty)

    grad_p0 = grad_e0 * model.c0
    grad_p1 = grad_e1 * model.c1
    grad_logits0 = p0 * (grad_p0 - np.dot(grad_p0, p0))
    grad_logits1 = p1 * (grad_p1 - np.dot(grad_p1, p1))
    return value, np.concatenate([grad_logits0, grad_logits1])


def optimize_raw(model: FittedDSPModel, *, num_starts: int = 20) -> tuple[Any, np.ndarray]:
    """Optimize unconstrained raw simplex weights with softmax logits."""
    rng = np.random.default_rng(CV_SEED)
    starts = [np.zeros(2 * len(model.domain_names), dtype=float)]
    starts.extend(
        np.concatenate(
            [
                rng.normal(scale=0.3, size=len(model.domain_names)),
                rng.normal(scale=0.3, size=len(model.domain_names)),
            ]
        )
        for _ in range(num_starts)
    )
    best = None
    for start in starts:
        result = minimize(
            lambda z: value_grad_logits(model, np.asarray(z, dtype=float))[0],
            start,
            jac=lambda z: value_grad_logits(model, np.asarray(z, dtype=float))[1],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    if best is None:
        raise RuntimeError("Raw optimization failed")
    logits = np.asarray(best.x, dtype=float)
    logits0 = logits[: len(model.domain_names)]
    logits1 = logits[len(model.domain_names) :]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)
    return best, np.stack([p0, p1], axis=0)


def metrics(packet: PacketData, model: FittedDSPModel, raw_result: Any, raw_weights: np.ndarray) -> dict[str, Any]:
    """Summarize fit, OOF, and raw optimum diagnostics."""
    train_pred = predict(model, packet.w)
    oof = oof_predictions(packet, model)
    residual = oof - packet.y
    train_residual = train_pred - packet.y
    fold_regrets: list[float] = []
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_SEED)
    for _train_idx, test_idx in kf.split(packet.w):
        chosen_idx = int(np.argmin(oof[test_idx]))
        fold_regrets.append(float(packet.y[test_idx][chosen_idx] - np.min(packet.y[test_idx])))
    tail_count = max(5, int(np.ceil(LOWER_TAIL_FRAC * len(packet.y))))
    tail_idx = np.argsort(oof)[:tail_count]
    raw_distances = average_phase_tv_distance(packet.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    return {
        "variant": model.variant.name,
        "description": model.variant.description,
        "fit_row_count": len(packet.y),
        "m_dependent_params_per_domain": model.m_dependent_params_per_domain,
        "total_param_count": model.total_param_count,
        "train_rmse": float(np.sqrt(np.mean(train_residual**2))),
        "train_spearman": float(spearmanr(packet.y, train_pred).statistic),
        "train_pearson": float(pearsonr(packet.y, train_pred).statistic),
        "cv_rmse": float(np.sqrt(np.mean(residual**2))),
        "cv_mae": float(np.mean(np.abs(residual))),
        "oof_spearman": float(spearmanr(packet.y, oof).statistic),
        "oof_pearson": float(pearsonr(packet.y, oof).statistic),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "cv_regret_at_1": float(packet.y[int(np.argmin(oof))] - np.min(packet.y)),
        "lower_tail_optimism": float(np.mean(np.maximum(packet.y[tail_idx] - oof[tail_idx], 0.0))),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.frame.iloc[nearest_idx][packet.name_col]),
        "raw_nearest_observed_value": float(packet.y[nearest_idx]),
        "raw_phase0_support_gt_1e3": int(np.sum(raw_weights[0] > 1e-3)),
        "raw_phase1_support_gt_1e3": int(np.sum(raw_weights[1] > 1e-3)),
        "phase0_max_weight": float(np.max(raw_weights[0])),
        "phase1_max_weight": float(np.max(raw_weights[1])),
        "optimum_success": bool(raw_result.success),
        "optimum_message": str(raw_result.message),
    }


def model_to_json(model: FittedDSPModel, model_metrics: dict[str, Any]) -> dict[str, Any]:
    """Serialize model to JSON-compatible dict."""
    params = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in model.params.items()}
    return {
        "variant": model.variant.name,
        "description": model.variant.description,
        "params": params,
        "intercept": model.intercept,
        "benefit_coef": model.benefit_coef.tolist(),
        "penalty_coef": model.penalty_coef.tolist(),
        "domain_names": model.domain_names,
        "c0": model.c0.tolist(),
        "c1": model.c1.tolist(),
        "metrics": model_metrics,
    }


def model_from_json(payload: dict[str, Any]) -> FittedDSPModel:
    """Load a fitted model from JSON payload."""
    variant_name = payload["variant"]
    variant = next((item for item in VARIANTS.values() if item.name == variant_name), None)
    if variant is None:
        raise ValueError(f"Unknown serialized DSP variant: {variant_name}")
    params = {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else float(value)
        for key, value in payload["params"].items()
    }
    return FittedDSPModel(
        variant=variant,
        params=params,
        intercept=float(payload["intercept"]),
        benefit_coef=np.asarray(payload["benefit_coef"], dtype=float),
        penalty_coef=np.asarray(payload["penalty_coef"], dtype=float),
        domain_names=list(payload["domain_names"]),
        c0=np.asarray(payload["c0"], dtype=float),
        c1=np.asarray(payload["c1"], dtype=float),
    )


def weights_to_frame(model: FittedDSPModel, weights: np.ndarray) -> pd.DataFrame:
    """Convert a two-phase weight matrix into a readable table."""
    rows = []
    for domain_name, phase0, phase1, c0, c1 in zip(
        model.domain_names,
        weights[0],
        weights[1],
        model.c0,
        model.c1,
        strict=True,
    ):
        rows.append(
            {
                "domain_name": domain_name,
                "phase_0_weight": float(phase0),
                "phase_1_weight": float(phase1),
                "phase_0_effective_epochs": float(phase0 * c0),
                "phase_1_effective_epochs": float(phase1 * c1),
            }
        )
    return pd.DataFrame.from_records(rows)


def observed_predictions(packet: PacketData, model: FittedDSPModel) -> pd.DataFrame:
    """Return observed-row predictions and residuals."""
    pred = predict(model, packet.w)
    out = packet.frame[[packet.name_col]].copy()
    out = out.rename(columns={packet.name_col: "run_name"})
    out["actual"] = packet.y
    out["prediction"] = pred
    out["residual_prediction_minus_actual"] = pred - packet.y
    out["actual_rank"] = out["actual"].rank(method="min")
    out["prediction_rank"] = out["prediction"].rank(method="min")
    return out.sort_values("prediction").reset_index(drop=True)


def read_weights_from_frame(frame: pd.DataFrame, domain_names: list[str]) -> np.ndarray:
    """Read phase weights from a dataframe with phase_0_/phase_1_ columns."""
    weights = np.zeros((len(frame), 2, len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain_name in enumerate(domain_names):
            column = f"{phase_name}_{domain_name}"
            if column not in frame.columns:
                raise ValueError(f"Missing weight column {column!r}")
            weights[:, phase_idx, domain_idx] = frame[column].to_numpy(dtype=float)
    return normalize_weights(weights)


def score_perturbations(model: FittedDSPModel, perturbation_csv: Path, output_csv: Path) -> None:
    """Score finite perturbation effects versus the CSV's base_run_name row if available.

    The input CSV must have phase_0_/phase_1_ columns for model.domain_names.
    If it has no explicit baseline row, the proportional baseline is reconstructed
    from target_mass_before for domain_bump rows.
    """
    frame = pd.read_csv(perturbation_csv, low_memory=False)
    weights = read_weights_from_frame(frame, model.domain_names)
    predictions = predict(model, weights)
    out_columns = [
        column for column in ("intervention_id", "intervention_type", "target_unit", "target_domain") if column in frame
    ]
    out = frame[out_columns].copy()
    out["dsp_prediction"] = predictions

    baseline_weights: np.ndarray | None = None
    if "run_name" in frame.columns and "base_run_name" in frame.columns:
        base_run_names = frame["base_run_name"].dropna().astype(str).unique()
        if len(base_run_names) == 1:
            baseline_rows = frame.loc[frame["run_name"].astype(str).eq(base_run_names[0])]
            if not baseline_rows.empty:
                baseline_weights = read_weights_from_frame(baseline_rows.iloc[:1].copy(), model.domain_names)[0]
    if baseline_weights is None and {"target_domain", "target_mass_before"}.issubset(frame.columns):
        domain_bumps = frame.loc[frame["target_domain"].notna()].copy()
        base = domain_bumps.set_index("target_domain")["target_mass_before"].reindex(model.domain_names).to_numpy()
        if np.isfinite(base).all():
            base = base / base.sum()
            baseline_weights = np.stack([base, base], axis=0)
    if baseline_weights is not None:
        base_prediction = float(predict(model, baseline_weights[None, :, :])[0])
        out["dsp_effect_vs_base"] = predictions - base_prediction
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


def fit_command(args: argparse.Namespace) -> None:
    """Fit and write DSP artifacts."""
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = VARIANTS[args.variant]
    packet = load_packet(data_dir, target=args.target, scale=args.scale, run_set=args.run_set)
    model, tuning = fit_variant(
        packet,
        variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    raw_result, raw_weights = optimize_raw(model, num_starts=args.optimum_starts)
    model_metrics = metrics(packet, model, raw_result, raw_weights)
    model_metrics["target"] = args.target
    model_metrics["scale"] = args.scale
    model_metrics["run_set"] = args.run_set
    tuning.to_csv(output_dir / "tuning.csv", index=False)
    pd.DataFrame([model_metrics]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(model_metrics, indent=2))
    (output_dir / "model.json").write_text(json.dumps(model_to_json(model, model_metrics), indent=2))
    weights_to_frame(model, raw_weights).to_csv(output_dir / "raw_optimum_weights.csv", index=False)
    observed_predictions(packet, model).to_csv(output_dir / "observed_predictions.csv", index=False)
    print(pd.DataFrame([model_metrics]).to_string(index=False))
    print(f"Wrote DSP artifacts to {output_dir}")


def score_command(args: argparse.Namespace) -> None:
    """Load a fitted model and score a perturbation manifest."""
    model = model_from_json(json.loads(Path(args.model_json).read_text()))
    score_perturbations(model, Path(args.perturbation_csv), Path(args.output_csv))
    print(f"Wrote perturbation scores to {args.output_csv}")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    fit = subparsers.add_parser("fit", help="fit a DSP model")
    fit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    fit.add_argument("--output-dir", type=Path, default=PACKET_ROOT / "outputs" / "dsp_canonical")
    fit.add_argument("--target", default=DEFAULT_TARGET)
    fit.add_argument("--scale", default=DEFAULT_SCALE)
    fit.add_argument(
        "--run-set",
        default=DEFAULT_RUN_SET,
        choices=["swarm_like_300m", "fit_swarm_60m_default", "all_signal"],
    )
    fit.add_argument("--variant", default="canonical", choices=sorted(VARIANTS))
    fit.add_argument("--maxiter", type=int, default=FIT_MAXITER)
    fit.add_argument("--coarse-top-k", type=int, default=START_TOP_K)
    fit.add_argument("--basin-hopping-iters", type=int, default=3)
    fit.add_argument("--optimum-starts", type=int, default=20)
    fit.set_defaults(func=fit_command)

    score = subparsers.add_parser("score-perturbations", help="score a perturbation manifest with a fitted DSP model")
    score.add_argument("--model-json", type=Path, required=True)
    score.add_argument("--perturbation-csv", type=Path, required=True)
    score.add_argument("--output-csv", type=Path, required=True)
    score.set_defaults(func=score_command)
    return parser


def main() -> None:
    """Run CLI."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command is None:
        args = parser.parse_args(["fit"])
    args.func(args)


if __name__ == "__main__":
    main()
