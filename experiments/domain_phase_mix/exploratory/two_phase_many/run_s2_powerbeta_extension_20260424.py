#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "scipy"]
# ///
"""Test S2-style structural laws with an explicit power-beta continuation term.

The model keeps the S2 anchored Chinchilla-style base over fixed-mixture scale:

    base(w, N, D_base) = E(w) + A(w) u_N + B(w) u_D + C(w) u_ND

and separates simulated epoching continuation as:

    h(w, N, D_base, mu) = G(w, N, D_base) * (mu ** -beta(w, N, D_base) - 1)

where D_base = realized_tokens / mu. For fixed w and mu=1 this is exactly a
power-law scale law in N and D_base; for fixed w,N,D_base it is a power law in
target-budget multiplier mu.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, lsq_linear

from run_s2_structural_upgrade_sprint_20260424 import (
    EnhancedFeatureFactory,
    fixed_drop_summary,
    load_module,
    metric_dict,
    optimum_diagnostics,
    plot_predicted_vs_actual,
)

SESSION2_SCRIPT = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/session10_candidate_revalidation_20260424"
    / "session2_structural/run_joint_mixture_scale_law.py"
)
DEFAULT_PACKET_ROOT = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/chatgpt_pro_hybrid_data_mixing_packet_v30"
)
DEFAULT_OUT_DIR = (
    Path("experiments/domain_phase_mix/exploratory/two_phase_many")
    / "chatgpt_pro_hybrid_data_mixing_packet_v30_local_artifacts"
    / "reference_outputs/s2_powerbeta_extension_20260424"
)
FAMILIES = ("broad_text", "tech_code", "reasoning")
TARGETS = (
    ("60m_1p2b", "60M/1.2B"),
    ("300m_6b", "100M/6B"),
    ("520m_10p4b", "340M/10.4B"),
    ("1_2b_24b", "900M/24B"),
)


@dataclass(frozen=True)
class VariantSpec:
    """Configuration for one S2 + continuation candidate."""

    name: str
    anchor_kind: str = "grp_famsqrt"
    head_a: str = "constant"
    head_b: str = "family"
    head_c: str = "constant"
    amp_mode: str = "scale_fam6"
    beta_mode: str = "scale_p1fam3"
    base_row_mode: str = "all"
    exponents: tuple[float, float, float, float] = (0.20, 0.25, 0.30, 0.65)
    ridge_anchor: float = 1e-4
    ridge_scale: float = 1e-5
    reg_amp: float = 0.01
    reg_beta: float = 0.01
    beta_min: float = 0.05
    beta_max: float = 1.50
    robust_beta_scale: float = 0.30
    donor_constant_count: int = 9


@dataclass
class ContinuationFit:
    """Fitted positive amplitude and bounded beta continuation model."""

    amp_mode: str
    beta_mode: str
    amp_coef: np.ndarray
    amp_mean: np.ndarray
    amp_std: np.ndarray
    beta_coef: np.ndarray
    beta_mean: np.ndarray
    beta_std: np.ndarray
    beta_min: float
    beta_max: float
    pair_rmse: float
    triple_rmse: float
    pair_count: int
    triple_count: int

    @property
    def param_count(self) -> int:
        return int(len(self.amp_coef) + len(self.beta_coef))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def as_array(value: np.ndarray | float, length: int) -> np.ndarray:
    return np.full(length, float(value), dtype=float) if np.ndim(value) == 0 else np.asarray(value, dtype=float)


def stable_standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0) if x.shape[1] else np.array([], dtype=float)
    std = x.std(axis=0) if x.shape[1] else np.array([], dtype=float)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def stable_standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std if x.shape[1] else x


def data_paths(data: object) -> np.ndarray:
    if "path" in data.runs.columns:
        return data.runs["path"].astype(str).to_numpy()
    return np.asarray(["unknown"] * len(data.y), dtype=str)


def d_base(data: object) -> np.ndarray:
    return np.asarray(data.D, dtype=float) / np.asarray(data.mu, dtype=float)


def entropy_rows(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 1e-30, 1.0)
    return -(weights * np.log(clipped)).sum(axis=1)


def family_stats(feature_factory: EnhancedFeatureFactory, weights: np.ndarray, include_entropy_max: bool) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    family = feature_factory.family_shares(weights)
    if not include_entropy_max:
        return family
    p0 = weights[:, 0, :]
    p1 = weights[:, 1, :]
    return np.column_stack([family, entropy_rows(p0), np.max(p0, axis=1), entropy_rows(p1), np.max(p1, axis=1)])


class S2PowerBetaModel:
    """Anchored S2 base law plus an explicit same-mixture continuation law."""

    def __init__(
        self,
        spec: VariantSpec,
        data: object,
        feature_factory: EnhancedFeatureFactory,
        module: ModuleType,
        train_mask: np.ndarray,
    ):
        self.spec = spec
        self.model_id = spec.name
        self.data = data
        self.ff = feature_factory
        self.module = module
        self.train_mask = np.asarray(train_mask, dtype=bool)
        self.alpha, self.beta, self.gamma, self.delta = map(float, spec.exponents)
        self.log_n_mean = float(np.mean(np.log(data.N[self.train_mask])))
        self.log_n_std = float(np.std(np.log(data.N[self.train_mask])))
        self.log_n_std = self.log_n_std if self.log_n_std > 1e-12 else 1.0
        dbase = d_base(data)
        self.log_dbase_mean = float(np.mean(np.log(dbase[self.train_mask])))
        self.log_dbase_std = float(np.std(np.log(dbase[self.train_mask])))
        self.log_dbase_std = self.log_dbase_std if self.log_dbase_std > 1e-12 else 1.0

    def scale_features(
        self, weights: np.ndarray, model_size: np.ndarray, base_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        u_n = (np.log(model_size) - self.log_n_mean) / self.log_n_std
        u_d = (np.log(base_tokens) - self.log_dbase_mean) / self.log_dbase_std
        return u_n, u_d

    def continuation_raw_features(
        self,
        weights: np.ndarray,
        model_size: np.ndarray,
        base_tokens: np.ndarray,
        mode: str,
    ) -> np.ndarray:
        u_n, u_d = self.scale_features(weights, model_size, base_tokens)
        family = family_stats(self.ff, weights, include_entropy_max=False)
        entmax = family_stats(self.ff, weights, include_entropy_max=True)[:, 6:]
        if mode == "const":
            return np.empty((weights.shape[0], 0), dtype=float)
        if mode == "scale":
            return np.column_stack([u_n, u_d])
        if mode == "scale_p1fam3":
            return np.column_stack([u_n, u_d, family[:, 3:6]])
        if mode == "scale_fam6":
            return np.column_stack([u_n, u_d, family])
        if mode == "scale_fam10":
            return np.column_stack([u_n, u_d, family, entmax])
        raise ValueError(f"unknown continuation feature mode: {mode}")

    def continuation_design(
        self,
        weights: np.ndarray,
        model_size: np.ndarray | float,
        realized_tokens: np.ndarray | float,
        multiplier: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.asarray(weights, dtype=float)
        count = weights.shape[0]
        model_array = as_array(model_size, count)
        token_array = as_array(realized_tokens, count)
        multiplier_array = as_array(multiplier, count)
        base_tokens = token_array / multiplier_array
        amp_raw = self.continuation_raw_features(weights, model_array, base_tokens, self.spec.amp_mode)
        beta_raw = self.continuation_raw_features(weights, model_array, base_tokens, self.spec.beta_mode)
        return amp_raw, beta_raw, multiplier_array

    def beta_values(
        self,
        weights: np.ndarray,
        model_size: np.ndarray | float,
        realized_tokens: np.ndarray | float,
        multiplier: np.ndarray | float,
    ) -> np.ndarray:
        _, beta_raw, _ = self.continuation_design(weights, model_size, realized_tokens, multiplier)
        beta_x = stable_standardize(beta_raw, self.continuation.beta_mean, self.continuation.beta_std)
        linear = np.full(weights.shape[0], float(self.continuation.beta_coef[0]), dtype=float)
        if beta_x.shape[1]:
            linear = linear + beta_x @ self.continuation.beta_coef[1:]
        return self.continuation.beta_min + (self.continuation.beta_max - self.continuation.beta_min) * sigmoid(linear)

    def continuation_predict(
        self,
        weights: np.ndarray,
        model_size: np.ndarray | float,
        realized_tokens: np.ndarray | float,
        multiplier: np.ndarray | float,
    ) -> np.ndarray:
        amp_raw, _, multiplier_array = self.continuation_design(weights, model_size, realized_tokens, multiplier)
        amp_x = stable_standardize(amp_raw, self.continuation.amp_mean, self.continuation.amp_std)
        linear = np.full(weights.shape[0], float(self.continuation.amp_coef[0]), dtype=float)
        if amp_x.shape[1]:
            linear = linear + amp_x @ self.continuation.amp_coef[1:]
        amplitude = softplus(linear)
        beta = self.beta_values(weights, model_size, realized_tokens, multiplier)
        return amplitude * (np.power(multiplier_array, -beta) - 1.0)

    def fit_continuation(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_idx = np.flatnonzero(self.train_mask)
        triples = triple_table(self.data, train_idx)
        triples_fit = triples[np.isfinite(triples["beta_hat"])].copy()
        if triples_fit.empty:
            raise ValueError("No same-mixture triples available to fit continuation beta")

        beta_idx = triples_fit["idx_mu1"].to_numpy(dtype=int)
        _, beta_raw, _ = self.continuation_design(
            self.data.W[beta_idx],
            self.data.N[beta_idx],
            self.data.D[beta_idx],
            self.data.mu[beta_idx],
        )
        beta_mean, beta_std = stable_standardize_fit(beta_raw)
        beta_x = stable_standardize(beta_raw, beta_mean, beta_std)
        beta_y = triples_fit["beta_hat"].to_numpy(dtype=float)
        safe_beta = float(np.clip(np.nanmedian(beta_y), self.spec.beta_min + 1e-4, self.spec.beta_max - 1e-4))
        q0 = (safe_beta - self.spec.beta_min) / (self.spec.beta_max - self.spec.beta_min)
        beta_start = np.zeros(beta_x.shape[1] + 1, dtype=float)
        beta_start[0] = math.log(q0 / (1.0 - q0))
        if beta_x.shape[1] >= 2:
            beta_start[1:3] = -0.25
        drop_weight = np.sqrt(
            np.maximum(triples_fit[["drop_0p5_to_1", "drop_1_to_2"]].min(axis=1).to_numpy(dtype=float), 1e-4)
        )
        drop_weight = drop_weight / np.median(drop_weight)

        def beta_pred(theta: np.ndarray) -> np.ndarray:
            linear = theta[0]
            if beta_x.shape[1]:
                linear = linear + beta_x @ theta[1:]
            return self.spec.beta_min + (self.spec.beta_max - self.spec.beta_min) * sigmoid(linear)

        def beta_residual(theta: np.ndarray) -> np.ndarray:
            residuals = list((beta_pred(theta) - beta_y) * drop_weight)
            if self.spec.reg_beta > 0.0 and len(theta) > 1:
                residuals.extend(np.sqrt(self.spec.reg_beta) * theta[1:])
            return np.asarray(residuals, dtype=float)

        beta_result = least_squares(
            beta_residual,
            beta_start,
            loss="soft_l1",
            f_scale=self.spec.robust_beta_scale,
            max_nfev=5000,
        )
        triples_fit["pred_beta"] = beta_pred(beta_result.x)

        pairs = pair_table(self.data, train_idx)
        if pairs.empty:
            raise ValueError("No same-mixture pairs available to fit continuation amplitude")
        pair_i = pairs["idx_low_mu"].to_numpy(dtype=int)
        amp_raw, _, _ = self.continuation_design(
            self.data.W[pair_i],
            self.data.N[pair_i],
            self.data.D[pair_i],
            self.data.mu[pair_i],
        )
        amp_mean, amp_std = stable_standardize_fit(amp_raw)
        amp_x = stable_standardize(amp_raw, amp_mean, amp_std)

        partial_continuation = ContinuationFit(
            amp_mode=self.spec.amp_mode,
            beta_mode=self.spec.beta_mode,
            amp_coef=np.zeros(amp_x.shape[1] + 1, dtype=float),
            amp_mean=amp_mean,
            amp_std=amp_std,
            beta_coef=beta_result.x,
            beta_mean=beta_mean,
            beta_std=beta_std,
            beta_min=self.spec.beta_min,
            beta_max=self.spec.beta_max,
            pair_rmse=float("nan"),
            triple_rmse=float(np.sqrt(np.mean((triples_fit["pred_beta"].to_numpy(float) - beta_y) ** 2))),
            pair_count=len(pairs),
            triple_count=len(triples_fit),
        )
        self.continuation = partial_continuation
        beta_pair = self.beta_values(
            self.data.W[pair_i],
            self.data.N[pair_i],
            self.data.D[pair_i],
            self.data.mu[pair_i],
        )
        mu_low = pairs["mu_low"].to_numpy(dtype=float)
        mu_high = pairs["mu_high"].to_numpy(dtype=float)
        scale_gap = np.power(mu_low, -beta_pair) - np.power(mu_high, -beta_pair)
        y_pair = pairs["actual_diff"].to_numpy(dtype=float)
        amp0 = float(np.clip(np.median(y_pair / np.maximum(scale_gap, 1e-6)), 0.003, 0.20))
        amp_start = np.zeros(amp_x.shape[1] + 1, dtype=float)
        amp_start[0] = math.log(np.expm1(amp0))
        if amp_x.shape[1] >= 2:
            amp_start[1:3] = 0.15

        def amp_pred(theta: np.ndarray) -> np.ndarray:
            linear = theta[0]
            if amp_x.shape[1]:
                linear = linear + amp_x @ theta[1:]
            return softplus(linear)

        def amp_residual(theta: np.ndarray) -> np.ndarray:
            residuals = list(amp_pred(theta) * scale_gap - y_pair)
            if self.spec.reg_amp > 0.0 and len(theta) > 1:
                residuals.extend(np.sqrt(self.spec.reg_amp) * theta[1:])
            return np.asarray(residuals, dtype=float)

        amp_result = least_squares(amp_residual, amp_start, loss="linear", max_nfev=5000)
        pairs = pairs.copy()
        pairs["beta"] = beta_pair
        pairs["amplitude"] = amp_pred(amp_result.x)
        pairs["predicted_diff"] = pairs["amplitude"] * scale_gap
        pairs["pair_residual"] = pairs["predicted_diff"] - pairs["actual_diff"]
        self.continuation = ContinuationFit(
            amp_mode=self.spec.amp_mode,
            beta_mode=self.spec.beta_mode,
            amp_coef=amp_result.x,
            amp_mean=amp_mean,
            amp_std=amp_std,
            beta_coef=beta_result.x,
            beta_mean=beta_mean,
            beta_std=beta_std,
            beta_min=self.spec.beta_min,
            beta_max=self.spec.beta_max,
            pair_rmse=float(np.sqrt(np.mean(pairs["pair_residual"].to_numpy(dtype=float) ** 2))),
            triple_rmse=float(np.sqrt(np.mean((triples_fit["pred_beta"].to_numpy(float) - beta_y) ** 2))),
            pair_count=len(pairs),
            triple_count=len(triples_fit),
        )
        return pairs, triples_fit

    def base_scale_terms(
        self, model_size: np.ndarray, base_tokens: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_n = (model_size / self.data.N0) ** (-self.alpha) - 1.0
        u_d = (base_tokens / self.data.D0) ** (-self.beta) - 1.0
        u_nd = (model_size / self.data.N0) ** (-self.gamma) * (base_tokens / self.data.D0) ** (-self.delta) - 1.0
        return u_n, u_d, u_nd

    def base_design(self, weights: np.ndarray, model_size: np.ndarray, base_tokens: np.ndarray) -> np.ndarray:
        head_a = self.ff.scale_head_features(weights, self.spec.head_a)
        head_b = self.ff.scale_head_features(weights, self.spec.head_b)
        head_c = self.ff.scale_head_features(weights, self.spec.head_c)
        u_n, u_d, u_nd = self.base_scale_terms(model_size, base_tokens)
        return np.column_stack([head_a * u_n[:, None], head_b * u_d[:, None], head_c * u_nd[:, None]])

    def fit_base(self) -> None:
        dbase = d_base(self.data)
        continuation = self.continuation_predict(self.data.W, self.data.N, self.data.D, self.data.mu)
        base_target = self.data.y - continuation
        x_anchor_all, anchor_names = self.ff.anchor_features(self.data.W, self.spec.anchor_kind)
        anchor_rows = self.train_mask & (self.data.scale_names_by_row == "300m_6b") & np.isclose(self.data.mu, 1.0)
        if self.spec.base_row_mode == "all":
            base_rows_mask = self.train_mask
        elif self.spec.base_row_mode == "mu1":
            base_rows_mask = self.train_mask & np.isclose(self.data.mu, 1.0)
        else:
            raise ValueError(f"unknown base_row_mode: {self.spec.base_row_mode}")
        coef_anchor, stats_anchor = self.module.ridge_fit(
            x_anchor_all[anchor_rows],
            base_target[anchor_rows],
            ridge=self.spec.ridge_anchor,
        )
        anchor_pred_all = self.module.ridge_predict_from_stats(x_anchor_all, coef_anchor, stats_anchor)
        design_all = self.base_design(self.data.W, self.data.N, dbase)
        rows = np.flatnonzero(base_rows_mask)
        matrix = design_all[rows]
        target = base_target[rows] - anchor_pred_all[rows]
        if self.spec.ridge_scale > 0.0:
            matrix = np.vstack([matrix, math.sqrt(self.spec.ridge_scale) * np.eye(design_all.shape[1])])
            target = np.concatenate([target, np.zeros(design_all.shape[1], dtype=float)])
        result = lsq_linear(matrix, target, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto")
        self.anchor_coef = coef_anchor
        self.anchor_stats = stats_anchor
        self.anchor_names = anchor_names
        self.scale_coef = np.clip(result.x, 0.0, np.inf)
        self.anchor_feature_count = x_anchor_all.shape[1] + 1
        self.scale_param_count = len(self.scale_coef)
        self.base_fit_success = bool(result.success)
        self.total_constant_count = (
            self.anchor_feature_count
            + self.scale_param_count
            + 4
            + self.continuation.param_count
            + self.spec.donor_constant_count
        )

    def fit(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pairs, triples = self.fit_continuation()
        self.fit_base()
        return pairs, triples

    def predict_custom(
        self,
        weights: np.ndarray,
        model_size: np.ndarray | float,
        realized_tokens: np.ndarray | float,
        multiplier: np.ndarray | float = 1.0,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float)
        count = weights.shape[0]
        model_array = as_array(model_size, count)
        token_array = as_array(realized_tokens, count)
        multiplier_array = as_array(multiplier, count)
        base_tokens = token_array / multiplier_array
        x_anchor, _ = self.ff.anchor_features(weights, self.spec.anchor_kind)
        anchor = self.module.ridge_predict_from_stats(x_anchor, self.anchor_coef, self.anchor_stats)
        base = anchor + self.base_design(weights, model_array, base_tokens) @ self.scale_coef
        continuation = self.continuation_predict(weights, model_array, token_array, multiplier_array)
        return base + continuation

    def predict_all(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, self.data.D, self.data.mu)

    def predict_all_base_only(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, d_base(self.data), np.ones_like(self.data.mu))


def fit_s2_base(
    module: ModuleType, data: object, feature_factory: EnhancedFeatureFactory, train_mask: np.ndarray
) -> object:
    return module.TwoStagePowerLaw(
        "s2_base",
        data,
        feature_factory,
        "grp_famsqrt",
        (0.20, 0.25, 0.30, 0.65),
        pair_weight=6.0,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        donor_constant_count=9,
        head_A="constant",
        head_B="family",
        head_C="constant",
    ).fit(train_mask)


def fit_s2_anchor_safety(
    module: ModuleType,
    data: object,
    feature_factory: EnhancedFeatureFactory,
    train_mask: np.ndarray,
) -> object:
    return module.TwoStagePowerLaw(
        "s2_anchor_safety",
        data,
        feature_factory,
        "grp_famsqrt_safety",
        (0.20, 0.25, 0.30, 0.65),
        pair_weight=6.0,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        donor_constant_count=9,
        head_A="constant",
        head_B="family",
        head_C="constant",
    ).fit(train_mask)


def triple_table(data: object, train_idx: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "idx": train_idx,
            "mixture_id": data.mixture_ids[train_idx].astype(str),
            "scale": data.scale_names_by_row[train_idx].astype(str),
            "path": data_paths(data)[train_idx],
            "mu": data.mu[train_idx].astype(float),
            "y": data.y[train_idx].astype(float),
        }
    )
    rows = []
    for (mixture_id, scale, path), group in frame.groupby(["mixture_id", "scale", "path"], sort=False):
        by_mu = {float(row.mu): row for row in group.itertuples(index=False)}
        if not all(mu in by_mu for mu in (0.5, 1.0, 2.0)):
            continue
        loss_0p5 = float(by_mu[0.5].y)
        loss_1 = float(by_mu[1.0].y)
        loss_2 = float(by_mu[2.0].y)
        drop_0p5_to_1 = loss_0p5 - loss_1
        drop_1_to_2 = loss_1 - loss_2
        beta_hat = math.log(drop_0p5_to_1 / drop_1_to_2, 2) if drop_0p5_to_1 > 0.0 and drop_1_to_2 > 0.0 else np.nan
        rows.append(
            {
                "mixture_id": mixture_id,
                "scale": scale,
                "path": path,
                "idx_mu1": int(by_mu[1.0].idx),
                "loss_mu0p5": loss_0p5,
                "loss_mu1": loss_1,
                "loss_mu2": loss_2,
                "drop_0p5_to_1": drop_0p5_to_1,
                "drop_1_to_2": drop_1_to_2,
                "beta_hat": float(beta_hat),
            }
        )
    return pd.DataFrame(rows)


def pair_table(data: object, train_idx: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "idx": train_idx,
            "mixture_id": data.mixture_ids[train_idx].astype(str),
            "scale": data.scale_names_by_row[train_idx].astype(str),
            "path": data_paths(data)[train_idx],
            "mu": data.mu[train_idx].astype(float),
            "y": data.y[train_idx].astype(float),
        }
    )
    rows = []
    for (mixture_id, scale, path), group in frame.groupby(["mixture_id", "scale", "path"], sort=False):
        if group["mu"].nunique() < 2:
            continue
        ordered = group.sort_values("mu").to_dict("records")
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                low = ordered[left]
                high = ordered[right]
                rows.append(
                    {
                        "mixture_id": mixture_id,
                        "scale": scale,
                        "path": path,
                        "idx_low_mu": int(low["idx"]),
                        "idx_high_mu": int(high["idx"]),
                        "mu_low": float(low["mu"]),
                        "mu_high": float(high["mu"]),
                        "actual_diff": float(low["y"] - high["y"]),
                    }
                )
    return pd.DataFrame(rows)


def variants() -> list[VariantSpec]:
    return [
        VariantSpec("s2pb_const_beta", beta_mode="const", amp_mode="scale_fam6", reg_amp=0.01, reg_beta=0.00),
        VariantSpec("s2pb_scale_beta", beta_mode="scale", amp_mode="scale_fam6", reg_amp=0.01, reg_beta=0.01),
        VariantSpec("s2pb_p1fam_beta", beta_mode="scale_p1fam3", amp_mode="scale_fam6", reg_amp=0.01, reg_beta=0.01),
        VariantSpec("s2pb_fam10_amp", beta_mode="scale_p1fam3", amp_mode="scale_fam10", reg_amp=0.01, reg_beta=0.01),
        VariantSpec(
            "s2pb_mu1_base_const_beta",
            beta_mode="const",
            amp_mode="scale_fam6",
            base_row_mode="mu1",
            reg_amp=0.01,
            reg_beta=0.00,
        ),
        VariantSpec(
            "s2pb_mu1_base_p1fam_beta",
            beta_mode="scale_p1fam3",
            amp_mode="scale_fam6",
            base_row_mode="mu1",
            reg_amp=0.01,
            reg_beta=0.01,
        ),
        VariantSpec(
            "s2pb_mu1_base_anchor_safety",
            anchor_kind="grp_famsqrt_safety",
            beta_mode="scale_p1fam3",
            amp_mode="scale_fam6",
            base_row_mode="mu1",
            reg_amp=0.01,
            reg_beta=0.01,
        ),
        VariantSpec(
            "s2pb_anchor_safety",
            anchor_kind="grp_famsqrt_safety",
            beta_mode="scale_p1fam3",
            amp_mode="scale_fam6",
            reg_amp=0.01,
            reg_beta=0.01,
        ),
        VariantSpec(
            "s2pb_family_quality",
            anchor_kind="grp_famsqrt_safety",
            head_a="family",
            head_b="family_quality",
            head_c="constant",
            beta_mode="scale_p1fam3",
            amp_mode="scale_fam6",
            reg_amp=0.03,
            reg_beta=0.01,
            ridge_scale=3e-5,
        ),
        VariantSpec(
            "s2pb_beta036_base",
            exponents=(0.20, 0.36, 0.30, 0.65),
            beta_mode="scale_p1fam3",
            amp_mode="scale_fam6",
            reg_amp=0.01,
            reg_beta=0.01,
        ),
    ]


def fit_powerbeta_variant(
    module: ModuleType,
    data: object,
    feature_factory: EnhancedFeatureFactory,
    train_mask: np.ndarray,
    spec: VariantSpec,
) -> tuple[S2PowerBetaModel, pd.DataFrame, pd.DataFrame]:
    model = S2PowerBetaModel(spec, data, feature_factory, module, train_mask)
    pairs, triples = model.fit()
    return model, pairs, triples


def evaluate_predictions(data: object, models: dict[str, object], predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    subsets = {
        "seed7_holdout": data.seed7_holdout,
        "fixed_340m_10p4b": data.fixed340,
        "random_supplement": data.random_supplement,
        "all_900m_24b_seed7_fit": data.all900_holdout,
    }
    rows = []
    for name, pred in predictions.items():
        model = models[name]
        params = int(getattr(model, "total_constant_count", getattr(model, "param_count", -1)))
        for subset, mask in subsets.items():
            row = metric_dict(data.y[mask], pred[mask])
            row.update(model=name, subset=subset, params=params)
            rows.append(row)
    return pd.DataFrame(rows)


def all900_protocol(
    module: ModuleType,
    data: object,
    feature_factory: EnhancedFeatureFactory,
    specs: list[VariantSpec],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    predictions = []
    idx900 = np.flatnonzero(data.all900_holdout)
    for spec in specs:
        model, _, _ = fit_powerbeta_variant(module, data, feature_factory, data.all900_train, spec)
        pred = np.asarray(model.predict_all(), dtype=float)
        row = metric_dict(data.y[data.all900_holdout], pred[data.all900_holdout])
        row.update(
            model=spec.name,
            subset="all_900m_24b",
            train_regime="all_non_900m_train",
            params=int(model.total_constant_count),
        )
        rows.append(row)
        frame = pd.DataFrame(
            {
                "registry_run_key": data.registry_keys[idx900].astype(str),
                "mixture_id": data.mixture_ids[idx900].astype(str),
                "scale": data.scale_names_by_row[idx900].astype(str),
                "mu": data.mu[idx900].astype(float),
                "actual": data.y[idx900].astype(float),
                "all_900m_24b": True,
                "model": spec.name,
                "prediction": pred[idx900],
                "residual": pred[idx900] - data.y[idx900],
            }
        )
        predictions.append(frame)
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def predictions_frame(data: object, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "registry_run_key": data.registry_keys.astype(str),
            "mixture_id": data.mixture_ids.astype(str),
            "scale": data.scale_names_by_row.astype(str),
            "mu": data.mu.astype(float),
            "actual": data.y.astype(float),
            "seed7_holdout": data.seed7_holdout,
            "fixed_340m_10p4b": data.fixed340,
            "all_900m_24b": data.all900_holdout,
        }
    )
    for name, pred in predictions.items():
        frame[name] = pred
        frame[f"{name}_residual"] = pred - data.y
    return frame


def structural_sanity(data: object, models: dict[str, object]) -> pd.DataFrame:
    seen = {}
    for idx, mixture_id in enumerate(data.mixture_ids.astype(str)):
        if mixture_id not in seen and data.seed7_train[idx]:
            seen[mixture_id] = idx
    mix_indices = np.asarray(list(seen.values()), dtype=int)
    if len(mix_indices) > 200:
        mix_indices = mix_indices[np.linspace(0, len(mix_indices) - 1, 200).round().astype(int)]
    weights = data.W[mix_indices]
    n_grid = np.array([22_813_184, 58_998_528, 102_648_576, 339_788_800, 906_037_248], dtype=float)
    d_grid = np.array([1_199_833_088, 2_599_944_192, 5_999_951_872, 10_399_776_768, 23_999_807_488], dtype=float)
    mu_grid = np.array([0.5, 1.0, 2.0], dtype=float)
    rows = []
    for name, model in models.items():
        if isinstance(model, S2PowerBetaModel):
            predict = model.predict_custom
        else:

            def predict(
                weight_batch: np.ndarray,
                model_size: float,
                tokens: float,
                multiplier: float = 1.0,
                *,
                _model=model,
            ) -> np.ndarray:
                return _model.predict_custom(
                    weight_batch,
                    model_size,
                    tokens,
                )

        n_preds = np.vstack([predict(weights, float(model_size), 5_999_951_872.0, 1.0) for model_size in n_grid]).T
        d_preds = np.vstack([predict(weights, 102_648_576.0, float(tokens), 1.0) for tokens in d_grid]).T
        mu_preds = np.vstack(
            [predict(weights, 339_788_800.0, 10_399_776_768.0 * float(mu), float(mu)) for mu in mu_grid]
        ).T
        for axis, values in (("N", n_preds), ("D_base", d_preds), ("mu", mu_preds)):
            deltas = np.diff(values, axis=1)
            rows.append(
                {
                    "model": name,
                    "axis": axis,
                    "checks": int(deltas.size),
                    "violations": int(np.sum(deltas > 1e-10)),
                    "violation_rate": float(np.mean(deltas > 1e-10)),
                    "max_positive_delta": float(np.max(deltas)),
                    "mean_delta": float(np.mean(deltas)),
                    "median_delta": float(np.median(deltas)),
                    "min_delta": float(np.min(deltas)),
                }
            )
    return pd.DataFrame(rows)


def powerbeta_diagnostics(models: dict[str, object], data: object) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        if not isinstance(model, S2PowerBetaModel):
            continue
        beta = model.beta_values(data.W, data.N, data.D, data.mu)
        cont = model.continuation_predict(data.W, data.N, data.D, data.mu)
        for subset, mask in {
            "seed7_train": data.seed7_train,
            "seed7_holdout": data.seed7_holdout,
            "fixed_340m_10p4b": data.fixed340,
            "all_900m_24b": data.all900_holdout,
        }.items():
            rows.append(
                {
                    "model": name,
                    "subset": subset,
                    "rows": int(np.sum(mask)),
                    "beta_mean": float(np.mean(beta[mask])),
                    "beta_min": float(np.min(beta[mask])),
                    "beta_max": float(np.max(beta[mask])),
                    "continuation_mean": float(np.mean(cont[mask])),
                    "continuation_min": float(np.min(cont[mask])),
                    "continuation_max": float(np.max(cont[mask])),
                    "pair_rmse": float(model.continuation.pair_rmse),
                    "triple_rmse": float(model.continuation.triple_rmse),
                    "pair_count": int(model.continuation.pair_count),
                    "triple_count": int(model.continuation.triple_count),
                }
            )
    return pd.DataFrame(rows)


def plot_all900_protocol(
    protocol_predictions: pd.DataFrame, metrics: pd.DataFrame, models: list[str], out_path: Path
) -> None:
    if protocol_predictions.empty:
        return
    subset = protocol_predictions[protocol_predictions["model"].isin(models)].copy()
    lo = float(min(subset["actual"].min(), subset["prediction"].min()) - 0.008)
    hi = float(max(subset["actual"].max(), subset["prediction"].max()) + 0.008)
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.3), constrained_layout=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("RdYlGn_r")
    scatter = None
    for ax, model in zip(axes, models, strict=False):
        rows = subset[subset["model"] == model]
        metric = metrics[metrics["model"] == model]
        title_metrics = "metrics unavailable"
        if not metric.empty:
            first = metric.iloc[0]
            title_metrics = f"RMSE={first.rmse:.4f}, Sp={first.spearman:.3f}, slope={first.slope:.3f}"
        scatter = ax.scatter(
            rows["actual"],
            rows["prediction"],
            c=rows["mu"],
            cmap=cmap,
            edgecolor="black",
            linewidth=0.4,
            s=46,
        )
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.0)
        ax.set_title(f"{model}\n{title_metrics}")
        ax.set_xlabel("Actual BPB")
        ax.set_ylabel("Predicted BPB")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.25)
    if scatter is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="target-budget multiplier")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def select_plot_models(metrics: pd.DataFrame, all900: pd.DataFrame, count: int = 5) -> list[str]:
    names = ["s2_base"]
    fixed = metrics[metrics["subset"] == "fixed_340m_10p4b"].sort_values("rmse")
    all900_ranked = all900.sort_values("rmse")
    for source in (fixed["model"], all900_ranked["model"]):
        for name in source:
            if name not in names:
                names.append(str(name))
            if len(names) >= count:
                return names
    return names


def write_report(
    out_dir: Path,
    metrics: pd.DataFrame,
    all900: pd.DataFrame,
    drops: pd.DataFrame,
    optima: pd.DataFrame,
    structural: pd.DataFrame,
    continuation: pd.DataFrame,
    plot_models: list[str],
) -> None:
    fixed = metrics[metrics["subset"] == "fixed_340m_10p4b"].sort_values("rmse")
    holdout = metrics[metrics["subset"] == "seed7_holdout"].sort_values("rmse")
    all900_sorted = all900.sort_values("rmse")
    raw_340 = optima[(optima["target_scale"] == "520m_10p4b") & (optima["opt_kind"] == "raw_random_search")]
    lines = [
        "# S2 Power-Beta Extension Sprint 2026-04-24",
        "",
        (
            "This sprint tests whether the power-beta continuation idea can be grafted onto the clean S2 "
            "anchored Chinchilla law."
        ),
        "",
        "Structural form:",
        "",
        "```text",
        "D_base = D / mu",
        "L(w,N,D,mu) = E(w) + A(w)u_N + B(w)u_Dbase + C(w)u_NDbase",
        "              + G(w,N,D_base) * (mu^-beta(w,N,D_base) - 1)",
        "```",
        "",
        "## Best Fixed-340M Metrics",
        "",
        fixed[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Best Seed-7 Holdout Metrics",
        "",
        holdout[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## All-900M Leave-Scale-Out Metrics",
        "",
        all900_sorted[
            ["model", "params", "rmse", "bias", "spearman", "slope", "std_ratio", "regret_at_1", "lower_tail_optimism"]
        ]
        .head(12)
        .to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Fixed-340M Drop Summary",
        "",
        drops.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Structural Sanity",
        "",
        structural.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Continuation Diagnostics",
        "",
        continuation.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw 340M Optimum Diagnostics",
        "",
        raw_340[
            [
                "model",
                "predicted_bpb",
                "hard_corner_flag",
                "phase0_family_collapse_flag",
                "phase1_family_collapse_flag",
                "phase1_tech_collapse_flag",
                "p0_support_inv_l2",
                "p1_support_inv_l2",
                "p0_broad_text_share",
                "p1_broad_text_share",
                "p0_tech_code_share",
                "p1_tech_code_share",
                "p0_reasoning_share",
                "p1_reasoning_share",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Plot Models",
        "",
        ", ".join(plot_models),
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def serialize_model(model: S2PowerBetaModel) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "formula": "base(w,N,D/mu)+G(w,N,D/mu)*(mu^-beta(w,N,D/mu)-1)",
        "spec": model.spec.__dict__,
        "param_count": int(model.total_constant_count),
        "anchor_feature_count": int(model.anchor_feature_count),
        "scale_param_count": int(model.scale_param_count),
        "continuation_param_count": int(model.continuation.param_count),
        "continuation_pair_rmse": float(model.continuation.pair_rmse),
        "continuation_triple_rmse": float(model.continuation.triple_rmse),
        "base_fit_success": bool(model.base_fit_success),
        "anchor_features": model.anchor_names,
        "scale_coefficients": model.scale_coef.tolist(),
        "continuation": {
            "amp_mode": model.continuation.amp_mode,
            "beta_mode": model.continuation.beta_mode,
            "amp_coef": model.continuation.amp_coef.tolist(),
            "amp_mean": model.continuation.amp_mean.tolist(),
            "amp_std": model.continuation.amp_std.tolist(),
            "beta_coef": model.continuation.beta_coef.tolist(),
            "beta_mean": model.continuation.beta_mean.tolist(),
            "beta_std": model.continuation.beta_std.tolist(),
            "beta_min": float(model.continuation.beta_min),
            "beta_max": float(model.continuation.beta_max),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, default=DEFAULT_PACKET_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    module = load_module(SESSION2_SCRIPT)
    data = module.load_packet(args.packet_root)
    feature_factory = EnhancedFeatureFactory(module.FeatureFactory(data), module)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "models").mkdir(parents=True, exist_ok=True)

    specs = variants()
    models: dict[str, object] = {
        "s2_base": fit_s2_base(module, data, feature_factory, data.seed7_train),
        "s2_anchor_safety": fit_s2_anchor_safety(module, data, feature_factory, data.seed7_train),
    }
    predictions: dict[str, np.ndarray] = {
        name: np.asarray(model.predict_all(), dtype=float) for name, model in models.items()
    }
    pair_frames = []
    triple_frames = []
    for spec in specs:
        model, pairs, triples = fit_powerbeta_variant(module, data, feature_factory, data.seed7_train, spec)
        models[spec.name] = model
        predictions[spec.name] = np.asarray(model.predict_all(), dtype=float)
        pairs.insert(0, "model", spec.name)
        triples.insert(0, "model", spec.name)
        pair_frames.append(pairs)
        triple_frames.append(triples)
        with (args.out_dir / "models" / f"{spec.name}_seed7_model.json").open("w") as handle:
            json.dump(serialize_model(model), handle, indent=2)

    metrics = evaluate_predictions(data, models, predictions)
    all900, all900_predictions = all900_protocol(module, data, feature_factory, specs)
    drop_details, drop_summary = fixed_drop_summary(data, predictions)
    pred_frame = predictions_frame(data, predictions)
    optima = optimum_diagnostics(data, feature_factory, models, np.random.default_rng(20260424))
    structural = structural_sanity(data, models)
    continuation = powerbeta_diagnostics(models, data)
    plot_models = select_plot_models(metrics, all900)

    metrics.to_csv(args.out_dir / "metrics_seed7_fit.csv", index=False)
    all900.to_csv(args.out_dir / "all900_protocol_metrics.csv", index=False)
    all900_predictions.to_csv(args.out_dir / "all900_protocol_predictions.csv", index=False)
    drop_details.to_csv(args.out_dir / "fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(args.out_dir / "fixed340_drop_summary.csv", index=False)
    pred_frame.to_csv(args.out_dir / "predictions_seed7_fit.csv", index=False)
    optima.to_csv(args.out_dir / "optimum_diagnostics.csv", index=False)
    structural.to_csv(args.out_dir / "structural_sanity.csv", index=False)
    continuation.to_csv(args.out_dir / "continuation_diagnostics.csv", index=False)
    pd.concat(pair_frames, ignore_index=True).to_csv(args.out_dir / "continuation_pair_fits.csv", index=False)
    pd.concat(triple_frames, ignore_index=True).to_csv(args.out_dir / "continuation_triple_fits.csv", index=False)
    with (args.out_dir / "variant_specs.json").open("w") as handle:
        json.dump([spec.__dict__ for spec in specs], handle, indent=2)

    plot_predicted_vs_actual(
        pred_frame,
        metrics,
        plot_models,
        "fixed_340m_10p4b",
        "fixed_340m_10p4b",
        args.out_dir / "plots/fixed340_pred_vs_actual_top_variants.png",
    )
    plot_predicted_vs_actual(
        pred_frame,
        metrics,
        plot_models,
        "all_900m_24b",
        "all_900m_24b_seed7_fit",
        args.out_dir / "plots/all900_pred_vs_actual_seed7_fit_top_variants.png",
    )
    plot_all900_protocol(
        all900_predictions,
        all900,
        [name for name in plot_models if name.startswith("s2pb")],
        args.out_dir / "plots/all900_pred_vs_actual_protocol_top_variants.png",
    )
    write_report(args.out_dir, metrics, all900, drop_summary, optima, structural, continuation, plot_models)

    print("Best fixed-340M:")
    print(
        metrics[metrics["subset"] == "fixed_340m_10p4b"]
        .sort_values("rmse")[["model", "params", "rmse", "spearman", "slope", "std_ratio"]]
        .head(10)
        .to_string(index=False)
    )
    print("\nAll-900M protocol:")
    print(
        all900.sort_values("rmse")[["model", "params", "rmse", "spearman", "slope", "std_ratio"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"\nWrote {args.out_dir}")


if __name__ == "__main__":
    main()
