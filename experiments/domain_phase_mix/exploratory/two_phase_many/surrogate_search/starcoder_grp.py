# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable GRP helpers for the 2-phase StarCoder packet."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.optimize import minimize, nnls
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PacketData,
    STARCODER_TARGET,
    load_two_phase_starcoder_packet,
    regression_metrics,
    safe_exp,
    softplus,
)

SHAPE_PARAMETER_COUNT = 5
DEFAULT_CV_SEED = 0
DEFAULT_TUNE_SEED = 0


def subset_packet(packet: PacketData, mask: np.ndarray) -> PacketData:
    """Return a row-filtered packet with shared metadata."""
    return replace(
        packet,
        frame=packet.frame.loc[mask].reset_index(drop=True),
        y=packet.y[mask],
        w=packet.w[mask],
    )


def load_completed_two_phase_starcoder_packet(target: str = STARCODER_TARGET) -> PacketData:
    """Load the completed 2-phase StarCoder packet."""
    packet = load_two_phase_starcoder_packet(target=target)
    if "status" not in packet.frame.columns:
        return packet
    mask = packet.frame["status"].eq("completed").to_numpy(dtype=bool)
    return subset_packet(packet, mask)


class StarcoderGRPSurrogate:
    """Two-family GRP surrogate for the 2-domain StarCoder packet."""

    def __init__(self, data: PacketData, params: dict[str, float]):
        self.data = data
        self.params = params.copy()
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def _retained_x(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]
        lam = float(self.params["lam"])
        eta = float(self.params["eta"])
        retained = np.exp(-lam * (1.0 - p1))
        return retained * e0 + eta * e1

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        alpha = float(self.params["alpha"])
        tau = float(self.params["tau"])
        x = self._retained_x(weights)

        broad_signal = np.log1p(alpha * x[:, 0])
        tech_signal = np.log1p(alpha * x[:, 1])
        penalty = softplus(np.log1p(x[:, 0]) - tau) ** 2 + softplus(np.log1p(x[:, 1]) - tau) ** 2
        return np.column_stack([-broad_signal, -tech_signal, penalty]).astype(float)

    def fit(self, weights: np.ndarray, y: np.ndarray) -> StarcoderGRPSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        design_centered = design - design_mean
        y_centered = y - y_mean

        reg = float(self.params["reg"])
        if reg > 0.0:
            design_centered = np.vstack([design_centered, np.sqrt(reg) * np.eye(design_centered.shape[1])])
            y_centered = np.concatenate([y_centered, np.zeros(design_centered.shape[1], dtype=float)])

        coef, _ = nnls(design_centered, y_centered)
        self.coef_ = coef
        self.intercept_ = float(y_mean - (design_mean @ coef).item())
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        return np.asarray(self.intercept_ + self.build_design(weights) @ self.coef_, dtype=float)


def optimize_starcoder_grp(packet: PacketData, *, seed: int = DEFAULT_TUNE_SEED) -> dict[str, float]:
    """Tune the GRP shape parameters with Nelder-Mead."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    def unpack(z: np.ndarray) -> dict[str, float]:
        return {
            "alpha": safe_exp(z[0], -8.0, 8.0),
            "eta": safe_exp(z[1], -8.0, 8.0),
            "lam": max(safe_exp(z[2], -12.0, 8.0) - 1e-8, 1e-8),
            "tau": float(np.clip(z[3], -4.0, 15.0)),
            "reg": safe_exp(z[4], -18.0, 0.0),
        }

    def objective(z: np.ndarray) -> float:
        params = unpack(z)
        yhat = np.zeros_like(packet.y)
        regrets: list[float] = []
        for _fold, (tr, te) in enumerate(kf.split(packet.w)):
            model = StarcoderGRPSurrogate(packet, params).fit(packet.w[tr], packet.y[tr])
            pred = model.predict(packet.w[te])
            yhat[te] = pred
            regrets.append(float(packet.y[te][np.argmin(pred)] - np.min(packet.y[te])))
        rmse = float(np.sqrt(np.mean((yhat - packet.y) ** 2)))
        return rmse + 0.02 * float(np.mean(regrets))

    starts = [
        np.array([0.0, 0.0, 0.0, 3.0, -8.0], dtype=float),
        np.array([0.0, 2.0, -1.0, 5.0, -8.0], dtype=float),
        np.array([1.0, 2.0, -2.0, 7.0, -10.0], dtype=float),
        np.array([-1.0, 1.0, -1.0, 3.0, -10.0], dtype=float),
    ]

    best_params: dict[str, float] | None = None
    best_obj = float("inf")
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 700, "xatol": 1e-4, "fatol": 1e-5},
        )
        params = unpack(result.x)
        value = float(result.fun)
        if value < best_obj:
            best_obj = value
            best_params = params

    if best_params is None:
        raise RuntimeError("Failed to optimize StarCoder GRP parameters")
    return best_params


def fit_starcoder_grp(
    packet: PacketData,
    *,
    params: dict[str, float] | None = None,
    seed: int = DEFAULT_TUNE_SEED,
) -> tuple[dict[str, float], StarcoderGRPSurrogate]:
    """Fit the 2-family GRP StarCoder surrogate."""
    final_params = optimize_starcoder_grp(packet, seed=seed) if params is None else dict(params)
    model = StarcoderGRPSurrogate(packet, final_params).fit(packet.w, packet.y)
    return final_params, model


def total_parameter_count(model: StarcoderGRPSurrogate) -> int:
    """Return total parameter count including intercept and nonlinear shapes."""
    if model.coef_ is None:
        raise RuntimeError("Model must be fit before counting parameters")
    return len(model.coef_) + 1 + SHAPE_PARAMETER_COUNT


def compute_starcoder_grp_metrics(
    *,
    cv_seed: int = DEFAULT_CV_SEED,
    tune_seed: int = DEFAULT_TUNE_SEED,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return train and fixed-param CV metrics for GRP on 2-phase StarCoder."""
    packet = load_completed_two_phase_starcoder_packet()
    params, model = fit_starcoder_grp(packet, params=params, seed=tune_seed)

    train_pred = model.predict(packet.w)
    train = regression_metrics(packet.frame, packet.name_col, packet.y, train_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(packet.y, dtype=float)
    fold_regrets: list[float] = []
    for tr, te in kf.split(packet.w):
        fold_model = StarcoderGRPSurrogate(packet, params).fit(packet.w[tr], packet.y[tr])
        pred = fold_model.predict(packet.w[te])
        oof[te] = pred
        chosen = int(np.argmin(pred))
        fold_regrets.append(float(packet.y[te][chosen] - np.min(packet.y[te])))

    cv = regression_metrics(packet.frame, packet.name_col, packet.y, oof)
    return {
        "model": "GRP",
        "dataset": "two_phase_starcoder",
        "status": "ok",
        "n_runs": len(packet.y),
        "n_params": total_parameter_count(model),
        "train_r2": float(train["r2"]),
        "train_rmse": float(train["rmse"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(train["regret_at_1"]),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "params": params,
    }
