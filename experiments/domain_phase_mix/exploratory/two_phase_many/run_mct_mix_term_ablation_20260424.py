#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ablate mixture-dependent terms in the barrier-free MCT-LRQ law.

This compares the current barrier-free MCT predictive law against controls that
do not learn smooth mixture functions:

1. full barrier-free MCT-LRQ: LRQ mixture anchor plus family-dependent D head;
2. LRQ anchor with only global scale terms;
3. identity-anchor transfer: one reference-scale anchor per observed mixture ID
   plus only global scale terms;
4. pure scale-only curve: no mixture anchor and only global scale terms.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, minimize

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_MCT_CODE_DIR = SCRIPT_DIR / "reference_outputs" / "mct_barrier_ablation_20260424" / "code"
FALLBACK_MCT_CODE_DIR = Path(
    "/tmp/chatgpt_pro_session_12_review/"
    "5_joint_mixture_scale_structural_v31_mct/"
    "joint_mixture_scale_structural_v31_mct/code"
)

MCT_BALANCED_EXPONENTS = (0.148968, 0.209383, 0.009859, 1.043436)
MCT_PAIR_WEIGHT = 4.0


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def jsonify(x):
    if isinstance(x, dict):
        return {str(k): jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonify(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def md_table(df: pd.DataFrame, cols: list[str] | None = None, floatfmt: str = ".6f") -> str:
    if cols is not None:
        df = df[cols].copy()
    if len(df) == 0:
        return "_empty_"
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"


def scale_terms(data, exponents: tuple[float, float, float, float], N: np.ndarray, D: np.ndarray):
    alpha, beta, gamma, delta = map(float, exponents)
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    uN = (N / data.N0) ** (-alpha) - 1.0
    uD = (D / data.D0) ** (-beta) - 1.0
    uND = (N / data.N0) ** (-gamma) * (D / data.D0) ** (-delta) - 1.0
    return uN, uD, uND


def make_pairs(data, mask: np.ndarray) -> list[tuple[int, int]]:
    df = pd.DataFrame(
        {
            "idx": np.arange(len(mask)),
            "scale": data.scale_names_by_row,
            "mix": data.mixture_ids,
            "D": data.D,
        }
    )
    df = df[np.asarray(mask, dtype=bool)].copy()
    pairs: list[tuple[int, int]] = []
    for _, group in df.groupby(["scale", "mix"]):
        if group["D"].nunique() < 2:
            continue
        rows = group.sort_values("D")["idx"].to_numpy()
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                pairs.append((int(rows[a]), int(rows[b])))
    return pairs


class GlobalScaleFitMixin:
    """Common fitting logic for models with only global scale coefficients."""

    def _fit_global_scale(
        self,
        train_mask: np.ndarray,
        anchor_all: np.ndarray,
        pair_weight: float,
        ridge_scale: float,
    ) -> None:
        data = self.data
        self.train_mask = np.asarray(train_mask, dtype=bool).copy()
        uN, uD, uND = scale_terms(data, self.exponents, data.N, data.D)
        design_all = np.column_stack([uN, uD, uND])
        y_resid = data.y - anchor_all

        rows = np.where(self.train_mask)[0]
        A = design_all[rows]
        b = y_resid[rows]
        pair_rows: list[np.ndarray] = []
        pair_y: list[float] = []
        for i, j in make_pairs(data, self.train_mask):
            weight = math.sqrt(pair_weight)
            pair_rows.append(weight * (design_all[i] - design_all[j]))
            pair_y.append(weight * (y_resid[i] - y_resid[j]))
        if pair_rows:
            A = np.vstack([A, np.asarray(pair_rows)])
            b = np.concatenate([b, np.asarray(pair_y)])
        if ridge_scale > 0:
            A = np.vstack([A, np.sqrt(ridge_scale) * np.eye(design_all.shape[1])])
            b = np.concatenate([b, np.zeros(design_all.shape[1])])
        result = lsq_linear(A, b, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto", verbose=0)
        self.scale_coef = np.clip(np.asarray(result.x, dtype=float), 0.0, np.inf)
        self.scale_fit_success = bool(result.success)
        self.scale_fit_cost = float(result.cost)
        self.pair_count = len(pair_rows)

    def _scale_prediction(self, N: np.ndarray | float, D: np.ndarray | float, count: int) -> np.ndarray:
        N_arr = np.full(count, float(N)) if np.ndim(N) == 0 else np.asarray(N, dtype=float)
        D_arr = np.full(count, float(D)) if np.ndim(D) == 0 else np.asarray(D, dtype=float)
        uN, uD, uND = scale_terms(self.data, self.exponents, N_arr, D_arr)
        return np.column_stack([uN, uD, uND]) @ self.scale_coef

    def offset(self, W: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(W).shape[0], dtype=float)

    def predict_all(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, self.data.D)

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        n = np.asarray(W).shape[0]
        return pd.DataFrame(
            {
                "A_N": np.full(n, float(self.scale_coef[0])),
                "B_D": np.full(n, float(self.scale_coef[1])),
                "C_ND": np.full(n, float(self.scale_coef[2])),
            }
        )


class ConstantScaleOnlyLaw(GlobalScaleFitMixin):
    """Pure scaling curve with no mixture dependence."""

    def __init__(
        self,
        model_id: str,
        data,
        exponents: tuple[float, float, float, float],
        pair_weight: float = MCT_PAIR_WEIGHT,
        ridge_scale: float = 1e-5,
    ):
        self.model_id = model_id
        self.data = data
        self.exponents = exponents
        self.pair_weight = float(pair_weight)
        self.ridge_scale = float(ridge_scale)
        self.anchor_feature_count = 1
        self.scale_param_count = 3
        self.donor_constant_count = 0
        self.barrier_constant_count = 0
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 4
        self.total_constant_count = self.fitted_param_count

    def fit(self, train_mask: np.ndarray):
        self.intercept = float(np.mean(self.data.y[np.asarray(train_mask, dtype=bool)]))
        anchor_all = np.full(len(self.data.y), self.intercept)
        self._fit_global_scale(train_mask, anchor_all, self.pair_weight, self.ridge_scale)
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        return self.intercept + self._scale_prediction(N, D, W.shape[0])

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": "b0 + a*((N/N0)^(-alpha)-1) + b*((D/D0)^(-beta)-1) + c*((N/N0)^(-gamma)*(D/D0)^(-delta)-1)",
            "mixture_dependence": "none",
            "intercept": self.intercept,
            "scale_coef": self.scale_coef,
            "exponents": self.exponents,
            "pair_weight": self.pair_weight,
            "ridge_scale": self.ridge_scale,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
        }


class ChinchillaApproach3ScaleOnlyLaw:
    """Pure scale-only Chinchilla Approach 3 law.

    L(N,D) = E + A * ((N/N0)^(-alpha) - 1) + B * ((D/D0)^(-beta) - 1)

    This deliberately has no mixture anchor, no mixture-dependent scale heads,
    and no MCT cross term.
    """

    def __init__(
        self,
        model_id: str,
        data,
        pair_weight: float = MCT_PAIR_WEIGHT,
        ridge_scale: float = 1e-5,
    ):
        self.model_id = model_id
        self.data = data
        self.pair_weight = float(pair_weight)
        self.ridge_scale = float(ridge_scale)
        self.anchor_feature_count = 1
        self.scale_param_count = 2
        self.donor_constant_count = 0
        self.barrier_constant_count = 0
        self.fitted_param_count = 5
        self.total_constant_count = self.fitted_param_count

    def _design_for_exponents(self, alpha: float, beta: float, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        uN = (np.asarray(N, dtype=float) / self.data.N0) ** (-float(alpha)) - 1.0
        uD = (np.asarray(D, dtype=float) / self.data.D0) ** (-float(beta)) - 1.0
        return np.column_stack([np.ones_like(uN), uN, uD])

    def _fit_linear(self, train_mask: np.ndarray, alpha: float, beta: float):
        data = self.data
        design_all = self._design_for_exponents(alpha, beta, data.N, data.D)
        rows = np.where(train_mask)[0]
        A = design_all[rows]
        b = data.y[rows]
        pair_rows: list[np.ndarray] = []
        pair_y: list[float] = []
        for i, j in make_pairs(data, train_mask):
            weight = math.sqrt(self.pair_weight)
            pair_rows.append(weight * (design_all[i] - design_all[j]))
            pair_y.append(weight * (data.y[i] - data.y[j]))
        if pair_rows:
            A = np.vstack([A, np.asarray(pair_rows)])
            b = np.concatenate([b, np.asarray(pair_y)])
        if self.ridge_scale > 0:
            ridge = np.diag([0.0, math.sqrt(self.ridge_scale), math.sqrt(self.ridge_scale)])
            A = np.vstack([A, ridge])
            b = np.concatenate([b, np.zeros(3)])
        result = lsq_linear(
            A,
            b,
            bounds=([-np.inf, 0.0, 0.0], [np.inf, np.inf, np.inf]),
            max_iter=2000,
            lsmr_tol="auto",
            verbose=0,
        )
        objective = float(np.mean((A @ result.x - b) ** 2))
        return result, objective, len(pair_rows)

    def fit(self, train_mask: np.ndarray):
        train_mask = np.asarray(train_mask, dtype=bool)
        starts = [
            (0.15, 0.21),
            (0.20, 0.25),
            (0.34, 0.28),
            (0.08, 0.50),
            (0.50, 0.20),
            (0.01, 1.00),
        ]
        best = None

        def objective(z: np.ndarray) -> float:
            _result, value, _pair_count = self._fit_linear(train_mask, float(z[0]), float(z[1]))
            return value

        for start in starts:
            exponent_result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=[(1e-4, 2.0), (1e-4, 2.0)],
                options={"maxiter": 200},
            )
            alpha, beta = map(float, exponent_result.x)
            linear_result, value, pair_count = self._fit_linear(train_mask, alpha, beta)
            candidate = (value, exponent_result, linear_result, pair_count)
            if best is None or candidate[0] < best[0]:
                best = candidate

        assert best is not None
        _value, exponent_result, linear_result, pair_count = best
        self.train_mask = train_mask.copy()
        self.alpha = float(exponent_result.x[0])
        self.beta = float(exponent_result.x[1])
        self.intercept = float(linear_result.x[0])
        self.scale_coef = np.asarray(linear_result.x[1:], dtype=float)
        self.scale_fit_success = bool(exponent_result.success and linear_result.success)
        self.scale_fit_cost = float(best[0])
        self.pair_count = int(pair_count)
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        n = W.shape[0]
        N_arr = np.full(n, float(N)) if np.ndim(N) == 0 else np.asarray(N, dtype=float)
        D_arr = np.full(n, float(D)) if np.ndim(D) == 0 else np.asarray(D, dtype=float)
        design = self._design_for_exponents(self.alpha, self.beta, N_arr, D_arr)
        return design @ np.asarray([self.intercept, self.scale_coef[0], self.scale_coef[1]], dtype=float)

    def predict_all(self) -> np.ndarray:
        return self.predict_custom(self.data.W, self.data.N, self.data.D)

    def offset(self, W: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(W).shape[0], dtype=float)

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        n = np.asarray(W).shape[0]
        return pd.DataFrame(
            {
                "A_N": np.full(n, float(self.scale_coef[0])),
                "B_D": np.full(n, float(self.scale_coef[1])),
                "C_ND": np.zeros(n, dtype=float),
            }
        )

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": "E + A*((N/N0)^(-alpha)-1) + B*((D/D0)^(-beta)-1)",
            "mixture_dependence": "none",
            "cross_term": "none",
            "intercept": self.intercept,
            "scale_coef_A_B": self.scale_coef,
            "alpha": self.alpha,
            "beta": self.beta,
            "pair_weight": self.pair_weight,
            "ridge_scale": self.ridge_scale,
            "scale_fit_success": self.scale_fit_success,
            "scale_fit_cost": self.scale_fit_cost,
            "pair_count": self.pair_count,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
        }


class IdentityAnchorChinchillaApproach3Law:
    """Identity-anchor transfer with strict Chinchilla Approach 3 scaling.

    L(mixture_id,N,D) = E_id(mixture_id)
        + A * ((N/N0)^(-alpha) - 1)
        + B * ((D/D0)^(-beta) - 1)

    This tests whether the identity-transfer control improves when the global
    scaling law is the clean two-term Approach 3 form rather than MCT's fixed
    three-term continuation head.
    """

    def __init__(
        self,
        model_id: str,
        data,
        pair_weight: float = MCT_PAIR_WEIGHT,
        ridge_scale: float = 1e-5,
    ):
        self.model_id = model_id
        self.data = data
        self.pair_weight = float(pair_weight)
        self.ridge_scale = float(ridge_scale)
        self.scale_param_count = 2
        self.donor_constant_count = 0
        self.barrier_constant_count = 0

    def _anchors_for_rows(self, mixture_ids: np.ndarray) -> np.ndarray:
        return np.asarray([self.anchor_by_mix.get(str(mix), self.fallback_anchor) for mix in mixture_ids], dtype=float)

    def _design_for_exponents(self, alpha: float, beta: float, N: np.ndarray, D: np.ndarray) -> np.ndarray:
        uN = (np.asarray(N, dtype=float) / self.data.N0) ** (-float(alpha)) - 1.0
        uD = (np.asarray(D, dtype=float) / self.data.D0) ** (-float(beta)) - 1.0
        return np.column_stack([uN, uD])

    def _fit_linear(self, train_mask: np.ndarray, alpha: float, beta: float, anchor_all: np.ndarray):
        data = self.data
        design_all = self._design_for_exponents(alpha, beta, data.N, data.D)
        y_resid = data.y - anchor_all
        rows = np.where(train_mask)[0]
        A = design_all[rows]
        b = y_resid[rows]
        pair_rows: list[np.ndarray] = []
        pair_y: list[float] = []
        for i, j in make_pairs(data, train_mask):
            weight = math.sqrt(self.pair_weight)
            pair_rows.append(weight * (design_all[i] - design_all[j]))
            pair_y.append(weight * (y_resid[i] - y_resid[j]))
        if pair_rows:
            A = np.vstack([A, np.asarray(pair_rows)])
            b = np.concatenate([b, np.asarray(pair_y)])
        if self.ridge_scale > 0:
            A = np.vstack([A, np.sqrt(self.ridge_scale) * np.eye(design_all.shape[1])])
            b = np.concatenate([b, np.zeros(design_all.shape[1])])
        result = lsq_linear(A, b, bounds=(0.0, np.inf), max_iter=2000, lsmr_tol="auto", verbose=0)
        objective = float(np.mean((A @ result.x - b) ** 2))
        return result, objective, len(pair_rows)

    def fit(self, train_mask: np.ndarray):
        data = self.data
        train_mask = np.asarray(train_mask, dtype=bool)
        anchor_rows = train_mask & (data.scale_names_by_row == "300m_6b") & np.isclose(data.mu, 1.0)
        if anchor_rows.sum() == 0:
            raise ValueError("identity-anchor Approach 3 fit requires at least one 100M/6B target-budget training row")
        anchor_frame = pd.DataFrame({"mixture_id": data.mixture_ids[anchor_rows], "y": data.y[anchor_rows]})
        self.anchor_by_mix = {str(k): float(v) for k, v in anchor_frame.groupby("mixture_id")["y"].mean().items()}
        self.fallback_anchor = float(anchor_frame["y"].mean())
        self.identity_anchor_count = len(self.anchor_by_mix)
        self.anchor_feature_count = self.identity_anchor_count + 1
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 2
        self.total_constant_count = self.fitted_param_count
        anchor_all = self._anchors_for_rows(data.mixture_ids)

        starts = [
            (0.15, 0.21),
            (0.20, 0.25),
            (0.34, 0.28),
            (0.08, 0.50),
            (0.50, 0.20),
            (0.01, 1.00),
        ]
        best = None

        def objective(z: np.ndarray) -> float:
            _result, value, _pair_count = self._fit_linear(train_mask, float(z[0]), float(z[1]), anchor_all)
            return value

        for start in starts:
            exponent_result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=[(1e-4, 2.0), (1e-4, 2.0)],
                options={"maxiter": 200},
            )
            alpha, beta = map(float, exponent_result.x)
            linear_result, value, pair_count = self._fit_linear(train_mask, alpha, beta, anchor_all)
            candidate = (value, exponent_result, linear_result, pair_count)
            if best is None or candidate[0] < best[0]:
                best = candidate

        assert best is not None
        _value, exponent_result, linear_result, pair_count = best
        self.train_mask = train_mask.copy()
        self.alpha = float(exponent_result.x[0])
        self.beta = float(exponent_result.x[1])
        self.scale_coef = np.asarray(linear_result.x, dtype=float)
        self.scale_fit_success = bool(exponent_result.success and linear_result.success)
        self.scale_fit_cost = float(best[0])
        self.pair_count = int(pair_count)
        return self

    def _scale_prediction(self, N: np.ndarray | float, D: np.ndarray | float, count: int) -> np.ndarray:
        N_arr = np.full(count, float(N)) if np.ndim(N) == 0 else np.asarray(N, dtype=float)
        D_arr = np.full(count, float(D)) if np.ndim(D) == 0 else np.asarray(D, dtype=float)
        return self._design_for_exponents(self.alpha, self.beta, N_arr, D_arr) @ self.scale_coef

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        anchor = np.full(W.shape[0], self.fallback_anchor)
        return anchor + self._scale_prediction(N, D, W.shape[0])

    def predict_all(self) -> np.ndarray:
        return self._anchors_for_rows(self.data.mixture_ids) + self._scale_prediction(
            self.data.N, self.data.D, len(self.data.y)
        )

    def offset(self, W: np.ndarray) -> np.ndarray:
        return np.zeros(np.asarray(W).shape[0], dtype=float)

    def head_values(self, W: np.ndarray) -> pd.DataFrame:
        n = np.asarray(W).shape[0]
        return pd.DataFrame(
            {
                "A_N": np.full(n, float(self.scale_coef[0])),
                "B_D": np.full(n, float(self.scale_coef[1])),
                "C_ND": np.zeros(n, dtype=float),
            }
        )

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": (
                "E_identity(mixture_id) + A*((N/N0)^(-alpha)-1) + B*((D/D0)^(-beta)-1); "
                "fallback identity anchor for unseen/custom mixtures"
            ),
            "mixture_dependence": "identity lookup, no learned smooth mixture features",
            "cross_term": "none",
            "identity_anchor_count": self.identity_anchor_count,
            "fallback_anchor": self.fallback_anchor,
            "scale_coef_A_B": self.scale_coef,
            "alpha": self.alpha,
            "beta": self.beta,
            "pair_weight": self.pair_weight,
            "ridge_scale": self.ridge_scale,
            "scale_fit_success": self.scale_fit_success,
            "scale_fit_cost": self.scale_fit_cost,
            "pair_count": self.pair_count,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
        }


class IdentityAnchorGlobalScaleLaw(GlobalScaleFitMixin):
    """Per-mixture reference anchor plus global scale terms.

    This is intentionally not a smooth mixture regression model. It is a
    lookup-table control: if a mixture has a target-reference row at 100M/6B,
    transfer that row by a global scaling law; otherwise use the reference mean.
    """

    def __init__(
        self,
        model_id: str,
        data,
        exponents: tuple[float, float, float, float],
        pair_weight: float = MCT_PAIR_WEIGHT,
        ridge_scale: float = 1e-5,
    ):
        self.model_id = model_id
        self.data = data
        self.exponents = exponents
        self.pair_weight = float(pair_weight)
        self.ridge_scale = float(ridge_scale)
        self.scale_param_count = 3
        self.donor_constant_count = 0
        self.barrier_constant_count = 0

    def _anchors_for_rows(self, mixture_ids: np.ndarray) -> np.ndarray:
        return np.asarray([self.anchor_by_mix.get(str(mix), self.fallback_anchor) for mix in mixture_ids], dtype=float)

    def fit(self, train_mask: np.ndarray):
        data = self.data
        train_mask = np.asarray(train_mask, dtype=bool)
        anchor_rows = train_mask & (data.scale_names_by_row == "300m_6b") & np.isclose(data.mu, 1.0)
        if anchor_rows.sum() == 0:
            raise ValueError("identity-anchor fit requires at least one 100M/6B target-budget training row")
        anchor_frame = pd.DataFrame({"mixture_id": data.mixture_ids[anchor_rows], "y": data.y[anchor_rows]})
        self.anchor_by_mix = {str(k): float(v) for k, v in anchor_frame.groupby("mixture_id")["y"].mean().items()}
        self.fallback_anchor = float(anchor_frame["y"].mean())
        self.identity_anchor_count = len(self.anchor_by_mix)
        self.anchor_feature_count = self.identity_anchor_count + 1
        self.fitted_param_count = self.anchor_feature_count + self.scale_param_count + 4
        self.total_constant_count = self.fitted_param_count
        anchor_all = self._anchors_for_rows(data.mixture_ids)
        self._fit_global_scale(train_mask, anchor_all, self.pair_weight, self.ridge_scale)
        return self

    def predict_custom(self, W: np.ndarray, N: np.ndarray | float, D: np.ndarray | float) -> np.ndarray:
        W = np.asarray(W, dtype=float)
        # Raw-simplex probes do not have mixture IDs, so this model can only use
        # the fallback identity anchor outside observed rows.
        anchor = np.full(W.shape[0], self.fallback_anchor)
        return anchor + self._scale_prediction(N, D, W.shape[0])

    def predict_all(self) -> np.ndarray:
        return self._anchors_for_rows(self.data.mixture_ids) + self._scale_prediction(
            self.data.N, self.data.D, len(self.data.y)
        )

    def artifact(self) -> dict:
        return {
            "model_id": self.model_id,
            "formula": (
                "E_identity(mixture_id) + a*uN + b*uD + c*uND; fallback identity anchor for unseen/custom mixtures"
            ),
            "mixture_dependence": "identity lookup, no learned smooth mixture features",
            "identity_anchor_count": self.identity_anchor_count,
            "fallback_anchor": self.fallback_anchor,
            "scale_coef": self.scale_coef,
            "exponents": self.exponents,
            "pair_weight": self.pair_weight,
            "ridge_scale": self.ridge_scale,
            "fitted_param_count_counting_exponents": self.fitted_param_count,
            "total_constant_count": self.total_constant_count,
        }


def make_mct_law(cbs, s2_mod, data, ff, model_id: str, train_mask: np.ndarray, head_b: str):
    return cbs.CompatibilityBarrierPowerLaw(
        s2_mod,
        model_id=model_id,
        data=data,
        ff=ff,
        strength=0.0,
        anchor_kind="lrq_scarcity",
        exponents=MCT_BALANCED_EXPONENTS,
        pair_weight=MCT_PAIR_WEIGHT,
        ridge_anchor=1e-4,
        ridge_scale=1e-5,
        head_A="constant",
        head_B=head_b,
        head_C="constant",
    ).fit(train_mask)


def metric_row(cbs, data, model_name: str, fit_protocol: str, split: str, mask: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model": model_name,
        "fit_protocol": fit_protocol,
        "split": split,
        **cbs.metric_dict(data.y[mask], pred[mask]),
    }


def evaluate_models(cbs, data, seed_models: dict[str, object], leave_models: dict[str, object]):
    rows: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for name, model in seed_models.items():
        pred = model.predict_all()
        for split, mask in [
            ("seed7_train", data.seed7_train),
            ("seed7_holdout", data.seed7_holdout),
            ("fixed340_holdout", data.fixed340),
            ("random_supplement", data.random_supplement),
        ]:
            rows.append(metric_row(cbs, data, name, "seed7", split, mask, pred))
        frames.append(cbs.prediction_frame(data, data.runs, name, "seed7", pred))
    for name, model in leave_models.items():
        pred = model.predict_all()
        for split, mask in [("non900_train", data.all900_train), ("all900_leave_scale_out", data.all900_holdout)]:
            rows.append(metric_row(cbs, data, name, "leave900out", split, mask, pred))
        frames.append(cbs.prediction_frame(data, data.runs, name, "leave900out", pred))
    return pd.DataFrame(rows), pd.concat(frames, ignore_index=True)


def parameter_counts(models: dict[str, object]) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        rows.append(
            {
                "model": name,
                "fitted_param_count_counting_exponents": int(getattr(model, "fitted_param_count", 0)),
                "total_constant_count": int(getattr(model, "total_constant_count", 0)),
                "anchor_feature_count_including_intercept_or_identity_count": int(
                    getattr(model, "anchor_feature_count", 0)
                ),
                "scale_param_count": int(getattr(model, "scale_param_count", 0)),
                "donor_constant_count": int(getattr(model, "donor_constant_count", 0)),
                "barrier_constant_count": int(getattr(model, "barrier_constant_count", 0)),
                "identity_anchor_count": int(getattr(model, "identity_anchor_count", 0)),
                "alpha": float(getattr(model, "alpha", np.nan)),
                "beta": float(getattr(model, "beta", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def plot_metric_bars(metrics: pd.DataFrame, outpath: Path):
    core = metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ]
    pivot = core.pivot_table(index="model", columns="split", values="rmse", aggfunc="first")
    order = [
        c for c in ["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"] if c in pivot
    ]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    x = np.arange(len(pivot))
    width = 0.17
    cmap = plt.get_cmap("RdYlGn_r")
    for i, split in enumerate(order):
        ax.bar(
            x + (i - (len(order) - 1) / 2) * width,
            pivot[split],
            width=width,
            label=split,
            color=cmap(i / max(1, len(order) - 1)),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=18, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("MCT mixture-term ablation RMSE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_pred_actual(predictions: pd.DataFrame, outpath: Path):
    splits = [("fixed340_holdout", "Fixed 340M", "seed7"), ("all900_holdout", "All 900M", "leave900out")]
    models = list(predictions["model"].drop_duplicates())
    fig, axes = plt.subplots(len(splits), len(models), figsize=(3.8 * len(models), 7.8), squeeze=False)
    for r, (mask_col, split_label, fit_protocol) in enumerate(splits):
        for c, model in enumerate(models):
            ax = axes[r, c]
            df = predictions[(predictions["model"] == model) & (predictions["fit_protocol"] == fit_protocol)]
            bg = df[~df[mask_col]]
            fg = df[df[mask_col]]
            ax.scatter(bg["actual_bpb"], bg["pred_bpb"], s=10, alpha=0.12, color="0.55")
            ax.scatter(fg["actual_bpb"], fg["pred_bpb"], s=34, alpha=0.9)
            lo = min(float(df["actual_bpb"].min()), float(df["pred_bpb"].min()))
            hi = max(float(df["actual_bpb"].max()), float(df["pred_bpb"].max()))
            ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1.0)
            ax.set_title(model, fontsize=8)
            ax.set_xlabel("actual BPB")
            if c == 0:
                ax.set_ylabel(f"{split_label}\npredicted BPB")
            ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def plot_drop_ratios(drop_summary: pd.DataFrame, outpath: Path):
    order_pairs = ["0.5x_to_1.0x", "0.5x_to_2.0x", "1.0x_to_2.0x"]
    models = list(drop_summary["model"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    x = np.arange(len(order_pairs))
    width = 0.15
    cmap = plt.get_cmap("RdYlGn_r")
    for j, model in enumerate(models):
        vals = []
        for pair in order_pairs:
            sub = drop_summary[(drop_summary["model"] == model) & (drop_summary["drop_pair"] == pair)]
            vals.append(float(sub["drop_ratio_mean"].iloc[0]) if len(sub) else np.nan)
        ax.bar(
            x + (j - (len(models) - 1) / 2) * width,
            vals,
            width=width,
            color=cmap(j / max(1, len(models) - 1)),
            label=model,
        )
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", " ") for p in order_pairs])
    ax.set_ylabel("predicted drop / actual drop")
    ax.set_title("Fixed-340M same-mixture target-budget drop ratios")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def write_report(
    outdir: Path,
    metrics: pd.DataFrame,
    drop_summary: pd.DataFrame,
    opt: pd.DataFrame,
    params: pd.DataFrame,
    monotone: pd.DataFrame,
):
    core = metrics[
        metrics["split"].isin(["seed7_holdout", "fixed340_holdout", "random_supplement", "all900_leave_scale_out"])
    ]
    metric_cols = [
        "model",
        "fit_protocol",
        "split",
        "n",
        "rmse",
        "spearman",
        "bias_pred_minus_actual",
        "slope_pred_on_actual",
        "std_ratio",
        "low_tail_rmse",
    ]
    opt_cols = [
        "model",
        "target_scale",
        "opt_kind",
        "predicted_bpb",
        "hard_corner_flag",
        "phase1_tech_collapse_flag",
        "any_family_collapse_flag",
        "nearest_observed_phase_mean_tv",
        "p0_broad_text_share",
        "p0_tech_code_share",
        "p0_reasoning_share",
        "p1_broad_text_share",
        "p1_tech_code_share",
        "p1_reasoning_share",
    ]
    lines = [
        "# MCT-LRQ Mixture-Term Ablation",
        "",
        "Date: 2026-04-24",
        "",
        "## Question",
        "",
        (
            "How much of barrier-free MCT-LRQ's predictive quality comes from actual smooth mixture-dependent "
            "terms, versus only scale continuation?"
        ),
        "",
        "## Variants",
        "",
        (
            "- `mct_lrq69_full_no_barrier`: current barrier-free MCT-LRQ predictive law, with LRQ mixture "
            "anchor and family-dependent D scale head."
        ),
        (
            "- `mct_lrq65_anchor_global_scale`: same LRQ mixture anchor, but A/B/C are all global constants. "
            "This removes mixture-dependent scale heads but keeps the mixture regression anchor."
        ),
        (
            "- `identity_anchor_global_scale`: lookup-table transfer from observed 100M/6B mixture IDs plus "
            "global scale terms. This learns no smooth mixture features; it only memorizes identities where "
            "available."
        ),
        (
            "- `identity_anchor_chinchilla_approach3`: same identity lookup-table anchor, but with strict "
            "two-term Approach 3 scaling and fitted global `alpha,beta`."
        ),
        "- `scale_only_global`: one global scaling curve with no mixture information.",
        (
            "- `scale_only_chinchilla_approach3`: strict pure-scale Chinchilla Approach 3 control: "
            "`E + A*uN + B*uD`, with fitted global `alpha,beta` and no MCT cross term."
        ),
        "",
        "## Main Result",
        "",
        (
            "The scale-only controls are not competitive. The identity-transfer control is useful for seen "
            "mixtures but does not solve generalization, and it cannot define meaningful raw-simplex optima "
            "because custom mixtures fall back to the mean anchor. The gap between `mct_lrq69_full_no_barrier` "
            "and `mct_lrq65_anchor_global_scale` measures the value of mixture-dependent scale heads; the much "
            "larger gap to the pure-scale controls measures the value of the LRQ/GRP-style mixture anchor."
        ),
        "",
        "## Parameters",
        "",
        md_table(params),
        "",
        "## Predictive Metrics",
        "",
        md_table(core.sort_values(["model", "fit_protocol", "split"]), cols=metric_cols),
        "",
        "## Fixed-340M Same-Mixture Drops",
        "",
        md_table(
            drop_summary.sort_values(["model", "drop_pair"]),
            cols=[
                "model",
                "drop_pair",
                "n",
                "actual_drop_mean",
                "pred_drop_mean",
                "drop_error_mean",
                "drop_ratio_mean",
                "drop_ratio_median",
                "drop_rmse",
            ],
        ),
        "",
        "## Raw And Constrained Optima",
        "",
        md_table(
            opt[opt["target_scale"].isin(["340M/10.4B", "900M/24B"])].sort_values(["model", "target_scale", "opt_kind"]),
            cols=opt_cols,
        ),
        "",
        "## Monotonicity Grid",
        "",
        md_table(monotone),
        "",
        "## Artifact Map",
        "",
        "- `csv/metric_summary.csv`: split metrics.",
        "- `csv/row_predictions.csv`: row-level predictions.",
        "- `csv/fixed340_drop_summary.csv`: same-mixture target-budget drop metrics.",
        "- `csv/optimum_diagnostics.csv`: raw/hull/trustblend optimum probes.",
        "- `plots/rmse_mix_term_ablation.png`: RMSE comparison.",
        "- `plots/pred_actual_mix_term_ablation.png`: fixed-340M and 900M predicted-vs-actual panels.",
        "- `plots/drop_ratios_mix_term_ablation.png`: fixed-340M drop-ratio comparison.",
    ]
    (outdir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--packet-root",
        type=Path,
        default=SCRIPT_DIR / "chatgpt_pro_hybrid_data_mixing_packet_v31",
    )
    parser.add_argument(
        "--mct-code-dir",
        type=Path,
        default=LOCAL_MCT_CODE_DIR if (LOCAL_MCT_CODE_DIR / "cbs_lrq_base.py").exists() else FALLBACK_MCT_CODE_DIR,
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=SCRIPT_DIR / "reference_outputs" / "mct_mix_term_ablation_20260424",
    )
    args = parser.parse_args()

    code_dir = args.mct_code_dir.resolve()
    if not (code_dir / "cbs_lrq_base.py").exists():
        raise FileNotFoundError(f"Missing MCT helper code at {code_dir}")

    outdir = args.outdir.resolve()
    for subdir in ["csv", "plots", "models", "code"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    cbs = import_from_path("mct_mix_ablation_cbs_lrq_base", code_dir / "cbs_lrq_base.py")
    packet_root = cbs.packet_root_from_arg(args.packet_root)
    s2_mod = cbs.import_s2(packet_root)
    data = s2_mod.load_packet(packet_root)
    base_ff = s2_mod.FeatureFactory(data)
    ff = cbs.LRQFeatureFactory(base_ff)

    seed_models = {
        "mct_lrq69_full_no_barrier": make_mct_law(
            cbs, s2_mod, data, ff, "mct_lrq69_full_no_barrier", data.seed7_train, "family"
        ),
        "mct_lrq65_anchor_global_scale": make_mct_law(
            cbs, s2_mod, data, ff, "mct_lrq65_anchor_global_scale", data.seed7_train, "constant"
        ),
        "identity_anchor_global_scale": (
            IdentityAnchorGlobalScaleLaw("identity_anchor_global_scale", data, MCT_BALANCED_EXPONENTS).fit(
                data.seed7_train
            )
        ),
        "identity_anchor_chinchilla_approach3": (
            IdentityAnchorChinchillaApproach3Law("identity_anchor_chinchilla_approach3", data).fit(data.seed7_train)
        ),
        "scale_only_global": (
            ConstantScaleOnlyLaw("scale_only_global", data, MCT_BALANCED_EXPONENTS).fit(data.seed7_train)
        ),
        "scale_only_chinchilla_approach3": (
            ChinchillaApproach3ScaleOnlyLaw("scale_only_chinchilla_approach3", data).fit(data.seed7_train)
        ),
    }
    leave_models = {
        "mct_lrq69_full_no_barrier": make_mct_law(
            cbs, s2_mod, data, ff, "mct_lrq69_full_no_barrier", data.all900_train, "family"
        ),
        "mct_lrq65_anchor_global_scale": make_mct_law(
            cbs, s2_mod, data, ff, "mct_lrq65_anchor_global_scale", data.all900_train, "constant"
        ),
        "identity_anchor_global_scale": (
            IdentityAnchorGlobalScaleLaw("identity_anchor_global_scale", data, MCT_BALANCED_EXPONENTS).fit(
                data.all900_train
            )
        ),
        "identity_anchor_chinchilla_approach3": (
            IdentityAnchorChinchillaApproach3Law("identity_anchor_chinchilla_approach3", data).fit(data.all900_train)
        ),
        "scale_only_global": (
            ConstantScaleOnlyLaw("scale_only_global", data, MCT_BALANCED_EXPONENTS).fit(data.all900_train)
        ),
        "scale_only_chinchilla_approach3": (
            ChinchillaApproach3ScaleOnlyLaw("scale_only_chinchilla_approach3", data).fit(data.all900_train)
        ),
    }

    metrics, predictions = evaluate_models(cbs, data, seed_models, leave_models)
    drop_detail, drop_summary, beta_detail = cbs.fixed340_drop_tables(data, seed_models)
    opt = cbs.optimum_diagnostics(seed_models, data, ff)
    monotone = s2_mod.monotonicity_grid(seed_models, data, ff)
    params = parameter_counts(seed_models)

    metrics.to_csv(outdir / "csv" / "metric_summary.csv", index=False)
    predictions.to_csv(outdir / "csv" / "row_predictions.csv", index=False)
    drop_detail.to_csv(outdir / "csv" / "fixed340_drop_pairs.csv", index=False)
    drop_summary.to_csv(outdir / "csv" / "fixed340_drop_summary.csv", index=False)
    beta_detail.to_csv(outdir / "csv" / "fixed340_beta_triples.csv", index=False)
    opt.to_csv(outdir / "csv" / "optimum_diagnostics.csv", index=False)
    monotone.to_csv(outdir / "csv" / "monotonicity_grid.csv", index=False)
    params.to_csv(outdir / "csv" / "parameter_counts.csv", index=False)

    for name, model in seed_models.items():
        artifact = model.artifact() if hasattr(model, "artifact") else {"model_id": name}
        (outdir / "models" / f"{name}_seed7.json").write_text(json.dumps(jsonify(artifact), indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__), outdir / "code" / Path(__file__).name)
    for helper in ["cbs_lrq_base.py", "run_mct_lrq_law.py"]:
        helper_path = code_dir / helper
        if helper_path.exists():
            shutil.copy2(helper_path, outdir / "code" / helper)

    plot_metric_bars(metrics, outdir / "plots" / "rmse_mix_term_ablation.png")
    plot_pred_actual(predictions, outdir / "plots" / "pred_actual_mix_term_ablation.png")
    plot_drop_ratios(drop_summary, outdir / "plots" / "drop_ratios_mix_term_ablation.png")
    write_report(outdir, metrics, drop_summary, opt, params, monotone)

    manifest = {
        "packet_root": str(packet_root),
        "mct_code_dir": str(code_dir),
        "outdir": str(outdir),
        "variants": list(seed_models),
    }
    (outdir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
