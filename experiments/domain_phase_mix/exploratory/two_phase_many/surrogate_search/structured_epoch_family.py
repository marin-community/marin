# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Low-parameter structured epoch-law surrogates for domain-mix studies."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.two_phase_many_observed_runs import TWO_PHASE_MANY_CSV_PATH

MANY_DOMAIN_SWARM_CSV = TWO_PHASE_MANY_CSV_PATH
STARCODER_TWO_PHASE_CSV = Path(__file__).resolve().parents[2] / "two_phase_starcoder_combined.csv"
MANY_DOMAIN_TARGET = "eval/uncheatable_eval/bpb"
STARCODER_TARGET = "eval/paloma/dolma_100_programing_languages/bpb"

SIGNAL_KIND_TOTAL_LOG = "total_log"
SIGNAL_KIND_RETAINED_TOTAL = "retained_total"
SIGNAL_KIND_THRESHOLD_TOTAL = "threshold_total"
SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL = "threshold_retained_total"

PENALTY_KIND_NONE = "none"
PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD = "per_domain_log_threshold"
PENALTY_KIND_GLOBAL_LOG_THRESHOLD = "global_log_threshold"
PENALTY_KIND_GROUP_LOG_THRESHOLD = "group_log_threshold"

PREMIUM_MODE_NONE = "none"
PREMIUM_MODE_GLOBAL = "global"
PREMIUM_MODE_PAIR = "pair"


@dataclass
class PacketData:
    """Feature-ready packet data for a surrogate family."""

    frame: pd.DataFrame
    name_col: str
    y: np.ndarray
    w: np.ndarray
    m: int
    c0: np.ndarray
    c1: np.ndarray
    domain_names: list[str]


@dataclass(frozen=True)
class ParameterCount:
    """Standardized parameter-count breakdown."""

    linear_coefficients: int
    intercept: int
    global_shape_parameters: int

    @property
    def reported_total(self) -> int:
        return self.linear_coefficients + self.intercept

    @property
    def total_with_shapes(self) -> int:
        return self.linear_coefficients + self.intercept + self.global_shape_parameters


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


def safe_exp(x: float, lo: float = -18.0, hi: float = 18.0) -> float:
    """Exponentiate with clipping."""
    return float(math.exp(max(lo, min(hi, float(x)))))


def regression_metrics(frame: pd.DataFrame, name_col: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Return standard regression and regret metrics."""
    residuals = y_pred - y_true
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    chosen_idx = int(np.argmin(y_pred))
    best_idx = int(np.argmin(y_true))
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "r2": float(1.0 - sse / sst),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "regret_at_1": float(y_true[chosen_idx] - y_true[best_idx]),
        "chosen_candidate": str(frame.iloc[chosen_idx][name_col]),
        "best_candidate": str(frame.iloc[best_idx][name_col]),
        "chosen_value": float(y_true[chosen_idx]),
        "best_value": float(y_true[best_idx]),
    }


def foldwise_regret(
    frame: pd.DataFrame,
    name_col: str,
    weights: np.ndarray,
    y: np.ndarray,
    fit_predict_fn,
    *,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Return out-of-fold regret diagnostics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    regrets: list[float] = []
    choices: list[str] = []
    oof = np.zeros_like(y)
    for fold, (tr, te) in enumerate(kf.split(weights)):
        predictor = fit_predict_fn(weights[tr], y[tr], seed + fold)
        pred = predictor(weights[te])
        oof[te] = pred
        chosen = int(np.argmin(pred))
        regrets.append(float(y[te][chosen] - np.min(y[te])))
        choices.append(str(frame.iloc[te[chosen]][name_col]))
    return {
        "oof_pred": oof,
        "cv_foldmean_regret_at_1": float(np.mean(regrets)),
        "cv_foldmedian_regret_at_1": float(np.median(regrets)),
        "cv_foldmax_regret_at_1": float(np.max(regrets)),
        "cv_fold_choices": json.dumps(choices),
    }


def load_two_phase_many_packet(
    csv_path: str | Path = MANY_DOMAIN_SWARM_CSV,
    *,
    target: str = MANY_DOMAIN_TARGET,
    name: str = "structured_epoch_many_domain",
) -> PacketData:
    """Load the 39-domain two-phase packet with real epoch multipliers."""
    frame, spec, _ = load_two_phase_many_candidate_summary_spec(
        csv_path,
        objective_metric=target,
        name=name,
    )
    name_col = "candidate_run_name" if "candidate_run_name" in frame.columns else "run_name"
    return PacketData(
        frame=frame,
        name_col=name_col,
        y=spec.y,
        w=spec.weights,
        m=spec.M,
        c0=np.asarray(spec.epoch_multipliers[0], dtype=float),
        c1=np.asarray(spec.epoch_multipliers[1], dtype=float),
        domain_names=list(spec.domain_names),
    )


def load_two_phase_starcoder_packet(
    csv_path: str | Path = STARCODER_TWO_PHASE_CSV,
    *,
    target: str = STARCODER_TARGET,
) -> PacketData:
    """Load the 2-domain Starcoder two-phase packet."""
    frame = pd.read_csv(csv_path)
    domain_names = ["nemotron_full", "starcoder"]
    weights = np.zeros((len(frame), 2, 2), dtype=float)
    for phase_idx, phase_name in enumerate(["phase_0", "phase_1"]):
        weights[:, phase_idx, 0] = frame[f"{phase_name}_nemotron_full"].to_numpy(float)
        weights[:, phase_idx, 1] = frame[f"{phase_name}_starcoder"].to_numpy(float)

    phase_0_nemotron_ratio = frame["phase_0_nemotron_epochs"] / frame["phase_0_nemotron_full"].replace(0, np.nan)
    phase_0_starcoder_ratio = frame["phase_0_starcoder_epochs"] / frame["phase_0_starcoder"].replace(0, np.nan)
    phase_1_nemotron_ratio = frame["phase_1_nemotron_epochs"] / frame["phase_1_nemotron_full"].replace(0, np.nan)
    phase_1_starcoder_ratio = frame["phase_1_starcoder_epochs"] / frame["phase_1_starcoder"].replace(0, np.nan)
    c0 = np.array(
        [
            float(phase_0_nemotron_ratio.dropna().iloc[0]),
            float(phase_0_starcoder_ratio.dropna().iloc[0]),
        ],
        dtype=float,
    )
    c1 = np.array(
        [
            float(phase_1_nemotron_ratio.dropna().iloc[0]),
            float(phase_1_starcoder_ratio.dropna().iloc[0]),
        ],
        dtype=float,
    )
    return PacketData(
        frame=frame,
        name_col="run_id",
        y=frame[target].to_numpy(float),
        w=weights,
        m=2,
        c0=c0,
        c1=c1,
        domain_names=domain_names,
    )


def _shape_parameter_count(params: dict[str, Any]) -> int:
    count = 5  # alpha, eta, lam, tau, reg
    if "sig_tau" in params:
        count += 1
    return count


class StructuredEpochSurrogate:
    """Independent-domain monotone additive epoch-law surrogate."""

    def __init__(self, data: PacketData, params: dict[str, Any]):
        self.data = data
        self.params = params.copy()
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def parameter_count(self) -> ParameterCount:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return ParameterCount(
            linear_coefficients=len(self.coef_),
            intercept=1,
            global_shape_parameters=_shape_parameter_count(self.params),
        )

    def build_signal(self, weights: np.ndarray) -> np.ndarray:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]
        alpha = float(self.params["alpha"])
        eta = float(self.params["eta"])
        lam = float(self.params["lam"])
        sig_tau = float(self.params.get("sig_tau", 0.0))
        kind = self.params["signal_kind"]

        if kind == SIGNAL_KIND_TOTAL_LOG:
            return np.log1p(alpha * (e0 + eta * e1))
        if kind == SIGNAL_KIND_THRESHOLD_TOTAL:
            return softplus(np.log1p(alpha * (e0 + eta * e1)) - sig_tau)
        if kind == SIGNAL_KIND_RETAINED_TOTAL:
            retained = np.exp(-lam * (1.0 - p1))
            return np.log1p(alpha * (retained * e0 + eta * e1))
        if kind == SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL:
            retained = np.exp(-lam * (1.0 - p1))
            return softplus(np.log1p(alpha * (retained * e0 + eta * e1)) - sig_tau)
        raise ValueError(f"Unknown signal_kind={kind!r}")

    def build_penalty(self, weights: np.ndarray) -> np.ndarray | None:
        kind = self.params["pen_kind"]
        if kind == PENALTY_KIND_NONE:
            return None

        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]
        lam = float(self.params["lam"])
        tau = float(self.params["tau"])
        retained = np.exp(-lam * (1.0 - p1))
        exposure = retained * e0 + e1

        if kind == PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD:
            return np.sum(softplus(np.log1p(exposure) - tau) ** 2, axis=1, keepdims=True)
        if kind == PENALTY_KIND_GLOBAL_LOG_THRESHOLD:
            return softplus(np.sum(np.log1p(exposure), axis=1, keepdims=True) - tau) ** 2
        raise ValueError(f"Unknown pen_kind={kind!r}")

    def fit(self, weights: np.ndarray, y: np.ndarray) -> StructuredEpochSurrogate:
        signal = self.build_signal(weights)
        penalty = self.build_penalty(weights)
        design = -signal if penalty is None else np.hstack([-signal, penalty])
        design_mean = design.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        design_centered = design - design_mean
        y_centered = y - y_mean

        reg = float(self.params.get("reg", 0.0))
        if reg > 0.0:
            design_aug = np.vstack([design_centered, np.sqrt(reg) * np.eye(design_centered.shape[1])])
            y_aug = np.concatenate([y_centered, np.zeros(design_centered.shape[1])])
        else:
            design_aug, y_aug = design_centered, y_centered

        coef, _ = nnls(design_aug, y_aug)
        self.coef_ = coef
        self.intercept_ = y_mean - float((design_mean @ coef).ravel()[0])
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fit")
        signal = self.build_signal(weights)
        penalty = self.build_penalty(weights)
        design = -signal if penalty is None else np.hstack([-signal, penalty])
        return self.intercept_ + design @ self.coef_

    def cv_predict(self, weights: np.ndarray, y: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> np.ndarray:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yhat = np.zeros_like(y)
        for tr, te in kf.split(weights):
            model = self.__class__(self.data, self.params).fit(weights[tr], y[tr])
            yhat[te] = model.predict(weights[te])
        return yhat


class CCPairStructuredSurrogate:
    """CC-aware structured epoch-law surrogate with shared pair features."""

    def __init__(self, data: PacketData, params: dict[str, Any]):
        self.data = data
        self.params = params.copy()
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.feature_names_: list[tuple[str, str, str]] | None = None

        pairs: list[tuple[int, int]] = []
        topics: list[str] = []
        singletons: list[int] = []
        for name in data.domain_names:
            if name.startswith("dolma3_cc/") and name.endswith("_high"):
                low = name[:-5] + "_low"
                pairs.append((data.domain_names.index(name), data.domain_names.index(low)))
                topics.append(name[len("dolma3_cc/") : -5])
                continue
            if name.startswith("dolma3_cc/") and name.endswith("_low"):
                continue
            singletons.append(data.domain_names.index(name))
        self.pairs = pairs
        self.pair_topics = topics
        self.singletons = singletons

    def parameter_count(self) -> ParameterCount:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return ParameterCount(
            linear_coefficients=len(self.coef_),
            intercept=1,
            global_shape_parameters=_shape_parameter_count(self.params),
        )

    def _signal_and_exposure(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]
        alpha = float(self.params["alpha"])
        eta = float(self.params["eta"])
        lam = float(self.params["lam"])
        sig_tau = float(self.params.get("sig_tau", 0.0))
        kind = self.params["signal_kind"]
        retained = np.exp(-lam * (1.0 - p1))

        if kind == SIGNAL_KIND_TOTAL_LOG:
            signal = np.log1p(alpha * (e0 + eta * e1))
        elif kind == SIGNAL_KIND_THRESHOLD_TOTAL:
            signal = softplus(np.log1p(alpha * (e0 + eta * e1)) - sig_tau)
        elif kind == SIGNAL_KIND_RETAINED_TOTAL:
            signal = np.log1p(alpha * (retained * e0 + eta * e1))
        elif kind == SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL:
            signal = softplus(np.log1p(alpha * (retained * e0 + eta * e1)) - sig_tau)
        else:
            raise ValueError(f"Unknown signal_kind={kind!r}")
        exposure = retained * e0 + e1
        return signal, exposure

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        signal, exposure = self._signal_and_exposure(weights)
        features: list[np.ndarray] = []
        names: list[tuple[str, str, str]] = []

        for idx in self.singletons:
            features.append(signal[:, idx : idx + 1])
            names.append(("signal", "singleton", self.data.domain_names[idx]))

        for (hi, lo), topic in zip(self.pairs, self.pair_topics, strict=True):
            features.append((signal[:, hi] + signal[:, lo])[:, None])
            names.append(("signal", "pair_base", topic))

        premium_mode = self.params.get("premium_mode", PREMIUM_MODE_NONE)
        if premium_mode == PREMIUM_MODE_PAIR:
            for (hi, _), topic in zip(self.pairs, self.pair_topics, strict=True):
                features.append(signal[:, hi : hi + 1])
                names.append(("signal", "pair_premium", topic))
        elif premium_mode == PREMIUM_MODE_GLOBAL:
            highs = np.stack([signal[:, hi] for hi, _ in self.pairs], axis=1)
            features.append(np.sum(highs, axis=1, keepdims=True))
            names.append(("signal", "pair_premium_global", "all_cc_high"))
        elif premium_mode != PREMIUM_MODE_NONE:
            raise ValueError(f"Unknown premium_mode={premium_mode!r}")

        penalty_kind = self.params.get("pen_kind", PENALTY_KIND_GROUP_LOG_THRESHOLD)
        if penalty_kind != PENALTY_KIND_NONE:
            tau = float(self.params["tau"])
            penalty_terms: list[np.ndarray] = []
            if penalty_kind == PENALTY_KIND_GROUP_LOG_THRESHOLD:
                for hi, lo in self.pairs:
                    penalty_terms.append(softplus(np.log1p(exposure[:, hi] + exposure[:, lo]) - tau) ** 2)
                for idx in self.singletons:
                    penalty_terms.append(softplus(np.log1p(exposure[:, idx]) - tau) ** 2)
            elif penalty_kind == PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD:
                for idx in range(self.data.m):
                    penalty_terms.append(softplus(np.log1p(exposure[:, idx]) - tau) ** 2)
            else:
                raise ValueError(f"Unknown pen_kind={penalty_kind!r}")
            features.append(np.sum(np.stack(penalty_terms, axis=1), axis=1, keepdims=True))
            names.append(("penalty", penalty_kind, "global"))

        feature_matrix = np.hstack(features)
        design = feature_matrix.copy()
        design[:, : feature_matrix.shape[1] - (1 if penalty_kind != PENALTY_KIND_NONE else 0)] *= -1.0
        self.feature_names_ = names
        return design

    def fit(self, weights: np.ndarray, y: np.ndarray) -> CCPairStructuredSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        design_centered = design - design_mean
        y_centered = y - y_mean

        reg = float(self.params.get("reg", 0.0))
        if reg > 0.0:
            design_aug = np.vstack([design_centered, np.sqrt(reg) * np.eye(design_centered.shape[1])])
            y_aug = np.concatenate([y_centered, np.zeros(design_centered.shape[1])])
        else:
            design_aug, y_aug = design_centered, y_centered

        coef, _ = nnls(design_aug, y_aug)
        self.coef_ = coef
        self.intercept_ = y_mean - float((design_mean @ coef).ravel()[0])
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fit")
        return self.intercept_ + self.build_design(weights) @ self.coef_

    def cv_predict(self, weights: np.ndarray, y: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> np.ndarray:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yhat = np.zeros_like(y)
        for tr, te in kf.split(weights):
            model = self.__class__(self.data, self.params).fit(weights[tr], y[tr])
            yhat[te] = model.predict(weights[te])
        return yhat


class CCPairTotalStructuredSurrogate:
    """CC-aware surrogate that saturates each high/low pair after summing exposure."""

    def __init__(self, data: PacketData, params: dict[str, Any]):
        self.data = data
        self.params = params.copy()
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.feature_names_: list[tuple[str, str, str]] | None = None

        pairs: list[tuple[int, int]] = []
        topics: list[str] = []
        singletons: list[int] = []
        used: set[int] = set()
        for idx, name in enumerate(data.domain_names):
            if idx in used:
                continue
            if name.startswith("dolma3_cc/") and name.endswith("_high"):
                low_name = name[:-5] + "_low"
                if low_name in data.domain_names:
                    low_idx = data.domain_names.index(low_name)
                    pairs.append((idx, low_idx))
                    topics.append(name[len("dolma3_cc/") : -5])
                    used.add(idx)
                    used.add(low_idx)
                    continue
            if name.startswith("dolma3_cc/") and name.endswith("_low"):
                high_name = name[:-4] + "high"
                if high_name in data.domain_names:
                    continue
            singletons.append(idx)
            used.add(idx)

        self.pairs = pairs
        self.pair_topics = topics
        self.singletons = singletons

    def parameter_count(self) -> ParameterCount:
        if self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return ParameterCount(
            linear_coefficients=len(self.coef_),
            intercept=1,
            global_shape_parameters=_shape_parameter_count(self.params),
        )

    def _retained_exposure(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p0 = weights[:, 0, :]
        p1 = weights[:, 1, :]
        e0 = p0 * self.data.c0[None, :]
        e1 = p1 * self.data.c1[None, :]
        eta = float(self.params["eta"])
        lam = float(self.params["lam"])
        retained = np.exp(-lam * (1.0 - p1))
        x = retained * e0 + eta * e1
        return x, retained, e0, e1

    def build_design(self, weights: np.ndarray) -> np.ndarray:
        alpha = float(self.params["alpha"])
        tau = float(self.params["tau"])
        x, _, _, _ = self._retained_exposure(weights)
        signal = np.log1p(alpha * x)

        features: list[np.ndarray] = []
        names: list[tuple[str, str, str]] = []

        for idx in self.singletons:
            features.append(signal[:, idx : idx + 1])
            names.append(("signal", "singleton", self.data.domain_names[idx]))

        for (hi, lo), topic in zip(self.pairs, self.pair_topics, strict=True):
            pair_signal = np.log1p(alpha * (x[:, hi] + x[:, lo]))
            features.append(pair_signal[:, None])
            names.append(("signal", "pair_total", topic))

        if self.pairs:
            highs = np.stack([signal[:, hi] for hi, _ in self.pairs], axis=1)
            features.append(np.sum(highs, axis=1, keepdims=True))
            names.append(("signal", "pair_premium_global", "all_cc_high"))

        penalty_terms: list[np.ndarray] = []
        for idx in self.singletons:
            penalty_terms.append(softplus(np.log1p(x[:, idx]) - tau) ** 2)
        for hi, lo in self.pairs:
            penalty_terms.append(softplus(np.log1p(x[:, hi] + x[:, lo]) - tau) ** 2)
        features.append(np.sum(np.stack(penalty_terms, axis=1), axis=1, keepdims=True))
        names.append(("penalty", PENALTY_KIND_GROUP_LOG_THRESHOLD, "global"))

        feature_matrix = np.hstack(features)
        design = feature_matrix.copy()
        design[:, :-1] *= -1.0
        self.feature_names_ = names
        return design

    def fit(self, weights: np.ndarray, y: np.ndarray) -> CCPairTotalStructuredSurrogate:
        design = self.build_design(weights)
        design_mean = design.mean(axis=0, keepdims=True)
        y_mean = float(y.mean())
        design_centered = design - design_mean
        y_centered = y - y_mean

        reg = float(self.params.get("reg", 0.0))
        if reg > 0.0:
            design_aug = np.vstack([design_centered, np.sqrt(reg) * np.eye(design_centered.shape[1])])
            y_aug = np.concatenate([y_centered, np.zeros(design_centered.shape[1])])
        else:
            design_aug, y_aug = design_centered, y_centered

        coef, _ = nnls(design_aug, y_aug)
        self.coef_ = coef
        self.intercept_ = y_mean - float((design_mean @ coef).ravel()[0])
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model is not fit")
        return self.intercept_ + self.build_design(weights) @ self.coef_

    def cv_predict(self, weights: np.ndarray, y: np.ndarray, *, n_splits: int = 5, seed: int = 0) -> np.ndarray:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yhat = np.zeros_like(y)
        for tr, te in kf.split(weights):
            model = self.__class__(self.data, self.params).fit(weights[tr], y[tr])
            yhat[te] = model.predict(weights[te])
        return yhat

    def coef_table(self) -> pd.DataFrame:
        if self.feature_names_ is None or self.coef_ is None:
            raise RuntimeError("Model is not fit")
        return pd.DataFrame(
            [
                {"kind0": kind0, "kind1": kind1, "name": name, "coef": float(coef)}
                for (kind0, kind1, name), coef in zip(self.feature_names_, self.coef_, strict=True)
            ]
        )


def domain_signal_coefs_globalpremium(model: CCPairStructuredSurrogate, data: PacketData) -> tuple[np.ndarray, float]:
    """Expand shared CC coefficients into domain-level signal coefficients."""
    if model.feature_names_ is None or model.coef_ is None:
        raise RuntimeError("Model is not fit")

    coef_tab = pd.DataFrame(
        [
            {"kind0": kind0, "kind1": kind1, "name": name, "coef": float(coef)}
            for (kind0, kind1, name), coef in zip(model.feature_names_, model.coef_, strict=True)
        ]
    )
    coeff = np.zeros(data.m, dtype=float)
    name_to_idx = {name: idx for idx, name in enumerate(data.domain_names)}

    singleton = coef_tab[coef_tab["kind1"] == "singleton"][["name", "coef"]]
    for _, row in singleton.iterrows():
        coeff[name_to_idx[str(row["name"])]] = float(row["coef"])

    pair_base = coef_tab[coef_tab["kind1"] == "pair_base"][["name", "coef"]].set_index("name")["coef"].to_dict()
    has_global = np.any(coef_tab["kind1"] == "pair_premium_global")
    premium_global = (
        float(coef_tab.loc[coef_tab["kind1"] == "pair_premium_global", "coef"].iloc[0]) if has_global else 0.0
    )
    for (hi, lo), topic in zip(model.pairs, model.pair_topics, strict=True):
        base = float(pair_base[topic])
        coeff[lo] = base
        coeff[hi] = base + premium_global

    has_penalty = np.any(coef_tab["kind0"] == "penalty")
    penalty_coef = float(coef_tab.loc[coef_tab["kind0"] == "penalty", "coef"].iloc[0]) if has_penalty else 0.0
    return coeff, penalty_coef


def optimize_cc_globalpremium_model(
    model: CCPairStructuredSurrogate,
    data: PacketData,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize a fitted CC-aware model over the two-phase simplices."""
    if model.intercept_ is None:
        raise RuntimeError("Model is not fit")

    coeff_signal, penalty_coef = domain_signal_coefs_globalpremium(model, data)
    n_domains = data.m
    c0 = data.c0
    c1 = data.c1
    alpha = float(model.params["alpha"])
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    sig_tau = float(model.params.get("sig_tau", 0.0))
    signal_kind = model.params["signal_kind"]
    rng = np.random.default_rng(seed)

    pair_map = model.pairs
    singletons = model.singletons

    def signal_and_grads(
        p0: np.ndarray, p1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        e0 = c0 * p0
        e1 = c1 * p1
        retained = np.exp(-lam * (1.0 - p1))
        d_retained = lam * retained

        if signal_kind == SIGNAL_KIND_RETAINED_TOTAL:
            z = retained * e0 + eta * e1
            signal = np.log1p(alpha * z)
            ds0 = alpha * (retained * c0) / (1.0 + alpha * z)
            ds1 = alpha * (d_retained * e0 + eta * c1) / (1.0 + alpha * z)
        elif signal_kind == SIGNAL_KIND_THRESHOLD_RETAINED_TOTAL:
            z = retained * e0 + eta * e1
            u = np.log1p(alpha * z) - sig_tau
            signal = softplus(u)
            common = (1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0)))) * alpha / (1.0 + alpha * z)
            ds0 = common * (retained * c0)
            ds1 = common * (d_retained * e0 + eta * c1)
        elif signal_kind == SIGNAL_KIND_THRESHOLD_TOTAL:
            z = e0 + eta * e1
            u = np.log1p(alpha * z) - sig_tau
            signal = softplus(u)
            common = (1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0)))) * alpha / (1.0 + alpha * z)
            ds0 = common * c0
            ds1 = common * (eta * c1)
        elif signal_kind == SIGNAL_KIND_TOTAL_LOG:
            z = e0 + eta * e1
            signal = np.log1p(alpha * z)
            ds0 = alpha * c0 / (1.0 + alpha * z)
            ds1 = alpha * eta * c1 / (1.0 + alpha * z)
        else:
            raise ValueError(f"Unsupported signal kind for optimization: {signal_kind!r}")

        return signal, ds0, ds1, retained, d_retained, e0, e1

    def penalty_and_grads(
        retained: np.ndarray,
        d_retained: np.ndarray,
        e0: np.ndarray,
        e1: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        exposure = retained * e0 + e1
        grad0 = np.zeros_like(e0)
        grad1 = np.zeros_like(e1)
        total = 0.0

        for idx in singletons:
            u = np.log1p(exposure[idx]) - tau
            sp = float(softplus(u))
            total += sp**2
            sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
            dq_dx = 2.0 * sp * sigmoid_u / (1.0 + exposure[idx])
            grad0[idx] += dq_dx * (retained[idx] * c0[idx])
            grad1[idx] += dq_dx * (d_retained[idx] * e0[idx] + c1[idx])

        for hi, lo in pair_map:
            group_exposure = exposure[hi] + exposure[lo]
            u = np.log1p(group_exposure) - tau
            sp = float(softplus(u))
            total += sp**2
            sigmoid_u = float(1.0 / (1.0 + np.exp(-np.clip(u, -50.0, 50.0))))
            dq_dx = 2.0 * sp * sigmoid_u / (1.0 + group_exposure)
            grad0[hi] += dq_dx * (retained[hi] * c0[hi])
            grad1[hi] += dq_dx * (d_retained[hi] * e0[hi] + c1[hi])
            grad0[lo] += dq_dx * (retained[lo] * c0[lo])
            grad1[lo] += dq_dx * (d_retained[lo] * e0[lo] + c1[lo])

        return total, grad0, grad1

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0, logits1 = z[:n_domains], z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 = p0 / np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 = p1 / np.sum(p1)

        signal, ds0, ds1, retained, d_retained, e0, e1 = signal_and_grads(p0, p1)
        value = float(model.intercept_ - np.dot(coeff_signal, signal))
        grad0 = -coeff_signal * ds0
        grad1 = -coeff_signal * ds1

        if penalty_coef > 0.0:
            penalty, penalty_grad0, penalty_grad1 = penalty_and_grads(retained, d_retained, e0, e1)
            value += penalty_coef * penalty
            grad0 += penalty_coef * penalty_grad0
            grad1 += penalty_coef * penalty_grad1

        dz0 = p0 * (grad0 - np.dot(grad0, p0))
        dz1 = p1 * (grad1 - np.dot(grad1, p1))
        return value, np.concatenate([dz0, dz1])

    starts: list[np.ndarray] = []
    uniform = np.ones(n_domains) / n_domains
    starts.append(np.concatenate([np.log(uniform), np.log(uniform)]))

    best_observed = data.w[int(np.argmin(data.y))].reshape(-1)
    starts.append(
        np.concatenate(
            [
                np.log(np.clip(best_observed[:n_domains], 1e-12, None)),
                np.log(np.clip(best_observed[n_domains:], 1e-12, None)),
            ]
        )
    )

    for run_name in ("baseline_unimax", "baseline_proportional"):
        idxs = data.frame.index[data.frame[data.name_col] == run_name]
        if len(idxs) == 0:
            continue
        observed = data.w[int(idxs[0])].reshape(-1)
        starts.append(
            np.concatenate(
                [
                    np.log(np.clip(observed[:n_domains], 1e-12, None)),
                    np.log(np.clip(observed[n_domains:], 1e-12, None)),
                ]
            )
        )

    for _ in range(n_random):
        phase0 = rng.gamma(1.0, 1.0, size=n_domains)
        phase1 = rng.gamma(1.0, 1.0, size=n_domains)
        starts.append(np.concatenate([np.log(phase0 / phase0.sum()), np.log(phase1 / phase1.sum())]))

    best_result = None
    for start in starts:
        result = minimize(value_grad_logits, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 = p0 / np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 = p1 / np.sum(p1)
    return best_result, p0, p1


def optimize_cc_pairtotal_model(
    model: CCPairTotalStructuredSurrogate,
    data: PacketData,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize a fitted pair-total CC-aware model over the two-phase simplices."""
    if model.intercept_ is None or model.coef_ is None:
        raise RuntimeError("Model is not fit")

    coef_tab = model.coef_table()
    name_to_idx = {name: idx for idx, name in enumerate(data.domain_names)}

    singleton_map = {
        str(row["name"]): float(row["coef"])
        for _, row in coef_tab[coef_tab["kind1"] == "singleton"][["name", "coef"]].iterrows()
    }
    single_idx = np.array([name_to_idx[name] for name in singleton_map], dtype=int)
    single_coef = np.array([singleton_map[data.domain_names[idx]] for idx in single_idx], dtype=float)

    pair_coef_map = coef_tab[coef_tab["kind1"] == "pair_total"][["name", "coef"]].set_index("name")["coef"].to_dict()
    pair_coef = np.array([float(pair_coef_map[topic]) for topic in model.pair_topics], dtype=float)

    has_premium = np.any(coef_tab["kind1"] == "pair_premium_global")
    premium_coef = (
        float(coef_tab.loc[coef_tab["kind1"] == "pair_premium_global", "coef"].iloc[0]) if has_premium else 0.0
    )
    penalty_coef = float(coef_tab.loc[coef_tab["kind0"] == "penalty", "coef"].iloc[0])

    alpha = float(model.params["alpha"])
    eta = float(model.params["eta"])
    lam = float(model.params["lam"])
    tau = float(model.params["tau"])
    c0 = data.c0
    c1 = data.c1
    n_domains = data.m
    pair_hi = np.array([hi for hi, _ in model.pairs], dtype=int)
    pair_lo = np.array([lo for _, lo in model.pairs], dtype=int)
    high_idx = pair_hi.copy()
    rng = np.random.default_rng(seed)

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0, logits1 = z[:n_domains], z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 = p0 / np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 = p1 / np.sum(p1)

        e0 = c0 * p0
        e1 = c1 * p1
        retained = np.exp(-lam * (1.0 - p1))
        d_retained = lam * retained
        exposure = retained * e0 + eta * e1
        signal = np.log1p(alpha * exposure)
        ds0 = alpha * (retained * c0) / (1.0 + alpha * exposure)
        ds1 = alpha * (d_retained * e0 + eta * c1) / (1.0 + alpha * exposure)

        value = float(model.intercept_)
        grad0 = np.zeros(n_domains, dtype=float)
        grad1 = np.zeros(n_domains, dtype=float)

        if len(single_idx):
            value -= float(np.dot(single_coef, signal[single_idx]))
            grad0[single_idx] -= single_coef * ds0[single_idx]
            grad1[single_idx] -= single_coef * ds1[single_idx]

        if len(model.pairs):
            pair_exposure = exposure[pair_hi] + exposure[pair_lo]
            pair_signal = np.log1p(alpha * pair_exposure)
            d_pair = alpha / (1.0 + alpha * pair_exposure)
            value -= float(np.dot(pair_coef, pair_signal))
            coeff_pair = -pair_coef * d_pair
            for pair_idx, (hi, lo) in enumerate(model.pairs):
                coeff = coeff_pair[pair_idx]
                grad0[hi] += coeff * (retained[hi] * c0[hi])
                grad1[hi] += coeff * (d_retained[hi] * e0[hi] + eta * c1[hi])
                grad0[lo] += coeff * (retained[lo] * c0[lo])
                grad1[lo] += coeff * (d_retained[lo] * e0[lo] + eta * c1[lo])

        if premium_coef > 0.0 and len(high_idx):
            value -= premium_coef * float(np.sum(signal[high_idx]))
            grad0[high_idx] -= premium_coef * ds0[high_idx]
            grad1[high_idx] -= premium_coef * ds1[high_idx]

        penalty_exposure = np.concatenate([exposure[single_idx], exposure[pair_hi] + exposure[pair_lo]])
        penalty_u = np.log1p(penalty_exposure) - tau
        penalty_sp = softplus(penalty_u)
        penalty_deriv = penalty_coef * 2.0 * penalty_sp * sigmoid(penalty_u) / (1.0 + penalty_exposure)
        value += float(penalty_coef * np.sum(penalty_sp**2))

        if len(single_idx):
            grad0[single_idx] += penalty_deriv[: len(single_idx)] * (retained[single_idx] * c0[single_idx])
            grad1[single_idx] += penalty_deriv[: len(single_idx)] * (
                d_retained[single_idx] * e0[single_idx] + eta * c1[single_idx]
            )

        offset = len(single_idx)
        for pair_idx, (hi, lo) in enumerate(model.pairs):
            coeff = penalty_deriv[offset + pair_idx]
            grad0[hi] += coeff * (retained[hi] * c0[hi])
            grad1[hi] += coeff * (d_retained[hi] * e0[hi] + eta * c1[hi])
            grad0[lo] += coeff * (retained[lo] * c0[lo])
            grad1[lo] += coeff * (d_retained[lo] * e0[lo] + eta * c1[lo])

        dz0 = p0 * (grad0 - np.dot(grad0, p0))
        dz1 = p1 * (grad1 - np.dot(grad1, p1))
        return value, np.concatenate([dz0, dz1])

    starts: list[np.ndarray] = []
    uniform = np.ones(n_domains) / n_domains
    starts.append(np.concatenate([np.log(uniform), np.log(uniform)]))

    best_observed = data.w[int(np.argmin(data.y))]
    starts.append(
        np.concatenate(
            [
                np.log(np.clip(best_observed[0], 1e-12, None)),
                np.log(np.clip(best_observed[1], 1e-12, None)),
            ]
        )
    )

    for run_name in ("baseline_unimax", "baseline_proportional", "baseline_olmix_loglinear"):
        idxs = data.frame.index[data.frame[data.name_col] == run_name]
        if len(idxs) == 0:
            continue
        observed = data.w[int(idxs[0])]
        starts.append(
            np.concatenate(
                [
                    np.log(np.clip(observed[0], 1e-12, None)),
                    np.log(np.clip(observed[1], 1e-12, None)),
                ]
            )
        )

    for _ in range(n_random):
        phase0 = rng.gamma(1.0, 1.0, size=n_domains)
        phase1 = rng.gamma(1.0, 1.0, size=n_domains)
        starts.append(np.concatenate([np.log(phase0 / phase0.sum()), np.log(phase1 / phase1.sum())]))

    best_result = None
    for start in starts:
        result = minimize(value_grad_logits, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if best_result is None or float(result.fun) < float(best_result.fun):
            best_result = result

    if best_result is None:
        raise RuntimeError("Optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 = p0 / np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 = p1 / np.sum(p1)
    return best_result, p0, p1


def evaluate_structured_model(data: PacketData, name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Evaluate an independent-domain structured epoch surrogate."""
    model = StructuredEpochSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()
    train = regression_metrics(data.frame, data.name_col, data.y, model.predict(data.w))
    cv_pred = model.cv_predict(data.w, data.y, seed=0, n_splits=5)
    cv = regression_metrics(data.frame, data.name_col, data.y, cv_pred)
    foldwise = foldwise_regret(
        data.frame,
        data.name_col,
        data.w,
        data.y,
        lambda wtr, ytr, seed: (lambda wte: StructuredEpochSurrogate(data, params).fit(wtr, ytr).predict(wte)),
        seed=0,
    )
    row = {
        "model": name,
        "reported_n_params": counts.reported_total,
        "linear_coefficients": counts.linear_coefficients,
        "intercept_params": counts.intercept,
        "global_shape_parameters": counts.global_shape_parameters,
        "total_with_shapes": counts.total_with_shapes,
        "params_json": json.dumps(params, sort_keys=True),
    }
    row.update({f"train_{k}": v for k, v in train.items()})
    row.update({f"cv_{k}": v for k, v in cv.items()})
    row.update({k: v for k, v in foldwise.items() if k != "oof_pred"})
    return row


def evaluate_cc_model(
    data: PacketData, name: str, params: dict[str, Any]
) -> tuple[dict[str, Any], CCPairStructuredSurrogate]:
    """Evaluate a CC-aware structured epoch surrogate."""
    model = CCPairStructuredSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()
    train = regression_metrics(data.frame, data.name_col, data.y, model.predict(data.w))
    cv_pred = model.cv_predict(data.w, data.y, seed=0, n_splits=5)
    cv = regression_metrics(data.frame, data.name_col, data.y, cv_pred)
    foldwise = foldwise_regret(
        data.frame,
        data.name_col,
        data.w,
        data.y,
        lambda wtr, ytr, seed: (lambda wte: CCPairStructuredSurrogate(data, params).fit(wtr, ytr).predict(wte)),
        seed=0,
    )
    row = {
        "model": name,
        "reported_n_params": counts.reported_total,
        "linear_coefficients": counts.linear_coefficients,
        "intercept_params": counts.intercept,
        "global_shape_parameters": counts.global_shape_parameters,
        "total_with_shapes": counts.total_with_shapes,
        "params_json": json.dumps(params, sort_keys=True),
    }
    row.update({f"train_{k}": v for k, v in train.items()})
    row.update({f"cv_{k}": v for k, v in cv.items()})
    row.update({k: v for k, v in foldwise.items() if k != "oof_pred"})
    return row, model


def evaluate_cc_pairtotal_model(
    data: PacketData, name: str, params: dict[str, Any]
) -> tuple[dict[str, Any], CCPairTotalStructuredSurrogate]:
    """Evaluate a CC-aware surrogate with pair-total saturation."""
    model = CCPairTotalStructuredSurrogate(data, params).fit(data.w, data.y)
    counts = model.parameter_count()
    train = regression_metrics(data.frame, data.name_col, data.y, model.predict(data.w))
    cv_pred = model.cv_predict(data.w, data.y, seed=0, n_splits=5)
    cv = regression_metrics(data.frame, data.name_col, data.y, cv_pred)
    foldwise = foldwise_regret(
        data.frame,
        data.name_col,
        data.w,
        data.y,
        lambda wtr, ytr, seed: (lambda wte: CCPairTotalStructuredSurrogate(data, params).fit(wtr, ytr).predict(wte)),
        seed=0,
    )
    row = {
        "model": name,
        "reported_n_params": counts.reported_total,
        "linear_coefficients": counts.linear_coefficients,
        "intercept_params": counts.intercept,
        "global_shape_parameters": counts.global_shape_parameters,
        "total_with_shapes": counts.total_with_shapes,
        "params_json": json.dumps(params, sort_keys=True),
    }
    row.update({f"train_{k}": v for k, v in train.items()})
    row.update({f"cv_{k}": v for k, v in cv.items()})
    row.update({k: v for k, v in foldwise.items() if k != "oof_pred"})
    return row, model


def optimize_starcoder_family(
    data: PacketData, signal_kind: str, *, use_sig_tau: bool = False, seed: int = 0
) -> dict[str, Any]:
    """Fit shared global shape parameters for a Starcoder signal family."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    def unpack(z: np.ndarray) -> dict[str, Any]:
        i = 0
        alpha = safe_exp(z[i], -8.0, 8.0)
        i += 1
        eta = safe_exp(z[i], -8.0, 8.0)
        i += 1
        lam = max(safe_exp(z[i], -12.0, 8.0) - 1e-8, 1e-8)
        i += 1
        tau = float(np.clip(z[i], -4.0, 15.0))
        i += 1
        reg = safe_exp(z[i], -18.0, 0.0)
        i += 1
        params: dict[str, Any] = {
            "signal_kind": signal_kind,
            "pen_kind": PENALTY_KIND_PER_DOMAIN_LOG_THRESHOLD,
            "alpha": alpha,
            "eta": eta,
            "lam": lam,
            "tau": tau,
            "reg": reg,
            "premium_mode": PREMIUM_MODE_NONE,
        }
        if use_sig_tau:
            params["sig_tau"] = float(np.clip(z[i], -4.0, 10.0))
        return params

    def objective(z: np.ndarray) -> float:
        params = unpack(z)
        yhat = np.zeros_like(data.y)
        fold_regrets: list[float] = []
        for tr, te in kf.split(data.w):
            model = CCPairStructuredSurrogate(data, params).fit(data.w[tr], data.y[tr])
            pred = model.predict(data.w[te])
            yhat[te] = pred
            fold_regrets.append(float(data.y[te][np.argmin(pred)] - np.min(data.y[te])))
        rmse = float(np.sqrt(np.mean((yhat - data.y) ** 2)))
        return rmse + 0.02 * float(np.mean(fold_regrets))

    starts = [
        np.array([0.0, 0.0, 0.0, 3.0, -8.0] + ([0.0] if use_sig_tau else [])),
        np.array([0.0, 2.0, -1.0, 5.0, -8.0] + ([1.0] if use_sig_tau else [])),
        np.array([1.0, 2.0, -2.0, 7.0, -10.0] + ([1.0] if use_sig_tau else [])),
        np.array([-1.0, 1.0, -1.0, 3.0, -10.0] + ([0.0] if use_sig_tau else [])),
    ]
    best_params: dict[str, Any] | None = None
    best_obj = float("inf")
    for start in starts:
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-5},
        )
        if np.isfinite(result.fun) and float(result.fun) < best_obj:
            best_obj = float(result.fun)
            best_params = unpack(result.x)
    if best_params is None:
        raise RuntimeError(f"Optimization failed for {signal_kind}")
    return best_params
