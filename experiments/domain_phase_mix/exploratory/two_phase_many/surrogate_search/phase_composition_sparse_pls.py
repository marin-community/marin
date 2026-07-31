# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seven-term phase-composition surrogate recovered from the Yixin packet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    PacketData,
    ParameterCount,
    load_two_phase_many_packet,
)

EPS = 1e-9
INNER_Z_CLIP = 3.0
OUTER_EXP_CLIP = 6.0

BASE_FEATURE_NAMES = (
    "phase_sum__dolmino_stack_edu_fim",
    "phase_sum__dolma3_stack_edu",
    "phase_sum__dolmino_stem_heavy_crawl",
    "phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking",
    "phase_gap__dolma3_stack_edu",
    "phase_sum__dolmino_synth_code",
)

SELECTED_TERM_NAMES = (
    "expneg_phase_sum__dolmino_stack_edu_fim",
    "expneg_phase_sum__dolma3_stack_edu",
    "exp(expneg_phase_sum__dolmino_stem_heavy_crawl)",
    "negrelu_phase_alr__phase_1_dolma3_arxiv__over__phase_1_dolmino_synth_thinking",
    "exp_phase_gap__dolma3_stack_edu",
    "invsq1p_phase_sum__dolmino_synth_code",
    "expneg_phase_sum__dolmino_stem_heavy_crawl",
)


@dataclass(frozen=True)
class OutlierFilterSummary:
    """Summary of the target outlier filter."""

    lower: float
    upper: float
    kept_rows: int
    dropped_rows: int


def filter_target_outliers(y: np.ndarray) -> tuple[np.ndarray, OutlierFilterSummary]:
    """Filter 3-IQR target outliers exactly as in the collaborator script."""
    mask = np.isfinite(y)
    q1, q3 = np.quantile(y[mask], [0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    keep = mask & (y >= lower) & (y <= upper)
    return keep, OutlierFilterSummary(
        lower=float(lower),
        upper=float(upper),
        kept_rows=int(np.sum(keep)),
        dropped_rows=int(len(y) - np.sum(keep)),
    )


def load_phase_composition_packet(target: str = MANY_DOMAIN_TARGET) -> PacketData:
    """Load the many-domain packet used by the collaborator scripts."""
    return load_two_phase_many_packet(target=target)


class PhaseCompositionSparsePLSSurrogate:
    """Seven-term phase-composition linear surrogate."""

    def __init__(self, data: PacketData):
        self.data = data
        self._domain_index = {domain_name: idx for idx, domain_name in enumerate(data.domain_names)}
        self._stack_fim_idx = self._domain_index["dolmino_stack_edu_fim"]
        self._stack_edu_idx = self._domain_index["dolma3_stack_edu"]
        self._stem_idx = self._domain_index["dolmino_stem_heavy_crawl"]
        self._arxiv_idx = self._domain_index["dolma3_arxiv"]
        self._thinking_idx = self._domain_index["dolmino_synth_thinking"]
        self._code_idx = self._domain_index["dolmino_synth_code"]
        self.base_feature_mean_: np.ndarray | None = None
        self.base_feature_std_: np.ndarray | None = None
        self.term_mean_: np.ndarray | None = None
        self.term_std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None

    def parameter_count(self) -> ParameterCount:
        """Return the standardized parameter-count breakdown."""
        return ParameterCount(linear_coefficients=len(SELECTED_TERM_NAMES), intercept=1, global_shape_parameters=0)

    def _base_feature_array(self, weights: np.ndarray) -> np.ndarray:
        arr = np.asarray(weights, dtype=float)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        p0 = arr[:, 0, :]
        p1 = arr[:, 1, :]
        x0 = 0.5 * p0
        x1 = 0.5 * p1

        features = np.empty((arr.shape[0], len(BASE_FEATURE_NAMES)), dtype=float)
        features[:, 0] = x0[:, self._stack_fim_idx] + x1[:, self._stack_fim_idx]
        features[:, 1] = x0[:, self._stack_edu_idx] + x1[:, self._stack_edu_idx]
        features[:, 2] = x0[:, self._stem_idx] + x1[:, self._stem_idx]
        features[:, 3] = np.log((x1[:, self._arxiv_idx] + EPS) / (x1[:, self._thinking_idx] + EPS))
        features[:, 4] = x1[:, self._stack_edu_idx] - x0[:, self._stack_edu_idx]
        features[:, 5] = x0[:, self._code_idx] + x1[:, self._code_idx]
        return features

    @staticmethod
    def _safe_std(values: np.ndarray) -> np.ndarray:
        std = np.asarray(values, dtype=float).std(axis=0)
        return np.where(std == 0.0, 1.0, std)

    @staticmethod
    def _selected_term_array_from_z(z: np.ndarray) -> np.ndarray:
        terms = np.empty((z.shape[0], len(SELECTED_TERM_NAMES)), dtype=float)
        stem_expneg = np.exp(np.clip(-z[:, 2], -INNER_Z_CLIP, INNER_Z_CLIP))

        terms[:, 0] = np.exp(np.clip(-z[:, 0], -INNER_Z_CLIP, INNER_Z_CLIP))
        terms[:, 1] = np.exp(np.clip(-z[:, 1], -INNER_Z_CLIP, INNER_Z_CLIP))
        terms[:, 2] = np.exp(np.clip(stem_expneg, -OUTER_EXP_CLIP, OUTER_EXP_CLIP))
        terms[:, 3] = np.maximum(-z[:, 3], 0.0)
        terms[:, 4] = np.exp(np.clip(z[:, 4], -INNER_Z_CLIP, INNER_Z_CLIP))
        terms[:, 5] = 1.0 / np.sqrt(1.0 + z[:, 5] ** 2)
        terms[:, 6] = stem_expneg
        return terms

    def _selected_term_array(
        self,
        weights: np.ndarray,
        *,
        base_feature_mean: np.ndarray,
        base_feature_std: np.ndarray,
    ) -> np.ndarray:
        base = self._base_feature_array(weights)
        z = (base - base_feature_mean[None, :]) / base_feature_std[None, :]
        return self._selected_term_array_from_z(z)

    def fit(self, weights: np.ndarray, y: np.ndarray) -> PhaseCompositionSparsePLSSurrogate:
        """Fit the seven-term full model."""
        base = self._base_feature_array(weights)
        self.base_feature_mean_ = base.mean(axis=0)
        self.base_feature_std_ = self._safe_std(base)
        terms = self._selected_term_array(
            weights, base_feature_mean=self.base_feature_mean_, base_feature_std=self.base_feature_std_
        )
        self.term_mean_ = terms.mean(axis=0)
        self.term_std_ = self._safe_std(terms)
        design = (terms - self.term_mean_[None, :]) / self.term_std_[None, :]
        fit = LinearRegression().fit(design, np.asarray(y, dtype=float))
        self.coef_ = np.asarray(fit.coef_, dtype=float)
        self.intercept_ = float(fit.intercept_)
        return self

    def predict(self, weights: np.ndarray) -> np.ndarray:
        """Predict the target for one or more two-phase schedules."""
        if self.base_feature_mean_ is None or self.base_feature_std_ is None:
            raise RuntimeError("Model must be fit before prediction")
        if self.term_mean_ is None or self.term_std_ is None or self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model must be fit before prediction")
        terms = self._selected_term_array(
            weights,
            base_feature_mean=self.base_feature_mean_,
            base_feature_std=self.base_feature_std_,
        )
        design = (terms - self.term_mean_[None, :]) / self.term_std_[None, :]
        return np.asarray(self.intercept_ + design @ self.coef_, dtype=float)

    def coefficient_table(self) -> dict[str, float]:
        """Return the fitted term coefficients."""
        if self.coef_ is None:
            raise RuntimeError("Model must be fit before coefficients are available")
        return {name: float(coef) for name, coef in zip(SELECTED_TERM_NAMES, self.coef_, strict=True)}

    def fit_summary(self, weights: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Return train metrics for the fitted model."""
        pred = self.predict(weights)
        rho = spearmanr(y, pred).statistic
        return {
            "train_r2": float(r2_score(y, pred)),
            "train_spearman": float(0.0 if rho is None or not np.isfinite(rho) else rho),
        }


def reproduction_cv_summary(
    data: PacketData,
    *,
    n_splits: int | None = None,
) -> tuple[dict[str, Any], PhaseCompositionSparsePLSSurrogate]:
    """Reproduce the collaborator's reported CV and full-fit coefficients."""
    y_full = np.asarray(data.y, dtype=float)
    keep, filt = filter_target_outliers(y_full)
    weights = np.asarray(data.w[keep], dtype=float)
    y = np.asarray(y_full[keep], dtype=float)

    model = PhaseCompositionSparsePLSSurrogate(data).fit(weights, y)
    terms = model._selected_term_array(
        weights,
        base_feature_mean=model.base_feature_mean_,
        base_feature_std=model.base_feature_std_,
    )
    splits = min(5, max(2, len(y) // 8)) if n_splits is None else n_splits
    cv = KFold(n_splits=splits, shuffle=True, random_state=0)
    fold_rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(terms), start=1):
        x_train = terms[train_idx]
        x_test = terms[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        mu = x_train.mean(axis=0)
        sd = np.where(x_train.std(axis=0) == 0.0, 1.0, x_train.std(axis=0))
        pred = LinearRegression().fit((x_train - mu) / sd, y_train).predict((x_test - mu) / sd)
        rho = spearmanr(y_test, pred).statistic
        fold_rows.append(
            {
                "fold": float(fold),
                "r2": float(r2_score(y_test, pred)),
                "spearman": float(0.0 if rho is None or not np.isfinite(rho) else rho),
            }
        )

    payload = {
        "target": MANY_DOMAIN_TARGET,
        "rows": len(y),
        "outlier_filter": {
            "lower": filt.lower,
            "upper": filt.upper,
            "kept_rows": filt.kept_rows,
            "dropped_rows": filt.dropped_rows,
        },
        "selected_terms": list(SELECTED_TERM_NAMES),
        "cv_r2_mean": float(np.mean([row["r2"] for row in fold_rows])),
        "cv_spearman_mean": float(np.mean([row["spearman"] for row in fold_rows])),
        "fold_metrics": fold_rows,
        "full_fit_intercept": float(model.intercept_),
        "full_fit_coefficients": model.coefficient_table(),
        **model.fit_summary(weights, y),
    }
    return payload, model


def optimize_phase_composition_sparse_pls_model(
    data: PacketData,
    model: PhaseCompositionSparsePLSSurrogate,
    *,
    n_random: int = 20,
    seed: int = 0,
) -> tuple[Any, np.ndarray, np.ndarray]:
    """Optimize the fitted seven-term model over the two phase simplices."""
    if model.base_feature_mean_ is None or model.base_feature_std_ is None:
        raise RuntimeError("Model must be fit before optimization")
    if model.term_mean_ is None or model.term_std_ is None or model.coef_ is None or model.intercept_ is None:
        raise RuntimeError("Model must be fit before optimization")

    n_domains = data.m
    base_mu = np.asarray(model.base_feature_mean_, dtype=float)
    base_sd = np.asarray(model.base_feature_std_, dtype=float)
    term_mu = np.asarray(model.term_mean_, dtype=float)
    term_sd = np.asarray(model.term_std_, dtype=float)
    coef = np.asarray(model.coef_, dtype=float)
    intercept = float(model.intercept_)
    rng = np.random.default_rng(seed)

    def _exp_clipped(value: float, *, negate: bool = False) -> tuple[float, float]:
        signed = -value if negate else value
        clipped = float(np.clip(signed, -INNER_Z_CLIP, INNER_Z_CLIP))
        out = float(np.exp(clipped))
        deriv = 0.0
        if -INNER_Z_CLIP < signed < INNER_Z_CLIP:
            deriv = -out if negate else out
        return out, deriv

    def value_grad_logits(z: np.ndarray) -> tuple[float, np.ndarray]:
        logits0 = z[:n_domains]
        logits1 = z[n_domains:]
        p0 = np.exp(logits0 - np.max(logits0))
        p0 /= np.sum(p0)
        p1 = np.exp(logits1 - np.max(logits1))
        p1 /= np.sum(p1)

        x0 = 0.5 * p0
        x1 = 0.5 * p1
        raw = np.empty(len(BASE_FEATURE_NAMES), dtype=float)
        raw[0] = x0[model._stack_fim_idx] + x1[model._stack_fim_idx]
        raw[1] = x0[model._stack_edu_idx] + x1[model._stack_edu_idx]
        raw[2] = x0[model._stem_idx] + x1[model._stem_idx]
        raw[3] = np.log((x1[model._arxiv_idx] + EPS) / (x1[model._thinking_idx] + EPS))
        raw[4] = x1[model._stack_edu_idx] - x0[model._stack_edu_idx]
        raw[5] = x0[model._code_idx] + x1[model._code_idx]
        raw_z = (raw - base_mu) / base_sd

        terms = np.empty(len(SELECTED_TERM_NAMES), dtype=float)
        dterm_draw = np.zeros((len(SELECTED_TERM_NAMES), len(BASE_FEATURE_NAMES)), dtype=float)

        terms[0], d_signed = _exp_clipped(raw_z[0], negate=True)
        dterm_draw[0, 0] = d_signed / base_sd[0]

        terms[1], d_signed = _exp_clipped(raw_z[1], negate=True)
        dterm_draw[1, 1] = d_signed / base_sd[1]

        stem_expneg, d_signed = _exp_clipped(raw_z[2], negate=True)
        terms[6] = stem_expneg
        dterm_draw[6, 2] = d_signed / base_sd[2]

        outer_input = stem_expneg
        outer_clipped = float(np.clip(outer_input, -OUTER_EXP_CLIP, OUTER_EXP_CLIP))
        terms[2] = float(np.exp(outer_clipped))
        if -OUTER_EXP_CLIP < outer_input < OUTER_EXP_CLIP:
            dterm_draw[2, 2] = terms[2] * dterm_draw[6, 2]

        terms[3] = max(-raw_z[3], 0.0)
        if raw_z[3] < 0.0:
            dterm_draw[3, 3] = -1.0 / base_sd[3]

        terms[4], d_signed = _exp_clipped(raw_z[4], negate=False)
        dterm_draw[4, 4] = d_signed / base_sd[4]

        denom = (1.0 + raw_z[5] ** 2) ** 1.5
        terms[5] = 1.0 / np.sqrt(1.0 + raw_z[5] ** 2)
        dterm_draw[5, 5] = (-raw_z[5] / denom) / base_sd[5]

        design = (terms - term_mu) / term_sd
        value = intercept + float(np.dot(design, coef))

        grad_terms = coef / term_sd
        grad_raw = grad_terms @ dterm_draw

        grad0 = np.zeros(n_domains, dtype=float)
        grad1 = np.zeros(n_domains, dtype=float)

        grad0[model._stack_fim_idx] += 0.5 * grad_raw[0]
        grad1[model._stack_fim_idx] += 0.5 * grad_raw[0]

        grad0[model._stack_edu_idx] += 0.5 * grad_raw[1]
        grad1[model._stack_edu_idx] += 0.5 * grad_raw[1]

        grad0[model._stem_idx] += 0.5 * grad_raw[2]
        grad1[model._stem_idx] += 0.5 * grad_raw[2]

        grad1[model._arxiv_idx] += 0.5 * grad_raw[3] / (x1[model._arxiv_idx] + EPS)
        grad1[model._thinking_idx] -= 0.5 * grad_raw[3] / (x1[model._thinking_idx] + EPS)

        grad0[model._stack_edu_idx] -= 0.5 * grad_raw[4]
        grad1[model._stack_edu_idx] += 0.5 * grad_raw[4]

        grad0[model._code_idx] += 0.5 * grad_raw[5]
        grad1[model._code_idx] += 0.5 * grad_raw[5]

        logits_grad0 = p0 * (grad0 - float(np.dot(grad0, p0)))
        logits_grad1 = p1 * (grad1 - float(np.dot(grad1, p1)))
        return value, np.concatenate([logits_grad0, logits_grad1])

    starts = [
        np.zeros(2 * n_domains, dtype=float),
        *[
            np.concatenate(
                [
                    np.log(rng.dirichlet(np.ones(n_domains))),
                    np.log(rng.dirichlet(np.ones(n_domains))),
                ]
            )
            for _ in range(n_random)
        ],
    ]

    best_result = None
    best_value = float("inf")
    for start in starts:
        result = minimize(value_grad_logits, start, jac=True, method="L-BFGS-B", options={"maxiter": 800})
        if result.fun < best_value:
            best_result = result
            best_value = float(result.fun)

    if best_result is None:
        raise RuntimeError("Optimization failed")

    logits0 = best_result.x[:n_domains]
    logits1 = best_result.x[n_domains:]
    p0 = np.exp(logits0 - np.max(logits0))
    p0 /= np.sum(p0)
    p1 = np.exp(logits1 - np.max(logits1))
    p1 /= np.sum(p1)
    return best_result, p0, p1
