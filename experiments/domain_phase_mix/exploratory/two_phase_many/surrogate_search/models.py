# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""Candidate surrogate models for many-domain two-phase data-mixture optimization.

Each model follows the interface:
    fit_fn(spec: DatasetSpec, **kwargs) -> (predict_fn, info_dict)

where predict_fn: (R', N, M) ndarray -> (R',) ndarray predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNetCV, RidgeCV

logger = logging.getLogger(__name__)

EPS = 1e-8

# ---------------------------------------------------------------------------
# Domain structure helpers
# ---------------------------------------------------------------------------
CC_PREFIX = "dolma3_cc/"
CC_TOPICS = [
    "art_and_design",
    "crime_and_law",
    "education_and_jobs",
    "electronics_and_hardware",
    "entertainment",
    "finance_and_business",
    "food_and_dining",
    "games",
    "health",
    "history_and_geography",
    "industrial",
    "literature",
    "science_math_and_technology",
]


@dataclass(frozen=True)
class DomainGrouping:
    """Describes the grouped structure of domains."""

    domain_names: list[str]
    # For each domain: which CC topic it belongs to (-1 if not CC)
    cc_topic_id: np.ndarray  # (M,) int, -1 for non-CC
    # For each domain: 1 if high quality, 0 if low quality, -1 if not CC
    quality_flag: np.ndarray  # (M,) int
    # For each domain: group id for coarser grouping
    group_id: np.ndarray  # (M,) int
    group_names: list[str]
    n_cc_topics: int
    n_noncc: int
    noncc_indices: list[int]


def build_domain_grouping(domain_names: list[str]) -> DomainGrouping:
    """Parse domain names to extract topic/quality structure."""
    M = len(domain_names)
    cc_topic_id = np.full(M, -1, dtype=int)
    quality_flag = np.full(M, -1, dtype=int)
    group_id = np.zeros(M, dtype=int)

    topic_to_id = {topic: i for i, topic in enumerate(CC_TOPICS)}

    # Groups: 0=CC-high, 1=CC-low, 2=dolma3-other, 3=dolmino-curated, 4=dolmino-synth
    group_names = ["cc_high", "cc_low", "dolma3_other", "dolmino_curated", "dolmino_synth"]
    dolmino_synth_names = {
        "dolmino_synth_code",
        "dolmino_synth_instruction",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
    }
    dolmino_curated_names = {
        "dolmino_common_crawl_hq",
        "dolmino_olmocr_pdfs_hq",
        "dolmino_stack_edu_fim",
        "dolmino_stem_heavy_crawl",
    }

    noncc_indices = []
    for d, name in enumerate(domain_names):
        if name.startswith(CC_PREFIX):
            suffix = name[len(CC_PREFIX) :]
            is_high = suffix.endswith("_high")
            topic_name = suffix.rsplit("_", 1)[0]
            if topic_name in topic_to_id:
                cc_topic_id[d] = topic_to_id[topic_name]
                quality_flag[d] = 1 if is_high else 0
                group_id[d] = 0 if is_high else 1
        else:
            noncc_indices.append(d)
            if name in dolmino_synth_names:
                group_id[d] = 4
            elif name in dolmino_curated_names:
                group_id[d] = 3
            else:
                group_id[d] = 2  # dolma3_arxiv, dolma3_finemath_3plus, etc.

    n_cc_topics = len(CC_TOPICS)
    n_noncc = len(noncc_indices)

    return DomainGrouping(
        domain_names=domain_names,
        cc_topic_id=cc_topic_id,
        quality_flag=quality_flag,
        group_id=group_id,
        group_names=group_names,
        n_cc_topics=n_cc_topics,
        n_noncc=n_noncc,
        noncc_indices=noncc_indices,
    )


# ---------------------------------------------------------------------------
# Stable numerics
# ---------------------------------------------------------------------------
def _softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50.0, 20.0))))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _ces(X: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    """CES aggregation: (sum_d w_d * X_d^rho)^(1/rho). X: (..., M), w: (M,)."""
    X = np.maximum(X, 1e-12)
    if abs(rho) < 1e-4:
        return np.exp(np.sum(w * np.log(X), axis=-1))
    inner = np.sum(w * np.power(X, rho), axis=-1)
    inner = np.maximum(inner, 1e-12)
    return np.power(inner, 1.0 / rho)


def _pseudo_huber(resid: np.ndarray, delta: float) -> np.ndarray:
    d = float(delta)
    return d * d * (np.sqrt(1.0 + (resid / d) ** 2) - 1.0)


def _huber_delta(y: np.ndarray) -> float:
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    sigma = 1.4826 * mad
    return float(np.clip(1.345 * sigma, 0.02, 0.30))


def _broadcast_epoch_mult(epoch_multipliers: np.ndarray, N: int, M: int) -> np.ndarray:
    C = np.asarray(epoch_multipliers, dtype=float)
    if C.ndim == 1:
        return np.tile(C[None, :], (N, 1))
    return C


# ---------------------------------------------------------------------------
# Model 1: Ridge / ElasticNet on log-epoch features
# ---------------------------------------------------------------------------
def _log_epoch_features(W: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Build log-epoch feature matrix from weights. Returns (R, N*M + M + 1)."""
    R, N, M = W.shape
    E = W * C[None, :, :]  # (R, N, M)
    features = []
    # Log-epoch per phase-domain
    for k in range(N):
        features.append(np.log(E[:, k, :] + EPS))  # (R, M)
    # Total epochs per domain
    total_E = E.sum(axis=1)  # (R, M)
    features.append(np.log(total_E + EPS))
    # Phase 1 fraction of total epochs
    if N > 1:
        phase1_frac = E[:, -1, :] / (total_E + EPS)
        features.append(phase1_frac)
    return np.column_stack(features)


def fit_ridge_log_epochs(spec, **kwargs):
    """Ridge regression on log-epoch features."""
    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)
    X_train = _log_epoch_features(spec.weights, C)
    y = spec.y
    model = RidgeCV(alphas=np.logspace(-3, 3, 50), fit_intercept=True)
    model.fit(X_train, y)
    n_params = X_train.shape[1] + 1

    def predict(W_new):
        W_new = np.asarray(W_new, dtype=float)
        if W_new.ndim == 2:
            W_new = W_new[None, :, :]
        return model.predict(_log_epoch_features(W_new, C))

    return predict, {"n_params": n_params, "alpha": model.alpha_}


def fit_elasticnet_log_epochs(spec, **kwargs):
    """ElasticNet on log-epoch features (feature selection baseline)."""
    C = _broadcast_epoch_mult(spec.epoch_multipliers, spec.N, spec.M)
    X_train = _log_epoch_features(spec.weights, C)
    y = spec.y
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        alphas=np.logspace(-5, 0, 50),
        cv=5,
        max_iter=10000,
        fit_intercept=True,
    )
    model.fit(X_train, y)
    n_nonzero = int(np.sum(model.coef_ != 0))
    n_params = n_nonzero + 1

    def predict(W_new):
        W_new = np.asarray(W_new, dtype=float)
        if W_new.ndim == 2:
            W_new = W_new[None, :, :]
        return model.predict(_log_epoch_features(W_new, C))

    return predict, {
        "n_params": n_params,
        "n_features": X_train.shape[1],
        "n_nonzero": n_nonzero,
        "alpha": model.alpha_,
        "l1_ratio": model.l1_ratio_,
    }


# Model 2: Simplified CES (S-CES)
# No interference, no per-domain phi, shared phase weight, all domain CES weights.
# Parameters: c0, logA, logB, (M-1) CES logits, rho, alpha (phase), tau = M+5
# ---------------------------------------------------------------------------
def fit_simplified_ces(spec, *, n_restarts=16, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """CES over log-epoch signals with shared phase weighting."""
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    n_params = 3 + (M - 1) + 1 + max(N - 1, 0) + 1  # c0,A,B, CES logits, rho, alpha, tau

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        logA = p[idx]
        idx += 1
        logB = p[idx]
        idx += 1
        ces_logits = np.zeros(M)
        ces_logits[: M - 1] = p[idx : idx + M - 1]
        idx += M - 1
        a = _softmax(ces_logits)
        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, logA, logB, a, rho, alpha, tau

    def forward(p, W_in):
        c0, logA, logB, a, rho, alpha, tau = unpack(p)
        A_coef = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B_coef = float(np.exp(np.clip(logB, -10.0, 10.0)))
        E = W_in * C[None, :, :]
        # Weighted exposure: sum_k alpha_k * E_{k,d}
        weighted_E = np.einsum("rnm,n->rm", E, alpha)  # (R', M)
        z = np.log1p(weighted_E)  # (R', M)
        U = _ces(z, a, rho)  # (R',)
        # Overfit penalty: total epochs across all domains
        total_E = E.sum(axis=(1, 2))  # (R',)
        P = _softplus(total_E - tau) ** 2
        return c0 - A_coef * U + B_coef * P

    def obj(p):
        yhat = forward(p, W)
        return float(np.sum(_pseudo_huber(yhat - y, huber_d))) + reg * float(np.sum(p * p))

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))
        idx += 1  # logA
        p0[idx] = float(rng.normal(-3.0, 1.0))
        idx += 1  # logB
        p0[idx : idx + M - 1] = rng.normal(0.0, 0.3, M - 1)
        idx += M - 1
        p0[idx] = float(rng.normal(0.0, 0.5))
        idx += 1  # rho
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.5, 0.5, N - 1)
            idx += N - 1  # phase 1 > phase 0
        p0[idx] = float(rng.normal(3.0, 1.0))
        idx += 1  # log(tau)
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("S-CES optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model 3: Quality-Factored CES (QF-CES)
# CES weights factored: topic_base + quality_offset for CC, individual for non-CC.
# Shared phase weight, no interference, single penalty threshold.
# ---------------------------------------------------------------------------
def fit_quality_factored_ces(spec, *, n_restarts=24, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """CES with topic-quality factored weights."""
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    grouping = build_domain_grouping(spec.domain_names)

    # Parameters:
    # c0, logA, logB: 3
    # CC topic logits: n_cc_topics - 1 (one is reference)
    # quality offset: 1 (positive = high quality contributes more)
    # non-CC domain logits: n_noncc
    # rho: 1
    # phase alpha: N-1
    # tau: 1
    n_cc = grouping.n_cc_topics
    n_noncc = grouping.n_noncc
    n_params = 3 + max(n_cc - 1, 0) + 1 + n_noncc + 1 + max(N - 1, 0) + 1

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        logA = p[idx]
        idx += 1
        logB = p[idx]
        idx += 1

        # Build CES weight logits from factored structure
        topic_logits = np.zeros(n_cc)
        if n_cc > 1:
            topic_logits[: n_cc - 1] = p[idx : idx + n_cc - 1]
            idx += n_cc - 1
        quality_offset = p[idx]
        idx += 1
        noncc_logits = p[idx : idx + n_noncc]
        idx += n_noncc

        # Assemble full logit vector
        full_logits = np.zeros(M)
        noncc_idx = 0
        for d in range(M):
            tid = grouping.cc_topic_id[d]
            if tid >= 0:
                # CC domain: topic base + quality offset
                full_logits[d] = topic_logits[tid] + quality_offset * grouping.quality_flag[d]
            else:
                full_logits[d] = noncc_logits[noncc_idx]
                noncc_idx += 1
        a = _softmax(full_logits)

        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, logA, logB, a, rho, alpha, tau

    def forward(p, W_in):
        c0, logA, logB, a, rho, alpha, tau = unpack(p)
        A_coef = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B_coef = float(np.exp(np.clip(logB, -10.0, 10.0)))
        E = W_in * C[None, :, :]
        weighted_E = np.einsum("rnm,n->rm", E, alpha)
        z = np.log1p(weighted_E)
        U = _ces(z, a, rho)
        total_E = E.sum(axis=(1, 2))
        P = _softplus(total_E - tau) ** 2
        return c0 - A_coef * U + B_coef * P

    def obj(p):
        yhat = forward(p, W)
        return float(np.sum(_pseudo_huber(yhat - y, huber_d))) + reg * float(np.sum(p * p))

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))
        idx += 1
        p0[idx] = float(rng.normal(-3.0, 1.0))
        idx += 1
        if n_cc > 1:
            p0[idx : idx + n_cc - 1] = rng.normal(0.0, 0.3, n_cc - 1)
            idx += n_cc - 1
        p0[idx] = float(rng.normal(0.3, 0.3))
        idx += 1  # quality offset (positive init)
        p0[idx : idx + n_noncc] = rng.normal(0.0, 0.5, n_noncc)
        idx += n_noncc
        p0[idx] = float(rng.normal(0.0, 0.5))
        idx += 1
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.5, 0.5, N - 1)
            idx += N - 1
        p0[idx] = float(rng.normal(3.0, 1.0))
        idx += 1
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("QF-CES optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model 4: Group-aggregated CES (G-CES)
# Aggregate domains within groups first, then CES across groups.
# ---------------------------------------------------------------------------
def fit_group_ces(spec, *, n_restarts=24, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """Two-level CES: within-group weighted sum, then CES across groups."""
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    grouping = build_domain_grouping(spec.domain_names)

    n_groups = len(grouping.group_names)
    # Build group membership
    group_members = [[] for _ in range(n_groups)]
    for d in range(M):
        group_members[grouping.group_id[d]].append(d)

    # Parameters:
    # c0, logA, logB: 3
    # Within-group domain weights: sum of (|group|-1) for each group
    n_within_params = sum(max(len(members) - 1, 0) for members in group_members)
    # Cross-group CES logits: n_groups - 1
    # quality_boost: 1 (multiplicative boost for high quality within CC groups)
    # rho: 1
    # phase alpha: N-1
    # tau: 1
    n_params = 3 + n_within_params + max(n_groups - 1, 0) + 1 + 1 + max(N - 1, 0) + 1

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        logA = p[idx]
        idx += 1
        logB = p[idx]
        idx += 1

        # Within-group weights (softmax within each group)
        within_weights = {}
        for g, members in enumerate(group_members):
            if len(members) <= 1:
                within_weights[g] = np.ones(max(len(members), 1))
            else:
                logits = np.zeros(len(members))
                logits[: len(members) - 1] = p[idx : idx + len(members) - 1]
                idx += len(members) - 1
                within_weights[g] = _softmax(logits)

        # Cross-group CES weights
        group_logits = np.zeros(n_groups)
        if n_groups > 1:
            group_logits[: n_groups - 1] = p[idx : idx + n_groups - 1]
            idx += n_groups - 1
        group_a = _softmax(group_logits)

        quality_boost = float(np.exp(np.clip(p[idx], -5.0, 5.0)))
        idx += 1
        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1

        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, logA, logB, within_weights, group_a, quality_boost, rho, alpha, tau

    def forward(p, W_in):
        c0, logA, logB, within_weights, group_a, quality_boost, rho, alpha, tau = unpack(p)
        A_coef = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B_coef = float(np.exp(np.clip(logB, -10.0, 10.0)))
        R_in = W_in.shape[0]
        E = W_in * C[None, :, :]  # (R_in, N, M)
        weighted_E = np.einsum("rnm,n->rm", E, alpha)  # (R_in, M)

        # Apply quality boost to high-quality CC domains
        boosted_E = weighted_E.copy()
        for d in range(M):
            if grouping.quality_flag[d] == 1:
                boosted_E[:, d] *= quality_boost

        # Aggregate within groups
        group_signals = np.zeros((R_in, n_groups))
        for g, members in enumerate(group_members):
            if not members:
                continue
            w = within_weights[g]
            member_z = np.log1p(boosted_E[:, members])  # (R_in, |group|)
            group_signals[:, g] = np.sum(w * member_z, axis=1)

        group_signals = np.maximum(group_signals, 1e-12)
        U = _ces(group_signals, group_a, rho)
        total_E = E.sum(axis=(1, 2))
        P = _softplus(total_E - tau) ** 2
        return c0 - A_coef * U + B_coef * P

    def obj(p):
        yhat = forward(p, W)
        return float(np.sum(_pseudo_huber(yhat - y, huber_d))) + reg * float(np.sum(p * p))

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))
        idx += 1
        p0[idx] = float(rng.normal(-3.0, 1.0))
        idx += 1
        p0[idx : idx + n_within_params] = rng.normal(0.0, 0.3, n_within_params)
        idx += n_within_params
        if n_groups > 1:
            p0[idx : idx + n_groups - 1] = rng.normal(0.0, 0.5, n_groups - 1)
            idx += n_groups - 1
        p0[idx] = float(rng.normal(0.0, 0.3))
        idx += 1  # quality_boost (near 1)
        p0[idx] = float(rng.normal(0.0, 0.5))
        idx += 1  # rho
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.5, 0.5, N - 1)
            idx += N - 1
        p0[idx] = float(rng.normal(3.0, 1.0))
        idx += 1
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("G-CES optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model 5: Additive log-epoch model with group regularization (ADD-GR)
# y = c0 + sum_d beta_d * log(1 + alpha*E0 + (1-alpha)*E1) + gamma * penalty
# beta_d regularized to be similar within groups.
# ---------------------------------------------------------------------------
def fit_additive_group_reg(spec, *, n_restarts=24, seed=0, maxiter=800, **kwargs):
    """Additive model with group-structured regularization on domain coefficients."""
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    grouping = build_domain_grouping(spec.domain_names)

    # Parameters: c0, beta (M), alpha (N-1), gamma, tau = M + N + 2
    n_params = 1 + M + max(N - 1, 0) + 1 + 1

    # Build group regularization matrix
    group_members = [[] for _ in range(len(grouping.group_names))]
    for d in range(M):
        group_members[grouping.group_id[d]].append(d)
    # Also pair CC high/low domains for extra regularization
    cc_pairs = []
    for topic_id in range(grouping.n_cc_topics):
        high_idx = low_idx = None
        for d in range(M):
            if grouping.cc_topic_id[d] == topic_id:
                if grouping.quality_flag[d] == 1:
                    high_idx = d
                else:
                    low_idx = d
        if high_idx is not None and low_idx is not None:
            cc_pairs.append((high_idx, low_idx))

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        beta = p[idx : idx + M]
        idx += M
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        gamma = float(np.exp(np.clip(p[idx], -10.0, 10.0)))
        idx += 1
        tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, beta, alpha, gamma, tau

    def forward(p, W_in):
        c0, beta, alpha, gamma, tau = unpack(p)
        E = W_in * C[None, :, :]
        weighted_E = np.einsum("rnm,n->rm", E, alpha)  # (R', M)
        z = np.log1p(weighted_E)  # (R', M)
        signal = z @ beta  # (R',) - additive contribution
        total_E = E.sum(axis=(1, 2))
        P = _softplus(total_E - tau) ** 2
        return c0 + signal + gamma * P

    def obj(p):
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))

        # Group regularization: pull beta within groups toward group mean
        group_reg = 0.0
        lam_group = 0.5
        for members in group_members:
            if len(members) <= 1:
                continue
            betas = p[1 + np.array(members)]
            group_mean = np.mean(betas)
            group_reg += np.sum((betas - group_mean) ** 2)

        # Pair regularization for CC high/low
        pair_reg = 0.0
        lam_pair = 0.3
        for hi, lo in cc_pairs:
            pair_reg += (p[1 + hi] - p[1 + lo]) ** 2

        # Light L2 on all params
        l2_reg = 0.01 * float(np.sum(p * p))

        return loss + lam_group * group_reg + lam_pair * pair_reg + l2_reg

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        idx += 1
        p0[idx : idx + M] = rng.normal(-0.01, 0.02, M)
        idx += M
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.5, 0.5, N - 1)
            idx += N - 1
        p0[idx] = float(rng.normal(-3.0, 1.0))
        idx += 1
        p0[idx] = float(rng.normal(3.0, 1.0))
        idx += 1
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("ADD-GR optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model 6: Per-domain Power-law CES (PD-CES)
# z_d = E_d^{gamma_group} instead of log(1+E_d).
# Per-group power exponent captures different saturation rates.
# ---------------------------------------------------------------------------
def fit_power_ces(spec, *, n_restarts=24, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """CES with per-group power-law signal extraction."""
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_epoch_mult(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    grouping = build_domain_grouping(spec.domain_names)
    n_groups = len(grouping.group_names)

    # Parameters:
    # c0, logA, logB: 3
    # CES logits: M-1
    # rho: 1
    # per-group power exponents: n_groups
    # phase alpha: N-1
    # tau: 1
    n_params = 3 + (M - 1) + 1 + n_groups + max(N - 1, 0) + 1

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        logA = p[idx]
        idx += 1
        logB = p[idx]
        idx += 1
        ces_logits = np.zeros(M)
        ces_logits[: M - 1] = p[idx : idx + M - 1]
        idx += M - 1
        a = _softmax(ces_logits)
        rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
        idx += 1
        # Per-group power exponents in (0, 1)
        gamma_raw = p[idx : idx + n_groups]
        idx += n_groups
        gamma = _sigmoid(gamma_raw)  # (n_groups,)
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
        idx += 1
        return c0, logA, logB, a, rho, gamma, alpha, tau

    def forward(p, W_in):
        c0, logA, logB, a, rho, gamma, alpha, tau = unpack(p)
        A_coef = float(np.exp(np.clip(logA, -10.0, 10.0)))
        B_coef = float(np.exp(np.clip(logB, -10.0, 10.0)))
        E = W_in * C[None, :, :]
        weighted_E = np.einsum("rnm,n->rm", E, alpha)  # (R', M)
        # Per-domain power law signal: z_d = (E_d + eps)^gamma_group(d)
        z = np.zeros_like(weighted_E)
        for d in range(M):
            g = grouping.group_id[d]
            z[:, d] = np.power(weighted_E[:, d] + EPS, gamma[g])
        U = _ces(z, a, rho)
        total_E = E.sum(axis=(1, 2))
        P = _softplus(total_E - tau) ** 2
        return c0 - A_coef * U + B_coef * P

    # Group-regularized objective
    def obj(p):
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        # Regularize CES logits within groups toward group mean
        ces_start = 3
        group_reg = 0.0
        for gid in range(n_groups):
            members = [d for d in range(M) if grouping.group_id[d] == gid]
            if len(members) <= 1:
                continue
            logits = p[ces_start + np.array([m for m in members if m < M - 1])]
            if len(logits) > 0:
                group_reg += np.sum((logits - np.mean(logits)) ** 2)
        return loss + reg * float(np.sum(p * p)) + 0.3 * group_reg

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[idx] = float(np.median(y))
        idx += 1
        p0[idx] = float(rng.normal(0.0, 1.0))
        idx += 1
        p0[idx] = float(rng.normal(-3.0, 1.0))
        idx += 1
        p0[idx : idx + M - 1] = rng.normal(0.0, 0.3, M - 1)
        idx += M - 1
        p0[idx] = float(rng.normal(0.0, 0.5))
        idx += 1
        p0[idx : idx + n_groups] = rng.normal(0.0, 0.5, n_groups)
        idx += n_groups  # gamma near 0.5
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.5, 0.5, N - 1)
            idx += N - 1
        p0[idx] = float(rng.normal(3.0, 1.0))
        idx += 1
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-10})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("PD-CES optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Ridge-LogEpoch": fit_ridge_log_epochs,
    "ElasticNet-LogEpoch": fit_elasticnet_log_epochs,
    "S-CES": fit_simplified_ces,
    "QF-CES": fit_quality_factored_ces,
    "G-CES": fit_group_ces,
    "ADD-GR": fit_additive_group_reg,
    "PD-CES": fit_power_ces,
}
