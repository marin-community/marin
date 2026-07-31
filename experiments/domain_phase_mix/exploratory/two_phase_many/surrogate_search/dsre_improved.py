# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Improved DS-RE-CEQ variants for the many-domain setting.

Key changes from DS-RE-CEQ:
1. Share per-domain parameters via domain groups (CC topic pairs, quality splits)
2. Replace squared softplus overfit penalty with power-law (Finetuner's Fallacy)
3. Simplify interference to shared or per-group (not per-domain)
4. Add group-regularized CES weights

Design principles:
- Preserve the core DS-RE-CEQ signal extraction (log-diminishing returns)
- Preserve CES aggregation (domain complementarity)
- Fix the parameter explosion: 162 → ~40-60 parameters
- Use the overfitting scaling law: gap ∝ E^β, β ∈ (0,1)
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

EPS = 1e-8

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

# Group IDs: 0=CC-high, 1=CC-low, 2=dolma3-other, 3=dolmino-curated, 4=dolmino-synth
DOLMINO_SYNTH = {
    "dolmino_synth_code",
    "dolmino_synth_instruction",
    "dolmino_synth_math",
    "dolmino_synth_qa",
    "dolmino_synth_thinking",
}
DOLMINO_CURATED = {
    "dolmino_common_crawl_hq",
    "dolmino_olmocr_pdfs_hq",
    "dolmino_stack_edu_fim",
    "dolmino_stem_heavy_crawl",
}
N_GROUPS = 5


def _domain_group_ids(domain_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Returns (group_id[M], cc_topic_id[M], group_members[G])."""
    M = len(domain_names)
    topic_map = {t: i for i, t in enumerate(CC_TOPICS)}
    group_id = np.zeros(M, dtype=int)
    cc_topic_id = np.full(M, -1, dtype=int)
    for d, name in enumerate(domain_names):
        if name.startswith("dolma3_cc/"):
            suffix = name[len("dolma3_cc/") :]
            is_high = suffix.endswith("_high")
            topic = suffix.rsplit("_", 1)[0]
            if topic in topic_map:
                cc_topic_id[d] = topic_map[topic]
            group_id[d] = 0 if is_high else 1
        elif name in DOLMINO_SYNTH:
            group_id[d] = 4
        elif name in DOLMINO_CURATED:
            group_id[d] = 3
        else:
            group_id[d] = 2
    members = [[] for _ in range(N_GROUPS)]
    for d in range(M):
        members[group_id[d]].append(d)
    return group_id, cc_topic_id, members


def _softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _softplus(x):
    return np.where(x > 20.0, x, np.log1p(np.exp(np.clip(x, -50.0, 20.0))))


def _ces(X, w, rho):
    X = np.maximum(X, 1e-12)
    if abs(rho) < 1e-4:
        return np.exp(np.sum(w * np.log(X), axis=-1))
    inner = np.sum(w * np.power(X, rho), axis=-1)
    return np.power(np.maximum(inner, 1e-12), 1.0 / rho)


def _pseudo_huber(r, d):
    return d * d * (np.sqrt(1.0 + (r / d) ** 2) - 1.0)


def _huber_delta(y):
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return float(np.clip(1.345 * 1.4826 * mad, 0.005, 0.10))


def _broadcast_C(epoch_multipliers, N, M):
    C = np.asarray(epoch_multipliers, dtype=float)
    if C.ndim == 1:
        return np.tile(C[None, :], (N, 1))
    return C


# ---------------------------------------------------------------------------
# Model V1: Lean DS-RE — shared pi, no interference, grouped CES, power-law overfit
# This is DS-RE-CEQ stripped to its core with the overfitting fixed.
# ---------------------------------------------------------------------------
def fit_lean_dsre(spec, *, n_restarts=32, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """Lean DS-RE: signal extraction + CES utility + power-law overfit penalty.

    Changes from DS-RE-CEQ:
    - Shared phase weight (not per-domain pi)
    - No interference/retention (r=1)
    - No satiety (phi=1, simplest diminishing returns)
    - Power-law overfit: gap = gamma * sum_d (E_d)^beta, beta in (0,1)
    - Group-regularized CES weights

    Parameters: c0, logA, logB (3) + CES logits (M-1) + rho (1)
                + phase alpha (N-1) + gamma (1) + beta (1) = M + 5
    """
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_C(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    group_id, _, group_members = _domain_group_ids(spec.domain_names)

    n_params = 3 + (M - 1) + 1 + max(N - 1, 0) + 2  # c0,A,B + CES + rho + alpha + gamma,beta

    def unpack(p):
        idx = 0
        c0, logA, logB = p[0], p[1], p[2]
        idx = 3
        ces_logits = np.zeros(M)
        ces_logits[: M - 1] = p[idx : idx + M - 1]
        idx += M - 1
        a = _softmax(ces_logits)
        rho = float(np.clip(5 * np.tanh(p[idx]), -10, 0.99))
        idx += 1
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        log_gamma = p[idx]
        idx += 1
        beta_raw = p[idx]
        idx += 1
        return c0, logA, logB, a, rho, alpha, log_gamma, beta_raw

    def forward(p, W_in):
        c0, logA, logB, a, rho, alpha, log_gamma, beta_raw = unpack(p)
        A = float(np.exp(np.clip(logA, -10, 10)))
        B = float(np.exp(np.clip(log_gamma, -10, 10)))
        beta = float(_sigmoid(beta_raw))  # in (0, 1) — sublinear overfit
        E = W_in * C[None, :, :]  # (R', N, M)
        # Phase-weighted total exposure
        wE = np.einsum("rnm,n->rm", E, alpha)  # (R', M)
        # Signal extraction: log(1 + E)
        z = np.log1p(wE)  # (R', M)
        U = _ces(z, a, rho)
        # Power-law overfit penalty: sum_d E_total_d ^ beta
        E_total = E.sum(axis=1)  # (R', M) — total epochs per domain
        overfit = np.sum(np.power(E_total + EPS, beta), axis=1)  # (R',)
        return c0 - A * U + B * overfit

    def obj(p):
        yhat = forward(p, W)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        # Group regularization on CES logits
        grp_reg = 0.0
        for members in group_members:
            if len(members) <= 1:
                continue
            logits_idx = [m for m in members if m < M - 1]
            if logits_idx:
                vals = p[3 + np.array(logits_idx)]
                grp_reg += np.sum((vals - np.mean(vals)) ** 2)
        return loss + reg * float(np.sum(p * p)) + 0.2 * grp_reg

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[0] = float(np.median(y))
        p0[1] = rng.normal(0, 1)
        p0[2] = rng.normal(-3, 1)
        idx = 3
        p0[idx : idx + M - 1] = rng.normal(0, 0.3, M - 1)
        idx += M - 1
        p0[idx] = rng.normal(0, 0.5)
        idx += 1  # rho
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.3, 0.5, N - 1)
            idx += N - 1
        p0[idx] = rng.normal(-4, 1)
        idx += 1  # log_gamma
        p0[idx] = rng.normal(0, 0.5)
        idx += 1  # beta_raw
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x.copy()
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("Lean DS-RE optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new, float).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model V2: Lean DS-RE with per-group overfit exponents
# ---------------------------------------------------------------------------
def fit_lean_dsre_v2(spec, *, n_restarts=32, seed=0, maxiter=800, reg=1e-4, **kwargs):
    """V2: per-group power-law overfit exponents + quality-factored CES weights.

    Changes from V1:
    - CES weights factored: topic_base + quality_offset for CC, individual for non-CC
    - Per-group overfit exponents beta_g (5 groups)
    - Per-group overfit scales gamma_g (5 groups)

    Parameters: c0, logA (2) + CES (n_cc_topics-1 + 1 + n_noncc) + rho (1)
                + alpha (N-1) + 5 * gamma + 5 * beta = ~32
    """
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_C(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    group_id, cc_topic_id, group_members = _domain_group_ids(spec.domain_names)

    n_cc = len(CC_TOPICS)
    noncc_idx = [d for d in range(M) if cc_topic_id[d] < 0]
    n_noncc = len(noncc_idx)

    # c0, logA (2) + topic logits (n_cc-1) + quality offset (1) + noncc logits (n_noncc)
    # + rho (1) + alpha (N-1) + gamma (N_GROUPS) + beta (N_GROUPS)
    n_params = 2 + max(n_cc - 1, 0) + 1 + n_noncc + 1 + max(N - 1, 0) + 2 * N_GROUPS

    def unpack(p):
        idx = 0
        c0 = p[idx]
        logA = p[idx + 1]
        idx += 2
        topic_logits = np.zeros(n_cc)
        if n_cc > 1:
            topic_logits[: n_cc - 1] = p[idx : idx + n_cc - 1]
            idx += n_cc - 1
        quality_offset = p[idx]
        idx += 1
        noncc_logits = p[idx : idx + n_noncc]
        idx += n_noncc
        # Assemble CES weight logits
        full_logits = np.zeros(M)
        ni = 0
        for d in range(M):
            tid = cc_topic_id[d]
            if tid >= 0:
                is_high = 1.0 if spec.domain_names[d].endswith("_high") else 0.0
                full_logits[d] = topic_logits[tid] + quality_offset * is_high
            else:
                full_logits[d] = noncc_logits[ni]
                ni += 1
        a = _softmax(full_logits)
        rho = float(np.clip(5 * np.tanh(p[idx]), -10, 0.99))
        idx += 1
        if N > 1:
            alpha_logits = np.zeros(N)
            alpha_logits[: N - 1] = p[idx : idx + N - 1]
            idx += N - 1
            alpha = _softmax(alpha_logits)
        else:
            alpha = np.ones(1)
        log_gammas = p[idx : idx + N_GROUPS]
        idx += N_GROUPS
        beta_raws = p[idx : idx + N_GROUPS]
        idx += N_GROUPS
        return c0, logA, a, rho, alpha, log_gammas, beta_raws

    def forward(p, W_in):
        c0, logA, a, rho, alpha, log_gammas, beta_raws = unpack(p)
        A = float(np.exp(np.clip(logA, -10, 10)))
        gammas = np.exp(np.clip(log_gammas, -10, 10))
        betas = _sigmoid(beta_raws)  # each in (0, 1)
        R_in = W_in.shape[0]
        E = W_in * C[None, :, :]
        wE = np.einsum("rnm,n->rm", E, alpha)
        z = np.log1p(wE)
        U = _ces(z, a, rho)
        # Per-group overfit
        E_total = E.sum(axis=1)  # (R_in, M)
        overfit = np.zeros(R_in)
        for g in range(N_GROUPS):
            members = group_members[g]
            if not members:
                continue
            Eg = E_total[:, members]  # (R_in, |g|)
            overfit += gammas[g] * np.sum(np.power(Eg + EPS, betas[g]), axis=1)
        return c0 - A * U + overfit

    def obj(p):
        yhat = forward(p, W)
        return float(np.sum(_pseudo_huber(yhat - y, huber_d))) + reg * float(np.sum(p * p))

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        idx = 0
        p0[0] = float(np.median(y))
        p0[1] = rng.normal(0, 1)
        idx = 2
        if n_cc > 1:
            p0[idx : idx + n_cc - 1] = rng.normal(0, 0.3, n_cc - 1)
            idx += n_cc - 1
        p0[idx] = rng.normal(0.3, 0.3)
        idx += 1
        p0[idx : idx + n_noncc] = rng.normal(0, 0.5, n_noncc)
        idx += n_noncc
        p0[idx] = rng.normal(0, 0.5)
        idx += 1
        if N > 1:
            p0[idx : idx + N - 1] = rng.normal(0.3, 0.5, N - 1)
            idx += N - 1
        p0[idx : idx + N_GROUPS] = rng.normal(-4, 1, N_GROUPS)
        idx += N_GROUPS
        p0[idx : idx + N_GROUPS] = rng.normal(0, 0.5, N_GROUPS)
        idx += N_GROUPS
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x.copy()
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("Lean DS-RE V2 optimization failed")

    final_p = best_p.copy()
    predict = lambda W_new: forward(final_p, np.asarray(W_new, float).reshape(-1, N, M))
    return predict, {"n_params": n_params, "final_p": final_p.copy()}


# ---------------------------------------------------------------------------
# Model V3: Additive linear + power-law overfit (Ridge + Overfit)
# Motivated by: Ridge-Raw works well, just needs the nonlinear overfit term.
# ---------------------------------------------------------------------------
def fit_ridge_plus_overfit(spec, *, n_restarts=32, seed=0, maxiter=800, reg=1e-4, alpha=0.35, **kwargs):
    """y = c0 + beta^T W_mix + sum_g gamma_g * sum_{d in g} E_d^{beta_g}

    Combines the Ridge-Raw linear term (which works well) with per-group
    power-law overfitting from the Finetuner's Fallacy.
    """
    rng = np.random.default_rng(seed)
    W, y = spec.weights, spec.y
    R, N, M = W.shape
    C = _broadcast_C(spec.epoch_multipliers, N, M)
    huber_d = _huber_delta(y)
    group_id, _, group_members = _domain_group_ids(spec.domain_names)

    E_all = W * C[None, :, :]  # (R, N, M)
    E_total = E_all.sum(axis=1)  # (R, M)
    W_mix = alpha * W[:, 0, :] + (1 - alpha) * W[:, 1, :] if N > 1 else W[:, 0, :]

    # Parameters: c0 (1) + beta (M) + gamma (N_GROUPS) + beta_overfit (N_GROUPS)
    n_params = 1 + M + 2 * N_GROUPS

    def unpack(p):
        idx = 0
        c0 = p[idx]
        idx += 1
        beta = p[idx : idx + M]
        idx += M
        log_gammas = p[idx : idx + N_GROUPS]
        idx += N_GROUPS
        beta_raws = p[idx : idx + N_GROUPS]
        idx += N_GROUPS
        return c0, beta, log_gammas, beta_raws

    def forward_precomputed(p, W_mix_in, E_total_in):
        c0, beta, log_gammas, beta_raws = unpack(p)
        gammas = np.exp(np.clip(log_gammas, -10, 10))
        betas = _sigmoid(beta_raws)
        linear = W_mix_in @ beta
        overfit = np.zeros(W_mix_in.shape[0])
        for g in range(N_GROUPS):
            members = group_members[g]
            if not members:
                continue
            Eg = E_total_in[:, members]
            overfit += gammas[g] * np.sum(np.power(Eg + EPS, betas[g]), axis=1)
        return c0 + linear + overfit

    def obj(p):
        yhat = forward_precomputed(p, W_mix, E_total)
        loss = float(np.sum(_pseudo_huber(yhat - y, huber_d)))
        # Group regularization on beta
        grp_reg = 0.0
        for members in group_members:
            if len(members) <= 1:
                continue
            vals = p[1 + np.array(members)]
            grp_reg += np.sum((vals - np.mean(vals)) ** 2)
        return loss + reg * float(np.sum(p * p)) + 0.1 * grp_reg

    best_val, best_p = np.inf, None
    for _ in range(n_restarts):
        p0 = np.zeros(n_params)
        p0[0] = float(np.median(y))
        p0[1 : 1 + M] = rng.normal(0, 0.01, M)
        p0[1 + M : 1 + M + N_GROUPS] = rng.normal(-5, 1, N_GROUPS)
        p0[1 + M + N_GROUPS :] = rng.normal(0, 0.5, N_GROUPS)
        try:
            res = minimize(obj, p0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val, best_p = float(res.fun), res.x.copy()
        except Exception:
            continue

    if best_p is None:
        raise RuntimeError("Ridge+Overfit optimization failed")

    final_p = best_p.copy()
    C_stored = C.copy()

    def predict(W_new):
        W_new = np.asarray(W_new, float).reshape(-1, N, M)
        E_new = W_new * C_stored[None, :, :]
        E_tot_new = E_new.sum(axis=1)
        W_mix_new = alpha * W_new[:, 0, :] + (1 - alpha) * W_new[:, 1, :] if N > 1 else W_new[:, 0, :]
        return forward_precomputed(final_p, W_mix_new, E_tot_new)

    return predict, {"n_params": n_params, "final_p": final_p.copy()}
