# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gaussian-process (Bayesian) formulation of the mixture surrogate.

The GP posterior *mean* is mathematically identical to the kernel-ridge prediction the
campaign settled on -- the ridge ``alpha`` is exactly the observation-noise variance
``sigma_n^2``. What the GP adds is a posterior *variance*, i.e. calibrated error bars on
every prediction, plus evidence-based hyperparameter selection and a basis for
uncertainty-aware mixture proposals (expected improvement).

Model
-----
    f ~ GP(mean = ybar, cov = sigma_f^2 * exp(-gamma * d2_hellinger))
    y = f + eps,   eps ~ N(0, sigma_n^2)

    posterior mean  mu*   = ybar + k*' (K + sigma_n^2 I)^-1 (y - ybar)     <- == KRR
    posterior var   var*  = sigma_f^2 - k*' (K + sigma_n^2 I)^-1 k*        <- new
    predictive var        = var* + sigma_n^2                               (for an observed y)

Note KRR only identifies the ratio sigma_n^2 / sigma_f^2, so it fixes the mean but leaves
the variance *scale* free. Fitting sigma_f^2 and sigma_n^2 separately by marginal
likelihood is what makes the error bars calibrated rather than arbitrary.
"""

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

JITTER = 1e-10


def _build_chol(d2: np.ndarray, theta: np.ndarray):
    """Cholesky of (sigma_f^2 * exp(-gamma d2) + sigma_n^2 I). theta = log[sf2, sn2, gamma]."""
    sf2, sn2, gamma = np.exp(theta)
    k = sf2 * np.exp(-gamma * d2)
    k[np.diag_indices_from(k)] += sn2 + JITTER
    return cholesky(k, lower=True), sf2, sn2, gamma


def neg_log_marginal_likelihood(theta: np.ndarray, d2: np.ndarray, y_centered: np.ndarray) -> float:
    """-log p(y | X, theta). Minimizing this is Bayesian model selection (the 'evidence')."""
    try:
        chol, *_ = _build_chol(d2, theta)
    except np.linalg.LinAlgError:
        return 1e12
    a = cho_solve((chol, True), y_centered)
    n = y_centered.size
    return float(0.5 * y_centered @ a + np.log(np.diag(chol)).sum() + 0.5 * n * np.log(2 * np.pi))


def fit_gp(d2: np.ndarray, y: np.ndarray) -> dict:
    """Fit (sigma_f^2, sigma_n^2, gamma) by maximizing the log marginal likelihood."""
    ybar = float(y.mean())
    yc = y - ybar
    med = float(np.median(d2[np.triu_indices(len(y), 1)]))
    # start from the campaign's frozen kernel settings: gamma = 0.25/median, noise ~ 10% of signal
    theta0 = np.log([max(yc.var(), 1e-6), max(0.1 * yc.var(), 1e-8), 0.25 / max(med, 1e-12)])
    res = minimize(
        neg_log_marginal_likelihood, theta0, args=(d2, yc), method="L-BFGS-B", bounds=[(np.log(1e-6), np.log(1e3))] * 3
    )
    chol, sf2, sn2, gamma = _build_chol(d2, res.x)
    return {
        "chol": chol,
        "alpha_dual": cho_solve((chol, True), yc),
        "ybar": ybar,
        "sigma_f2": float(sf2),
        "sigma_n2": float(sn2),
        "gamma": float(gamma),
        "nlml": float(res.fun),
        "median_d2": med,
        "ridge_equivalent_alpha": float(sn2 / sf2),  # the KRR alpha this corresponds to
    }


def condition_gp(d2: np.ndarray, y: np.ndarray, sigma_f2: float, sigma_n2: float, gamma: float) -> dict:
    """Condition the GP on data with FIXED hyperparameters.

    Sequential design re-conditions on every newly acquired point but only needs to re-optimize
    the hyperparameters occasionally; this is the cheap half (one Cholesky, no optimization).
    """
    ybar = float(y.mean())
    yc = y - ybar
    chol, sf2, sn2, g = _build_chol(d2, np.log([sigma_f2, sigma_n2, gamma]))
    return {
        "chol": chol,
        "alpha_dual": cho_solve((chol, True), yc),
        "ybar": ybar,
        "sigma_f2": float(sf2),
        "sigma_n2": float(sn2),
        "gamma": float(g),
    }


def predict_gp(fit: dict, d2_star: np.ndarray, include_noise: bool = True):
    """Posterior mean and standard deviation at new points. d2_star is (n_star, n_train)."""
    k_star = fit["sigma_f2"] * np.exp(-fit["gamma"] * d2_star)
    mu = k_star @ fit["alpha_dual"] + fit["ybar"]
    v = solve_triangular(fit["chol"], k_star.T, lower=True)
    var = fit["sigma_f2"] - np.einsum("ij,ij->j", v, v)
    if include_noise:
        var = var + fit["sigma_n2"]
    return mu, np.sqrt(np.clip(var, 1e-12, None))


def krr_predict(d2_tr: np.ndarray, d2_star: np.ndarray, y: np.ndarray, gamma: float, alpha: float):
    """Plain kernel-ridge prediction, for the equivalence check."""
    ybar = y.mean()
    k = np.exp(-gamma * d2_tr)
    k[np.diag_indices_from(k)] += alpha
    dual = np.linalg.solve(k, y - ybar)
    return np.exp(-gamma * d2_star) @ dual + ybar
