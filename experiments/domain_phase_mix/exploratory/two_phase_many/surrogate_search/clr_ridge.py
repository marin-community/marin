# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "scikit-learn"]
# ///
"""CLR-Ridge: Centered Log-Ratio Ridge model for compositional data mixture prediction.

Mathematical form:
    W_mix = alpha * W_phase0 + (1 - alpha) * W_phase1
    X = CLR(W_mix) = log(W_mix) - mean(log(W_mix))       [compositional centering]
    X_reduced = PCA_train(X)                               [project to M-1 dimensions]
    y_hat = c0 + X_reduced @ beta                          [Ridge-regularized prediction]

Parameters:
    alpha: phase weight in (0, 1), controls importance of phase 0 vs phase 1
    c0: intercept (scalar)
    beta: coefficient vector (M-1 dimensional after PCA)
    lambda: Ridge regularization strength (tuned via leave-one-out CV)

Key properties:
    - Works on the simplex: CLR transform properly handles compositional data
    - Phase weighting: alpha < 0.5 means phase 1 (annealing) matters more
    - Ridge regularization automatically shrinks irrelevant domain coefficients
    - PCA(M-1) removes the singular direction from CLR space
    - Total effective parameters: ~M (determined by Ridge effective degrees of freedom)

For the two-phase many-domain MMLU prediction task:
    - alpha ≈ 0.06 optimizes R² and Spearman correlation
    - alpha ≈ 0.14 optimizes Regret@1 (achieves < 0.05)
    - Alpha trade-off: lower alpha → better overall prediction, higher alpha → better tail selection
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

EPS = 1e-10
RIDGE_ALPHAS = np.logspace(-3, 5, 50)


@dataclass(frozen=True)
class CLRRidgeModel:
    """Fitted CLR-Ridge model."""

    alpha: float
    pca: PCA
    scaler: StandardScaler
    ridge: RidgeCV
    n_phases: int
    n_domains: int

    @property
    def n_params(self) -> int:
        """Effective parameter count: PCA components + intercept."""
        return self.pca.n_components_ + 1

    def predict(self, W: np.ndarray) -> np.ndarray:
        """Predict from weight array of shape (R, N, M) or (R, N*M)."""
        W = np.asarray(W, dtype=float)
        if W.ndim == 2:
            W = W.reshape(-1, self.n_phases, self.n_domains)
        if W.ndim != 3:
            raise ValueError(f"Expected 3D weight array, got {W.ndim}D")
        W0 = W[:, 0, :]
        W1 = W[:, 1, :] if self.n_phases > 1 else W[:, 0, :]
        X_clr = _clr(self.alpha * W0 + (1 - self.alpha) * W1)
        X_pca = self.pca.transform(X_clr)
        X_scaled = self.scaler.transform(X_pca)
        return self.ridge.predict(X_scaled)


def _clr(W: np.ndarray) -> np.ndarray:
    """Centered log-ratio transform for compositional data."""
    log_W = np.log(W + EPS)
    return log_W - log_W.mean(axis=1, keepdims=True)


def fit_clr_ridge(
    spec,
    *,
    alpha: float = 0.14,
    n_components: int | None = None,
    seed: int = 0,
    **kwargs,
) -> tuple:
    """Fit CLR-Ridge model to a DatasetSpec.

    Args:
        spec: DatasetSpec with weights (R, N, M) and y (R,).
        alpha: Phase weighting. Lower alpha → more emphasis on phase 1.
            - Use alpha ≈ 0.06 for best R²/Spearman.
            - Use alpha ≈ 0.14 for best Regret@1.
        n_components: PCA components. Default M-1 (removes singular CLR direction).
        seed: Random seed (unused, for interface compatibility).

    Returns:
        (predict_fn, info_dict) matching the GeneralModelSpec interface.
    """
    W = spec.weights
    y = spec.y
    R, N, M = W.shape

    if n_components is None:
        n_components = M - 1

    W0 = W[:, 0, :]
    W1 = W[:, 1, :] if N > 1 else W[:, 0, :]

    X_clr = _clr(alpha * W0 + (1 - alpha) * W1)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_clr)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)
    ridge = RidgeCV(alphas=RIDGE_ALPHAS)
    ridge.fit(X_scaled, y)

    model = CLRRidgeModel(
        alpha=alpha,
        pca=pca,
        scaler=scaler,
        ridge=ridge,
        n_phases=N,
        n_domains=M,
    )

    def predict_fn(W_new: np.ndarray) -> np.ndarray:
        return model.predict(W_new)

    info = {
        "n_params": model.n_params,
        "alpha": alpha,
        "ridge_alpha": float(ridge.alpha_),
        "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        "model": model,
    }
    return predict_fn, info


def fit_clr_ridge_r2(spec, **kwargs):
    """CLR-Ridge optimized for R²/Spearman (alpha=0.06)."""
    return fit_clr_ridge(spec, alpha=0.06, **kwargs)


def fit_clr_ridge_regret(spec, **kwargs):
    """CLR-Ridge optimized for Regret@1 (alpha=0.14)."""
    return fit_clr_ridge(spec, alpha=0.14, **kwargs)


def fit_clr_ridge_balanced(spec, **kwargs):
    """CLR-Ridge balanced trade-off (alpha=0.10)."""
    return fit_clr_ridge(spec, alpha=0.10, **kwargs)
