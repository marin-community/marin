# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable DS-RE-CEQ evaluation helpers for 2-phase StarCoder."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.scaling_models import fit_dsre_ceq
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    STARCODER_TARGET,
)

STARCODER_CSV = "experiments/domain_phase_mix/exploratory/two_phase_starcoder_combined.csv"


def load_two_phase_starcoder_xy() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load completed 2-phase StarCoder rows as the 2D DS-RE-CEQ design."""
    frame = pd.read_csv(STARCODER_CSV)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "completed"].reset_index(drop=True)
    x = frame[["phase_0_starcoder", "phase_1_starcoder"]].to_numpy(dtype=float)
    y = frame[STARCODER_TARGET].to_numpy(dtype=float)
    return frame, x, y


def _regression_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residuals = y_pred - y_true
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "r2": float(1.0 - ss_res / ss_tot),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def compute_starcoder_dsre_ceq_metrics(
    *,
    cv_seed: int = 0,
    old_cv_seed: int = 42,
    fit_seed: int = 0,
) -> dict[str, Any]:
    """Return live DS-RE-CEQ train and CV metrics on 2-phase StarCoder.

    The returned ``cv_*`` fields use pooled out-of-fold predictions, matching
    the current GRP table convention. ``old_cv_*`` records the older March 2
    mean-per-fold convention for reference.
    """
    _frame, x, y = load_two_phase_starcoder_xy()

    predict_fn, params = fit_dsre_ceq(x, y, seed=fit_seed)
    train_pred = predict_fn(x)
    train = _regression_summary(y, train_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=cv_seed)
    oof = np.zeros_like(y, dtype=float)
    fold_regrets: list[float] = []
    for fold_idx, (tr, te) in enumerate(kf.split(x)):
        fold_predict_fn, _ = fit_dsre_ceq(x[tr], y[tr], seed=fold_idx)
        pred = fold_predict_fn(x[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][np.argmin(pred)] - np.min(y[te])))

    cv = _regression_summary(y, oof)
    cv_regret_at_1 = float(y[np.argmin(oof)] - np.min(y))

    old_kf = KFold(n_splits=5, shuffle=True, random_state=old_cv_seed)
    old_r2s: list[float] = []
    old_rmses: list[float] = []
    old_spearmans: list[float] = []
    old_regrets: list[float] = []
    for fold_idx, (tr, te) in enumerate(old_kf.split(x)):
        fold_predict_fn, _ = fit_dsre_ceq(x[tr], y[tr], seed=fold_idx)
        pred = fold_predict_fn(x[te])
        fold_summary = _regression_summary(y[te], pred)
        old_r2s.append(float(fold_summary["r2"]))
        old_rmses.append(float(fold_summary["rmse"]))
        old_spearmans.append(float(fold_summary["spearman"]))
        old_regrets.append(float(y[te][np.argmin(pred)] - np.min(y[te])))

    return {
        "model": "DS-RE-CEQ",
        "dataset": "two_phase_starcoder",
        "status": "ok",
        "n_runs": len(y),
        "n_params": len(params),
        "train_r2": float(train["r2"]),
        "train_rmse": float(train["rmse"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(y[np.argmin(train_pred)] - np.min(y)),
        "cv_r2": float(cv["r2"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": cv_regret_at_1,
        "cv_foldmean_regret_at_1": float(np.mean(fold_regrets)),
        "params": params.tolist(),
        "old_cv_r2_meanfold": float(np.mean(old_r2s)),
        "old_cv_rmse_meanfold": float(np.mean(old_rmses)),
        "old_cv_spearman_meanfold": float(np.mean(old_spearmans)),
        "old_cv_regret_meanfold": float(np.mean(old_regrets)),
    }
