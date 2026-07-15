# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib"]
# ///
"""Grug holdout test, PHASE 1: frozen-model predictions for the 40 quarantined runs.

Pre-registered protocol: ``scratch/mixture_features/grug/test_protocol.md`` plus the
amendment on issue #7067 (target re-registered to ``zmacro_english_20`` before any label
opening). This script NEVER touches ``QUARANTINE_test_labels.parquet``.

Frozen models, all fit on the FULL 800 training runs with the same hyperparameter
selection procedures as the phase-2 CV code (inner 5-fold CV / GCV, seed 0):
  - PRIMARY 4_hellinger_kernel_k1000: kernel ridge on mean per-phase squared-Hellinger
    distances between K=1000 content histograms (retrodiction.kernel_cv_predict).
  - 1_weights_ridge: GCV ridge on raw per-phase bucket weights (grug_fit.ridge_gcv).
  - 2_weights_lgbm: LightGBM on raw weights, grid by inner 5-fold CV (grug_fit grid).
  - 0_train_mean: constant train-mean predictor.
  - ctrl_shuffled / ctrl_matched: the primary's estimator on shuffled-columns /
    matched-random V (10 seeds each; per-seed rows + seed-averaged row).

Target: y(r) = mean over observed t in the 20-task english list of
(bpb_t(r) - mu_t) / sd_t with (mu_t, sd_t) FROZEN in target_candidates.json.

Outputs (scratch/mixture_features/grug/):
  test_predictions.parquet   one row per (experiment_index, model); schema metadata
                             carries git SHA, target formula, frozen hyperparameters.
  frozen_model_hyperparams.json  same metadata as a standalone JSON for upload.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import logging
import subprocess

import featurize
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from grug_fit import (
    GRUG_DIR,
    RIDGE_ALPHAS,
    TRAIN_RUNS,
    load_grug_artifacts,
    per_phase_hist,
    ridge_gcv,
    weight_matrix,
)
from retrodiction import (
    KR_ALPHAS,
    KR_GAMMA_FACTORS,
    N_INNER_FOLDS,
    SEED,
    _kr_fit_predict,
    _sq_hellinger,
    kernel_cv_predict,
)
from sklearn.model_selection import GridSearchCV, KFold

logger = logging.getLogger("grug_holdout_predict")

TEST_FEATURES = GRUG_DIR / "test_runs_features_only.parquet"
TARGET_CANDIDATES = GRUG_DIR / "target_candidates.json"
MANIFEST = GRUG_DIR / "holdout_manifest.json"
OUT_PREDICTIONS = GRUG_DIR / "test_predictions.parquet"
OUT_HYPERPARAMS = GRUG_DIR / "frozen_model_hyperparams.json"

TARGET_NAME = "zmacro_english_20"
N_CONTROL_SEEDS = 10
LGBM_GRID = {"num_leaves": [7, 15, 31], "min_child_samples": [5, 10]}  # == grug_fit.LGBM_GRID
# Frozen train-CV OOF Spearman of the primary (amendment; R3 bar = this - 0.15).
PRIMARY_TRAIN_CV_SPEARMAN = 0.8180054500085157


def zmacro_english(runs: pd.DataFrame, task_list: list[str], mu: dict, sd: dict) -> np.ndarray:
    """y(r) = NaN-aware mean over observed t in task_list of (bpb_t - mu_t)/sd_t."""
    y = np.full(len(runs), np.nan)
    for i, ev_json in enumerate(runs["evals"].to_numpy()):
        ev = json.loads(ev_json)
        zs = [(ev[t]["bpb"] - mu[t]) / sd[t] for t in task_list if t in ev and "bpb" in ev[t]]
        if not zs:
            raise ValueError(f"run {i} has no observed target tasks")
        y[i] = float(np.mean(zs))
    return y


def kernel_fit_predict_capture(d2: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """retrodiction.kernel_cv_predict with the selected (gamma, alpha) captured.

    Identical grids, folds, and strict-< first-minimum tie-breaking; the returned
    predictions are asserted equal to the library function.
    """
    d_tr = d2[np.ix_(tr, tr)]
    med = float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
    gammas = np.asarray(KR_GAMMA_FACTORS) / max(med, 1e-12)
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(tr))))
    best, best_sse = None, np.inf
    for gi, g in enumerate(gammas):
        k_full = np.exp(-g * d_tr)
        for al in KR_ALPHAS:
            sse = 0.0
            for itr, iva in folds:
                p = _kr_fit_predict(k_full[np.ix_(itr, itr)], y[tr][itr], k_full[np.ix_(iva, itr)], al)
                sse += ((p - y[tr][iva]) ** 2).sum()
            if sse < best_sse:
                best, best_sse = (gi, float(al)), sse
    gi, al = best
    g = float(gammas[gi])
    pred = _kr_fit_predict(np.exp(-g * d_tr), y[tr], np.exp(-g * d2[np.ix_(te, tr)]), al)
    ref = kernel_cv_predict(d2, tr, te, y)
    np.testing.assert_allclose(pred, ref, rtol=1e-12, atol=1e-14)
    hp = {
        "gamma": g,
        "gamma_factor": float(KR_GAMMA_FACTORS[gi]),
        "alpha": al,
        "median_train_sq_hellinger": med,
        "inner_cv_sse": float(best_sse),
    }
    return pred, hp


def ridge_fit_predict_capture(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, dict]:
    """grug_fit.ridge_gcv with the selected alpha captured; asserted equal to the original."""
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean()
    yc = y_tr - ym
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    g = u.T @ yc
    n = len(y_tr)
    s2 = s**2
    null_resid = float(yc @ yc - g @ g)
    best_alpha, best_score = RIDGE_ALPHAS[0], np.inf
    for al in RIDGE_ALPHAS:
        resid = float(np.sum((al / (s2 + al) * g) ** 2)) + null_resid
        tr_a = float(np.sum(s2 / (s2 + al)))
        denom = n - tr_a
        score = n * resid / denom**2 if denom > 1e-9 else np.inf
        if score < best_score:
            best_score, best_alpha = score, al
    pred = (b @ vt.T) @ (g * s / (s2 + best_alpha)) + ym
    ref, ref_score = ridge_gcv(x_tr, y_tr, x_te)
    np.testing.assert_allclose(pred, ref, rtol=1e-12, atol=1e-14)
    assert abs(best_score - ref_score) < 1e-12
    return pred, {"alpha": float(best_alpha), "gcv_score": float(best_score)}


def lgbm_fit_predict_capture(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> tuple[np.ndarray, dict]:
    """grug_fit.predict_lgbm (same estimator, grid, inner folds), fit on the full train."""
    import lightgbm as lgb

    gs = GridSearchCV(
        lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, random_state=SEED, verbosity=-1, n_jobs=1),
        LGBM_GRID,
        cv=KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED),
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    gs.fit(x_tr, y_tr)
    hp = {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "best_grid_params": {k: int(v) for k, v in gs.best_params_.items()},
        "inner_cv_neg_rmse": float(gs.best_score_),
    }
    return gs.predict(x_te), hp


def _control_predict(
    ctrl: str, seed: int, v1000: np.ndarray, w_all: np.ndarray, tr, te, y_all
) -> tuple[np.ndarray, dict]:
    v = featurize.shuffled_columns_v(v1000, seed) if ctrl == "shuffled" else featurize.matched_random_v(v1000, seed)
    d2 = _sq_hellinger(per_phase_hist(w_all, v))
    return kernel_fit_predict_capture(d2, tr, te, y_all)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()

    cand = json.loads(TARGET_CANDIDATES.read_text())
    rec = cand["recommended_target"]
    assert rec["name"] == "e_zmacro_english"
    task_list, z_mu, z_sd = rec["task_list"], rec["train_z_mu"], rec["train_z_sd"]
    assert len(task_list) == 20

    manifest = json.loads(MANIFEST.read_text())

    hists, views, _centroids, _rff_means, _rff_order, _buckets_table = load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)

    train = pd.read_parquet(TRAIN_RUNS)
    test = pd.read_parquet(TEST_FEATURES)
    assert len(train) == 800 and len(test) == 40
    assert sorted(test["experiment_index"].tolist()) == sorted(manifest["test_experiment_indices"])
    assert not set(test["experiment_index"]) & set(train["experiment_index"])

    y_train = zmacro_english(train, task_list, z_mu, z_sd)
    # sanity: frozen z-stats must reproduce the train-run per-task stats (same 800 runs)
    evs = [json.loads(e) for e in train["evals"]]
    for t in task_list:
        vals = np.array([ev[t]["bpb"] for ev in evs if t in ev and "bpb" in ev[t]])
        np.testing.assert_allclose(vals.mean(), z_mu[t], rtol=1e-9)
        np.testing.assert_allclose(vals.std(), z_sd[t], rtol=1e-9)
    logger.info("target %s on 800 train runs: mean %.4f std %.4f", TARGET_NAME, y_train.mean(), y_train.std())

    w_train = weight_matrix(train, buckets)
    w_test = weight_matrix(test, buckets)
    w_all = np.concatenate([w_train, w_test], axis=0)
    tr = np.arange(800)
    te = np.arange(800, 840)
    y_all = np.concatenate([y_train, np.zeros(40)])  # test slots never read (y[tr] only)

    x_w_train = w_train.reshape(len(w_train), -1)
    x_w_test = w_test.reshape(len(w_test), -1)

    hyper: dict = {
        "git_sha": git_sha,
        "target": {
            "name": TARGET_NAME,
            "formula": "mean over observed t in task_list of (bpb_t - mu_t)/sd_t; frozen train stats",
            "task_list": task_list,
            "train_z_mu": z_mu,
            "train_z_sd": z_sd,
            "lower_is_better": True,
        },
        "train_cv_oof_spearman_primary": PRIMARY_TRAIN_CV_SPEARMAN,
        "r3_bar": PRIMARY_TRAIN_CV_SPEARMAN - 0.15,
        "n_train": 800,
        "n_test": 40,
        "seed": SEED,
        "models": {},
    }
    preds: dict[str, np.ndarray] = {}

    # ---- primary: Hellinger kernel ridge K=1000 per phase ----
    d2_all = _sq_hellinger(per_phase_hist(w_all, v1000))
    preds["4_hellinger_kernel_k1000"], hp = kernel_fit_predict_capture(d2_all, tr, te, y_all)
    hyper["models"]["4_hellinger_kernel_k1000"] = {
        "estimator": "kernel ridge, k=exp(-gamma*d2), d2=mean per-phase squared Hellinger, K=1000",
        "selection": (
            f"inner {N_INNER_FOLDS}-fold CV (seed {SEED}) over gamma_factors {list(KR_GAMMA_FACTORS)} x alphas logspace(-3,2,6), fit on all 800"
        ),
        **hp,
    }
    logger.info("primary frozen: %s", hp)

    # ---- baselines ----
    preds["0_train_mean"] = np.full(40, y_train.mean())
    hyper["models"]["0_train_mean"] = {"estimator": "constant train mean", "value": float(y_train.mean())}

    preds["1_weights_ridge"], hp = ridge_fit_predict_capture(x_w_train, y_train, x_w_test)
    hyper["models"]["1_weights_ridge"] = {
        "estimator": "ridge on standardized per-phase raw bucket weights (336 dims)",
        "selection": "closed-form GCV over alphas logspace(-3,3,25), fit on all 800",
        **hp,
    }

    preds["2_weights_lgbm"], hp = lgbm_fit_predict_capture(x_w_train, y_train, x_w_test)
    hyper["models"]["2_weights_lgbm"] = {
        "estimator": "LGBMRegressor on per-phase raw bucket weights",
        "selection": f"inner {N_INNER_FOLDS}-fold CV (seed {SEED}) over grid {LGBM_GRID}, fit on all 800",
        **hp,
    }

    # ---- controls (primary's estimator on control featurizations, 10 seeds each) ----
    arms = [(c, s) for c in ("shuffled", "matched") for s in range(N_CONTROL_SEEDS)]
    out = joblib.Parallel(n_jobs=min(len(arms), joblib.cpu_count()))(
        joblib.delayed(_control_predict)(c, s, v1000, w_all, tr, te, y_all) for c, s in arms
    )
    ctrl_preds: dict[str, list[np.ndarray]] = {"shuffled": [], "matched": []}
    for (c, s), (p, hp) in zip(arms, out, strict=True):
        ctrl_preds[c].append(p)
        preds[f"ctrl_{c}_s{s}"] = p
        hyper["models"][f"ctrl_{c}_s{s}"] = {
            "estimator": f"primary kernel ridge on {c} control V (seed {s})",
            **hp,
        }
    for c in ("shuffled", "matched"):
        preds[f"ctrl_{c}_mean10"] = np.mean(np.stack(ctrl_preds[c]), axis=0)
        hyper["models"][f"ctrl_{c}_mean10"] = {"estimator": f"mean prediction over the {N_CONTROL_SEEDS} ctrl_{c} seeds"}
    logger.info("controls done (%d arms)", len(arms))

    # ---- write predictions ----
    exp_idx = test["experiment_index"].to_numpy()
    rows = [
        {"experiment_index": int(ei), "model": m, "prediction": float(p[i])}
        for m, p in sorted(preds.items())
        for i, ei in enumerate(exp_idx)
    ]
    tbl = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    tbl = tbl.replace_schema_metadata(
        {**(tbl.schema.metadata or {}), b"grug_holdout_phase1": json.dumps(hyper).encode()}
    )
    pq.write_table(tbl, OUT_PREDICTIONS)
    OUT_HYPERPARAMS.write_text(json.dumps(hyper, indent=2))

    sha = hashlib.sha256(OUT_PREDICTIONS.read_bytes()).hexdigest()
    print(f"\nsha256(test_predictions.parquet) = {sha}")
    print(f"rows: {len(rows)} ({len(preds)} models x 40 runs)\n")
    print(f"{'model':32s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s}")
    for m in sorted(preds):
        p = preds[m]
        print(f"{m:32s} {p.mean():9.4f} {p.std():9.4f} {p.min():9.4f} {p.max():9.4f}")


if __name__ == "__main__":
    main()
