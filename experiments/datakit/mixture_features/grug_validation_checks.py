# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "joblib"]
# ///
"""Adversarial-review validation checks for the grug surrogate campaign (train-only).

Three analyses prescribed by the methodology review (logbook 2026-07-15/16). Uses ONLY
the 800 training runs (QUARANTINE_test_labels.parquet stays closed); the realized holdout
number 0.7205 enters only as a published constant to compare against.

  1  Epoch-augmentation architecture bake-off on zmacro_english_20: SUM kernel
     (alpha*K_hell + (1-alpha)*K_hinge), concat-ARD single RBF with a hinge-block
     bandwidth scale, and a two-head additive model (kernel + ridge on the kernel's
     inner-OOF training residuals). Same 15 folds as every phase-2 comparison; paired
     per-fold deltas vs the plain Hellinger kernel (reference 0.8147).
  2  Selection-corrected optimism bootstrap over the declared target-candidate set:
     re-run target selection (best 5-fold CV kernel Spearman) inside each bootstrap
     replicate; optimism = in-replicate CV score minus out-of-bag score of the chosen
     target. Kernel hyperparameters are FROZEN per candidate from the full-train fit
     (stated simplification for compute).
  3  Noise/SNR proxies from existing data: (a) near-duplicate mixture pairs in the
     800-run weight matrix as a repeat-noise anchor; (b) per-task kernel-OOF residual
     covariance decomposition (shared run-level factor vs task-idiosyncratic) and the
     implied reliability bound for zmacro_english_20. Both are inferior to seed repeats.

Metric convention: per-fold Spearman, mean over the 15 folds (matches the reference
numbers kernel 0.8147 / hist-ridge 0.7396 / hist+hinge 0.7701 / product kernel 0.7673).

Outputs: scratch/mixture_features/grug/validation_checks.{json,md}.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import featurize  # noqa: E402
import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from grug_fit import (  # noqa: E402
    GRUG_DIR,
    TRAIN_RUNS,
    flat,
    hinge_epoch_features,
    json_default,
    load_grug_artifacts,
    per_phase_hist,
    ridge_gcv,
    stage_b_epochs,
    weight_matrix,
)
from grug_target_analysis import (  # noqa: E402
    build_task_matrix,
    nan_macro,
    pc_scores,
    run_per_task_cv,
    task_family,
    z_macro,
)
from retrodiction import (  # noqa: E402
    KR_ALPHAS,
    KR_GAMMA_FACTORS,
    N_INNER_FOLDS,
    SEED,
    _kr_fit_predict,
    _sq_hellinger,
    kernel_cv_predict,
    spearman_cols,
)
from scipy.spatial.distance import pdist, squareform  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.model_selection import KFold, RepeatedKFold  # noqa: E402

logger = logging.getLogger("grug_validation_checks")

N_SPLITS, N_REPEATS = 5, 3
TARGET_NAME = "zmacro_english_20"
REF_KERNEL = 0.8147  # per-fold mean, frozen primary on these folds
REF_HOLDOUT = 0.7205  # realized one-shot holdout Spearman (published, labels not re-opened)
N_HOLDOUT = 40

# Analysis 1 grids
SUM_ALPHA_GRID = (0.5, 0.7, 0.9, 0.95)
SUM_GAMMA_FACTORS = (0.5, 1.0, 2.0)  # x 1/median(train sq-dist), same convention as predict_additive_kernel
ARD_SCALE_GRID = (0.1, 0.3, 1.0, 3.0)  # hinge-block bandwidth scale s (blocks median-normalized on train)

# Analysis 2
N_BOOT = 200
BOOT_SEED0 = 1000
CANDIDATE_NAMES = ("macro_bpb", "english_macro", "zmacro_all", "zmacro_english_20", "pc1", "pc2")

# Analysis 3
TV_DUP_THRESHOLD = 0.05
TV_EXACT_EPS = 1e-9
N_CLOSEST_PAIRS = 20


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def build_zmacro_target(runs: pd.DataFrame, rec: dict) -> np.ndarray:
    """zmacro_english_20 with the FROZEN train stats from target_candidates.json."""
    task_list = rec["task_list"]
    mu, sd = rec["train_z_mu"], rec["train_z_sd"]
    y = np.empty(len(runs))
    for i, ev_json in enumerate(runs["evals"]):
        ev = json.loads(ev_json)
        zs = [(ev[t]["bpb"] - mu[t]) / sd[t] for t in task_list if t in ev and "bpb" in ev[t]]
        y[i] = float(np.mean(zs))
    return y


def sq_euclid_std(x: np.ndarray) -> np.ndarray:
    """Squared-Euclidean distances over full-data-standardized features (as _sq_euclid_qual)."""
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (x - mu) / sd
    g = z @ z.T
    sq = np.diag(g)
    return np.clip(sq[:, None] + sq[None, :] - 2 * g, 0.0, None)


def _offdiag_median(d: np.ndarray) -> float:
    return float(np.median(d[~np.eye(len(d), dtype=bool)])) or 1e-12


def per_fold_spearman(pred: np.ndarray, y_te: np.ndarray) -> float:
    return float(spearman_cols(pred[:, None], y_te)[0])


# ---------------------------------------------------------------------------
# Analysis 1: epoch-augmentation architecture bake-off
# ---------------------------------------------------------------------------


def _inner_folds(n: int) -> list:
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    return list(kf.split(np.arange(n)))


def _sse_over_folds(k_full: np.ndarray, y_tr: np.ndarray, folds: list, alpha: float) -> float:
    sse = 0.0
    for itr, iva in folds:
        p = _kr_fit_predict(k_full[np.ix_(itr, itr)], y_tr[itr], k_full[np.ix_(iva, itr)], alpha)
        sse += ((p - y_tr[iva]) ** 2).sum()
    return sse


def predict_sum_kernel(d2h: np.ndarray, d2q: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """K = a*exp(-gh d2_hell) + (1-a)*exp(-gq d2_hinge); (a, gh, gq, ridge alpha) by inner CV."""
    tr, te = np.asarray(tr), np.asarray(te)
    dh_tr, dq_tr = d2h[np.ix_(tr, tr)], d2q[np.ix_(tr, tr)]
    med_h, med_q = _offdiag_median(dh_tr), _offdiag_median(dq_tr)
    folds = _inner_folds(len(tr))
    y_tr = y[tr]
    best, best_sse = None, np.inf
    for gh in np.asarray(SUM_GAMMA_FACTORS) / med_h:
        kh = np.exp(-gh * dh_tr)
        for gq in np.asarray(SUM_GAMMA_FACTORS) / med_q:
            kq = np.exp(-gq * dq_tr)
            for a in SUM_ALPHA_GRID:
                k_full = a * kh + (1.0 - a) * kq
                for al in KR_ALPHAS:
                    sse = _sse_over_folds(k_full, y_tr, folds, al)
                    if sse < best_sse:
                        best, best_sse = (gh, gq, a, al), sse
    gh, gq, a, al = best
    k_tr = a * np.exp(-gh * dh_tr) + (1.0 - a) * np.exp(-gq * dq_tr)
    k_te = a * np.exp(-gh * d2h[np.ix_(te, tr)]) + (1.0 - a) * np.exp(-gq * d2q[np.ix_(te, tr)])
    return _kr_fit_predict(k_tr, y_tr, k_te, al)


def predict_concat_ard(d2h4: np.ndarray, d2q: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """Single RBF over [sqrt per-phase hist | s * standardized hinge]: d2 = d2h/med + s^2 d2q/med.

    d2h4 is the squared-Euclidean distance of the concatenated sqrt-hists (= 4 x mean
    Hellinger^2); block medians computed on the train fold; (s, gamma, alpha) by inner CV.
    """
    tr, te = np.asarray(tr), np.asarray(te)
    dh_tr, dq_tr = d2h4[np.ix_(tr, tr)], d2q[np.ix_(tr, tr)]
    med_h, med_q = _offdiag_median(dh_tr), _offdiag_median(dq_tr)
    folds = _inner_folds(len(tr))
    y_tr = y[tr]
    best, best_sse = None, np.inf
    for s in ARD_SCALE_GRID:
        d_tr = dh_tr / med_h + (s * s) * dq_tr / med_q
        med_s = _offdiag_median(d_tr)
        for g in np.asarray(KR_GAMMA_FACTORS) / med_s:
            k_full = np.exp(-g * d_tr)
            for al in KR_ALPHAS:
                sse = _sse_over_folds(k_full, y_tr, folds, al)
                if sse < best_sse:
                    best, best_sse = (s, g, al), sse
    s, g, al = best
    d_tr = dh_tr / med_h + (s * s) * dq_tr / med_q
    d_te = d2h4[np.ix_(te, tr)] / med_h + (s * s) * d2q[np.ix_(te, tr)] / med_q
    return _kr_fit_predict(np.exp(-g * d_tr), y_tr, np.exp(-g * d_te), al)


def predict_two_head(d2h: np.ndarray, xq: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """Kernel ridge on Hellinger + ridge head on hinge features fit on inner-OOF residuals.

    Stage 1 selects (gamma, alpha) exactly as kernel_cv_predict; stage 2 computes the
    kernel's inner-OOF predictions on the training fold with the selected pair (no test
    contact), fits ridge_gcv(hinge -> residual) and adds it to the kernel test prediction.
    """
    tr, te = np.asarray(tr), np.asarray(te)
    d_tr = d2h[np.ix_(tr, tr)]
    med = _offdiag_median(d_tr)
    gammas = np.asarray(KR_GAMMA_FACTORS) / med
    folds = _inner_folds(len(tr))
    y_tr = y[tr]
    best, best_sse = None, np.inf
    for g in gammas:
        k_full = np.exp(-g * d_tr)
        for al in KR_ALPHAS:
            sse = _sse_over_folds(k_full, y_tr, folds, al)
            if sse < best_sse:
                best, best_sse = (g, al), sse
    g, al = best
    k_tr = np.exp(-g * d_tr)
    oof_tr = np.empty(len(tr))
    for itr, iva in folds:
        oof_tr[iva] = _kr_fit_predict(k_tr[np.ix_(itr, itr)], y_tr[itr], k_tr[np.ix_(iva, itr)], al)
    resid = y_tr - oof_tr
    pred_kernel_te = _kr_fit_predict(k_tr, y_tr, np.exp(-g * d2h[np.ix_(te, tr)]), al)
    pred_resid_te, _ = ridge_gcv(xq[tr], resid, xq[te])
    return pred_kernel_te + pred_resid_te


A1_VARIANTS = ("plain_kernel", "sum_kernel", "concat_ard", "two_head")


def _a1_fold(fold_id: int, tr, te, d2h, d2h4, d2q, xq, y) -> dict:
    t0 = time.monotonic()
    preds = {
        "plain_kernel": kernel_cv_predict(d2h, np.asarray(tr), np.asarray(te), y),
        "sum_kernel": predict_sum_kernel(d2h, d2q, y, tr, te),
        "concat_ard": predict_concat_ard(d2h4, d2q, y, tr, te),
        "two_head": predict_two_head(d2h, xq, y, tr, te),
    }
    logger.info("A1 fold %d done %.1fs", fold_id + 1, time.monotonic() - t0)
    return {"fold_id": fold_id, "te": np.asarray(te), "preds": preds}


def analysis_1(d2h, d2h4, d2q, xq, y, folds, n_jobs) -> dict:
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_a1_fold)(fid, tr, te, d2h, d2h4, d2q, xq, y) for fid, (tr, te) in enumerate(folds)
    )
    out.sort(key=lambda d: d["fold_id"])
    per_fold = {v: np.array([per_fold_spearman(o["preds"][v], y[o["te"]]) for o in out]) for v in A1_VARIANTS}
    res = {"per_fold_spearman": {v: per_fold[v].tolist() for v in A1_VARIANTS}, "summary": {}, "deltas_vs_plain": {}}
    for v in A1_VARIANTS:
        res["summary"][v] = {"mean": float(per_fold[v].mean()), "std": float(per_fold[v].std())}
    for v in A1_VARIANTS[1:]:
        d = per_fold[v] - per_fold["plain_kernel"]
        res["deltas_vs_plain"][v] = {
            "d_mean": float(d.mean()),
            "d_std": float(d.std()),
            "wins": int((d > 0).sum()),
            "wilcoxon_p": float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0,
        }
    plain_mean = res["summary"]["plain_kernel"]["mean"]
    if abs(plain_mean - REF_KERNEL) > 0.002:
        logger.warning("plain kernel per-fold mean %.4f differs from reference %.4f", plain_mean, REF_KERNEL)
    return res


# ---------------------------------------------------------------------------
# Analysis 2: selection-corrected optimism bootstrap
# ---------------------------------------------------------------------------


def select_kernel_hyperparams(d2: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(gamma, alpha) by the standard inner-CV rule (kernel_cv_predict) on the FULL train."""
    med = _offdiag_median(d2)
    gammas = np.asarray(KR_GAMMA_FACTORS) / med
    folds = _inner_folds(len(y))
    best, best_sse = None, np.inf
    for g in gammas:
        k_full = np.exp(-g * d2)
        for al in KR_ALPHAS:
            sse = _sse_over_folds(k_full, y, folds, al)
            if sse < best_sse:
                best, best_sse = (float(g), float(al)), sse
    return best


def candidate_targets(ymat, macro, eng_cols, all_cols, fit_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Six declared candidate targets over ALL runs, sample stats fit on fit_rows (multiset)."""
    mu = np.nanmean(ymat[fit_rows], axis=0)
    sd = np.nanstd(ymat[fit_rows], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return {
        "macro_bpb": macro,
        "english_macro": nan_macro(ymat, eng_cols),
        "zmacro_all": z_macro(ymat, all_cols, mu, sd),
        "zmacro_english_20": z_macro(ymat, eng_cols, mu, sd),
        "pc1": pc_scores(ymat, fit_rows, macro, 0),
        "pc2": pc_scores(ymat, fit_rows, macro, 1),
    }


def _boot_replicate(b: int, k_by_cand: dict, hyper: dict, ymat, macro, eng_cols, all_cols, n: int) -> dict:
    rng = np.random.default_rng(BOOT_SEED0 + b)
    draws = rng.integers(0, n, n)
    uniq = np.unique(draws)
    oob = np.setdiff1d(np.arange(n), uniq)
    kf = KFold(N_SPLITS, shuffle=True, random_state=b)
    cv_scores = {c: [] for c in CANDIDATE_NAMES}
    # folds over UNIQUE in-sample runs (a run's duplicate draws stay on the train side of
    # its fold, so bootstrap duplicates never leak across the CV split)
    for tr_u, va_u in kf.split(uniq):
        tr_ids, va_ids = uniq[tr_u], uniq[va_u]
        fit_rows = draws[np.isin(draws, tr_ids)]
        targets = candidate_targets(ymat, macro, eng_cols, all_cols, fit_rows)
        for c in CANDIDATE_NAMES:
            yc = targets[c]
            k = k_by_cand[c]
            al = hyper[c][1]
            p = _kr_fit_predict(k[np.ix_(fit_rows, fit_rows)], yc[fit_rows], k[np.ix_(va_ids, fit_rows)], al)
            cv_scores[c].append(per_fold_spearman(p, yc[va_ids]))
    cv_mean = {c: float(np.mean(cv_scores[c])) for c in CANDIDATE_NAMES}
    chosen = max(cv_mean, key=cv_mean.get)
    targets_full = candidate_targets(ymat, macro, eng_cols, all_cols, draws)
    yc = targets_full[chosen]
    k = k_by_cand[chosen]
    p_oob = _kr_fit_predict(k[np.ix_(draws, draws)], yc[draws], k[np.ix_(oob, draws)], hyper[chosen][1])
    oob_score = per_fold_spearman(p_oob, yc[oob])
    return {"b": b, "chosen": chosen, "cv_score": cv_mean[chosen], "oob_score": oob_score, "n_oob": len(oob)}


def analysis_2(d2h, ymat, macro, eng_cols, all_cols, n_jobs) -> dict:
    n = len(macro)
    full_targets = candidate_targets(ymat, macro, eng_cols, all_cols, np.arange(n))
    hyper = {}
    for c in CANDIDATE_NAMES:
        hyper[c] = select_kernel_hyperparams(d2h, full_targets[c])
        logger.info("A2 hyperparams %s: gamma=%.4g alpha=%.4g", c, *hyper[c])
    k_by_gamma = {g: np.exp(-g * d2h) for g in sorted({hyper[c][0] for c in CANDIDATE_NAMES})}
    k_by_cand = {c: k_by_gamma[hyper[c][0]] for c in CANDIDATE_NAMES}

    reps = joblib.Parallel(n_jobs=n_jobs, backend="threading")(
        joblib.delayed(_boot_replicate)(b, k_by_cand, hyper, ymat, macro, eng_cols, all_cols, n) for b in range(N_BOOT)
    )
    gaps = np.array([r["cv_score"] - r["oob_score"] for r in reps])
    oobs = np.array([r["oob_score"] for r in reps])
    cvs = np.array([r["cv_score"] for r in reps])
    chosen_counts = pd.Series([r["chosen"] for r in reps]).value_counts().to_dict()
    optimism, opt_se = float(gaps.mean()), float(gaps.std(ddof=1) / np.sqrt(N_BOOT))
    debiased = REF_KERNEL - optimism
    # realized-holdout consistency: Fisher-z SE of a Spearman on n=40 plus the replicate
    # OOB spread as an estimate of external-sample variability
    se_holdout = float(np.sqrt(1.06 / (N_HOLDOUT - 3)))  # Fieller/Fisher approx for Spearman
    z_gap = (REF_HOLDOUT - debiased) / se_holdout
    return {
        "n_bootstrap": N_BOOT,
        "kernel_hyperparams_note": (
            "kernel (gamma, alpha) FROZEN per candidate from the full-train "
            "inner-CV fit; only the dual weights are refit inside replicates (compute cut, stated)"
        ),
        "cv_convention": (
            "in-replicate score = mean of 5 per-fold Spearmans; folds over unique "
            "in-sample runs so duplicates never straddle the split; sample-dependent target stats "
            "(z mu/sd, PC loadings) refit on each fold's train rows"
        ),
        "frozen_hyperparams": {c: {"gamma": hyper[c][0], "alpha": hyper[c][1]} for c in CANDIDATE_NAMES},
        "chosen_target_counts": chosen_counts,
        "chosen_frac_zmacro_english_20": float(chosen_counts.get("zmacro_english_20", 0) / N_BOOT),
        "cv_score_mean": float(cvs.mean()),
        "oob_score_mean": float(oobs.mean()),
        "oob_score_std": float(oobs.std(ddof=1)),
        "optimism_mean": optimism,
        "optimism_se": opt_se,
        "optimism_ci95": [optimism - 1.96 * opt_se, optimism + 1.96 * opt_se],
        "debiased_expected_holdout": debiased,
        "realized_holdout": REF_HOLDOUT,
        "holdout_spearman_se_n40": se_holdout,
        "realized_minus_debiased": REF_HOLDOUT - debiased,
        "z_realized_vs_debiased": float(z_gap),
        "replicates": reps,
    }


# ---------------------------------------------------------------------------
# Analysis 3: noise / SNR proxies
# ---------------------------------------------------------------------------


def analysis_3a(w: np.ndarray, y: np.ndarray, macro: np.ndarray) -> dict:
    """Near-duplicate mixture pairs in the 800-run weight matrix (TV per phase)."""
    tv0 = 0.5 * squareform(pdist(w[:, 0, :], "cityblock"))
    tv1 = 0.5 * squareform(pdist(w[:, 1, :], "cityblock"))
    tv_max = np.maximum(tv0, tv1)
    iu = np.triu_indices(len(y), k=1)
    vals = tv_max[iu]
    dup_mask = vals < TV_DUP_THRESHOLD
    exact_mask = vals < TV_EXACT_EPS
    order = np.argsort(vals)[:N_CLOSEST_PAIRS]
    closest = [
        {
            "i": int(iu[0][o]),
            "j": int(iu[1][o]),
            "tv_max": float(vals[o]),
            "abs_dy_zmacro": float(abs(y[iu[0][o]] - y[iu[1][o]])),
            "abs_dy_macro_bpb": float(abs(macro[iu[0][o]] - macro[iu[1][o]])),
        }
        for o in order
    ]
    out = {
        "tv_threshold": TV_DUP_THRESHOLD,
        "n_pairs_below_threshold": int(dup_mask.sum()),
        "n_exact_duplicates": int(exact_mask.sum()),
        "min_tv_max": float(vals.min()),
        "tv_quantiles": {q: float(np.quantile(vals, q)) for q in (0.0, 0.001, 0.01, 0.05, 0.5)},
        "closest_pairs": closest,
        "y_std_zmacro": float(y.std()),
    }
    if dup_mask.any():
        dy = np.abs(y[iu[0][dup_mask]] - y[iu[1][dup_mask]])
        out["implied_noise_sd_zmacro"] = float(np.sqrt(np.mean(dy**2) / 2.0))
        out["dup_pair_abs_dy_median"] = float(np.median(dy))
    return out


def analysis_3b(d2h, h1000, ymat, tasks, rec, y, folds, kernel_oof_y: np.ndarray, n_jobs) -> dict:
    """Per-task kernel-OOF residual covariance decomposition for the 20 target tasks."""
    task_list = rec["task_list"]
    cols = [tasks.index(t) for t in task_list]
    ymat20 = ymat[:, cols]
    sd = np.array([rec["train_z_sd"][t] for t in task_list])
    _per_task, oof = run_per_task_cv(d2h, h1000, ymat20, task_list, folds, n_jobs)
    pred = np.nanmean(oof["kernel"], axis=0)  # (n, 20)
    resid_z = (pred - ymat20) / sd[None, :]

    # pairwise-complete residual covariance
    n_t = len(task_list)
    cov = np.full((n_t, n_t), np.nan)
    for a in range(n_t):
        for b in range(a, n_t):
            m = ~np.isnan(resid_z[:, a]) & ~np.isnan(resid_z[:, b])
            ra, rb = resid_z[m, a], resid_z[m, b]
            cov[a, b] = cov[b, a] = float(np.mean((ra - ra.mean()) * (rb - rb.mean())))
    offdiag = cov[np.triu_indices(n_t, k=1)]
    var_shared = float(np.mean(offdiag))  # one-factor estimate of the shared run-level residual
    var_diag = float(np.mean(np.diag(cov)))
    var_idio = var_diag - var_shared

    v_target = float(y.var())
    resid_y = np.nanmean(kernel_oof_y, axis=0) - y
    var_resid_y = float(resid_y.var())
    noise_task_only = var_idio / n_t
    noise_with_shared = var_shared + var_idio / n_t

    def _bounds(noise_var: float) -> dict:
        rel = max(1.0 - noise_var / v_target, 0.0)
        return {
            "noise_var": noise_var,
            "reliability": rel,
            "implied_max_spearman": float(np.sqrt(rel)),
            "snr": float(np.sqrt((v_target - noise_var) / noise_var)) if noise_var > 0 else np.inf,
        }

    return {
        "n_tasks": n_t,
        "target_var": v_target,
        "kernel_oof_resid_var_target": var_resid_y,
        "resid_cov_shared_offdiag_mean": var_shared,
        "resid_var_diag_mean": var_diag,
        "resid_var_idiosyncratic": var_idio,
        "consistency_check_var_resid_y_vs_decomposition": {
            "var_resid_y": var_resid_y,
            "var_shared_plus_idio_over_T": noise_with_shared,
        },
        "bound_if_shared_factor_is_run_noise": _bounds(noise_with_shared),
        "bound_task_idiosyncratic_only": _bounds(noise_task_only),
        "split_half_reference_implied_max": 0.880,
        "achieved_kernel_perfold": REF_KERNEL,
        "caveats": (
            "resid diag includes model approximation error (overstates task noise -> "
            "conservative); the shared residual factor cannot be split into run-seed noise vs "
            "mixture-caused-but-unmodeled signal without seed repeats. Inferior to a seed-repeat panel."
        ),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(res: dict) -> str:
    lines = []
    A = lines.append
    A("# Grug validation checks (adversarial-review follow-ups; 800 train runs only)\n")
    A(f"Target {TARGET_NAME} (frozen train z-stats); metric = per-fold Spearman mean over the")
    A("same 15 folds (RepeatedKFold(5,3,seed 0)) as every prior comparison. Holdout labels")
    A("NOT re-opened; realized holdout 0.7205 used as a published constant.\n")

    if "analysis_1" in res:
        a1 = res["analysis_1"]
        A("## 1. Epoch-augmentation architecture bake-off\n")
        A("| model | per-fold Spearman | delta vs plain | wins/15 | Wilcoxon p |")
        A("|-------|-------------------|----------------|---------|------------|")
        for v in A1_VARIANTS:
            s = a1["summary"][v]
            if v == "plain_kernel":
                A(f"| {v} | {s['mean']:.4f} +/- {s['std']:.4f} | - | - | - |")
            else:
                d = a1["deltas_vs_plain"][v]
                A(
                    f"| {v} | {s['mean']:.4f} +/- {s['std']:.4f} | {d['d_mean']:+.4f} | "
                    f"{d['wins']}/15 | {d['wilcoxon_p']:.4f} |"
                )
        A("")
        A(f"- verdict: {res['verdicts']['analysis_1']}\n")

    if "analysis_2" in res:
        a2 = res["analysis_2"]
        A("## 2. Selection-corrected optimism bootstrap (target + model selection)\n")
        A(f"- {a2['n_bootstrap']} bootstrap replicates; {a2['kernel_hyperparams_note']}")
        A(f"- {a2['cv_convention']}")
        A(f"- chosen-target distribution: {a2['chosen_target_counts']}")
        A(
            f"- optimism = in-replicate CV - OOB of the chosen target: "
            f"{a2['optimism_mean']:+.4f} +/- {a2['optimism_se']:.4f} "
            f"(95% CI [{a2['optimism_ci95'][0]:+.4f}, {a2['optimism_ci95'][1]:+.4f}])"
        )
        A(
            f"- de-biased expectation for external data: {REF_KERNEL:.4f} - {a2['optimism_mean']:.4f} = "
            f"{a2['debiased_expected_holdout']:.4f}; realized holdout {REF_HOLDOUT:.4f} "
            f"(diff {a2['realized_minus_debiased']:+.4f}, ~{a2['z_realized_vs_debiased']:+.2f} SE at n=40)"
        )
        A(f"- verdict: {res['verdicts']['analysis_2']}\n")

    if "analysis_3" in res:
        a3a, a3b = res["analysis_3"]["near_duplicates"], res["analysis_3"]["residual_decomposition"]
        A("## 3. Noise / SNR proxies (no seed repeats exist; both proxies inferior to them)\n")
        A("### 3a. Near-duplicate mixture pairs (TV per phase)\n")
        A(
            f"- pairs with max-phase TV < {a3a['tv_threshold']}: {a3a['n_pairs_below_threshold']} "
            f"(exact duplicates: {a3a['n_exact_duplicates']}); min TV = {a3a['min_tv_max']:.4f}"
        )
        if a3a["n_pairs_below_threshold"]:
            A(
                f"- implied noise SD of {TARGET_NAME}: {a3a['implied_noise_sd_zmacro']:.4f} "
                f"(target SD {a3a['y_std_zmacro']:.4f})"
            )
        else:
            A("- no repeat-noise anchor available from the design (D-optimal design spreads mixtures)")
        A(f"- closest pairs (top {min(5, len(a3a['closest_pairs']))}):")
        for p in a3a["closest_pairs"][:5]:
            A(f"  - runs ({p['i']},{p['j']}): TV {p['tv_max']:.3f}, |dy| zmacro {p['abs_dy_zmacro']:.3f}")
        A("")
        A("### 3b. Per-task kernel-OOF residual covariance decomposition (20 target tasks)\n")
        A(
            f"- target var {a3b['target_var']:.4f}; kernel OOF resid var {a3b['kernel_oof_resid_var_target']:.4f}; "
            f"shared (off-diag mean) {a3b['resid_cov_shared_offdiag_mean']:.4f}; "
            f"idiosyncratic {a3b['resid_var_idiosyncratic']:.4f}"
        )
        bs, bt = a3b["bound_if_shared_factor_is_run_noise"], a3b["bound_task_idiosyncratic_only"]
        A(
            f"- if the shared residual factor is run-level noise: reliability {bs['reliability']:.3f}, "
            f"implied max Spearman {bs['implied_max_spearman']:.3f}, SNR {bs['snr']:.2f}"
        )
        A(
            f"- task-idiosyncratic noise only: reliability {bt['reliability']:.3f}, "
            f"implied max Spearman {bt['implied_max_spearman']:.3f}, SNR {bt['snr']:.2f} "
            f"(split-half reference bound {a3b['split_half_reference_implied_max']:.2f})"
        )
        A(f"- caveats: {a3b['caveats']}")
        A(f"- verdict: {res['verdicts']['analysis_3']}\n")
    return "\n".join(lines)


def make_verdicts(res: dict) -> dict:
    v = {}
    if "analysis_1" in res:
        a1 = res["analysis_1"]
        beats = [(n, d) for n, d in a1["deltas_vs_plain"].items() if d["d_mean"] > 0 and d["wilcoxon_p"] < 0.05]
        if beats:
            best = max(beats, key=lambda t: t[1]["d_mean"])
            v["analysis_1"] = (
                f"{best[0]} beats the plain kernel ({best[1]['d_mean']:+.4f}, p={best[1]['wilcoxon_p']:.4f})"
            )
        else:
            v["analysis_1"] = "no epoch-augmentation construction beats the plain Hellinger kernel"
    if "analysis_2" in res:
        a2 = res["analysis_2"]
        consistent = abs(a2["z_realized_vs_debiased"]) < 1.96
        v["analysis_2"] = (
            f"selection optimism ~{a2['optimism_mean']:+.3f}; realized holdout {REF_HOLDOUT} is "
            f"{'consistent' if consistent else 'INCONSISTENT'} with the optimism-corrected prediction "
            f"{a2['debiased_expected_holdout']:.3f} ({a2['z_realized_vs_debiased']:+.2f} SE at n=40)"
        )
    if "analysis_3" in res:
        a3a = res["analysis_3"]["near_duplicates"]
        a3b = res["analysis_3"]["residual_decomposition"]
        anchor = (
            f"noise anchor FOUND (n={a3a['n_pairs_below_threshold']}, implied SD "
            f"{a3a.get('implied_noise_sd_zmacro', float('nan')):.3f})"
            if a3a["n_pairs_below_threshold"]
            else f"no near-duplicate noise anchor (min TV {a3a['min_tv_max']:.3f})"
        )
        bs = a3b["bound_if_shared_factor_is_run_noise"]
        v["analysis_3"] = (
            f"{anchor}; residual decomposition implies max Spearman {bs['implied_max_spearman']:.3f} "
            f"(SNR {bs['snr']:.2f}) if the shared residual factor is run noise"
        )
    return v


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyses", default="1,2,3", help="comma list of analyses to run")
    args = ap.parse_args()
    todo = set(args.analyses.split(","))
    n_jobs = joblib.cpu_count()

    hists, views, _centroids, _rff_means, _rff_order, buckets_table = load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)

    runs = pd.read_parquet(TRAIN_RUNS)
    w = weight_matrix(runs, buckets)
    macro = runs["macro_bpb"].to_numpy(dtype=np.float64)
    rec = json.loads((GRUG_DIR / "target_candidates.json").read_text())["recommended_target"]
    y = build_zmacro_target(runs, rec)
    ymat, tasks = build_task_matrix(runs)
    eng_cols = [j for j, t in enumerate(tasks) if task_family(t) == "english"]
    all_cols = list(range(len(tasks)))
    assert [tasks[j] for j in eng_cols] == rec["task_list"], "english family != registered 20-task list"

    e, _ = stage_b_epochs(w, buckets, buckets_table)
    hphase = per_phase_hist(w, v1000)
    h1000 = flat(hphase)
    d2_hell = _sq_hellinger(hphase)
    d2_hell4 = 4.0 * d2_hell  # squared-Euclid of concatenated per-phase sqrt-hists
    hinge = hinge_epoch_features(w, e)  # (n, 6)
    d2_hinge = sq_euclid_std(hinge)

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(np.arange(len(y))))

    res: dict = {
        "target": TARGET_NAME,
        "reference": {
            "kernel_perfold": REF_KERNEL,
            "hist_ridge_perfold": 0.7396,
            "hist_ridge_hinge_perfold": 0.7701,
            "product_kernel_perfold": 0.7673,
            "realized_holdout": REF_HOLDOUT,
        },
    }

    if "1" in todo:
        t0 = time.monotonic()
        res["analysis_1"] = analysis_1(d2_hell, d2_hell4, d2_hinge, hinge, y, folds, n_jobs)
        logger.info("analysis 1 done %.0fs", time.monotonic() - t0)

    if "2" in todo:
        t0 = time.monotonic()
        res["analysis_2"] = analysis_2(d2_hell, ymat, macro, eng_cols, all_cols, n_jobs)
        logger.info("analysis 2 done %.0fs", time.monotonic() - t0)

    if "3" in todo:
        t0 = time.monotonic()
        a3a = analysis_3a(w, y, macro)
        # kernel OOF on the target itself, for the residual-variance cross-check
        oof_y = np.full((N_REPEATS, len(y)), np.nan)
        fold_preds = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(kernel_cv_predict)(d2_hell, np.asarray(tr), np.asarray(te), y) for tr, te in folds
        )
        for fid, (_tr, te) in enumerate(folds):
            oof_y[fid // N_SPLITS, np.asarray(te)] = fold_preds[fid]
        a3b = analysis_3b(d2_hell, h1000, ymat, tasks, rec, y, folds, oof_y, n_jobs)
        res["analysis_3"] = {"near_duplicates": a3a, "residual_decomposition": a3b}
        logger.info("analysis 3 done %.0fs", time.monotonic() - t0)

    res["verdicts"] = make_verdicts(res)
    (GRUG_DIR / "validation_checks.json").write_text(json.dumps(res, indent=2, default=json_default))
    report = write_report(res)
    (GRUG_DIR / "validation_checks.md").write_text(report)
    print(report)
    logger.info("wrote %s", GRUG_DIR / "validation_checks.md")


if __name__ == "__main__":
    main()
