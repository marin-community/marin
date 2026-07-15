# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "joblib"]
# ///
"""Grug-swarm target diagnosis: why is the macro_bpb predictability ceiling ~0.30?

Phase 2b of the grug campaign (logbook 2026-07-15). Uses ONLY the 800 training runs
(``QUARANTINE_test_labels.parquet`` stays closed). Diagnoses the low OOF Spearman of the
phase-2 surrogates on ``macro_bpb`` and evaluates formula-defined alternative targets so
one can be re-registered before the holdout test.

Stages (same 5-fold x 3-repeat CV as grug_fit, seed 0, identical splits):
  1  Per-task predictability: OOF Spearman of the two best phase-2 predictors
     (Hellinger-kernel K1000, hist-ridge K1000) for every bpb task + macro_bpb.
  2  Structure: per-task bpb correlation matrix, hierarchical clustering, name-rule
     families (english / multilingual / code / math), residual correlations.
  3  Candidate targets (all fixed formulas, task-selection nested inside CV folds):
     english macro, coverage-restricted macros, coherence-filtered macro, PC1/PC2,
     family macros, z-scored macros.
  4  Noise floor: split-half (Spearman-Brown) reliability of each macro target +
     implied max achievable Spearman; decomposition of the 0.303 ceiling.
  5  Quick wins: phase-2 top-3 variants on the recommended target; per-task /
     per-family modeling + rank aggregation vs direct macro modeling.

Outputs under scratch/mixture_features/grug/: per_task_cv.parquet,
target_candidates.json, target_report.md.
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import time

import featurize
import joblib
import numpy as np
import pandas as pd
from grug_fit import (
    GRUG_DIR,
    RIDGE_ALPHAS,
    TRAIN_RUNS,
    _sq_euclid_qual,
    flat,
    json_default,
    load_grug_artifacts,
    per_phase_hist,
    predict_additive_kernel,
    ridge_gcv,
    weight_matrix,
)
from retrodiction import (
    KR_ALPHAS,
    KR_GAMMA_FACTORS,
    N_INNER_FOLDS,
    SEED,
    _sq_hellinger,
    kernel_cv_predict,
    spearman_cols,
)
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.model_selection import KFold, RepeatedKFold

logger = logging.getLogger("grug_target_analysis")

N_SPLITS, N_REPEATS = 5, 3
COHERENCE_THRESHOLDS = (0.3, 0.5)
MIN_COHERENT_TASKS = 3
N_SPLIT_HALF = 200
N_CLUSTERS = 4
MIN_TASKS_FOR_RECOMMENDATION = 5


def task_family(t: str) -> str:
    if t.startswith(("belebele_", "include_")):
        return "multilingual"
    if t.startswith("logprob_humaneval"):
        return "code"
    if t.startswith("logprob_gsm8k"):
        return "math"
    return "english"


# ---------------------------------------------------------------------------
# Label matrix
# ---------------------------------------------------------------------------


def build_task_matrix(runs: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(n_runs, n_tasks) bpb matrix over the union of bpb tasks; NaN where missing."""
    evs = [json.loads(e) for e in runs["evals"]]
    tasks = sorted({k for ev in evs for k, v in ev.items() if "bpb" in v})
    y = np.full((len(evs), len(tasks)), np.nan)
    for i, ev in enumerate(evs):
        for j, t in enumerate(tasks):
            if t in ev and "bpb" in ev[t]:
                y[i, j] = ev[t]["bpb"]
    return y, tasks


def nan_macro(y: np.ndarray, cols: list[int]) -> np.ndarray:
    """Per-run mean bpb over the observed subset of ``cols`` (the macro_bpb convention)."""
    return np.nanmean(y[:, cols], axis=1)


def z_macro(y: np.ndarray, cols: list[int], mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Per-run mean of z-scored (given stats) bpb over observed subset of ``cols``."""
    z = (y[:, cols] - mu[cols]) / sd[cols]
    return np.nanmean(z, axis=1)


# ---------------------------------------------------------------------------
# Multi-target predictors (exact vectorizations of grug_fit.ridge_gcv and
# retrodiction.kernel_cv_predict: same grids, same first-minimum tie-breaking)
# ---------------------------------------------------------------------------


def ridge_gcv_multi(x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray) -> np.ndarray:
    """grug_fit.ridge_gcv for a (n, T) target matrix; per-target alpha by GCV."""
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean(axis=0)
    yc = y_tr - ym
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    g = u.T @ yc  # (r, T)
    n, s2 = len(y_tr), s**2
    null_resid = (yc**2).sum(axis=0) - (g**2).sum(axis=0)
    scores = np.full((len(RIDGE_ALPHAS), y_tr.shape[1]), np.inf)
    for i, al in enumerate(RIDGE_ALPHAS):
        resid = (((al / (s2 + al))[:, None] * g) ** 2).sum(axis=0) + null_resid
        denom = n - float((s2 / (s2 + al)).sum())
        if denom > 1e-9:
            scores[i] = n * resid / denom**2
    best = scores.argmin(axis=0)
    bvt = b @ vt.T
    preds = np.empty((len(x_te), y_tr.shape[1]))
    for bi in np.unique(best):
        cols = best == bi
        al = RIDGE_ALPHAS[bi]
        preds[:, cols] = bvt @ (g[:, cols] * (s / (s2 + al))[:, None]) + ym[cols]
    return preds


def kernel_cv_predict_multi(d2: np.ndarray, tr: np.ndarray, te: np.ndarray, y: np.ndarray) -> np.ndarray:
    """retrodiction.kernel_cv_predict for a (n, T) target matrix; per-target (gamma, alpha)."""
    d_tr = d2[np.ix_(tr, tr)]
    med = float(np.median(d_tr[~np.eye(len(tr), dtype=bool)]))
    gammas = np.asarray(KR_GAMMA_FACTORS) / max(med, 1e-12)
    kf = KFold(N_INNER_FOLDS, shuffle=True, random_state=SEED)
    folds = list(kf.split(np.arange(len(tr))))
    y_tr = y[tr]
    n_t = y_tr.shape[1]
    sse = np.empty((len(gammas), len(KR_ALPHAS), n_t))
    for gi, g in enumerate(gammas):
        k_full = np.exp(-g * d_tr)
        for ai, al in enumerate(KR_ALPHAS):
            s = np.zeros(n_t)
            for itr, iva in folds:
                ym = y_tr[itr].mean(axis=0)
                dual = np.linalg.solve(k_full[np.ix_(itr, itr)] + al * np.eye(len(itr)), y_tr[itr] - ym)
                p = k_full[np.ix_(iva, itr)] @ dual + ym
                s += ((p - y_tr[iva]) ** 2).sum(axis=0)
            sse[gi, ai] = s
    best = sse.reshape(-1, n_t).argmin(axis=0)  # first min == single-target tie-breaking
    preds = np.empty((len(te), n_t))
    for b in np.unique(best):
        gi, ai = divmod(int(b), len(KR_ALPHAS))
        cols = best == b
        g, al = gammas[gi], KR_ALPHAS[ai]
        ym = y_tr[:, cols].mean(axis=0)
        dual = np.linalg.solve(np.exp(-g * d_tr) + al * np.eye(len(tr)), y_tr[:, cols] - ym)
        preds[:, cols] = np.exp(-g * d2[np.ix_(te, tr)]) @ dual + ym
    return preds


def _selftest_multi(d2: np.ndarray, x: np.ndarray, y: np.ndarray, folds: list) -> None:
    """Multi-target predictors must reproduce the phase-2 single-target code paths."""
    tr, te = folds[0]
    pk1 = kernel_cv_predict(d2, np.asarray(tr), np.asarray(te), y)
    pk2 = kernel_cv_predict_multi(d2, np.asarray(tr), np.asarray(te), y[:, None])[:, 0]
    np.testing.assert_allclose(pk1, pk2, rtol=1e-10, atol=1e-12)
    pr1 = ridge_gcv(x[tr], y[tr], x[te])[0]
    pr2 = ridge_gcv_multi(x[tr], y[tr][:, None], x[te])[:, 0]
    np.testing.assert_allclose(pr1, pr2, rtol=1e-8, atol=1e-10)
    logger.info("self-test passed: multi-target == phase-2 single-target on fold 0")


# ---------------------------------------------------------------------------
# Stage 1: per-task CV
# ---------------------------------------------------------------------------


def _fold_multi(fold_id: int, tr, te, d2, x, ymat) -> dict:
    t0 = time.monotonic()
    out = {
        "fold_id": fold_id,
        "te": np.asarray(te),
        "kernel": kernel_cv_predict_multi(d2, np.asarray(tr), np.asarray(te), ymat),
        "ridge": ridge_gcv_multi(x[tr], ymat[tr], x[te]),
    }
    logger.info("multi fold %d done %.1fs", fold_id + 1, time.monotonic() - t0)
    return out


def _fold_partial(fold_id: int, tr, te, d2, x, yfull: np.ndarray) -> dict:
    """Single-target fold for one partial-coverage task: restrict to observed runs."""
    obs = ~np.isnan(yfull)
    tr = np.asarray([i for i in tr if obs[i]])
    te = np.asarray([i for i in te if obs[i]])
    if len(te) == 0:
        return {"fold_id": fold_id, "te": te, "kernel": np.array([]), "ridge": np.array([])}
    pk = kernel_cv_predict(d2, tr, te, np.where(obs, yfull, 0.0))
    pr = ridge_gcv_multi(x[tr], yfull[tr][:, None], x[te])[:, 0]
    return {"fold_id": fold_id, "te": te, "kernel": pk, "ridge": pr}


def oof_spearman(oof: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(mean, std) over repeats of Spearman(OOF prediction, y), NaN-target aware."""
    vals = []
    for r in range(oof.shape[0]):
        m = ~np.isnan(oof[r]) & ~np.isnan(y)
        vals.append(float(spearman_cols(oof[r][m][:, None], y[m])[0]))
    return float(np.mean(vals)), float(np.std(vals))


def run_per_task_cv(d2, x, ymat, tasks, folds, n_jobs) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """OOF per-task predictions for both predictors. Returns (table, oof arrays)."""
    n = ymat.shape[0]
    full_cols = [j for j in range(ymat.shape[1]) if not np.isnan(ymat[:, j]).any()]
    part_cols = [j for j in range(ymat.shape[1]) if j not in full_cols]
    logger.info("per-task CV: %d full-coverage (multi), %d partial (single)", len(full_cols), len(part_cols))

    oof = {m: np.full((N_REPEATS, n, ymat.shape[1]), np.nan) for m in ("kernel", "ridge")}
    fold_out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fold_multi)(fid, tr, te, d2, x, ymat[:, full_cols]) for fid, (tr, te) in enumerate(folds)
    )
    for out in fold_out:
        rep, te = out["fold_id"] // N_SPLITS, out["te"]
        for m in ("kernel", "ridge"):
            oof[m][rep][np.ix_(te, full_cols)] = out[m]

    jobs = [(j, fid, tr, te) for j in part_cols for fid, (tr, te) in enumerate(folds)]
    part_out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fold_partial)(fid, tr, te, d2, x, ymat[:, j]) for j, fid, tr, te in jobs
    )
    for (j, fid, _tr, _te), out in zip(jobs, part_out, strict=True):
        rep, te = fid // N_SPLITS, out["te"]
        if len(te):
            oof["kernel"][rep][te, j] = out["kernel"]
            oof["ridge"][rep][te, j] = out["ridge"]

    rows = []
    for j, t in enumerate(tasks):
        yj = ymat[:, j]
        row = {
            "task": t,
            "family": task_family(t),
            "coverage": int((~np.isnan(yj)).sum()),
            "bpb_mean": float(np.nanmean(yj)),
            "bpb_std": float(np.nanstd(yj)),
        }
        for m in ("kernel", "ridge"):
            mean, std = oof_spearman(oof[m][:, :, j], yj)
            row[f"{m}_spearman"], row[f"{m}_spearman_std"] = mean, std
        rows.append(row)
    return pd.DataFrame(rows), oof


# ---------------------------------------------------------------------------
# Stage 2: structure
# ---------------------------------------------------------------------------


def spearman_corr_matrix(ymat: np.ndarray) -> np.ndarray:
    """Pairwise-complete Spearman correlation over task columns."""
    n_t = ymat.shape[1]
    c = np.eye(n_t)
    for a in range(n_t):
        for b in range(a + 1, n_t):
            m = ~np.isnan(ymat[:, a]) & ~np.isnan(ymat[:, b])
            c[a, b] = c[b, a] = spearmanr(ymat[m, a], ymat[m, b]).statistic
    return c


def cluster_tasks(corr: np.ndarray, tasks: list[str]) -> dict[str, int]:
    d = np.clip(1.0 - corr, 0.0, None)
    np.fill_diagonal(d, 0.0)
    lk = linkage(squareform(d, checks=False), method="average")
    labels = fcluster(lk, t=N_CLUSTERS, criterion="maxclust")
    return {t: int(c) for t, c in zip(tasks, labels, strict=True)}


def family_block_stats(corr: np.ndarray, tasks: list[str]) -> dict:
    fams = sorted({task_family(t) for t in tasks})
    idx = {f: [j for j, t in enumerate(tasks) if task_family(t) == f] for f in fams}
    out = {}
    for fa in fams:
        for fb in fams:
            block = corr[np.ix_(idx[fa], idx[fb])]
            if fa == fb:
                n = len(idx[fa])
                if n < 2:
                    continue
                vals = block[np.triu_indices(n, k=1)]
            else:
                vals = block.ravel()
            out[f"{fa}|{fb}"] = float(np.mean(vals))
    return out


# ---------------------------------------------------------------------------
# Stage 3: candidate targets
# ---------------------------------------------------------------------------


def coherent_task_cols(ymat_tr: np.ndarray, thr: float) -> list[int]:
    """Cols whose Spearman with the leave-one-out macro (on given runs) >= thr."""
    macro_all = np.nanmean(ymat_tr, axis=1)
    n_obs = (~np.isnan(ymat_tr)).sum(axis=1)
    cols = []
    for j in range(ymat_tr.shape[1]):
        yj = ymat_tr[:, j]
        m = ~np.isnan(yj)
        loo = (macro_all[m] * n_obs[m] - yj[m]) / (n_obs[m] - 1)
        if spearmanr(yj[m], loo).statistic >= thr:
            cols.append(j)
    if len(cols) < MIN_COHERENT_TASKS:
        cols = list(range(ymat_tr.shape[1]))
    return cols


def pc_scores(ymat: np.ndarray, tr: np.ndarray, macro: np.ndarray, comp: int) -> np.ndarray:
    """PC ``comp`` scores of the z-scored (train stats) task matrix, train-fitted loadings.

    NaNs impute to 0 (train mean). Sign aligned so train scores correlate positively with
    train macro_bpb (lower better is preserved).
    """
    mu, sd = np.nanmean(ymat[tr], axis=0), np.nanstd(ymat[tr], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    z = (ymat - mu) / sd
    z = np.where(np.isnan(z), 0.0, z)
    zc = z[tr] - z[tr].mean(axis=0)
    _u, _s, vt = np.linalg.svd(zc, full_matrices=False)
    load = vt[comp]
    scores = (z - z[tr].mean(axis=0)) @ load
    if spearmanr(scores[tr], macro[tr]).statistic < 0:
        scores = -scores
    return scores


def _fold_nested(fold_id: int, tr, te, d2, ymat, macro) -> dict:
    """Nested-selection candidates for one fold: selection on 4/5, evaluate on 1/5."""
    tr, te = np.asarray(tr), np.asarray(te)
    out = {"fold_id": fold_id, "te": te}
    for thr in COHERENCE_THRESHOLDS:
        cols = coherent_task_cols(ymat[tr], thr)
        y = nan_macro(ymat, cols)
        p = kernel_cv_predict(d2, tr, te, y)
        out[f"coherent_{thr}"] = (p, y[te], len(cols))
    for comp in (0, 1):
        y = pc_scores(ymat, tr, macro, comp)
        p = kernel_cv_predict(d2, tr, te, y)
        out[f"pc{comp + 1}"] = (p, y[te], np.nan)
    return out


def per_fold_spearman(preds: dict) -> dict[str, tuple[float, float]]:
    """Mean/std over folds of Spearman(pred, fold target) for nested candidates."""
    out = {}
    for name, per_fold in preds.items():
        sps = [float(spearman_cols(p[:, None], yte)[0]) for p, yte in per_fold]
        out[name] = (float(np.mean(sps)), float(np.std(sps)))
    return out


# ---------------------------------------------------------------------------
# Stage 4: split-half reliability
# ---------------------------------------------------------------------------


def split_half_reliability(ymat: np.ndarray, cols: list[int], z_stats=None) -> dict:
    """Spearman-Brown reliability of a macro target from random task split-halves.

    Splits the task list in half N_SPLIT_HALF times; per run each half-macro is the
    (NaN-aware) mean over observed tasks in that half. Caveats: (i) per-task idiosyncratic
    mixture response counts as noise here (conservative for a model that sees the mixture),
    (ii) run-level seed noise common to all tasks is invisible (anti-conservative).
    """
    rng = np.random.default_rng(SEED)
    cols = np.asarray(cols)
    r_p, r_s = [], []
    y = ymat[:, cols]
    if z_stats is not None:
        mu, sd = z_stats
        y = (y - mu[cols]) / sd[cols]
    for _ in range(N_SPLIT_HALF):
        perm = rng.permutation(len(cols))
        a, b = perm[: len(cols) // 2], perm[len(cols) // 2 :]
        ma, mb = np.nanmean(y[:, a], axis=1), np.nanmean(y[:, b], axis=1)
        m = ~np.isnan(ma) & ~np.isnan(mb)
        r_p.append(pearsonr(ma[m], mb[m]).statistic)
        r_s.append(spearmanr(ma[m], mb[m]).statistic)
    rp, rs = float(np.mean(r_p)), float(np.mean(r_s))
    sb = lambda r: 2 * r / (1 + r)  # noqa: E731
    return {
        "half_r_pearson": rp,
        "half_r_spearman": rs,
        "reliability_sb_pearson": sb(rp),
        "reliability_sb_spearman": sb(rs),
        "max_spearman_sqrt_rel": float(np.sqrt(max(sb(rs), 0.0))),
        "n_tasks": len(cols),
    }


# ---------------------------------------------------------------------------
# Stage 5: quick wins on the recommended target
# ---------------------------------------------------------------------------


def _fold_variants(fold_id: int, tr, te, feats: dict, y: np.ndarray) -> dict:
    return {
        "fold_id": fold_id,
        "te": np.asarray(te),
        "4_hellinger_kernel_k1000": kernel_cv_predict(feats["d2_hell"], np.asarray(tr), np.asarray(te), y),
        "7_kernel_plus_quality": predict_additive_kernel(feats["d2_hell"], feats["d2_qual"], y, tr, te),
        "5_rff_ridge": ridge_gcv_multi(feats["rff"][tr], y[tr][:, None], feats["rff"][te])[:, 0],
        "3_hist_ridge_k1000": ridge_gcv_multi(feats["h1000"][tr], y[tr][:, None], feats["h1000"][te])[:, 0],
        "1_weights_ridge": ridge_gcv_multi(feats["weights"][tr], y[tr][:, None], feats["weights"][te])[:, 0],
    }


def run_variants_on_target(feats: dict, y: np.ndarray, folds: list, n_jobs: int) -> dict:
    names = ("4_hellinger_kernel_k1000", "7_kernel_plus_quality", "5_rff_ridge", "3_hist_ridge_k1000", "1_weights_ridge")
    fold_out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fold_variants)(fid, tr, te, feats, y) for fid, (tr, te) in enumerate(folds)
    )
    oof = {v: np.full((N_REPEATS, len(y)), np.nan) for v in names}
    for out in fold_out:
        rep, te = out["fold_id"] // N_SPLITS, out["te"]
        for v in names:
            oof[v][rep, te] = out[v]
    return {v: oof_spearman(oof[v], y) for v in names}


def rank_aggregate(oof_tasks: np.ndarray, cols: list[int], y_ref: np.ndarray) -> tuple[float, float]:
    """Mean per-repeat Spearman of the mean-rank aggregate of per-task OOF predictions.

    oof_tasks: (n_repeats, n, n_tasks) OOF preds; aggregate = mean over ``cols`` of the
    per-task prediction ranks; scored against y_ref.
    """
    vals = []
    for r in range(oof_tasks.shape[0]):
        ranks = []
        for j in cols:
            p = oof_tasks[r, :, j]
            m = ~np.isnan(p)
            rk = np.full(len(p), np.nan)
            rk[m] = rankdata(p[m]) / m.sum()
            ranks.append(rk)
        agg = np.nanmean(np.stack(ranks, axis=1), axis=1)
        m = ~np.isnan(agg) & ~np.isnan(y_ref)
        vals.append(float(spearman_cols(agg[m][:, None], y_ref[m])[0]))
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n_jobs = min(15, joblib.cpu_count())

    hists, views, _centroids, rff_means, rff_order, buckets_table = load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, order = featurize.composition_matrix(hists, k=1000, views=views)
    assert order == buckets
    v1000 = np.asarray(v1000)
    rff = rff_means[[rff_order[b] for b in buckets]]
    tiers = buckets_table.set_index("bucket").loc[buckets, "quality_tier"].to_numpy()

    runs = pd.read_parquet(TRAIN_RUNS)
    w = weight_matrix(runs, buckets)
    macro = runs["macro_bpb"].to_numpy(dtype=np.float64)
    ymat, tasks = build_task_matrix(runs)
    fams = [task_family(t) for t in tasks]
    logger.info("task matrix: %d runs x %d tasks (families: %s)", *ymat.shape, pd.Series(fams).value_counts().to_dict())

    h1000 = flat(per_phase_hist(w, v1000))
    d2_hell = _sq_hellinger(per_phase_hist(w, v1000))
    feats = {
        "weights": w.reshape(len(w), -1),
        "h1000": h1000,
        "d2_hell": d2_hell,
        "d2_qual": _sq_euclid_qual(w, tiers),
        "rff": np.concatenate([w[:, 0, :] @ rff, w[:, 1, :] @ rff], axis=1),
    }

    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(np.arange(len(macro))))
    _selftest_multi(d2_hell, h1000, macro, folds)

    # ---- Stage 1: per-task CV ----
    per_task, oof_tasks = run_per_task_cv(d2_hell, h1000, ymat, tasks, folds, n_jobs)
    logger.info("stage 1 done")

    # ---- Stage 2: structure ----
    corr = spearman_corr_matrix(ymat)
    clusters = cluster_tasks(corr, tasks)
    per_task["cluster"] = per_task["task"].map(clusters)
    per_task.to_parquet(GRUG_DIR / "per_task_cv.parquet", index=False)
    blocks = family_block_stats(corr, tasks)
    # residual structure of the kernel model (mean OOF over repeats)
    resid = np.nanmean(oof_tasks["kernel"], axis=0) - ymat
    resid_corr = spearman_corr_matrix(resid)
    resid_blocks = family_block_stats(resid_corr, tasks)

    fam_cols = {f: [j for j, t in enumerate(tasks) if task_family(t) == f] for f in sorted(set(fams))}
    all_cols = list(range(len(tasks)))
    full_cols = [j for j in all_cols if not np.isnan(ymat[:, j]).any()]
    mu_all, sd_all = np.nanmean(ymat, axis=0), np.nanstd(ymat, axis=0)
    sd_all = np.where(sd_all > 0, sd_all, 1.0)
    target_cols = {
        "macro_bpb": all_cols,
        "macro_full_coverage": full_cols,
        "a_macro_english": fam_cols["english"],
        "a_macro_english_full": [j for j in fam_cols["english"] if j in full_cols],
        "d_macro_multilingual": fam_cols["multilingual"],
        "d_macro_belebele": [j for j, t in enumerate(tasks) if t.startswith("belebele_")],
        "d_macro_include": [j for j, t in enumerate(tasks) if t.startswith("include_")],
        "d_code_humaneval": fam_cols["code"],
        "d_math_gsm8k": fam_cols["math"],
        "e_zmacro_all": all_cols,
        "e_zmacro_english": fam_cols["english"],
    }

    # ---- Stage 3: fixed-formula candidates (multi-target, same folds) ----
    fixed = {
        n: z_macro(ymat, c, mu_all, sd_all) if n.startswith("e_") else nan_macro(ymat, c) for n, c in target_cols.items()
    }
    fixed["macro_bpb"] = macro  # the reported per-run varying-task-set convention
    fnames = list(fixed)
    ycand = np.stack([fixed[k] for k in fnames], axis=1)
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fold_multi)(fid, tr, te, d2_hell, h1000, ycand) for fid, (tr, te) in enumerate(folds)
    )
    cand_res = {}
    oof_cand = {m: np.full((N_REPEATS, len(macro), len(fnames)), np.nan) for m in ("kernel", "ridge")}
    for o in out:
        rep, te = o["fold_id"] // N_SPLITS, o["te"]
        for m in ("kernel", "ridge"):
            oof_cand[m][rep][np.ix_(te, range(len(fnames)))] = o[m]
    for j, name in enumerate(fnames):
        cand_res[name] = {m: oof_spearman(oof_cand[m][:, :, j], ycand[:, j]) for m in ("kernel", "ridge")}

    # nested-selection candidates (selection inside each fold)
    nested_out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fold_nested)(fid, tr, te, d2_hell, ymat, macro) for fid, (tr, te) in enumerate(folds)
    )
    nested_names = [f"coherent_{t}" for t in COHERENCE_THRESHOLDS] + ["pc1", "pc2"]
    nested_preds = {n: [] for n in nested_names}
    nested_ntasks = {n: [] for n in nested_names}
    for o in sorted(nested_out, key=lambda d: d["fold_id"]):
        for n in nested_names:
            p, yte, k = o[n]
            nested_preds[n].append((p, yte))
            nested_ntasks[n].append(k)
    nested_sp = per_fold_spearman(nested_preds)
    # per-fold Spearman for fixed candidates too (apples-to-apples with nested)
    fixed_perfold = {}
    for j, name in enumerate(fnames):
        sps = []
        for o in out:
            te = o["te"]
            sps.append(float(spearman_cols(o["kernel"][:, j][:, None], ycand[te, j])[0]))
        fixed_perfold[name] = (float(np.mean(sps)), float(np.std(sps)))

    # coherence selection on the FULL train (the registered fixed rule)
    coherent_full = {thr: [tasks[j] for j in coherent_task_cols(ymat, thr)] for thr in COHERENCE_THRESHOLDS}

    # ---- Stage 4: split-half reliability ----
    rel = {
        n: split_half_reliability(ymat, c, z_stats=(mu_all, sd_all) if n.startswith("e_") else None)
        for n, c in target_cols.items()
        if len(c) >= 2
    }

    # ---- Recommendation: best kernel OOF Spearman among fixed multi-task candidates ----
    eligible = {
        n: cand_res[n]["kernel"][0]
        for n in fnames
        if n != "macro_bpb" and len(target_cols[n]) >= MIN_TASKS_FOR_RECOMMENDATION
    }
    recommended = max(eligible, key=eligible.get)
    rec_cols = target_cols[recommended]
    logger.info("recommended target: %s (kernel OOF %.4f)", recommended, eligible[recommended])

    # ---- Stage 5: quick wins on the recommended target ----
    y_rec = fixed[recommended]
    variants = run_variants_on_target(feats, y_rec, folds, n_jobs)
    agg = {
        "per_task_rank_agg_on_target_tasks_vs_recommended": rank_aggregate(oof_tasks["kernel"], rec_cols, y_rec),
        "per_task_rank_agg_all_tasks_vs_macro": rank_aggregate(oof_tasks["kernel"], all_cols, macro),
    }
    # family-level rank aggregation: use the fixed-candidate OOF preds of the 4 family macros
    fam_targets = ["a_macro_english", "d_macro_multilingual", "d_code_humaneval", "d_math_gsm8k"]
    fam_idx = [fnames.index(n) for n in fam_targets]
    fam_oof = oof_cand["kernel"][:, :, fam_idx]  # (rep, n, 4)
    agg["family_rank_agg_vs_macro"] = rank_aggregate(fam_oof, list(range(len(fam_idx))), macro)
    agg["family_rank_agg_vs_recommended"] = rank_aggregate(fam_oof, list(range(len(fam_idx))), y_rec)

    # ---- outputs ----
    candidates_doc = {
        "protocol": {
            "cv": f"{N_SPLITS}-fold x {N_REPEATS} repeats, RepeatedKFold(random_state={SEED}), identical to phase 2",
            "quarantine_respected": True,
            "n_train_runs": len(macro),
            "n_bpb_tasks_union": len(tasks),
            "per_run_bpb_task_count": {
                "min": int((~np.isnan(ymat)).sum(axis=1).min()),
                "max": int((~np.isnan(ymat)).sum(axis=1).max()),
            },
        },
        "families": {f: [tasks[j] for j in c] for f, c in fam_cols.items()},
        "target_task_lists": {n: [tasks[j] for j in c] for n, c in target_cols.items()},
        "fixed_candidates": {
            n: {
                "kernel_oof_spearman": cand_res[n]["kernel"],
                "ridge_oof_spearman": cand_res[n]["ridge"],
                "kernel_perfold_spearman": fixed_perfold[n],
            }
            for n in fnames
        },
        "nested_candidates": {
            n: {"perfold_spearman": nested_sp[n], "mean_n_tasks_selected": float(np.nanmean(nested_ntasks[n]))}
            for n in nested_names
        },
        "coherent_task_lists_full_train": coherent_full,
        "reliability": rel,
        "structure": {
            "family_block_corr": blocks,
            "residual_family_block_corr": resid_blocks,
            "spearman_macro_english_vs_belebele": float(
                spearmanr(fixed["a_macro_english"], fixed["d_macro_belebele"]).statistic
            ),
        },
        "recommended_target": {
            "name": recommended,
            "kernel_oof_spearman": cand_res[recommended]["kernel"],
            "task_list": [tasks[j] for j in rec_cols],
            "z_stats_note": "z-scored targets use per-task mean/std computed on the 800 train runs",
            "train_z_mu": {tasks[j]: float(mu_all[j]) for j in rec_cols} if recommended.startswith("e_") else None,
            "train_z_sd": {tasks[j]: float(sd_all[j]) for j in rec_cols} if recommended.startswith("e_") else None,
        },
        "variants_on_recommended": variants,
        "rank_aggregation": agg,
    }
    (GRUG_DIR / "target_candidates.json").write_text(json.dumps(candidates_doc, indent=2, default=json_default))
    write_report(
        per_task,
        cand_res,
        fixed_perfold,
        nested_sp,
        nested_ntasks,
        rel,
        blocks,
        resid_blocks,
        recommended,
        rec_cols,
        tasks,
        variants,
        agg,
        coherent_full,
    )
    logger.info("wrote %s", GRUG_DIR)


def write_report(
    per_task,
    cand_res,
    fixed_perfold,
    nested_sp,
    nested_ntasks,
    rel,
    blocks,
    resid_blocks,
    recommended,
    rec_cols,
    tasks,
    variants,
    agg,
    coherent_full,
) -> None:
    lines = []
    A = lines.append
    A("# Grug target analysis (phase 2b, 800 training runs; holdout untouched)\n")
    A("Same CV as phase 2 (5-fold x 3 repeats, seed 0). Predictors: Hellinger-kernel K1000")
    A("(phase-2 primary) and hist-ridge K1000. All numbers OOF Spearman unless noted.\n")

    A("## 1. Per-task predictability (52-55 bpb tasks)\n")
    pt = per_task.sort_values("kernel_spearman", ascending=False)
    A("| task | family | cov | bpb std | kernel | ridge |")
    A("|------|--------|-----|---------|--------|-------|")
    for _, r in pt.iterrows():
        A(
            f"| {r['task']} | {r['family']} | {r['coverage']} | {r['bpb_std']:.3f} | "
            f"{r['kernel_spearman']:+.3f} | {r['ridge_spearman']:+.3f} |"
        )
    A("")
    A(
        f"- macro_bpb baseline: kernel {cand_res['macro_bpb']['kernel'][0]:.4f} +/- "
        f"{cand_res['macro_bpb']['kernel'][1]:.4f}, ridge {cand_res['macro_bpb']['ridge'][0]:.4f} +/- "
        f"{cand_res['macro_bpb']['ridge'][1]:.4f}"
    )
    for f in sorted(per_task["family"].unique()):
        sub = per_task[per_task["family"] == f]
        A(
            f"- family {f}: n={len(sub)}, kernel mean {sub['kernel_spearman'].mean():+.3f} "
            f"(min {sub['kernel_spearman'].min():+.3f}, max {sub['kernel_spearman'].max():+.3f})"
        )
    A("")

    A("## 2. Structure (Spearman corr across runs)\n")
    A("Family-block mean correlations (bpb values):")
    for k, v in sorted(blocks.items()):
        A(f"- {k}: {v:+.3f}")
    A("Family-block mean correlations (kernel OOF residuals):")
    for k, v in sorted(resid_blocks.items()):
        A(f"- {k}: {v:+.3f}")
    A("")

    A("## 3. Candidate targets (kernel / ridge OOF Spearman; per-fold mean for nested)\n")
    A("| candidate | kernel OOF | ridge OOF | kernel per-fold |")
    A("|-----------|-----------|-----------|-----------------|")
    for n, res in cand_res.items():
        A(
            f"| {n} | {res['kernel'][0]:.4f} +/- {res['kernel'][1]:.4f} | "
            f"{res['ridge'][0]:.4f} +/- {res['ridge'][1]:.4f} | "
            f"{fixed_perfold[n][0]:.4f} +/- {fixed_perfold[n][1]:.4f} |"
        )
    A("")
    A("Nested-selection candidates (selection on 4/5 inside every fold; per-fold Spearman):")
    for n, (m, s) in nested_sp.items():
        extra = f", mean tasks selected {np.nanmean(nested_ntasks[n]):.1f}" if "coherent" in n else ""
        A(f"- {n}: {m:.4f} +/- {s:.4f}{extra}")
    A("")
    for thr, lst in coherent_full.items():
        A(f"- coherence rule (full-train, thr={thr}) selects {len(lst)} tasks: {', '.join(lst)}")
    A("")

    A("## 4. Noise floor (split-half over tasks, Spearman-Brown)\n")
    A("| target | n tasks | half-r (S) | SB reliability | implied max Spearman |")
    A("|--------|---------|------------|----------------|----------------------|")
    for n, r in rel.items():
        A(
            f"| {n} | {r['n_tasks']} | {r['half_r_spearman']:.3f} | {r['reliability_sb_spearman']:.3f} | "
            f"{r['max_spearman_sqrt_rel']:.3f} |"
        )
    A("")
    A("Caveats: task-split-half counts real per-task mixture response as 'noise' (conservative)")
    A("but cannot see run-level seed noise common to all tasks (anti-conservative).\n")

    A("## 5. Recommended target + quick wins\n")
    A(
        f"- RECOMMENDED: **{recommended}** = mean bpb over the fixed task list below "
        "(z-scored with train-run stats if the name starts with e_)."
    )
    A(f"- task list ({len(rec_cols)}): {', '.join(tasks[j] for j in rec_cols)}")
    A("- variants on the recommended target (OOF Spearman):")
    for v, (m, s) in variants.items():
        A(f"  - {v}: {m:.4f} +/- {s:.4f}")
    A("- rank aggregation:")
    for k, (m, s) in agg.items():
        A(f"  - {k}: {m:.4f} +/- {s:.4f}")
    (GRUG_DIR / "target_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
