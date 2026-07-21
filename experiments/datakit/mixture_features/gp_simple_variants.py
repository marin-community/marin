# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two cheap variants of the surrogate, held to the 'simple unless it clearly wins' bar.

OPTION 1 -- LINEAR MEAN instead of a constant.
    Today the model is kernel-ridge around a constant mean ybar, so far from data it reverts to
    the average. That showed up concretely in the inversion study: the best predicted candidate
    was -0.689 while the best OBSERVED run was -1.058, i.e. the surrogate would not propose
    anything as good as what we had already seen. A linear trend on COARSE content (the K=40
    histogram, which is the aggregate-group signal f18 identified) gives the model something to
    extrapolate along, with the kernel left to fit the residual. Cost: one ridge fit.

OPTION 2 -- SPLIT-CONFORMAL intervals instead of GP posterior variance.
    The GP's error bars needed marginal-likelihood fitting of a signal amplitude and a noise
    term, and in-distribution they came out ~= the seed-noise floor we had already measured.
    Split-conformal gets an interval from residual quantiles alone: no hyperparameters, and a
    distribution-free finite-sample coverage guarantee. If it matches the GP's coverage and
    width, it is the simpler way to get the one benefit the GP actually delivered.

Both are scored against the SAME baseline (constant-mean kernel ridge) on the same folds.
Target: z-scored bpb, LOWER IS BETTER.
"""

import json

import numpy as np
from gp_inversion_study import load_all
from gp_surrogate import fit_gp, predict_gp
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

N_FOLDS = 5
SEED = 0
CONFORMAL_ALPHAS = (0.32, 0.05)  # target 68% and 95% intervals
RIDGE_ALPHAS = np.logspace(-3, 3, 25)


def coarse_features(h_train, n_comp=20):
    """Low-dimensional summary of per-phase content: the top principal components of h.

    PCA is unsupervised -- it never sees y, exactly like the frozen codebook itself -- so this
    stays a fixed feature transform rather than something fit to the target.
    """
    feats = []
    for p in range(h_train.shape[1]):
        x = h_train[:, p, :] - h_train[:, p, :].mean(0)
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        feats.append(x @ vt[:n_comp].T)
    return np.concatenate(feats, axis=1)


def ridge_fit_predict(x_tr, y_tr, x_te):
    """Ridge with alpha by simple GCV on the training rows; returns train and test predictions."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-12
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean()
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    uty = u.T @ (y_tr - ym)
    best, best_score = None, np.inf
    for al in RIDGE_ALPHAS:
        shrink = s**2 / (s**2 + al)
        resid = (y_tr - ym) - u @ (shrink * uty)
        dof = shrink.sum()
        score = (resid**2).sum() / max(len(y_tr) - dof, 1e-6)  # GCV-ish
        if score < best_score:
            best, best_score = al, score
    shrink = s**2 / (s**2 + best)
    coef = vt.T @ (shrink * uty / np.maximum(s, 1e-12))
    return a @ coef + ym, b @ coef + ym


def evaluate(d2, y, x_coarse):
    """5-fold: baseline (constant mean) vs linear-mean variant, plus conformal vs GP intervals."""
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    rows = {k: [] for k in ("base_pred", "lin_pred", "truth", "gp_sd", "conf_hi", "conf_lo")}
    conf_width = {a: [] for a in CONFORMAL_ALPHAS}
    for tr, te in kf.split(np.arange(len(y))):
        # ---- baseline: constant-mean kernel ridge (== the current model) ----
        f = fit_gp(d2[np.ix_(tr, tr)], y[tr])
        mu_b, sd_b = predict_gp(f, d2[np.ix_(te, tr)], include_noise=True)

        # ---- option 1: linear mean on coarse content + kernel on the residual ----
        lin_tr, lin_te = ridge_fit_predict(x_coarse[tr], y[tr], x_coarse[te])
        r_tr = y[tr] - lin_tr
        fr = fit_gp(d2[np.ix_(tr, tr)], r_tr)
        mu_r, _ = predict_gp(fr, d2[np.ix_(te, tr)], include_noise=True)
        mu_l = lin_te + mu_r

        # ---- option 2: split-conformal on the BASELINE model ----
        # split the training fold into proper-train and calibration
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(len(tr))
        n_cal = len(tr) // 4
        cal, prop = tr[perm[:n_cal]], tr[perm[n_cal:]]
        fp = fit_gp(d2[np.ix_(prop, prop)], y[prop])
        mu_cal, _ = predict_gp(fp, d2[np.ix_(cal, prop)], include_noise=True)
        resid_cal = np.abs(y[cal] - mu_cal)
        mu_te_p, _ = predict_gp(fp, d2[np.ix_(te, prop)], include_noise=True)
        for a in CONFORMAL_ALPHAS:
            q = float(np.quantile(resid_cal, 1 - a, method="higher"))
            conf_width[a].append(q)
            if a == CONFORMAL_ALPHAS[1]:  # keep the 95% band for coverage bookkeeping
                rows["conf_hi"].append(mu_te_p + q)
                rows["conf_lo"].append(mu_te_p - q)

        rows["base_pred"].append(mu_b)
        rows["lin_pred"].append(mu_l)
        rows["truth"].append(y[te])
        rows["gp_sd"].append(sd_b)

    out = {k: np.concatenate(v) for k, v in rows.items()}
    base, lin, truth, sd = out["base_pred"], out["lin_pred"], out["truth"], out["gp_sd"]
    res = {
        "baseline_rmse": float(np.sqrt(np.mean((base - truth) ** 2))),
        "baseline_spearman": float(spearmanr(base, truth).statistic),
        "linear_mean_rmse": float(np.sqrt(np.mean((lin - truth) ** 2))),
        "linear_mean_spearman": float(spearmanr(lin, truth).statistic),
        "gp_coverage_95": float(np.mean(np.abs(truth - base) < 1.96 * sd)),
        "gp_mean_halfwidth_95": float(np.mean(1.96 * sd)),
        "conformal_coverage_95": float(np.mean((truth < out["conf_hi"]) & (truth > out["conf_lo"]))),
        "conformal_mean_halfwidth_95": float(np.mean(conf_width[CONFORMAL_ALPHAS[1]])),
        "conformal_mean_halfwidth_68": float(np.mean(conf_width[CONFORMAL_ALPHAS[0]])),
        "prediction_range_baseline": [float(base.min()), float(base.max())],
        "prediction_range_linear": [float(lin.min()), float(lin.max())],
        "best_observed_y": float(y.min()),
    }
    return res


def main():
    d2, y, _v, _w, h_train = load_all()
    x_coarse = coarse_features(h_train)
    print(f"n = {len(y)}; coarse linear features = {x_coarse.shape[1]} (top-20 content PCs per phase)\n")

    r = evaluate(d2, y, x_coarse)

    print("=== OPTION 1: linear mean vs constant mean (5-fold held-out) ===")
    print(f"  baseline (constant mean): RMSE {r['baseline_rmse']:.4f}   Spearman {r['baseline_spearman']:.4f}")
    print(f"  linear mean + kernel    : RMSE {r['linear_mean_rmse']:.4f}   Spearman {r['linear_mean_spearman']:.4f}")
    d_rmse = r["baseline_rmse"] - r["linear_mean_rmse"]
    d_rho = r["linear_mean_spearman"] - r["baseline_spearman"]
    print(f"  delta: RMSE {d_rmse:+.4f} (positive = linear better), Spearman {d_rho:+.4f}")
    print(f"  prediction range  baseline {r['prediction_range_baseline']}")
    print(f"                    linear   {r['prediction_range_linear']}   (best observed {r['best_observed_y']:.3f})")

    print("\n=== OPTION 2: split-conformal vs GP posterior intervals ===")
    print(f"  GP        95% coverage {100*r['gp_coverage_95']:.1f}%   mean half-width {r['gp_mean_halfwidth_95']:.4f}")
    print(
        f"  conformal 95% coverage {100*r['conformal_coverage_95']:.1f}%   mean half-width "
        f"{r['conformal_mean_halfwidth_95']:.4f}"
    )
    print(f"  conformal 68% half-width {r['conformal_mean_halfwidth_68']:.4f}")

    with open("scratch/mixture_features/grug/gp_simple_variants.json", "w") as fh:
        json.dump(r, fh, indent=1)
    print("\nwrote scratch/mixture_features/grug/gp_simple_variants.json")


if __name__ == "__main__":
    main()
