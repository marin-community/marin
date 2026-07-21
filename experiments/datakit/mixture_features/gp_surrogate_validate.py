# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the Bayesian (GP) formulation of the mixture surrogate.

Three questions, in order of importance:
  1. EQUIVALENCE  -- does the GP posterior mean reproduce the kernel-ridge prediction?
                     (it must; they are the same estimator)
  2. CALIBRATION  -- are the new error bars trustworthy? Held-out coverage of the 68%/95%
                     credible intervals should be ~68%/~95%, and RMS z-score ~1.
  3. INFORMATIVE  -- does the predicted sigma actually track error / distance-from-data?
                     (a constant sigma would be useless even if 'calibrated')
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

sys.path.insert(0, "experiments/datakit/mixture_features")
import featurize
import grug_fit as gf
from gp_surrogate import fit_gp, krr_predict, predict_gp
from retrodiction import _sq_hellinger

N_FOLDS = 5
SEED = 0


def load_grug():
    gf.phase_step_split = lambda: (38144, 9615)
    gf.TOTAL_STEPS = 47759
    hists, views, _cent, _rff, _rffo, _bt = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, _ = featurize.composition_matrix(hists, k=1000, views=views)
    train = pd.read_parquet("scratch/mixture_features/grug/train_runs.parquet")
    w = gf.weight_matrix(train, buckets)
    d2 = _sq_hellinger(gf.per_phase_hist(w, np.asarray(v1000, float)))

    with open("scratch/mixture_features/grug/target_candidates.json") as fh:
        tgt = json.load(fh)["recommended_target"]
    mu, sd, tasks = tgt["train_z_mu"], tgt["train_z_sd"], tgt["task_list"]

    def score(ej):
        ev = json.loads(ej)
        vals = [(ev[t]["bpb"] - mu[t]) / sd[t] for t in tasks if isinstance(ev.get(t), dict) and "bpb" in ev[t]]
        return float(np.mean(vals)) if vals else np.nan

    y = np.array([score(x) for x in train.evals])
    ok = np.isfinite(y)
    return d2[np.ix_(ok, ok)], y[ok]


def main():
    d2, y = load_grug()
    n = len(y)
    print(f"n = {n} runs\n")

    # ---- fit on all data: hyperparameters by marginal likelihood (the Bayesian way) ----
    fit = fit_gp(d2, y)
    print("=== 1. hyperparameters by MARGINAL LIKELIHOOD (evidence), not CV ===")
    print(f"  sigma_f^2 (signal var) = {fit['sigma_f2']:.4f}")
    print(f"  sigma_n^2 (noise var)  = {fit['sigma_n2']:.4f}   -> noise sd = {np.sqrt(fit['sigma_n2']):.4f} z")
    print(f"  gamma                  = {fit['gamma']:.4f}   (campaign frozen: {0.25/fit['median_d2']:.4f})")
    print(f"  ridge-equivalent alpha = {fit['ridge_equivalent_alpha']:.4f}   (campaign CV picked: 0.1)")
    print(f"  -log marginal lik      = {fit['nlml']:.2f}")

    # ---- 2. equivalence with kernel ridge ----
    gp_mu, _ = predict_gp(fit, d2)
    krr = krr_predict(d2, d2, y, fit["gamma"], fit["ridge_equivalent_alpha"])
    print("\n=== 2. EQUIVALENCE: GP posterior mean vs kernel-ridge prediction ===")
    print(f"  max |GP mean - KRR| = {np.abs(gp_mu - krr).max():.2e}")
    print(f"  correlation          = {np.corrcoef(gp_mu, krr)[0,1]:.6f}")

    # ---- 3. held-out calibration (the question that matters) ----
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    mus, sds, ys, nnd = [], [], [], []
    for tr, te in kf.split(np.arange(n)):
        f = fit_gp(d2[np.ix_(tr, tr)], y[tr])  # refit hyperparams inside the fold
        m, s = predict_gp(f, d2[np.ix_(te, tr)], include_noise=True)
        mus.append(m)
        sds.append(s)
        ys.append(y[te])
        nnd.append(d2[np.ix_(te, tr)].min(axis=1))  # distance to nearest training run
    mu_cv = np.concatenate(mus)
    sd_cv = np.concatenate(sds)
    y_cv = np.concatenate(ys)
    nn_cv = np.concatenate(nnd)

    z = (y_cv - mu_cv) / sd_cv
    rmse = float(np.sqrt(np.mean((y_cv - mu_cv) ** 2)))
    print("\n=== 3. CALIBRATION of the credible intervals (5-fold held-out) ===")
    print(f"  CV RMSE                     = {rmse:.4f} z")
    print(f"  RMS z-score                 = {np.sqrt((z**2).mean()):.3f}   (1.0 = perfectly calibrated)")
    print(f"  coverage of 68% interval    = {100*np.mean(np.abs(z) < 1.0):.1f}%   (nominal 68.3%)")
    print(f"  coverage of 95% interval    = {100*np.mean(np.abs(z) < 1.96):.1f}%   (nominal 95.0%)")
    print(f"  mean predicted sd           = {sd_cv.mean():.4f} z   (vs actual RMSE {rmse:.4f})")

    # ---- 4. is the uncertainty informative? ----
    print("\n=== 4. IS THE UNCERTAINTY INFORMATIVE? ===")
    print(f"  spearman(predicted sd, |error|)        = {spearmanr(sd_cv, np.abs(y_cv-mu_cv)).statistic:+.3f}")
    print(
        f"  spearman(predicted sd, dist-to-nearest)= {spearmanr(sd_cv, nn_cv).statistic:+.3f}"
        f"   (positive = more uncertain further from data)"
    )
    print(
        f"  predicted sd range                     = [{sd_cv.min():.4f}, {sd_cv.max():.4f}]"
        f"  (spread {sd_cv.max()/sd_cv.min():.2f}x)"
    )

    out = {
        "n": n,
        "hyperparameters": {k: fit[k] for k in ("sigma_f2", "sigma_n2", "gamma", "ridge_equivalent_alpha", "nlml")},
        "equivalence_max_abs_diff_vs_krr": float(np.abs(gp_mu - krr).max()),
        "cv_rmse": rmse,
        "rms_z": float(np.sqrt((z**2).mean())),
        "coverage_68": float(np.mean(np.abs(z) < 1.0)),
        "coverage_95": float(np.mean(np.abs(z) < 1.96)),
        "spearman_sd_abs_error": float(spearmanr(sd_cv, np.abs(y_cv - mu_cv)).statistic),
        "spearman_sd_nn_distance": float(spearmanr(sd_cv, nn_cv).statistic),
    }
    with open("scratch/mixture_features/grug/gp_surrogate_validation.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote scratch/mixture_features/grug/gp_surrogate_validation.json")


if __name__ == "__main__":
    main()
