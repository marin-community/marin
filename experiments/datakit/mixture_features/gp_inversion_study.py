# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inversion: use the surrogate backwards to CHOOSE mixtures, and ask whether the Bayesian
(GP) form buys anything over kernel ridge for the two practical jobs.

The GP posterior mean is identical to the kernel-ridge prediction, so anything that depends
only on the mean is a tie by construction. The question is whether the posterior *variance*
changes what you would actually do. Two jobs:

  A. FIND THE BEST MIXTURE GIVEN FIXED DATA.
     argmin(mean) is identical for KRR and GP. The GP can additionally pick by lower
     confidence bound (mean + kappa*sd, minimizing), which refuses to chase an optimistic
     mean out into unexplored space. We report where each pick lands and how far from data.

  B. DECIDE WHERE TO RUN MORE EXPERIMENTS.
     Retrospective sequential design over the 800 runs as a pool: start from a small random
     seed set, then repeatedly acquire more runs under each strategy and track
       (i)  best TRUE y found so far          -> the optimization job
       (ii) Spearman on the not-yet-acquired  -> the model-quality / space-sampling job
     Strategies: expected improvement and max-variance (need the GP's sd), greedy-mean (the
     KRR-style choice: just take the best predicted point), and random (what a Dirichlet
     swarm does today).

Caveat: this is a POOL-based replay over the 800 existing runs, not continuous optimization
over the simplex -- it measures which runs you would have chosen, which is the standard
offline way to compare acquisition functions.

Target convention: y is z-scored bpb, so LOWER IS BETTER (we minimize).
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, "experiments/datakit/mixture_features")
import featurize
import grug_fit as gf
from gp_surrogate import condition_gp, fit_gp, predict_gp
from retrodiction import _sq_hellinger

N_INIT = 30
N_MAX = 200
BATCH = 5
N_SEEDS = 3
REFIT_EVERY = 4  # refit hyperparameters every N acquisition rounds
KAPPA = 2.0  # lower-confidence-bound width for risk-adjusted selection


def _fit_on(idx, d2, y):
    return fit_gp(d2[np.ix_(idx, idx)], y[idx])


def _predict_on(fit, obs, cand, d2):
    return predict_gp(fit, d2[np.ix_(cand, obs)], include_noise=False)


def expected_improvement(mu, sd, best):
    """EI for MINIMIZATION: how much we expect to improve on the incumbent best (lowest) y."""
    z = (best - mu) / np.maximum(sd, 1e-12)
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def _condition_on(idx, d2, y, hyp):
    """Re-condition on the current observed set, reusing the last-fitted hyperparameters."""
    return condition_gp(d2[np.ix_(idx, idx)], y[idx], hyp["sigma_f2"], hyp["sigma_n2"], hyp["gamma"])


def run_strategy(strategy, d2, y, rng):
    n = len(y)
    obs = list(rng.choice(n, N_INIT, replace=False))
    hyp = _fit_on(obs, d2, y)
    fit = hyp
    trace = []
    round_i = 0
    while len(obs) < N_MAX:
        cand = np.setdiff1d(np.arange(n), obs)
        if strategy == "random":
            pick = rng.choice(cand, BATCH, replace=False)
        else:
            mu, sd = _predict_on(fit, obs, cand, d2)
            if strategy == "ei":
                score = -expected_improvement(mu, sd, y[obs].min())  # minimize -EI
            elif strategy == "max_var":
                score = -sd
            elif strategy == "greedy_mean":
                score = mu  # KRR-style: just take the best predicted mixture
            else:
                raise ValueError(strategy)
            pick = cand[np.argsort(score)[:BATCH]]
        obs.extend(pick.tolist())
        round_i += 1
        if round_i % REFIT_EVERY == 0:
            hyp = _fit_on(obs, d2, y)  # re-optimize hyperparameters occasionally (expensive)
        fit = _condition_on(obs, d2, y, hyp)  # always re-condition on the new data (cheap)
        rest = np.setdiff1d(np.arange(n), obs)
        mu_rest, _ = _predict_on(fit, obs, rest, d2)
        trace.append(
            {
                "n_obs": len(obs),
                "best_found": float(y[obs].min()),
                "pool_spearman": float(spearmanr(mu_rest, y[rest]).statistic),
            }
        )
    return trace


def part_b(d2, y):
    strategies = ["ei", "max_var", "greedy_mean", "random"]
    out = {s: {} for s in strategies}
    for s in strategies:
        traces = [run_strategy(s, d2, y, np.random.default_rng(1000 + k)) for k in range(N_SEEDS)]
        grid = [t["n_obs"] for t in traces[0]]
        out[s] = {
            "n_obs": grid,
            "best_found": np.mean([[p["best_found"] for p in t] for t in traces], axis=0).tolist(),
            "pool_spearman": np.mean([[p["pool_spearman"] for p in t] for t in traces], axis=0).tolist(),
        }
    return out


def part_a(d2, y, v_phase, rng):
    """Score a Dirichlet candidate bank; compare argmin(mean) vs argmin(LCB)."""
    fit = fit_gp(d2, y)
    # candidates drawn around the empirical design (concentration x mean training weights)
    mean_w = v_phase["w"].mean(axis=0)  # (2, n_buckets)
    cands = np.stack(
        [np.stack([rng.dirichlet(50 * np.maximum(mean_w[p], 1e-6)) for p in range(2)]) for _ in range(v_phase["n_cand"])]
    )  # (n_cand, 2, n_buckets)
    h_cand = np.einsum("cpb,kb->cpk", cands, v_phase["v"])  # (n_cand, 2, k); v is (k, buckets)
    h_tr = v_phase["h_train"]
    # mean per-phase squared Hellinger between candidates and training runs
    d2_star = np.zeros((len(cands), len(y)))
    for p in range(2):
        s_c, s_t = np.sqrt(np.clip(h_cand[:, p, :], 0, None)), np.sqrt(np.clip(h_tr[:, p, :], 0, None))
        d2_star += np.clip(1.0 - s_c @ s_t.T, 0.0, None)
    d2_star /= 2.0
    mu, sd = predict_gp(fit, d2_star, include_noise=False)
    ucb_pick = int(np.argmin(mu + KAPPA * sd))  # risk-adjusted (pessimistic) choice
    mean_pick = int(np.argmin(mu))
    return {
        "n_candidates": len(cands),
        "mean_pick": {
            "pred": float(mu[mean_pick]),
            "sd": float(sd[mean_pick]),
            "dist_to_nearest_run": float(d2_star[mean_pick].min()),
        },
        "lcb_pick": {
            "pred": float(mu[ucb_pick]),
            "sd": float(sd[ucb_pick]),
            "dist_to_nearest_run": float(d2_star[ucb_pick].min()),
        },
        "same_pick": mean_pick == ucb_pick,
        "best_observed_y": float(y.min()),
        "median_train_nn_dist": float(np.median(np.sort(d2, axis=1)[:, 1])),
    }


def load_all():
    """Load the grug design: pairwise content distance, target, and the content basis."""
    gf.phase_step_split = lambda: (38144, 9615)
    gf.TOTAL_STEPS = 47759
    hists, views, _c, _r, _ro, _bt = gf.load_grug_artifacts()
    buckets = [h.domain for h in hists]
    v1000, _ = featurize.composition_matrix(hists, k=1000, views=views)
    v = np.asarray(v1000, float)  # (k, n_buckets)
    train = pd.read_parquet("scratch/mixture_features/grug/train_runs.parquet")
    w = gf.weight_matrix(train, buckets)  # (n, 2, n_buckets)
    h_train = gf.per_phase_hist(w, v)  # (n, 2, k)
    d2 = _sq_hellinger(h_train)

    with open("scratch/mixture_features/grug/target_candidates.json") as fh:
        tgt = json.load(fh)["recommended_target"]
    mu, sd, tasks = tgt["train_z_mu"], tgt["train_z_sd"], tgt["task_list"]

    def score(ej):
        ev = json.loads(ej)
        vals = [(ev[t]["bpb"] - mu[t]) / sd[t] for t in tasks if isinstance(ev.get(t), dict) and "bpb" in ev[t]]
        return float(np.mean(vals)) if vals else np.nan

    y = np.array([score(x) for x in train.evals])
    ok = np.isfinite(y)
    return d2[np.ix_(ok, ok)], y[ok], v, w[ok], h_train[ok]


def main():
    d2, y, v, w, h_train = load_all()
    print(f"n = {len(y)} runs; target is z-scored bpb (LOWER = better)\n")

    print("=== A. INVERSION: find the best mixture from fixed data ===")
    a = part_a(d2, y, {"v": v, "w": w, "h_train": h_train, "n_cand": 4000}, np.random.default_rng(42))
    print(f"  candidates scored: {a['n_candidates']}   best OBSERVED run y = {a['best_observed_y']:+.3f}")
    print(
        f"  argmin(mean)  [KRR == GP]: pred {a['mean_pick']['pred']:+.3f}  sd {a['mean_pick']['sd']:.3f}"
        f"  dist-to-nearest-run {a['mean_pick']['dist_to_nearest_run']:.4f}"
    )
    print(
        f"  argmin(mean+{KAPPA}sd) [GP only]: pred {a['lcb_pick']['pred']:+.3f}  sd {a['lcb_pick']['sd']:.3f}"
        f"  dist-to-nearest-run {a['lcb_pick']['dist_to_nearest_run']:.4f}"
    )
    print(f"  same mixture chosen? {a['same_pick']}   (median train NN dist {a['median_train_nn_dist']:.4f})")

    print("\n=== B. SEQUENTIAL DESIGN: where to run the next experiments ===")
    b = part_b(d2, y)
    grid = b["ei"]["n_obs"]
    marks = [i for i, g in enumerate(grid) if g in (50, 100, 150, 200)]
    print(f"  {'n_runs':>7} | " + " | ".join(f"{s:>12}" for s in b))
    print("  best TRUE y found so far (lower=better):")
    for i in marks:
        print(f"  {grid[i]:>7} | " + " | ".join(f"{b[s]['best_found'][i]:>12.3f}" for s in b))
    print("  model quality: Spearman on the NOT-yet-acquired pool:")
    for i in marks:
        print(f"  {grid[i]:>7} | " + " | ".join(f"{b[s]['pool_spearman'][i]:>12.3f}" for s in b))

    with open("scratch/mixture_features/grug/gp_inversion_study.json", "w") as fh:
        json.dump({"part_a_inversion": a, "part_b_sequential": b}, fh, indent=1)
    print("\nwrote scratch/mixture_features/grug/gp_inversion_study.json")


if __name__ == "__main__":
    main()
