# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two follow-ups on whether the Bayesian (GP) form beats kernel ridge in practice.

Since the GP posterior mean IS the kernel-ridge prediction, only the posterior variance can
discriminate. These are the two places it should matter most:

  PART 4 (de-confound). The earlier sequential study scored model quality on each strategy's
  own leftover pool, so greedy/EI were graded on the region they refused to sample. Here every
  strategy is scored on ONE FIXED held-out set that is never acquirable, so the comparison is
  apples-to-apples.

  PART 2 (batch). Swarms launch many runs in parallel, not one at a time. Picking the top-K by
  predicted mean -- the natural kernel-ridge move -- should return K near-duplicate mixtures,
  because nothing in the mean penalizes redundancy. The GP can enforce diversity: its posterior
  variance drops around an already-selected point WITHOUT needing that point's label, so a
  sequential variance-aware batch spreads out automatically. We measure batch diversity, the
  best run actually obtained, and the model improvement each batch buys.

Target convention: y is z-scored bpb, so LOWER IS BETTER (we minimize).
"""

import json

import numpy as np
from gp_inversion_study import load_all
from gp_surrogate import condition_gp, fit_gp, predict_gp
from scipy.stats import norm, spearmanr

N_HOLDOUT = 200
N_INIT = 30
N_MAX = 200
BATCH = 5
N_SEEDS = 3
REFIT_EVERY = 4

BATCH_SIZE = 50
BATCH_N_INIT = 60
BATCH_SEEDS = 5


def expected_improvement(mu, sd, best):
    """EI for MINIMIZATION."""
    z = (best - mu) / np.maximum(sd, 1e-12)
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def _cond(idx, d2, y, hyp):
    return condition_gp(d2[np.ix_(idx, idx)], y[idx], hyp["sigma_f2"], hyp["sigma_n2"], hyp["gamma"])


# ---------------------------------------------------------------- part 4: fixed held-out
def run_fixed_holdout(strategy, d2, y, rng, holdout):
    n = len(y)
    pool = np.setdiff1d(np.arange(n), holdout)
    obs = list(rng.choice(pool, N_INIT, replace=False))
    hyp = fit_gp(d2[np.ix_(obs, obs)], y[obs])
    fit = _cond(obs, d2, y, hyp)
    trace, round_i = [], 0
    while len(obs) < N_MAX:
        cand = np.setdiff1d(pool, obs)
        if strategy == "random":
            pick = rng.choice(cand, BATCH, replace=False)
        else:
            mu, sd = predict_gp(fit, d2[np.ix_(cand, obs)], include_noise=False)
            if strategy == "ei":
                score = -expected_improvement(mu, sd, y[obs].min())
            elif strategy == "max_var":
                score = -sd
            elif strategy == "greedy_mean":
                score = mu
            else:
                raise ValueError(strategy)
            pick = cand[np.argsort(score)[:BATCH]]
        obs.extend(pick.tolist())
        round_i += 1
        if round_i % REFIT_EVERY == 0:
            hyp = fit_gp(d2[np.ix_(obs, obs)], y[obs])
        fit = _cond(obs, d2, y, hyp)
        mu_h, _ = predict_gp(fit, d2[np.ix_(holdout, obs)], include_noise=False)
        trace.append(
            {
                "n_obs": len(obs),
                "best_found": float(y[obs].min()),
                "holdout_spearman": float(spearmanr(mu_h, y[holdout]).statistic),
                "holdout_rmse": float(np.sqrt(np.mean((mu_h - y[holdout]) ** 2))),
            }
        )
    return trace


def part4(d2, y):
    rng0 = np.random.default_rng(7)
    holdout = rng0.choice(len(y), N_HOLDOUT, replace=False)
    out = {}
    for s in ["ei", "max_var", "greedy_mean", "random"]:
        tr = [run_fixed_holdout(s, d2, y, np.random.default_rng(2000 + k), holdout) for k in range(N_SEEDS)]
        out[s] = {
            "n_obs": [p["n_obs"] for p in tr[0]],
            "best_found": np.mean([[p["best_found"] for p in t] for t in tr], axis=0).tolist(),
            "holdout_spearman": np.mean([[p["holdout_spearman"] for p in t] for t in tr], axis=0).tolist(),
            "holdout_rmse": np.mean([[p["holdout_rmse"] for p in t] for t in tr], axis=0).tolist(),
        }
    return out


# ---------------------------------------------------------------- part 2: batch selection
def select_batch(kind, d2, y, obs, cand, hyp, rng):
    if kind == "random":
        return rng.choice(cand, BATCH_SIZE, replace=False)
    if kind == "greedy_mean":  # the kernel-ridge move: just take the K best predicted
        fit = _cond(obs, d2, y, hyp)
        mu, _ = predict_gp(fit, d2[np.ix_(cand, obs)], include_noise=False)
        return cand[np.argsort(mu)[:BATCH_SIZE]]
    if kind == "gp_batch":  # variance-aware sequential batch (kriging-believer EI)
        chosen, s_idx, s_y = [], list(obs), list(y[obs])
        for _ in range(BATCH_SIZE):
            rem = np.setdiff1d(cand, chosen)
            f = condition_gp(d2[np.ix_(s_idx, s_idx)], np.array(s_y), hyp["sigma_f2"], hyp["sigma_n2"], hyp["gamma"])
            m, s = predict_gp(f, d2[np.ix_(rem, s_idx)], include_noise=False)
            pick = rem[int(np.argmax(expected_improvement(m, s, min(s_y))))]
            chosen.append(int(pick))
            s_idx.append(int(pick))
            s_y.append(float(m[list(rem).index(pick)]))  # fantasy label = posterior mean
        return np.array(chosen)
    raise ValueError(kind)


def part2(d2, y):
    rng0 = np.random.default_rng(11)
    holdout = rng0.choice(len(y), N_HOLDOUT, replace=False)
    pool_all = np.setdiff1d(np.arange(len(y)), holdout)
    out = {}
    for kind in ["gp_batch", "greedy_mean", "random"]:
        div, best, sp_gain = [], [], []
        for k in range(BATCH_SEEDS):
            rng = np.random.default_rng(3000 + k)
            obs = list(rng.choice(pool_all, BATCH_N_INIT, replace=False))
            hyp = fit_gp(d2[np.ix_(obs, obs)], y[obs])
            f0 = _cond(obs, d2, y, hyp)
            m0, _ = predict_gp(f0, d2[np.ix_(holdout, obs)], include_noise=False)
            sp_before = spearmanr(m0, y[holdout]).statistic
            cand = np.setdiff1d(pool_all, obs)
            b = select_batch(kind, d2, y, obs, cand, hyp, rng)
            iu = np.triu_indices(len(b), 1)
            div.append(float(d2[np.ix_(b, b)][iu].mean()))  # within-batch content spread
            best.append(float(y[b].min()))  # best run the batch actually got
            obs2 = obs + list(b)
            hyp2 = fit_gp(d2[np.ix_(obs2, obs2)], y[obs2])
            f1 = _cond(obs2, d2, y, hyp2)
            m1, _ = predict_gp(f1, d2[np.ix_(holdout, obs2)], include_noise=False)
            sp_gain.append(float(spearmanr(m1, y[holdout]).statistic - sp_before))
        out[kind] = {
            "batch_diversity_mean_pairwise_d2": float(np.mean(div)),
            "best_y_in_batch": float(np.mean(best)),
            "holdout_spearman_gain": float(np.mean(sp_gain)),
        }
    return out


def main():
    d2, y, _v, _w, _h = load_all()
    print(f"n = {len(y)} runs; target z-scored bpb (LOWER = better); fixed held-out = {N_HOLDOUT}\n")

    print("=== PART 4: sequential design scored on a FIXED held-out set (de-confounded) ===")
    p4 = part4(d2, y)
    grid = p4["ei"]["n_obs"]
    marks = [i for i, g in enumerate(grid) if g in (50, 100, 150, 200)]
    print(f"  {'n_runs':>7} | " + " | ".join(f"{s:>12}" for s in p4))
    print("  best TRUE y found (lower=better):")
    for i in marks:
        print(f"  {grid[i]:>7} | " + " | ".join(f"{p4[s]['best_found'][i]:>12.3f}" for s in p4))
    print("  held-out Spearman (FIXED set, higher=better):")
    for i in marks:
        print(f"  {grid[i]:>7} | " + " | ".join(f"{p4[s]['holdout_spearman'][i]:>12.3f}" for s in p4))

    print(f"\n=== PART 2: one parallel batch of {BATCH_SIZE} runs ===")
    p2 = part2(d2, y)
    print(f"  {'strategy':>12} | {'batch spread':>13} | {'best y in batch':>15} | {'holdout rho gain':>17}")
    for k, v in p2.items():
        print(
            f"  {k:>12} | {v['batch_diversity_mean_pairwise_d2']:>13.4f} | "
            f"{v['best_y_in_batch']:>15.3f} | {v['holdout_spearman_gain']:>17.4f}"
        )

    with open("scratch/mixture_features/grug/gp_design_study.json", "w") as fh:
        json.dump({"part4_fixed_holdout": p4, "part2_batch": p2}, fh, indent=1)
    print("\nwrote scratch/mixture_features/grug/gp_design_study.json")


if __name__ == "__main__":
    main()
