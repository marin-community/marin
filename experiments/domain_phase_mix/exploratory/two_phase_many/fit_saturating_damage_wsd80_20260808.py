# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""WSD80-SUR-104: SUR-102's channels with repetition damage that saturates, shared shape, multi-start.

Registered before fitting with the prediction that saturating damage must raise the predicted late code
share toward the observed 0.500, because every damage term in this project grew without bound and
unbounded damage over-penalises exactly the heavy-late-code region. The prediction held.

Reaches 23 of 24 gate-passes over six seeds: optimum distance 6/6, Regret@1 6/6, gain error 6/6, interior
OOF RMSE 5/6. NOT promotable -- the registered seed-0 protocol fails RMSE at 0.008229 against 0.007954.

Two method notes, both of which changed the answer. The sign-constrained solve runs on scaled columns
because the repetition column spans fourteen orders of magnitude with the exponent free; unscaled it
crashed the solver outright on one fold draw. And the inner-fold objective is bimodal in the saturation
scale, so selection refines from starts spanning that parameter's whole bounded range. That multi-start
was justified BEFORE it was applied, by comparing the two basins on the inner-fold objective with no gate
consulted: the identified basin scores lower in both affected seeds, so the optimiser had simply missed
it. Pass seeds as arguments; default 0-4.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import differential_evolution, minimize  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as h,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_gated_absorption_wsd80_20260807 as M,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import multitarget_ile_wsd80_20260806 as wsd  # noqa: E402

P = M.PANEL
W = P.weights
I = M.INTERIOR
Y = M.TARGETS.values
names = list(M.TARGETS.names)
prim = names.index(h.PRIMARY_TARGET)
y = Y[:, prim]
J = Y.shape[1]
BROAD, CODE = 0, 1
# Last entry is the log10 saturation scale; a large value is the old unbounded damage form.
BOUNDS = (*M.BOUNDS, (-1.0, 8.0))


def design(weights, v):
    near, gb, gc, lo, tau, pd, kappa, beta, nu, le, kb, ls = v
    off = 10.0**lo
    eps = 10.0**le
    s = 10.0**ls

    def ex(domain, horizon):
        return M.EPOCHS[domain] * ((1 - horizon) * weights[:, 0, domain] + horizon * weights[:, 1, domain])

    excess = np.maximum(ex(CODE, pd) - 1.0, 0.0) ** tau
    # saturating: reproduces the unbounded form for small excess, levels off for large
    damage = -np.expm1(-excess / s)
    free = np.column_stack([np.ones(len(weights)), weights[:, 1, BROAD]])
    cons = np.column_stack(
        [
            (ex(CODE, near) + off) ** -gc,
            (ex(BROAD, near) + off) ** -gb,
            (ex(CODE, 1.0) + off) ** -gc,
            (ex(BROAD, 1.0) + off) ** -gb,
            (M.absorbed(weights, CODE, kappa, beta) + off) ** -gc,
            (M.absorbed(weights, BROAD, kb, beta) + off) ** -gb,
            (ex(CODE, 0.0) + eps) ** -nu,
            damage,
        ]
    )
    return free, cons


def select(rows, seed):
    folds = h.wsd80_folds("random", W[rows], np.arange(len(rows)), M.N_INNER_FOLDS, seed)
    sub = Y[rows]
    subI = I[rows]

    def cv(v):
        F, C = design(W[rows], v)
        if not (np.isfinite(F).all() and np.isfinite(C).all()):
            return 1e6
        t = 0.0
        for a_, b_ in folds:
            sc = np.maximum(sub[a_].std(axis=0), 1e-9)
            for k in range(J):
                bb, aa = M.fit_head(F[a_], C[a_], sub[a_, k])
                r = (F[b_] @ bb + C[b_] @ aa - sub[b_, k])[subI[b_]]
                if len(r):
                    t += float(r @ r) / (sc[k] ** 2)
        return t

    best = differential_evolution(
        cv, BOUNDS, rng=np.random.default_rng(20260808), popsize=12, maxiter=70, tol=1e-11, polish=True, init="sobol"
    ).x
    # The objective is bimodal in the saturation scale, so refine from starts spanning that parameter's
    # WHOLE range and keep whichever scores best on the inner-fold objective. Selection never sees a gate,
    # and the starts are fixed by the bounds rather than by any value known to work.
    fbest = cv(best)
    for start in np.linspace(BOUNDS[-1][0], BOUNDS[-1][1], 5):
        v0 = best.copy()
        v0[-1] = start
        r = minimize(cv, v0, bounds=list(BOUNDS), method="L-BFGS-B", options={"maxiter": 300, "eps": 1e-6})
        if r.fun < fbest:
            best, fbest = r.x, float(r.fun)
    return best


def ok(passed):
    return "P" if passed else "F"


ir = np.flatnonzero(I)
ob = int(ir[np.argmin(y[ir])])
t0, t1 = P.phase_0[ob, 1], P.phase_1[ob, 1]
N = 801
ax = np.linspace(0, 1, N)
G0, G1 = np.meshgrid(ax, ax, indexing="ij")
Wg = wsd.grid_weights(G0.ravel(), G1.ravel())
ta = np.linspace(0, 1, N * N // 4)
Wt = wsd.grid_weights(ta, ta)
print("SUR-104 saturating damage. PRE-STATED: late share should rise from 0.449-0.451 toward 0.500")
print("gates RMSE<=0.007954 | R@1<=0.004842 | dist<=0.05 | |gerr|<=0.004439\n")
tot = 0
for seed in [int(x) for x in sys.argv[1:] or range(5)]:
    outer = h.wsd80_folds("random", W, np.arange(len(y)), M.N_FOLDS, seed)
    pred = np.empty_like(y)
    for tr, te in outer:
        v = select(tr, seed)
        F, C = design(W, v)
        bb, aa = M.fit_head(F[tr], C[tr], y[tr])
        pred[te] = F[te] @ bb + C[te] @ aa
    rm = float(np.sqrt(np.mean((pred - y)[I] ** 2)))
    rk = ir[np.argsort(pred[ir])]
    r1 = float(y[rk[0]] - y[ob])
    r5 = float(y[rk[:5]].min() - y[ob])
    v = select(np.arange(len(y)), seed)
    F, C = design(W, v)
    bb, aa = M.fit_head(F, C, y)
    Fg, Cg = design(Wg, v)
    Ft, Ct = design(Wt, v)
    pr = Fg @ bb + Cg @ aa
    ti = Ft @ bb + Ct @ aa
    i = int(np.argmin(pr))
    o0, o1 = G0.ravel()[i], G1.ravel()[i]
    d = float(np.hypot(o0 - t0, o1 - t1))
    ge = abs(float(ti.min() - pr.min()) - 0.009594)
    n = sum([rm <= 0.007954, r1 <= 0.004842, d <= 0.05, ge <= 0.004439])
    tot += n
    print(
        f"seed {seed}: RMSE {rm:.6f}{ok(rm<=0.007954)} R@1 {r1:.6f}{ok(r1<=0.004842)} dist {d:.6f}{ok(d<=0.05)} "
        f"gerr {ge:.6f}{ok(ge<=0.004439)} [{n}/4] opt ({o0:.3f},{o1:.3f}) R@5 {r5:.6f} sat {10**v[11]:.1f}"
    )
print(f"\n{tot} gate-passes")
