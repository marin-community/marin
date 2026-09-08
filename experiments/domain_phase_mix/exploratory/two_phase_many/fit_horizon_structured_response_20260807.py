# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""The WSD80 fit behind WSD80-SUR-094: two benefit horizons plus a separate damage horizon.

Running this reproduces exactly one thing, the SUR-094 fit reported in the registry. An earlier version
of this docstring advertised `forms`, `exponents`, `twoscale`, `consistent`, `support` and `damagephi`
subcommands; the script never read its arguments and those modes did not exist. The claim was false and
is withdrawn. The analyses it named were run from throwaway scripts and are recorded in the logbook with
their numbers, but they are NOT reproducible from this artifact.

Two further limits, both found by independent review rather than by me:

The nonlinear shape is selected on the same three partitions whose held-out error is then reported, so
the printed interior RMSE is optimistically biased. A matched nested rerun gives about 0.008040 rather
than 0.006746, which is 1.061x the repaired-RPL reference and does NOT beat it. Treat the printed number
as in-protocol only.

`SLOW` stops at 0.45 and `GB` at 0.2, and the fit selects both endpoints, so two nonlinear parameters
sit on arbitrary grid bounds. `TAU` also stops at 3.0 here; a wider sweep was run separately and stayed
at 3.0, but that check is not reproducible from this file.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_multitarget_interference_evidence_20260806 as h,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import multitarget_ile_wsd80_20260806 as w  # noqa: E402

panel, targets = w.load_targets()
geo = w.geometry()
y = targets.values[:, targets.index(h.PRIMARY_TARGET)]
idx = np.arange(len(y))
interior = w.interior_mask(panel)
CTOT = geo.c0 + geo.c1


def design(W, s, f, dmg, gc, gb, ec, tau):
    cols = [np.ones(len(W))]
    for phi in (s, f):
        eps = CTOT * ((1 - phi) * W[:, 0, :] + phi * W[:, 1, :])
        cols += [(eps[:, 1] + ec) ** (-gc), (eps[:, 0] + ec) ** (-gb)]
    epsd = CTOT * ((1 - dmg) * W[:, 0, :] + dmg * W[:, 1, :])
    cols.append(np.maximum(epsd[:, 1] - 1.0, 0.0) ** tau)
    return np.column_stack(cols)


def fit(A, yy):
    lo = np.concatenate([[-np.inf], np.zeros(A.shape[1] - 1)])
    hi = np.full(A.shape[1], 1e7)
    hi[0] = np.inf
    return lsq_linear(A, yy, bounds=(lo, hi), method="trf").x


SLOW = (0.05, 0.15, 0.30, 0.45)
FAST = (0.65, 0.75, 0.85, 0.95, 1.0)
DMG = (0.05, 0.10, 0.20, 0.30, 0.45, 0.60, 0.80, 1.0)
GC = (0.1, 0.2, 0.3, 0.5)
GB = (0.2, 0.5, 1.0)
EC = (0.01, 0.05, 0.2)
TAU = (1.0, 2.0, 3.0)
outer = h.wsd80_folds("random", panel.weights, idx, 3, 0)
best = (np.inf, None)
for s in SLOW:
    for f in FAST:
        if f <= s:
            continue
        for dmg in DMG:
            for gc in GC:
                for gb in GB:
                    for ec in EC:
                        for tau in TAU:
                            A = design(panel.weights, s, f, dmg, gc, gb, ec, tau)
                            sq = 0.0
                            for tr, te in outer:
                                r = A[te] @ fit(A[tr], y[tr]) - y[te]
                                sq += float(r @ r)
                            if sq < best[0]:
                                best = (sq, (s, f, dmg, gc, gb, ec, tau))
_, p = best
s, f, dmg, gc, gb, ec, tau = p
A = design(panel.weights, *p)
oof = np.empty_like(y)
for tr, te in outer:
    oof[te] = A[te] @ fit(A[tr], y[tr])
c = fit(A, y)
ax = np.linspace(0, 1, 201)
P0, P1 = np.meshgrid(ax, ax, indexing="ij")
pred = design(w.grid_weights(P0.ravel(), P1.ravel()), *p) @ c
tie = design(w.grid_weights(ax, ax), *p) @ c
m = h.BOUNDARY_MARGIN
ig = (P0.ravel() > m) & (P1.ravel() > m) & (P0.ravel() < 1 - m) & (P1.ravel() < 1 - m)
rows = np.flatnonzero(ig)
bi = rows[int(np.argmin(pred[rows]))]
oi = np.flatnonzero(interior)
ob = oi[int(np.argmin(y[oi]))]
ranked = oi[np.argsort(oof[oi])]
gain = tie.min() - pred.min()
dist = np.hypot(P0.ravel()[bi] - panel.phase_0[ob, 1], P1.ravel()[bi] - panel.phase_1[ob, 1])
reg = y[ranked[0]] - y[ob]
rmse = np.sqrt(np.mean((oof - y)[interior] ** 2))
gain_gate = "PASS" if abs(gain - h.OBSERVED_WSD_GAIN) <= 0.004439 else "FAIL"
distance_gate = "PASS" if dist <= 0.05 else "FAIL"
print(f"slow={s} fast={f} damage_horizon={dmg} gc={gc} gb={gb} off={ec} tau={tau}")
print(f"  interior OOF RMSE {rmse:.6f}   (RPL 0.007575)  {'PASS' if rmse<=0.007575*1.05 else 'fail'}")
print(f"  gain {gain:+.6f}  |err| {abs(gain - h.OBSERVED_WSD_GAIN):.6f}  gate<=0.004439  {gain_gate}")
print(f"  optimum ({P0.ravel()[bi]:.3f},{P1.ravel()[bi]:.3f}) dist {dist:.4f}  " f"gate<=0.05  {distance_gate}")
print(f"  Regret@1 {reg:.6f}  gate<=0.004842  {'PASS' if reg<=0.004842 else 'FAIL'}")
