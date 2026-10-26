# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantify the switch transient: train-loss spike size/duration per seed and adaptation time constants."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

S = Path(sys.argv[1])
MANIFEST = (
    Path(__file__).resolve().parent
    / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9/trajectory_manifest.csv"
)
man = pd.read_csv(MANIFEST)


def ci(x):
    x = np.asarray(x, float)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def series(path):
    return pd.read_parquet(path).dropna(subset=["train/loss"]).set_index("_step")["train/loss"]


arms = {}
for block, coord in (("m100", "c020"), ("m200", "c011"), ("m400", "c007")):
    arms[f"conf_{block}_untied"] = (
        [
            S / "wandb" / f"dsopt_r3d28260_{block}_untied_{coord}_s{s}.train.parquet"
            for s in ("0821", "0822", "0823", "0824", "0825")
        ],
        22608,
    )
for block, coord in (("m100", "c109"), ("m200", "c079"), ("m400", "c043")):
    arms[f"conf_{block}_tied"] = (
        [
            S / "wandb" / f"dsopt_r3d28260_{block}_tied_{coord}_s{s}.train.parquet"
            for s in ("0821", "0822", "0823", "0824", "0825")
        ],
        22608,
    )
for role in man[man.arm == "b"].policy_role.unique():
    ids = man[(man.arm == "b") & (man.policy_role == role)]
    arms[f"h5_{role}"] = (
        [S / "wandb_gcf" / f"{t}.train.parquet" for t in ids.trajectory_id],
        int(ids.boundary_step.iloc[0]),
    )
for arm in ("p2", "p3", "p4"):
    for sup in ("m100a", "full"):
        ids = man[(man.arm == arm) & (man.support_id == sup)]
        arms[f"{arm}_{sup}"] = ([S / "wandb_gcf" / f"{t}.train.parquet" for t in ids.trajectory_id], 22608)

print(
    "=== per-seed train-loss jump at the switch: first 8 post-switch steps minus the 512 pre-switch steps; "
    "and steps until the 8-step running mean first falls below the pre-switch mean"
)
rows = []
for name, (paths, sw) in arms.items():
    jumps, durs, j64 = [], [], []
    for p in paths:
        if not p.exists():
            continue
        s = series(p)
        pre = s.loc[sw - 512 : sw - 1].mean()
        first = s.loc[sw : sw + 7].mean()
        jumps.append(first - pre)
        j64.append(s.loc[sw : sw + 63].mean() - pre)
        post = s.loc[sw : sw + 2047]
        run = post.rolling(8).mean()
        below = run[run < pre]
        durs.append(int(below.index[0] - sw + 1) if len(below) else 9999)
    if not jumps:
        continue
    m, h = ci(jumps)
    m64, h64 = ci(j64)
    print(
        f"{name:32s} n={len(jumps):2d} pre={pre:.3f} jump(0-7)={m:+.3f}±{h:.3f}  "
        f"jump(0-63)={m64:+.3f}±{h64:.3f}  steps-below-pre: median {int(np.median(durs))} max {max(durs)}"
    )
    rows.append(
        dict(
            arm=name,
            n=len(jumps),
            jump8=m,
            jump8_ci=h,
            jump64=m64,
            jump64_ci=h64,
            dur_median=np.median(durs),
            dur_max=max(durs),
        )
    )
pd.DataFrame(rows).to_csv(S / "train_loss_jump.csv", index=False)

print(
    "\n=== adaptation time constant of the seed-mean train loss after the switch: "
    "fit L(t)=L_inf + A exp(-(t-sw)/tau) + b (t-sw) on [sw+8, sw+2048]"
)


def fit_tau(mean_series, sw, span=2048):
    seg = mean_series.loc[sw + 8 : sw + span]
    t = (seg.index.values - sw).astype(float)
    y = seg.values

    def f(t, Linf, A, tau, b):
        return Linf + A * np.exp(-t / tau) + b * t

    p0 = [y[-1], y[0] - y[-1], 100.0, 0.0]
    popt, _ = optimize.curve_fit(f, t, y, p0=p0, maxfev=20000)
    return popt


for name, (paths, sw) in arms.items():
    frames = [series(p) for p in paths if p.exists()]
    if not frames:
        continue
    m = pd.concat(frames, axis=1).mean(axis=1)
    try:
        Linf, A, tau, b = fit_tau(m, sw)
        print(f"{name:32s} tau={tau:7.1f} steps  A={A:+.3f}  slope={b * 1000:+.4f}/1000 steps  L_inf={Linf:.3f}")
    except Exception as e:
        print(name, "fit failed", e)
# excess over the tied comparator (removes the shared decay): P3 - P2 and P3 - P4, confirmation untied - tied
print(
    "\n=== excess of the switched run over its tied comparator, seed-paired, "
    "64-step bins (first 8 bins) and tau of the excess"
)
pairs = [
    ("p3_m100a", "p2_m100a"),
    ("p3_m100a", "p4_m100a"),
    ("p3_full", "p2_full"),
    ("conf_m100_untied", "conf_m100_tied"),
    ("conf_m200_untied", "conf_m200_tied"),
    ("conf_m400_untied", "conf_m400_tied"),
]
for a, b in pairs:
    pa, sw = arms[a]
    pb, _ = arms[b]
    ma = pd.concat([series(p) for p in pa if p.exists()], axis=1).mean(axis=1)
    mb = pd.concat([series(p) for p in pb if p.exists()], axis=1).mean(axis=1)
    ex = (ma - mb).dropna()
    bins = [ex.loc[s : s + 63].mean() for s in range(sw, sw + 512, 64)]
    try:
        Linf, A, tau, slope = fit_tau(ex, sw, span=3000)
        print(
            f"{a}-{b}: "
            + " ".join(f"{v:+.3f}" for v in bins)
            + f" | tau={tau:.0f} A={A:+.3f} E_inf={Linf:+.3f} slope={slope * 1000:+.4f}/1000"
        )
    except Exception as e:
        print(a, b, "fit failed", e)
