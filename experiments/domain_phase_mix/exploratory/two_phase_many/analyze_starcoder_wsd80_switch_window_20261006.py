# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Switch-window traces of the 7.41B confirmation pairs from W&B histories."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

W = Path(sys.argv[1])
BLOCKS = {"m100": ("c109", "c020"), "m200": ("c079", "c011"), "m400": ("c043", "c007")}
SEEDS = ["0821", "0822", "0823", "0824", "0825"]
BOUNDARY = 22608


def load_eval(name):
    df = pd.read_parquet(W / f"{name}.eval.parquet")
    keys = [c for c in df.columns if c.startswith("eval/")]
    df = df.dropna(subset=keys, how="all")
    return df.set_index("_step")


def metric_cols(df):
    pl = next(c for c in df.columns if "programing_languages" in c and c.endswith("/bpb"))
    c4 = next(c for c in df.columns if "c4_en" in c and c.endswith("/bpb"))
    return {"PL": pl, "C4": c4}


def ci(x):
    x = np.asarray(x)
    n = len(x)
    m = x.mean()
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


rows = []
for block, (tied, untied) in BLOCKS.items():
    ev_t = {s: load_eval(f"dsopt_r3d28260_{block}_tied_{tied}_s{s}") for s in SEEDS}
    ev_u = {s: load_eval(f"dsopt_r3d28260_{block}_untied_{untied}_s{s}") for s in SEEDS}
    cols = metric_cols(ev_t[SEEDS[0]])
    steps = sorted(set(ev_t[SEEDS[0]].index) & set(ev_u[SEEDS[0]].index))
    steps = [s for s in steps if s >= 19000]
    print(f"\n=== block {block}: tied {tied} vs untied {untied}; steps {steps}")
    for mname, col in cols.items():
        print(f"--- {mname} ({col})")
        print(f"{'step':>6} {'tied':>8} {'untied':>8} {'untied-tied':>12} {'95% CI':>20}")
        for st in steps:
            t = np.array([ev_t[s].loc[st, col] for s in SEEDS])
            u = np.array([ev_u[s].loc[st, col] for s in SEEDS])
            m, lo, hi = ci(u - t)
            print(f"{st:>6} {t.mean():8.4f} {u.mean():8.4f} {m:12.4f} [{lo:8.4f}, {hi:8.4f}]")
            rows.append(dict(block=block, metric=mname, step=st, tied=t.mean(), untied=u.mean(), diff=m, lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(W / "confirmation_switch_window.csv", index=False)

# train loss around the switch, seed-averaged, binned
print("\n=== train/loss, seed-mean, 64-step bins from 21,600 to 24,800 (bin label = first step)")


def tl(block, kind, coord):
    frames = []
    for s in SEEDS:
        d = pd.read_parquet(W / f"dsopt_r3d28260_{block}_{kind}_{coord}_s{s}.train.parquet")
        d = d.dropna(subset=["train/loss"]).set_index("_step")["train/loss"]
        frames.append(d)
    return pd.concat(frames, axis=1).mean(axis=1)


for block, (tied, untied) in BLOCKS.items():
    lt, lu = tl(block, "tied", tied), tl(block, "untied", untied)
    print(f"--- {block}")
    print(f"{'bin':>6} {'tied':>8} {'untied':>8} {'u-t':>8}   | 8-step bins after switch:")
    for b in range(21600, 24800, 64):
        a, c = lt.loc[b : b + 63].mean(), lu.loc[b : b + 63].mean()
        print(f"{b:>6} {a:8.4f} {c:8.4f} {c - a:8.4f}")
    print("  fine bins (8 steps) from 22,544 to 22,800:")
    for b in range(22544, 22800, 8):
        a, c = lt.loc[b : b + 7].mean(), lu.loc[b : b + 7].mean()
        print(f"  {b:>6} {a:8.4f} {c:8.4f} {c - a:8.4f}")
