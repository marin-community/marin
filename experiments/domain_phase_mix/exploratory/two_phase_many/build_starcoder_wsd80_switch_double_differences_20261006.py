# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble the eval double-difference table (switched minus tied comparator, relative to the last pre-switch eval)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

S = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
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


def load_eval(path):
    df = pd.read_parquet(path)
    keys = [c for c in df.columns if c.startswith("eval/")]
    return df.dropna(subset=keys, how="all").set_index("_step")


rows = []


def add(arm, label, p0, p1, lr_state, switch, u, t, seeds):
    """u, t: dict seed -> eval frame (switched, tied comparator)."""
    PL = next(c for c in next(iter(u.values())).columns if "programing_languages" in c and c.endswith("/bpb"))
    C4 = next(c for c in next(iter(u.values())).columns if "c4_en" in c and c.endswith("/bpb"))
    steps = sorted(set(next(iter(u.values())).index))
    pre = max(s for s in steps if s <= switch)
    for metric, col in (("PL", PL), ("C4", C4)):
        for st in [s for s in steps if s > switch]:
            dd = [(u[s].loc[st, col] - t[s].loc[st, col]) - (u[s].loc[pre, col] - t[s].loc[pre, col]) for s in seeds]
            within = [u[s].loc[st, col] - u[s].loc[pre, col] for s in seeds]
            m, h = ci(dd)
            mw, hw = ci(within)
            rows.append(
                dict(
                    arm=arm,
                    label=label,
                    p0=p0,
                    p1=p1,
                    contrast=p1 - p0,
                    lr_state=lr_state,
                    switch_step=switch,
                    pre_step=pre,
                    step=st,
                    lag=st - switch,
                    metric=metric,
                    n=len(seeds),
                    double_diff=m,
                    double_diff_ci=h,
                    within_change=mw,
                    within_change_ci=hw,
                    switched_mean=np.mean([u[s].loc[st, col] for s in seeds]),
                    tied_mean=np.mean([t[s].loc[st, col] for s in seeds]),
                )
            )


# confirmation pairs
SEEDS = ["0821", "0822", "0823", "0824", "0825"]
for block, (tied, untied, p0, p1) in {
    "m100": ("c109", "c020", 0.01008, 0.85968),
    "m200": ("c079", "c011", 0.0, 0.5),
    "m400": ("c043", "c007", 0.0, 0.2814080848021789),
}.items():
    u = {s: load_eval(S / "wandb" / f"dsopt_r3d28260_{block}_untied_{untied}_s{s}.eval.parquet") for s in SEEDS}
    t = {s: load_eval(S / "wandb" / f"dsopt_r3d28260_{block}_tied_{tied}_s{s}.eval.parquet") for s in SEEDS}
    add(f"conf_{block}", f"{block} {p0:.2f}->{p1:.2f} vs tied", p0, p1, "decay onset", 22608, u, t, SEEDS)
# gradient-conflict evals (persisted)
ev = pd.read_parquet(S / "gcf_evals/gcf_v9_evals.parquet").merge(
    man[
        [
            "trajectory_id",
            "arm",
            "policy_role",
            "support_id",
            "boundary_step",
            "phase_0_starcoder",
            "phase_1_starcoder",
            "training_seed",
        ]
    ],
    on="trajectory_id",
)


def frames(sel):
    out = {}
    for _tid, g in sel.groupby("trajectory_id"):
        g = g.sort_values("step").drop_duplicates("step", keep="last")
        out[int(g.training_seed.iloc[0]) % 100] = g.set_index("step")
    return out


h5 = ev[ev.arm == "b"]
tied = frames(h5[h5.policy_role == "boundary_tied_018"])
for role, g in h5[h5.policy_role != "boundary_tied_018"].groupby("policy_role"):
    u = frames(g)
    seeds = sorted(set(u) & set(tied))
    sw = int(g.boundary_step.iloc[0])
    p0 = float(g.phase_0_starcoder.iloc[0])
    p1 = float(g.phase_1_starcoder.iloc[0])
    beta = role.split("_")[-1].replace("0p", "0.")
    lr = "constant LR" if sw < 22528 else ("decay onset" if sw == 22528 else "in decay")
    add(f"h5_{beta}", f"H5 beta {beta}: {p0:.2f}->{p1:.2f} vs tied 0.18", p0, p1, lr, sw, u, tied, seeds)
for sup in ("m100a", "full"):
    p3 = frames(ev[(ev.arm == "p3") & (ev.support_id == sup)])
    for comp in ("p2", "p4"):
        t = frames(ev[(ev.arm == comp) & (ev.support_id == sup)])
        seeds = sorted(set(p3) & set(t))
        add(
            f"p3_vs_{comp}_{sup}",
            f"P3 {sup} 0.02->0.82 vs {'tied 0.70' if comp == 'p2' else 'tied 0.18'}",
            0.02,
            0.82,
            "decay onset",
            22608,
            p3,
            t,
            seeds,
        )
df = pd.DataFrame(rows)
df.to_csv(OUT / "eval_double_differences.csv", index=False)
print(
    df[df.lag <= 1500]
    .pivot_table(index=["arm", "lag"], columns="metric", values=["double_diff", "double_diff_ci", "within_change"])
    .round(4)
    .to_string()
)
