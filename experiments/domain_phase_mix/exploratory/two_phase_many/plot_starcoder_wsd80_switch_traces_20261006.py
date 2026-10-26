# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Figure: what happens at an abrupt StarCoder switch (train loss, web and code evals)."""

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

S = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = (
    Path(__file__).resolve().parent
    / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9/trajectory_manifest.csv"
)
man = pd.read_csv(MANIFEST)
INK, GRID, PAPER = "#111111", "#b8b8b8", "white"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "text.usetex": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
    }
)
ARMS = [  # key in the double-difference tables, label, color
    ("p3_vs_p4_m100a", "0.02→0.82 at decay onset (P3 vs tied 0.18)", "#2a78d6"),
    ("conf_m200", "2x: 0.00→0.50 at decay onset", "#eb6834"),
    ("conf_m400", "4x: 0.00→0.28 at decay onset", "#1baf7a"),
    ("h5_0.60", "0.02→0.42 at 0.6T, constant LR", "#eda100"),
    ("h5_0.85", "0.12→0.52 at 0.85T, in decay", "#e87ba4"),
]
TRAIN_KEYS = {"p3_vs_p4_m100a": "p3_m100a"}  # train-loss series to draw for arms whose key is a comparison


def train_paths(key):
    key = TRAIN_KEYS.get(key, key)
    if key.startswith("conf_"):
        block = key.split("_")[1]
        coord = {"m100": "c020", "m200": "c011", "m400": "c007"}[block]
        seeds = ("0821", "0822", "0823", "0824", "0825")
        return [S / "wandb" / f"dsopt_r3d28260_{block}_untied_{coord}_s{s}.train.parquet" for s in seeds], 22608
    if key.startswith("p3_"):
        ids = man[(man.arm == "p3") & (man.support_id == key.split("_")[1])]
        return [S / "wandb_gcf" / f"{t}.train.parquet" for t in ids.trajectory_id], 22608
    beta = key.split("_")[1]
    role = f"boundary_beta_{beta.replace('.', 'p')}"
    ids = man[(man.arm == "b") & (man.policy_role == role)]
    return [S / "wandb_gcf" / f"{t}.train.parquet" for t in ids.trajectory_id], int(ids.boundary_step.iloc[0])


dd = pd.read_csv(S / "tables/eval_double_differences.csv")
if len(sys.argv) > 3:
    offline = pd.read_csv(sys.argv[3])
    offline = offline[offline.lag > 0].rename(columns={"ci": "double_diff_ci"})
    dd = pd.concat(
        [
            dd[["arm", "metric", "lag", "double_diff", "double_diff_ci"]],
            offline[["arm", "metric", "lag", "double_diff", "double_diff_ci"]],
        ]
    )
    dd = dd.drop_duplicates(["arm", "metric", "lag"], keep="last")
ENDPOINTS = Path(sys.argv[4]) if len(sys.argv) > 4 else None
fig, axes = plt.subplots(1, 4 if ENDPOINTS else 3, figsize=(9.6 if ENDPOINTS else 7.2, 2.6))
# A: train loss, 8-step bins
ax = axes[0]
for key, label, color in ARMS:
    paths, sw = train_paths(key)
    m = pd.concat(
        [pd.read_parquet(p).dropna(subset=["train/loss"]).set_index("_step")["train/loss"] for p in paths if p.exists()],
        axis=1,
    ).mean(axis=1)
    xs = np.arange(-128, 520, 8)
    ys = [m.loc[sw + x : sw + x + 7].mean() for x in xs]
    ax.plot(xs + 4, ys, color=color, lw=1.4, label=label)
ax.axvline(0, color=GRID, lw=0.8, ls="--")
ax.set_xlabel("steps since the switch", color=INK)
ax.set_ylabel("training loss, nats per token", color=INK)
ax.set_title("A  Training loss on the current mixture", loc="left", fontsize=7.5, fontweight="bold")
# B/C: double differences
for ax, metric, title in (
    (axes[1], "C4", "B  C4 English BPB gap vs tied, change"),
    (axes[2], "PL", "C  PL BPB gap vs tied, change"),
):
    for key, label, color in ARMS:
        sub = dd[(dd.arm == key) & (dd.metric == metric) & (dd.lag <= 4000)].sort_values("lag")
        ax.errorbar(
            sub.lag,
            sub.double_diff,
            yerr=sub.double_diff_ci,
            color=color,
            lw=1.2,
            marker="o",
            ms=3.5,
            capsize=1.5,
            label=label,
        )
    ax.axhline(0, color=GRID, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("steps since the switch", color=INK)
    ax.set_ylabel("change in the paired gap, BPB", color=INK)
    ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold")
if ENDPOINTS:
    ax = axes[3]
    from scipy import stats

    end = pd.read_csv(ENDPOINTS)
    end = end[end.status == "done"]
    baseline = (
        "minus_october_abrupt"
        if end.get("minus_october_abrupt", pd.Series(dtype=float)).notna().sum() >= 20
        else "minus_august_abrupt"
    )
    widths = {"abrupt": 0, "ramp0128": 128, "ramp0256": 256, "ramp0512": 512, "ramp2048": 2048, "ramp5632": 5632}
    xs, ys, los, his = [], [], [], []
    for arm, group in end.groupby("arm"):
        if arm == "abrupt":
            continue
        d = group[baseline].dropna().values
        h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        xs.append(widths[arm])
        ys.append(d.mean())
        los.append(h)
        his.append(h)
    positions = {0: 0, 128: 1, 256: 2, 512: 3, 2048: 4, 5632: 5}
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    ys = np.array(ys)[order]
    err = np.array(los)[order]
    px = [positions[int(x)] for x in xs]
    ax.errorbar(px, ys, yerr=err, color="#2a78d6", marker="o", ms=4, lw=1.4, capsize=2, zorder=3)
    for arm, group in end.groupby("arm"):
        if arm != "abrupt":
            ax.plot(
                np.full(len(group), positions[widths[arm]]) + 0.18,
                group[baseline],
                ls="none",
                marker="o",
                ms=2.2,
                color="#2a78d6",
                alpha=0.45,
            )
    canary = end[end.arm == "abrupt"]
    if len(canary):
        ax.plot(
            np.zeros(len(canary)),
            canary.minus_august_abrupt.values,
            marker="s",
            ms=3.5,
            color="#eb6834",
            ls="none",
            alpha=0.8,
        )
    ax.axhline(0, color=GRID, lw=0.8, ls="--")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels(["abrupt", "128", "256", "512", "2,048", "5,632"], fontsize=6.5)
    ax.set_xlim(-0.5, 5.6)
    ax.set_xlabel("ramp width, steps", color=INK)
    ax.set_ylabel(
        "final PL BPB, ramp minus abrupt"
        if baseline == "minus_october_abrupt"
        else "final PL BPB, ramp minus August abrupt",
        color=INK,
    )
    ax.set_title("D  Final PL BPB vs ramp width (4x)", loc="left", fontsize=7.5, fontweight="bold")

for ax in axes:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(w_pad=1.0, rect=(0, 0.1, 1, 1))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"transition_switch_traces.{ext}", dpi=300, bbox_inches="tight")
print("saved", OUT)
