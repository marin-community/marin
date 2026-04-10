# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize text reference sweep hparams vs eval/loss with SHAP feature importance.

Usage:
    uv run --with lightgbm --with shap --with pandas --with matplotlib \
        experiments/dna/exp109_bolinas_scaling_analysis/text_sweep_analysis.py [--refresh]

Caches wandb data locally; pass --refresh to re-fetch.
"""

import json
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from experiments.dna.exp109_bolinas_scaling_analysis.reference_hparams import shap_analysis, smooth_main_effect

WANDB_PROJECT = "marin-community/marin"
WANDB_FILTER = {"display_name": {"$regex": "mega-sweep.*v3"}}
CACHE_PATH = Path("/tmp/text_sweep_mega_v3.json")
OUT_DIR = Path("experiments/dna/exp109_bolinas_scaling_analysis/results/text")

METRIC = "eval/uncheatable_eval/macro_loss"
SEQ_LEN = 4096
HYPERS = ["lr", "adam_lr", "beta1", "beta2", "epsilon", "max_grad_norm", "z_loss_weight"]
LOG_HYPERS = {"lr", "adam_lr", "epsilon", "z_loss_weight"}


def _safe_float(x) -> float:
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except Exception:
        return np.nan


def fetch_data(project: str, filters: dict) -> list[dict]:
    import wandb

    api = wandb.Api(timeout=300)
    runs = list(api.runs(path=project, filters=filters))
    rows = []
    for r in runs:
        c = dict(r.config or {})
        s = r.summary or {}
        tr = c.get("trainer", {}) or {}
        mo = c.get("model", {}) or {}
        opt = c.get("optimizer", {}) or {}

        train_steps = _safe_float(tr.get("num_train_steps"))
        train_bs = _safe_float(tr.get("train_batch_size"))
        tokens = (train_steps * train_bs * SEQ_LEN) if np.isfinite(train_steps) and np.isfinite(train_bs) else np.nan

        rows.append(
            {
                "id": r.id,
                "name": r.name,
                "display_name": getattr(r, "display_name", None),
                "group": getattr(r, "group", None),
                "state": r.state,
                "_step": _safe_float(s.get("_step")),
                "train_length": train_steps,
                "batch_size": train_bs,
                "hidden_dim": _safe_float(mo.get("hidden_dim")),
                "seq_len": SEQ_LEN,
                "tokens": tokens,
                METRIC: _safe_float(s.get(METRIC)),
                "lr": _safe_float(opt.get("learning_rate")),
                "adam_lr": _safe_float(opt.get("adam_lr")),
                "beta1": _safe_float(opt.get("beta1")),
                "beta2": _safe_float(opt.get("beta2")),
                "epsilon": _safe_float(opt.get("epsilon")),
                "max_grad_norm": _safe_float(opt.get("max_grad_norm")),
                "z_loss_weight": _safe_float(c.get("z_loss_weight")),
            }
        )
    return rows


def load_data(refresh: bool = False) -> pd.DataFrame:
    if not refresh and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} runs from cache ({CACHE_PATH})")
    else:
        data = fetch_data(WANDB_PROJECT, WANDB_FILTER)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fetched {len(data)} runs, cached to {CACHE_PATH}")
    df = pd.DataFrame(data)
    df[METRIC] = pd.to_numeric(df[METRIC], errors="coerce")
    return df


def filter_slice(
    df: pd.DataFrame,
    bs: int = 32,
    hid: int = 512,
    token_budget: int = 5_000_000_000,
    rel_tol: float = 0.02,
) -> pd.DataFrame:
    d = df[(df["state"] == "finished") & df[METRIC].notna()].copy()
    need = ["batch_size", "hidden_dim", "train_length", "tokens", *HYPERS]
    d = d.dropna(subset=need).copy()
    d["batch_size"] = d["batch_size"].round().astype(int)
    d["hidden_dim"] = d["hidden_dim"].round().astype(int)

    for h in LOG_HYPERS:
        d = d[d[h] > 0].copy()

    # Token budget bucketing
    token_targets = np.array([1e9, 2.5e9, 5e9, float(token_budget)])
    token_targets = np.unique(token_targets)
    toks = d["tokens"].to_numpy(float)
    nearest_idx = np.abs(toks[:, None] - token_targets[None, :]).argmin(axis=1)
    nearest_target = token_targets[nearest_idx]
    rel_err = np.abs(toks - nearest_target) / nearest_target
    d["token_budget"] = nearest_target.astype(np.int64)
    d["token_rel_err"] = rel_err
    d = d[(d["token_budget"] == token_budget) & (d["token_rel_err"] <= rel_tol)].copy()

    g = d[(d["batch_size"] == bs) & (d["hidden_dim"] == hid)].copy()
    print(f"bs={bs}, hid={hid}, token_budget={token_budget:,}, rel_tol={rel_tol}, n={len(g)}")
    if len(g) < 2:
        raise SystemExit("Not enough runs in this slice.")
    return g


def plot(g: pd.DataFrame, bs: int, hid: int, token_budget: int) -> None:
    X = g[HYPERS]
    y = g[METRIC].values

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, min_child_samples=2, random_state=42, verbose=-1
    ).fit(X, y)
    sa = shap_analysis(model, X)

    # Sort by importance ascending for horizontal bar chart
    sorted_importance = sa.importance.sort_values()
    sorted_colors = plt.cm.tab10([HYPERS.index(h) for h in sorted_importance.index])

    # Layout: SHAP bar on top, scatter grid below
    scatter_rows = (len(HYPERS) + 3) // 4
    fig = plt.figure(figsize=(12, 7.5))
    gs = GridSpec(
        1 + scatter_rows,
        4,
        figure=fig,
        height_ratios=[1] + [2] * scatter_rows,
        top=0.92,
        bottom=0.06,
        hspace=0.3,
        wspace=0.08,
    )

    # SHAP importance bar chart
    ax_imp = fig.add_subplot(gs[0, :])
    bars = ax_imp.barh(
        sorted_importance.index, sorted_importance.values, color=sorted_colors, edgecolor="k", linewidth=0.5
    )
    ax_imp.set_xlabel("SHAP importance (% mean |SHAP|)", fontsize=9)
    ax_imp.tick_params(labelsize=8)
    for bar, val in zip(bars, sorted_importance.values, strict=True):
        ax_imp.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7)

    # Hparam scatter plots with SHAP main-effect line
    clip_y = float(np.percentile(y, 95) * 1.02)
    ylim = (y.min() - 0.002, clip_y + 0.004)
    colors = plt.cm.tab10(np.arange(len(HYPERS)))

    for i, hparam in enumerate(HYPERS):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        xs = g[hparam].values

        normal = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv <= clip_y]
        clipped_pts = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv > clip_y]

        if normal:
            ax.scatter(*zip(*normal, strict=True), alpha=0.5, s=20, edgecolors="k", linewidths=0.2, color=colors[i])
        if clipped_pts:
            cx, _cy = zip(*clipped_pts, strict=True)
            ax.scatter(
                cx, [clip_y] * len(cx), alpha=0.5, s=20, edgecolors="k", linewidths=0.2, color=colors[i], marker="^"
            )

        # Smoothed SHAP main-effect curve
        me = sa.main_effects[hparam].values
        curve = smooth_main_effect(xs, me)
        ax.plot(curve.index, curve.values, color="k", linewidth=1.5, alpha=0.8, zorder=4)

        ax.set_xlabel(hparam, fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        if hparam in LOG_HYPERS:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(METRIC, fontsize=9)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Text reference sweep — hparam vs {METRIC}\n" f"bs={bs}, hidden_dim={hid}, tokens={token_budget:,}, n={len(g)}",
        fontsize=11,
        y=0.98,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUT_DIR / "reference_sweep_hparams.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(plot_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved to {plot_path}")


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    bs, hid, token_budget = 32, 512, 5_000_000_000
    df = load_data(refresh=refresh)
    g = filter_slice(df, bs=bs, hid=hid, token_budget=token_budget)
    plot(g, bs=bs, hid=hid, token_budget=token_budget)
