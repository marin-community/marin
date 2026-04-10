# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize reference sweep hparams vs eval/loss with SHAP feature importance.

Usage:
    uv run --with lightgbm --with shap --with pandas \
        experiments/dna/exp109_bolinas_scaling_analysis/reference_sweep_hparams.py [--refresh]

Caches wandb data locally; pass --refresh to re-fetch.
"""

import json
import math
import sys
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.gridspec import GridSpec


@dataclass(frozen=True)
class ShapAnalysis:
    """Results of SHAP analysis on a fitted model."""

    shap_values: pd.DataFrame
    importance: pd.Series
    main_effects: pd.DataFrame
    base_value: float


def shap_analysis(model: lgb.LGBMRegressor, X: pd.DataFrame) -> ShapAnalysis:
    """Compute SHAP values, importance, and main effects from a fitted model.

    Args:
        model: A fitted LGBMRegressor.
        X: Feature matrix with named columns. Index is preserved in outputs.
    """
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X.values)
    base = float(explainer.expected_value)

    shap_df = pd.DataFrame(sv, columns=X.columns, index=X.index)
    assert shap_df.index.equals(X.index)

    raw_importance = np.abs(sv).mean(axis=0)
    importance = pd.Series(
        raw_importance / raw_importance.sum() * 100,
        index=X.columns,
    ).sort_values(ascending=False)

    main_effects = shap_df.add(base)

    return ShapAnalysis(
        shap_values=shap_df,
        importance=importance,
        main_effects=main_effects,
        base_value=base,
    )


def smooth_main_effect(
    feature_values: np.ndarray,
    main_effect: np.ndarray,
    window_frac: float = 0.2,
) -> pd.Series:
    """Centered rolling mean of SHAP main-effect values, sorted by feature."""
    s = pd.Series(main_effect, index=feature_values).sort_index()
    window = max(int(len(s) * window_frac) | 1, 3)
    return s.rolling(window, center=True, min_periods=1).mean()


VERSION = "v0.6"
WANDB_PROJECT = "eric-czech/marin"
WANDB_GROUP = f"dna-bolinas-reference-sweep-{VERSION}"
CACHE_PATH = Path(f"/tmp/sweep_{VERSION}_finished.json")
RESULTS_DIR = Path(f"experiments/dna/exp109_bolinas_scaling_analysis/results/reference/{VERSION}")
PLOT_PATH = RESULTS_DIR / "reference_sweep_hparams.png"
CSV_PATH = RESULTS_DIR / "reference_sweep_data.csv"
SUMMARY_PATH = RESULTS_DIR / "reference_sweep_summary.txt"

HPARAM_KEYS = ["initializer_range", "lr", "adam_lr", "beta1", "beta2", "eps", "mgn", "zloss"]
HPARAM_LABELS = ["init_range", "lr", "adam_lr", "beta1", "beta2", "epsilon", "max_grad_norm", "z_loss"]
LOG_SCALE = {"initializer_range", "lr", "adam_lr", "eps", "zloss"}
CLIP_Y = 1.29


def fetch_data(project: str, group: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(project, filters={"group": group})
    data = []
    for r in runs:
        if r.state != "finished":
            continue
        hparams = {}
        for tag in r.tags:
            if "=" in tag:
                k, v = tag.split("=", 1)
                try:
                    hparams[k] = float(v)
                except ValueError:
                    pass
        loop = next((int(t.split("=")[1]) for t in r.tags if t.startswith("loop=")), None)
        data.append(
            {
                "eval/loss": r.summary.get("eval/loss"),
                "eval/macro_loss": r.summary.get("eval/macro_loss"),
                "name": r.name,
                "completed_at": r.heartbeatAt,
                "loop": loop,
                **hparams,
            }
        )
    return data


def load_data(refresh: bool = False) -> pd.DataFrame:
    if not refresh and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} runs from cache ({CACHE_PATH})")
    else:
        data = fetch_data(WANDB_PROJECT, WANDB_GROUP)
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Fetched {len(data)} finished runs, cached to {CACHE_PATH}")
    df = pd.DataFrame(data)
    df["eval/loss"] = pd.to_numeric(df["eval/loss"], errors="coerce")
    df["eval/macro_loss"] = pd.to_numeric(df["eval/macro_loss"], errors="coerce")
    return df


def plot(df: pd.DataFrame, metric: str = "eval/loss") -> None:
    n_total = len(df)
    valid = df[df[metric].apply(lambda v: isinstance(v, float) and math.isfinite(v))]
    n_feasible = len(valid)

    X = valid[HPARAM_KEYS]
    y = valid[metric].values
    ylim = (y.min() - 0.002, CLIP_Y + 0.004)

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, min_child_samples=2, random_state=42, verbose=-1
    ).fit(X, y)
    sa = shap_analysis(model, X)

    # Map internal keys to display labels for importance bar chart
    key_to_label = dict(zip(HPARAM_KEYS, HPARAM_LABELS, strict=True))
    sorted_importance = sa.importance.sort_values()
    sorted_labels = [key_to_label[k] for k in sorted_importance.index]
    sorted_colors = plt.cm.tab10([HPARAM_KEYS.index(k) for k in sorted_importance.index])

    fig = plt.figure(figsize=(12, 10))
    gs_top = GridSpec(1, 4, figure=fig, top=0.95, bottom=0.75, wspace=0.08)
    gs_bot = GridSpec(3, 4, figure=fig, height_ratios=[1.5, 2, 2], top=0.67, bottom=0.05, hspace=0.3, wspace=0.08)

    # --- Timeline: eval/loss vs completion time, colored by loop ---
    ax_time = fig.add_subplot(gs_top[0, :])
    times = [datetime.fromisoformat(d.replace("Z", "+00:00")) for d in valid["completed_at"]]
    loops = valid["loop"].fillna(0).astype(int).values
    max_loop = int(loops.max()) if len(loops) else 0
    cmap = plt.cm.viridis

    for t, yv, loop in zip(times, y, loops, strict=True):
        clipped = yv > CLIP_Y
        plot_y = CLIP_Y if clipped else yv
        c = cmap(loop / max(max_loop, 1))
        marker = "^" if clipped else "o"
        ax_time.scatter(t, plot_y, c=[c], s=25, edgecolors="k", linewidths=0.3, marker=marker, alpha=0.7, zorder=3)
        if clipped:
            ax_time.annotate(
                f"{yv:.2f}", (t, CLIP_Y), textcoords="offset points", xytext=(0, -8), ha="center", fontsize=5
            )

    # Lower envelope (cumulative min over time)
    sorted_idx = sorted(range(len(times)), key=lambda i: times[i])
    env_times, env_losses = [], []
    cur_min = float("inf")
    for i in sorted_idx:
        if y[i] < cur_min:
            cur_min = y[i]
            env_times.append(times[i])
            env_losses.append(cur_min)
    if env_times and times[sorted_idx[-1]] > env_times[-1]:
        env_times.append(times[sorted_idx[-1]])
        env_losses.append(cur_min)
    ax_time.step(env_times, env_losses, where="post", color="red", linewidth=1.5, alpha=0.8, zorder=4)

    ax_time.set_ylim(ylim)
    ax_time.set_ylabel(metric, fontsize=9)
    ax_time.set_xlabel("Completion time", fontsize=9)
    ax_time.tick_params(labelsize=7)
    for label in ax_time.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")

    # Colorbar for loop
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_loop))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_time, pad=0.02, aspect=20, fraction=0.03)
    cbar.set_label("Loop", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --- SHAP importance bar chart ---
    ax_imp = fig.add_subplot(gs_bot[0, :])
    bars = ax_imp.barh(sorted_labels, sorted_importance.values, color=sorted_colors, edgecolor="k", linewidth=0.5)
    ax_imp.set_xlabel("SHAP importance (% mean |SHAP|)", fontsize=9)
    ax_imp.tick_params(labelsize=8)
    for bar, val in zip(bars, sorted_importance.values, strict=True):
        ax_imp.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7)

    # --- Hparam scatter plots ---
    colors = plt.cm.tab10(np.arange(len(HPARAM_LABELS)))
    for i, (hparam, label) in enumerate(zip(HPARAM_KEYS, HPARAM_LABELS, strict=True)):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs_bot[row, col])
        xs = valid[hparam].values
        normal = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv <= CLIP_Y]
        clipped_pts = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv > CLIP_Y]
        if normal:
            ax.scatter(*zip(*normal, strict=True), alpha=0.7, s=25, edgecolors="k", linewidths=0.3, color=colors[i])
        if clipped_pts:
            cx, _cy = zip(*clipped_pts, strict=True)
            ax.scatter(
                cx, [CLIP_Y] * len(cx), alpha=0.7, s=25, edgecolors="k", linewidths=0.3, color=colors[i], marker="^"
            )
            for xv, yv in clipped_pts:
                ax.annotate(
                    f"{yv:.2f}", (xv, CLIP_Y), textcoords="offset points", xytext=(0, -8), ha="center", fontsize=5
                )
        # Smoothed SHAP main-effect curve
        me = sa.main_effects[hparam].values
        curve = smooth_main_effect(xs, me)
        ax.plot(curve.index, curve.values, color="k", linewidth=1.5, alpha=0.8, zorder=4)

        ax.set_xlabel(label, fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        if hparam in LOG_SCALE:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(metric, fontsize=9)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Bolinas DNA reference sweep\n{n_feasible} feasible / {n_total} finished runs — metric: {metric}",
        fontsize=11,
        y=1.01,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    fig.savefig(PLOT_PATH.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved to {PLOT_PATH}")


def export_csv(df: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(df)} rows to {CSV_PATH}")


def summarize(df: pd.DataFrame) -> None:
    lines = ["Reference sweep summary", "=" * 40, ""]

    # Overall counts and best loss by initializer_range
    overall = (
        df.groupby("initializer_range")
        .agg(total=("name", "size"), feasible=("eval/loss", "count"), best_loss=("eval/loss", "min"))
        .sort_index()
    )
    lines.append("Runs by initializer_range:")
    lines.append(overall.to_string())
    lines.append("")

    # Counts by initializer_range x loop
    pivot = df.pivot_table(index="initializer_range", columns="loop", values="name", aggfunc="count", fill_value=0)
    pivot.columns = [f"loop_{int(c)}" for c in pivot.columns]
    lines.append("Runs by initializer_range x loop:")
    lines.append(pivot.to_string())
    lines.append("")

    # NaN loss breakdown by initializer_range x loop
    nan_df = df[df["eval/loss"].isna()]
    if len(nan_df) > 0:
        nan_pivot = nan_df.pivot_table(
            index="initializer_range", columns="loop", values="name", aggfunc="count", fill_value=0
        )
        nan_pivot.columns = [f"loop_{int(c)}" for c in nan_pivot.columns]
        nan_pivot["total_nan"] = nan_pivot.sum(axis=1)
        lines.append(f"NaN loss runs ({len(nan_df)} total) by initializer_range x loop:")
        lines.append(nan_pivot.to_string())
    else:
        lines.append("No NaN loss runs.")
    lines.append("")

    report = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(report)
    print(f"Saved summary to {SUMMARY_PATH}")
    print(report)


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    df = load_data(refresh=refresh)
    export_csv(df)
    summarize(df)
    plot(df)
