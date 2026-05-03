# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize the 3 canonical 1e20 midtraining runs.

Produces six PNG plots (saved next to this script):

1. ``train_loss_vs_step.png`` — train/loss_smooth vs step. Shape check: warmup
   → peak → decay.
2. ``train_loss_vs_cumlr.png`` — train/loss_smooth vs cumulative LR. If the
   schedule-aware framing is right-ish, the three LR-factor curves should come
   closer to overlaying on this axis than on raw step.
3. ``paloma_c4_en_vs_step.png`` — retention metric over training. Should rise;
   higher LR factor → more retention damage.
4. ``predictor_fit_overlay.png`` — for lr=0.67 train/loss_smooth, overlay:
   raw data, target window, and the predictor fits (B1 sqrt-t,
   B2 profiled-c, B2r log-prior, B3 c=1) at prefix=50%.
5. ``c_profile_scan.png`` — validation/objective curves over c so the
   identifiability of the exponent is visible.
6. ``predictor_method_mae.png`` — MAE comparison from the generated CSVs.

On macOS the script will ``open`` the output directory automatically.
Run: ``uv run python scripts/analysis/plot_midtrain_curves.py``.
"""

import logging
import os
import platform
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the loaders + fit functions from the predictor module.
from midtrain_loss_predictor import (
    RUNS_1E20,
    TARGET_WINDOW,
    TOTAL_STEPS,
    WARMUP_STEPS,
    c_profile_records,
    compute_cumulative_lr,
    cumlr_fit_data,
    fit_cumlr_fixed_c,
    fit_cumlr_power,
    fit_cumlr_regularized_c,
    fit_sqrt_t,
    load_run,
    normalized_remaining_lr,
    power_prediction,
    smooth_train_loss,
)

logger = logging.getLogger(__name__)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

COLORS = {
    "lr=0.5": "#1f77b4",
    "lr=0.67": "#2ca02c",
    "lr=0.83": "#d62728",
}


def plot_train_loss_vs_step(runs_data, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for spec, df in runs_data:
        ax.plot(df["_step"], df["train/loss_smooth"], color=COLORS[spec.name], lw=1.5, label=spec.name)
    ax.axvline(WARMUP_STEPS, color="gray", ls="--", lw=0.8, alpha=0.6, label="warmup end")
    ax.axvspan(*TARGET_WINDOW, color="orange", alpha=0.12, label="target window")
    ax.set_xlabel("training step")
    ax.set_ylabel("train/loss (EMA halflife=100)")
    ax.set_title("Delphi 1e20 midtraining — train/loss vs step")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_train_loss_vs_cumlr(runs_data, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for spec, df in runs_data:
        u = compute_cumulative_lr(df)
        # Left: raw u axis (shows that LR factor scales U ∝ lr_factor)
        axes[0].plot(u, df["train/loss_smooth"], color=COLORS[spec.name], lw=1.5, label=spec.name)
        # Right: normalized u / U (collapse attempt)
        U = float(u.iloc[-1])
        axes[1].plot(u / U, df["train/loss_smooth"], color=COLORS[spec.name], lw=1.5, label=spec.name)

    axes[0].set_xlabel("cumulative learning_rate u(t)")
    axes[0].set_ylabel("train/loss (smoothed)")
    axes[0].set_title("vs raw cumulative LR (U scales with lr_factor)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("normalized cumulative LR  u / U")
    axes[1].set_ylabel("train/loss (smoothed)")
    axes[1].set_title("vs u/U — if the curves collapse, schedule-shape dominates")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("Does cumulative-LR collapse the 3 factors?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_paloma_vs_step(runs_data, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for spec, df in runs_data:
        sub = df[df["eval/paloma/c4_en/loss"].notna()]
        ax.plot(
            sub["_step"],
            sub["eval/paloma/c4_en/loss"],
            color=COLORS[spec.name],
            marker="o",
            lw=1.5,
            ms=4,
            label=spec.name,
        )
    ax.axhline(2.8586, color="black", ls=":", lw=0.8, alpha=0.7, label="pretrain c4_en (2.8586)")
    ax.axvline(WARMUP_STEPS, color="gray", ls="--", lw=0.8, alpha=0.6, label="warmup end")
    ax.set_xlabel("training step")
    ax.set_ylabel("eval/paloma/c4_en/loss")
    ax.set_title("Delphi 1e20 midtraining — retention metric (higher = more damage)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_predictor_overlay(runs_data, out_path: str, run_name: str = "lr=0.67", prefix_frac: float = 0.5) -> None:
    """Show the fit lines superimposed on the data for one (run, prefix) case."""
    match = [(s, d) for s, d in runs_data if s.name == run_name]
    if not match:
        raise RuntimeError(f"run {run_name!r} not found")
    _, df = match[0]
    metric = "train/loss_smooth"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: vs step (for B1 + B3 c=1 in step space)
    ax = axes[0]
    ax.plot(df["_step"], df[metric], color="#444", lw=1.2, label="data (smoothed)")
    ax.axvline(prefix_frac * TOTAL_STEPS, color="gray", ls="--", lw=0.8, label=f"prefix={prefix_frac}")
    ax.axvline(WARMUP_STEPS, color="red", ls=":", lw=0.8, alpha=0.6, label="warmup end")
    ax.axvspan(*TARGET_WINDOW, color="orange", alpha=0.12, label="target window")

    # B1 fit
    pred_b1, (a, b) = fit_sqrt_t(df, prefix_frac, metric)
    t_plot = np.arange(WARMUP_STEPS, TOTAL_STEPS + 1)
    ax.plot(t_plot, a + b / np.sqrt(t_plot), color="#ff7f0e", lw=2, ls="--", label=f"B1: a+b/√t (pred {pred_b1:.3f})")
    ax.scatter([TOTAL_STEPS], [pred_b1], color="#ff7f0e", s=60, zorder=5)

    ax.set_xlabel("step")
    ax.set_ylabel("train/loss (smoothed)")
    ax.set_title(f"{run_name}: raw-step view (B1 fit at prefix={prefix_frac})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Right: vs cumulative LR for B2 + B3 c=1 (schedule-aware)
    ax = axes[1]
    u_full = compute_cumulative_lr(df)
    U = float(u_full.iloc[-1])
    ax.plot(u_full, df[metric], color="#444", lw=1.2, label="data (smoothed)")
    cutoff_u = u_full[df["_step"] <= prefix_frac * TOTAL_STEPS].iloc[-1]
    ax.axvline(cutoff_u, color="gray", ls="--", lw=0.8, label=f"prefix={prefix_frac}")
    warmup_u = u_full[df["_step"] <= WARMUP_STEPS].iloc[-1]
    ax.axvline(warmup_u, color="red", ls=":", lw=0.8, alpha=0.6, label="warmup end")

    # B2: profiled c, selected by prefix-tail validation
    pred_b2, (L2, A2, c2, fit_start_u2, rmse2) = fit_cumlr_power(df, prefix_frac, metric)
    u_plot = np.linspace(warmup_u, U, 500)
    x_plot2 = normalized_remaining_lr(u_plot, U, fit_start_u2)
    ax.plot(
        u_plot,
        power_prediction(x_plot2, L2, A2, c2),
        color="#2ca02c",
        lw=2,
        ls="--",
        label=f"B2 profiled: c={c2:.2f}, val RMSE={rmse2:.3g}, pred {pred_b2:.3f}",
    )
    # B2r: profiled c with weak prior centered at c=1
    pred_b2r, (L2r, A2r, c2r, fit_start_u2r, rmse2r) = fit_cumlr_regularized_c(df, prefix_frac, metric)
    x_plot2r = normalized_remaining_lr(u_plot, U, fit_start_u2r)
    ax.plot(
        u_plot,
        power_prediction(x_plot2r, L2r, A2r, c2r),
        color="#17becf",
        lw=2,
        ls=(0, (3, 1, 1, 1)),
        label=f"B2r log-prior: c={c2r:.2f}, val RMSE={rmse2r:.3g}, pred {pred_b2r:.3f}",
    )
    # B3: c=1
    pred_b3, (L3, A3, c3, fit_start_u3) = fit_cumlr_fixed_c(df, prefix_frac, metric, c_fixed=1.0)
    x_plot3 = normalized_remaining_lr(u_plot, U, fit_start_u3)
    ax.plot(
        u_plot,
        power_prediction(x_plot3, L3, A3, c3),
        color="#9467bd",
        lw=2,
        ls="-.",
        label=f"B3 fixed: c=1, pred {pred_b3:.3f}",
    )

    # Actual target
    target = df[(df["_step"] >= TARGET_WINDOW[0]) & (df["_step"] <= TARGET_WINDOW[1])][metric].mean()
    ax.axhline(target, color="black", ls=":", lw=1.0, alpha=0.7, label=f"actual final {target:.3f}")

    ax.set_xlabel("cumulative learning_rate  u(t)")
    ax.set_ylabel("train/loss (smoothed)")
    ax.set_title(f"{run_name}: schedule-aware view (B2, B3 fits at prefix={prefix_frac})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_c_profile_scan(runs_data, out_path: str, prefix_frac: float = 0.5) -> None:
    """Show whether c is identified by prefix-tail validation."""
    metric = "train/loss_smooth"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for spec, df in runs_data:
        x, y, _, _ = cumlr_fit_data(df, prefix_frac, metric)

        raw = c_profile_records(x, y)
        raw_best = raw.loc[raw["objective"].idxmin()]
        axes[0].plot(raw["c"], raw["validation_rmse"], color=COLORS[spec.name], lw=1.6, label=spec.name)
        axes[0].scatter([raw_best["c"]], [raw_best["validation_rmse"]], color=COLORS[spec.name], s=35, zorder=5)

        regularized = c_profile_records(x, y, prior_c=1.0)
        reg_best = regularized.loc[regularized["objective"].idxmin()]
        axes[1].plot(
            regularized["c"],
            np.sqrt(regularized["objective"]),
            color=COLORS[spec.name],
            lw=1.6,
            label=spec.name,
        )
        axes[1].scatter([reg_best["c"]], [np.sqrt(reg_best["objective"])], color=COLORS[spec.name], s=35, zorder=5)

    axes[0].axvline(1.0, color="black", ls=":", lw=1.0, alpha=0.7, label="c=1 prior")
    axes[1].axvline(1.0, color="black", ls=":", lw=1.0, alpha=0.7, label="c=1 prior")
    axes[0].set_title("B2: prefix-tail validation RMSE")
    axes[1].set_title("B2r: sqrt(validation MSE + log-c prior)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("c")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("train/loss_smooth error at held-out prefix tail")
    fig.suptitle(f"Profiled exponent scan, prefix={prefix_frac}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_method_mae(out_path: str) -> None:
    """Bar chart from ``midtrain_loss_predictor_self_prefix.csv``."""
    csv_path = os.path.join(OUT_DIR, "midtrain_loss_predictor_self_prefix.csv")
    if not os.path.exists(csv_path):
        raise RuntimeError(f"{csv_path} missing; run midtrain_loss_predictor.py first")

    df = pd.read_csv(csv_path)
    df = df[df["metric"] == "train/loss_smooth"]
    methods = [
        "B1_sqrt_t",
        "B2_profiled_c",
        "B2r_profiled_c_logprior1",
        "B3_cumlr_c=1",
        "B3_cumlr_c=0.5",
    ]
    summary = df[df["method"].isin(methods)].groupby(["prefix", "method"])["abs_err"].mean().reset_index()
    pivot = summary.pivot(index="prefix", columns="method", values="abs_err").reindex(columns=methods)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.14
    for i, method in enumerate(methods):
        ax.bar(x + (i - 2) * width, pivot[method], width=width, label=method)
    ax.axhspan(0.005, 0.010, color="green", alpha=0.12, label="noise floor")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0%}" for p in pivot.index])
    ax.set_xlabel("prefix")
    ax.set_ylabel("self-prefix MAE on train/loss_smooth")
    ax.set_title("Predictor MAE after replacing bounded nonlinear B2")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Loading %d runs from W&B...", len(RUNS_1E20))
    runs_data = []
    for spec in RUNS_1E20:
        df = load_run(spec)
        df = smooth_train_loss(df)
        runs_data.append((spec, df))
    logger.info("Loaded. Generating plots...")

    outputs = [
        ("train_loss_vs_step.png", plot_train_loss_vs_step),
        ("train_loss_vs_cumlr.png", plot_train_loss_vs_cumlr),
        ("paloma_c4_en_vs_step.png", plot_paloma_vs_step),
        ("predictor_fit_overlay.png", lambda rd, p: plot_predictor_overlay(rd, p, "lr=0.67", 0.5)),
        ("c_profile_scan.png", lambda rd, p: plot_c_profile_scan(rd, p, 0.5)),
    ]
    out_paths = []
    for fname, fn in outputs:
        p = os.path.join(OUT_DIR, fname)
        fn(runs_data, p)
        out_paths.append(p)
        logger.info("  %s", p)

    mae_path = os.path.join(OUT_DIR, "predictor_method_mae.png")
    plot_method_mae(mae_path)
    out_paths.append(mae_path)
    logger.info("  %s", mae_path)

    if platform.system() == "Darwin":
        logger.info("Opening plots...")
        subprocess.run(["open", *out_paths], check=False)
    else:
        logger.info("Non-macOS: open the files manually from %s", OUT_DIR)


if __name__ == "__main__":
    main()
