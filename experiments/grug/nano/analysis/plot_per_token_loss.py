# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot sorted + cumulative per-token loss for the p16 muon vs adamh dump.

Reads the parquets produced by ``experiments/grug/nano/scripts/per_token_loss.py``
and emits:

  1. ``p16_per_token_loss_overall.png`` — both models on one pair of axes
     (sorted curve + cumulative curve), pooled across all eval sets.
  2. ``p16_per_token_loss_overall_logy.png`` — sorted loss with log-y to
     expose the right-side tail.
  3. ``p16_per_token_loss_per_eval.png`` — 23-panel grid, one panel per
     eval set, showing both models' sorted loss curves on each.
  4. ``p16_per_token_loss_cumulative_per_eval.png`` — 23-panel grid of
     cumulative-loss curves per eval set.

Usage::

    uv run python experiments/grug/nano/analysis/plot_per_token_loss.py
"""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MUON_PARQUET = "gs://marin-us-east5/grug/pertoken-may7-nano-muon-tuned-p16-80857d/" "may7-nano-muon-tuned-p16.parquet"
ADAMH_PARQUET = (
    "gs://marin-us-east5/grug/pertoken-may7-nano-adamh-heuristic-p16-4a85ab/" "may7-nano-adamh-heuristic-p16.parquet"
)
VOCAB_SIZE = 128_256
UNIFORM_BASELINE = math.log(VOCAB_SIZE)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.read_parquet(MUON_PARQUET)
    a = pd.read_parquet(ADAMH_PARQUET)
    m = m[m.weight > 0]
    a = a[a.weight > 0]
    return m, a


def _plot_sorted(ax, m_loss: np.ndarray, a_loss: np.ndarray, title: str, *, log_y: bool = False) -> None:
    m_sorted = np.sort(m_loss)
    a_sorted = np.sort(a_loss)
    n = len(m_sorted)
    x = np.arange(n)
    ax.plot(x, m_sorted, label=f"muon (mean {m_sorted.mean():.3f})", color="C0", alpha=0.9, lw=1.0)
    ax.plot(x, a_sorted, label=f"adamh (mean {a_sorted.mean():.3f})", color="C1", alpha=0.9, lw=1.0)
    ax.axhline(
        UNIFORM_BASELINE, color="gray", ls="--", lw=0.7, label=f"uniform = ln({VOCAB_SIZE}) = {UNIFORM_BASELINE:.2f}"
    )
    ax.set_xlabel("token rank (sorted ascending per model)")
    ax.set_ylabel("loss (nats)")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("symlog", linthresh=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _plot_cumulative(ax, m_loss: np.ndarray, a_loss: np.ndarray, title: str) -> None:
    m_sorted = np.sort(m_loss)
    a_sorted = np.sort(a_loss)
    n = len(m_sorted)
    x = np.arange(n)
    m_cum = np.cumsum(m_sorted)
    a_cum = np.cumsum(a_sorted)
    ax.plot(x, m_cum, label=f"muon (total {m_cum[-1]:.0f} nats)", color="C0", lw=1.2)
    ax.plot(x, a_cum, label=f"adamh (total {a_cum[-1]:.0f} nats)", color="C1", lw=1.2)
    # Mark where the top-x% of tokens accounts for 50% / 90% of total loss.
    for cum, color in ((m_cum, "C0"), (a_cum, "C1")):
        total = cum[-1]
        for frac, ls in ((0.5, ":"), (0.9, "--")):
            idx = int(np.searchsorted(cum, total * (1 - frac)))
            ax.axvline(idx, color=color, ls=ls, lw=0.7, alpha=0.5)
    ax.set_xlabel("token rank")
    ax.set_ylabel("cumulative loss (nats)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def overall_plots(m: pd.DataFrame, a: pd.DataFrame) -> None:
    m_loss = m["loss"].values
    a_loss = a["loss"].values
    n = len(m_loss)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    _plot_sorted(axes[0], m_loss, a_loss, f"p16: sorted per-token loss (n={n:,} positions, all eval sets)")
    _plot_cumulative(axes[1], m_loss, a_loss, "p16: cumulative per-token loss")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "p16_per_token_loss_overall.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_sorted(ax, m_loss, a_loss, f"p16: sorted per-token loss, log-y (n={n:,})", log_y=True)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "p16_per_token_loss_overall_logy.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def per_eval_grid(m: pd.DataFrame, a: pd.DataFrame, *, kind: str) -> None:
    """``kind`` is 'sorted' or 'cumulative'."""
    eval_sets = sorted(m["eval_set"].unique())
    n_sets = len(eval_sets)
    cols = 4
    rows = (n_sets + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = axes.flatten()
    plotter = _plot_sorted if kind == "sorted" else _plot_cumulative
    for i, es in enumerate(eval_sets):
        ax = axes[i]
        m_loss = m[m.eval_set == es]["loss"].values
        a_loss = a[a.eval_set == es]["loss"].values
        title = f"{es}\n(n={len(m_loss):,})"
        plotter(ax, m_loss, a_loss, title)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("")  # too cluttered to repeat per panel
        ax.set_ylabel("")
        # Compact legend
        for ln in ax.get_legend().get_lines():
            ln.set_linewidth(1.5)
        ax.legend(fontsize=6, loc="lower right" if kind == "sorted" else "upper left")
    for j in range(n_sets, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"p16: {'sorted' if kind == 'sorted' else 'cumulative'} per-token loss, per eval set", fontsize=12, y=1.0
    )
    fig.tight_layout()
    suffix = "per_eval" if kind == "sorted" else "cumulative_per_eval"
    out = os.path.join(OUT_DIR, f"p16_per_token_loss_{suffix}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    m, a = _load()
    print(f"loaded: muon={len(m):,} rows, adamh={len(a):,} rows; eval_sets={m['eval_set'].nunique()}")
    overall_plots(m, a)
    per_eval_grid(m, a, kind="sorted")
    per_eval_grid(m, a, kind="cumulative")
