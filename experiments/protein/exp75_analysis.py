# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""exp75 results analysis: fetch W&B runs -> CSV, plot final val loss vs completion time.

Two-step so plotting never needs the network: ``--refresh`` pulls the sweep's runs
from W&B into a per-version CSV; without it, the cached CSV is read and only the
figure is redrawn.

Usage::

    # refresh the CSV from W&B, then (re)draw:
    uv run --with wandb --with pandas --with matplotlib \\
        python -m experiments.protein.exp75_analysis --refresh
    # redraw from the cached CSV (no W&B call):
    uv run --with wandb --with pandas --with matplotlib \\
        python -m experiments.protein.exp75_analysis [--version v1]

Outputs (gitignored) under ``experiments/protein/exp75_results/<version>/``:

    exp75_runs.csv               every run of the sweep version, any state (one row each)
    baseline.json                snapshot of the 1.5B unmasked baseline run's current
                                 ``eval/contacts-v1-val/loss`` and run_progress (that run is
                                 still training, so the value/progress move until it finishes)
    loss_vs_completion.{png,svg} final ``eval/contacts-v1-val/loss`` vs run completion
                                 time, colored by epoch, with the best-so-far frontier
                                 traced and the 1.5B baseline drawn as a horizontal line.
                                 COMPLETED runs only (may loosen later).
    exp75_curves.csv             per-run eval-loss-vs-tokens history (one row per eval point)
    loss_curves_by_epoch.{png,svg} 2x2 facets by epoch count of eval loss vs tokens for
                                 every run; best run per facet highlighted, baseline hline.
    best_loss_vs_epochs.{png,svg} best final loss reached at each epoch count, x on a log2
                                 scale, baseline hline.
    lr_wd_grid.csv               long format (epochs, lr, wd, loss) for finished runs
    lr_wd_heatmap.{png,svg}      2x2 LR x WD heatmaps of final val loss, one per epoch count
    contact_metrics_vs_loss.{png,svg} 1x3 scatter of contact AUC and R-precision (all/long) vs
                                 eval loss for the per-epoch winners (static CONTACT_METRICS table)

The CSV carries ``run_progress``, ``tokens_exact``, ``params_exact``, ``epochs``,
``lr`` and ``wd`` (epochs/lr/wd/*_exact lifted from W&B tags) so other analyses can
reuse it without re-querying.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ENTITY = "eric-czech"
PROJECT = "marin"
GROUP = "exp75-contacts-v1-tune"
VAL_LOSS_KEY = "eval/contacts-v1-val/loss"
TOKENS_KEY = "throughput/total_tokens"  # cumulative tokens, logged on every eval row
RESULTS_DIR = Path(__file__).parent / "exp75_results"

# 1.5B unmasked run on the same eval — the baseline exp75 is trying to beat. Lives in a
# different entity/project; we snapshot its current val loss into baseline.json.
BASELINE_RUN = "open-athena/MarinFold/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2"

# Tag keys lifted into their own CSV columns (see exp75_sweep.build_trial tags).
TAG_KEYS = ["epochs", "lr", "wd", "params_exact", "tokens_exact", "tpu", "band"]

# Stable, colorblind-friendly color per epoch rung.
EPOCH_COLORS = {1: "#4C72B0", 2: "#DD8452", 4: "#55A868", 8: "#C44E52"}

# Val-loss values above this are uninteresting (mostly early/in-progress runs); the y-axis
# tops out here and any higher point is pinned to the line and drawn as an up-triangle (^)
# to signal "actually higher", so a few bad runs don't compress the scale for everything else.
CLIP_HIGH = 3.18

# Same idea for the per-token curves, but a looser cap (early high-LR evals spike past 3.4);
# clipped points are pinned to the cap and marked with an up-triangle.
CURVE_CLIP_HIGH = 3.4

# Contact-prediction metrics for the per-epoch sweep winners, scored through the MarinFold
# harness (single realization, no ensembling). Static table from Open-Athena/MarinFold#89
# (comment 4793283235); `loss` is that epoch's best eval/contacts-v1-val/loss. AUC is the
# long-range variant; R-precision is reported all / long.
CONTACT_METRICS = [
    {"ckpt": "E1", "epochs": 1, "loss": 3.046, "auc_long": 0.615, "rprec_all": 0.028, "rprec_long": 0.022},
    {"ckpt": "E2", "epochs": 2, "loss": 2.942, "auc_long": 0.623, "rprec_all": 0.029, "rprec_long": 0.024},
    {"ckpt": "E4", "epochs": 4, "loss": 2.924, "auc_long": 0.620, "rprec_all": 0.031, "rprec_long": 0.023},
    {"ckpt": "E8", "epochs": 8, "loss": 2.757, "auc_long": 0.881, "rprec_all": 0.339, "rprec_long": 0.269},
]


def _scatter_clipped(ax, sub, *, size, zorder, alpha=1.0, color, label=None):
    """Scatter ``sub`` (rows with completed_dt/val_loss), pinning val_loss > CLIP_HIGH to the
    clip line as up-triangles. The circle layer always carries ``label`` so the legend entry
    survives even when every point of a group is clipped."""
    below = sub[sub["val_loss"] <= CLIP_HIGH]
    above = sub[sub["val_loss"] > CLIP_HIGH]
    ax.scatter(
        below["completed_dt"],
        below["val_loss"],
        s=size,
        color=color,
        edgecolor="none",
        alpha=alpha,
        zorder=zorder,
        label=label,
    )
    if not above.empty:
        ax.scatter(
            above["completed_dt"],
            [CLIP_HIGH] * len(above),
            s=size * 1.25,
            marker="^",
            color=color,
            edgecolor="none",
            alpha=alpha,
            zorder=zorder,
        )


def _iso_to_unix(iso: str | None) -> float | None:
    if not iso:
        return None
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).replace(tzinfo=timezone.utc).timestamp()


def _unix_to_iso(unix: float | None) -> str | None:
    if not unix:
        return None
    return datetime.fromtimestamp(unix, timezone.utc).isoformat()


def fetch(version: str) -> pd.DataFrame:
    """Pull every run of the given sweep version from W&B into a DataFrame."""
    import wandb  # noqa: PLC0415 -- lazy: only --refresh needs W&B; plotting reads the CSV

    api = wandb.Api()
    suffix = f"-{version}"
    rows = []
    for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if not r.name.endswith(suffix):  # partition by sweep version (trial-name suffix)
            continue
        tags = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in r.tags or [] if "=" in x}
        s = r.summary
        row = {
            "name": r.name,
            "state": r.state,
            "val_loss": s.get(VAL_LOSS_KEY),
            "run_progress": s.get("run_progress"),
            "step": s.get("_step"),
            "created_at": _unix_to_iso(_iso_to_unix(r.created_at)),
            "completed_at": _unix_to_iso(s.get("_timestamp")),  # last heartbeat = completion for finished runs
            "runtime_s": s.get("_runtime"),
        }
        for k in TAG_KEYS:
            row[k] = tags.get(k)
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in ["epochs", "params_exact", "tokens_exact", "step"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    for c in ["lr", "wd", "val_loss", "run_progress", "runtime_s"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def load_or_refresh(version: str, refresh: bool) -> pd.DataFrame:
    csv = RESULTS_DIR / version / "exp75_runs.csv"
    if refresh or not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = fetch(version)
        df.to_csv(csv, index=False)
        print(f"fetched {len(df)} runs -> {csv}")
    else:
        df = pd.read_csv(csv)
        print(f"read {len(df)} runs <- {csv} (pass --refresh to re-fetch)")
    return df


def fetch_curves(version: str) -> pd.DataFrame:
    """Per-run eval-loss-vs-tokens histories for the sweep version (one row per eval point).

    Runs that never logged an eval (crashed/failed early) contribute nothing.
    """
    import wandb  # noqa: PLC0415 -- lazy: only --refresh needs W&B; plotting reads the CSV

    api = wandb.Api()
    suffix = f"-{version}"
    rows = []
    for r in api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP}):
        if not r.name.endswith(suffix):
            continue
        h = r.history(keys=[VAL_LOSS_KEY, TOKENS_KEY], pandas=True)
        if VAL_LOSS_KEY not in h.columns:
            continue
        h = h.dropna(subset=[VAL_LOSS_KEY])
        if h.empty:
            continue
        tags = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in r.tags or [] if "=" in x}
        for _, pt in h.iterrows():
            rows.append(
                {
                    "name": r.name,
                    "state": r.state,
                    "epochs": tags.get("epochs"),
                    "lr": tags.get("lr"),
                    "wd": tags.get("wd"),
                    "tokens": pt.get(TOKENS_KEY),
                    "val_loss": pt[VAL_LOSS_KEY],
                }
            )
    df = pd.DataFrame(rows)
    for c in ["epochs", "tokens", "val_loss", "lr", "wd"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    return df


def load_or_refresh_curves(version: str, refresh: bool) -> pd.DataFrame:
    csv = RESULTS_DIR / version / "exp75_curves.csv"
    if refresh or not csv.exists():
        csv.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_curves(version)
        df.to_csv(csv, index=False)
        print(f"fetched {df['name'].nunique()} run curves ({len(df)} pts) -> {csv}")
    else:
        df = pd.read_csv(csv)
        print(f"read {df['name'].nunique()} run curves ({len(df)} pts) <- {csv} (pass --refresh to re-fetch)")
    return df


def fetch_baseline() -> dict:
    """Current val loss + run_progress of the 1.5B baseline run (a point-in-time snapshot)."""
    import wandb  # noqa: PLC0415 -- lazy: only --refresh touches W&B; plotting reads the JSON

    api = wandb.Api()
    run = api.run(BASELINE_RUN)
    s = run.summary
    return {
        "name": run.name,
        "state": run.state,
        "val_loss": s.get(VAL_LOSS_KEY),
        "run_progress": s.get("run_progress"),
    }


def load_or_refresh_baseline(version: str, refresh: bool) -> dict | None:
    path = RESULTS_DIR / version / "baseline.json"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        baseline = fetch_baseline()
        path.write_text(json.dumps(baseline, indent=2))
        print(
            f"fetched baseline {baseline['name']} loss={baseline['val_loss']} "
            f"progress={baseline['run_progress']} -> {path}"
        )
        return baseline
    baseline = json.loads(path.read_text())
    print(f"read baseline <- {path} (pass --refresh to re-fetch)")
    return baseline


def plot(df: pd.DataFrame, version: str, baseline: dict | None = None) -> None:
    """Final val loss vs run completion time, colored by epoch, best-so-far traced."""
    done = df[(df["state"] == "finished") & df["val_loss"].notna() & df["completed_at"].notna()].copy()
    done["completed_dt"] = pd.to_datetime(done["completed_at"], utc=True)
    done = done.sort_values("completed_dt").reset_index(drop=True)

    # In-progress runs: plotted at their last-heartbeat time with current (partial) val loss.
    # Shown faint and kept out of the best-so-far frontier / best annotation — not yet meaningful.
    running = df[(df["state"] == "running") & df["val_loss"].notna() & df["completed_at"].notna()].copy()
    running["completed_dt"] = pd.to_datetime(running["completed_at"], utc=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfd",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 5.2))  # wide/short reads better inline in a GH issue

    if done.empty:
        ax.text(0.5, 0.5, "no completed runs yet", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        for ep in sorted(done["epochs"].dropna().unique()):
            sub = done[done["epochs"] == ep]
            _scatter_clipped(
                ax,
                sub,
                size=95,
                zorder=3,
                color=EPOCH_COLORS.get(int(ep), "#8c8c8c"),
                label=f"{int(ep)} epoch" + ("" if ep == 1 else "s"),
            )

        # Best-so-far frontier: running minimum of final loss over completion time.
        run_min = done["val_loss"].cummin()
        ax.step(
            done["completed_dt"], run_min, where="post", color="#2b2b2b", lw=1.7, ls="--", zorder=2, label="best so far"
        )

        best = done.loc[done["val_loss"].idxmin()]
        # Box parked in the right margin, left edge aligned with the legend (x=1.01 in axes
        # coords); the arrow still points back to the best run in the data.
        ax.annotate(
            f"best  {best['val_loss']:.4f}\nlr={best['lr']:.2e}\nwd={best['wd']:.2e}\n{int(best['epochs'])} epochs",
            (best["completed_dt"], best["val_loss"]),
            xycoords="data",
            textcoords=ax.transAxes,
            xytext=(1.016, 0.28),
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e0", ec="#b5172f", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="#b5172f", lw=1.4),
        )

        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=timezone.utc))

    if not running.empty:
        for ep in sorted(running["epochs"].dropna().unique()):
            sub = running[running["epochs"] == ep]
            _scatter_clipped(ax, sub, size=70, zorder=2.2, alpha=0.22, color=EPOCH_COLORS.get(int(ep), "#8c8c8c"))
        # Single grey proxy so the legend explains the faint points without per-epoch duplicates.
        ax.scatter([], [], s=70, color="#8c8c8c", edgecolor="none", alpha=0.45, label=f"in progress ({len(running)})")

    # Clip line + a triangle proxy so the up-triangles read as "value is above the cap".
    if (df["val_loss"] > CLIP_HIGH).any():
        ax.axhline(CLIP_HIGH, color="#cccccc", lw=0.9, zorder=1)
        ax.scatter([], [], marker="^", s=80, color="#8c8c8c", edgecolor="none", label=f"≥ {CLIP_HIGH:.2f} (clipped)")
        ax.set_ylim(top=CLIP_HIGH + 0.012)

    if baseline and baseline.get("val_loss") is not None:
        bl = baseline["val_loss"]
        # Solid seaborn-deep purple: stands out yet sits in the same family as the epoch colors.
        ax.axhline(bl, color="#8172B3", lw=2.0, ls="-", zorder=1.5)
        run_id = BASELINE_RUN.split("/")[-1]
        # Label parked bottom-left, with a thin vertical leader up to the purple line. The
        # leader is pinned far left (x in axes coords) to stay clear of the data points;
        # y is in data coords so it meets the line exactly.
        conn_tx = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        y_lo, y_hi = ax.get_ylim()
        x_conn = 0.012
        y_label_top = y_lo + 0.12 * (y_hi - y_lo)
        ax.plot([x_conn, x_conn], [y_label_top, bl], transform=conn_tx, color="#8172B3", lw=1.1, zorder=1.4)
        ax.text(
            x_conn + 0.008,  # nudged right of the leader so the line doesn't clip the text
            y_label_top,
            f"1.5B baseline  {bl:.4f}\n{run_id}",
            transform=conn_tx,
            ha="left",
            va="top",
            fontsize=9.5,
            color="#5e4b8b",
        )

    ax.set_xlabel("run completion time (UTC)")
    ax.set_ylabel("final  eval/contacts-v1-val/loss")
    n_done = 0 if done.empty else len(done)
    ax.set_title(f"exp75 {version} — final val loss by completion time ({n_done} completed runs)")
    ax.grid(False)
    for spine in ax.spines.values():  # thin full border around the plot area
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")
    # Placed just outside the axes (top-right) so it never overlaps the in-progress dots;
    # savefig(bbox_inches="tight") expands the export to include it.
    ax.legend(
        frameon=True,
        framealpha=0.92,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        edgecolor="#dddddd",
    )
    fig.tight_layout()

    outdir = RESULTS_DIR / version
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = outdir / f"loss_vs_completion.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def plot_curves(curves: pd.DataFrame, baseline: dict | None, version: str) -> None:
    """2x2 facets by epoch count: eval loss vs tokens for every run.

    x and y are both free per facet (token budgets differ ~8x and the loss range narrows as
    epochs grow); x is reported in billions of tokens. Within each facet the best run by final
    loss is highlighted in the epoch color (others stay grey) and the baseline is a dotted
    purple line. Loss is clipped to CURVE_CLIP_HIGH, with clipped points drawn as up-triangles.
    """
    epochs = sorted(EPOCH_COLORS)  # [1, 2, 4, 8]
    bl = baseline.get("val_loss") if baseline else None

    def draw_run(ax, g, *, color, lw, zorder, marker=None):
        g = g.sort_values("tokens")
        x = g["tokens"] / 1e9
        ax.plot(x, g["val_loss"].clip(upper=CURVE_CLIP_HIGH), color=color, lw=lw, zorder=zorder, marker=marker, ms=4)
        clipped = g[g["val_loss"] > CURVE_CLIP_HIGH]
        if not clipped.empty:
            ax.scatter(
                clipped["tokens"] / 1e9,
                [CURVE_CLIP_HIGH] * len(clipped),
                marker="^",
                s=24,
                color=color,
                zorder=zorder + 1,
            )

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "#fbfbfd"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.ravel()

    for ax, ep in zip(axes, epochs, strict=True):
        for spine in ax.spines.values():  # thin full border, no grid (matches the other figure)
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#333333")
        ax.grid(False)
        ax.set_title(f"{ep} epoch" + ("" if ep == 1 else "s"), fontsize=13, fontweight="bold")
        ax.set_xlabel("tokens (B)")

        sub = curves[curves["epochs"] == ep]
        if sub.empty:
            ax.text(0.5, 0.5, "no runs", ha="center", va="center", transform=ax.transAxes, color="#999999")
            continue

        # final loss per run = val_loss at its last (max-token) eval; best is the lowest among
        # finished runs (fall back to all if none finished yet).
        finals = sub.sort_values("tokens").groupby("name").tail(1)
        pool = finals[finals["state"] == "finished"]
        pool = pool if not pool.empty else finals
        best_name = pool.loc[pool["val_loss"].idxmin(), "name"]

        for name, g in sub.groupby("name"):
            if name == best_name:
                continue
            draw_run(ax, g, color="#d3d3d3", lw=1.0, zorder=2)

        draw_run(ax, sub[sub["name"] == best_name], color=EPOCH_COLORS[ep], lw=2.4, zorder=4, marker="o")

        if bl is not None:
            ax.axhline(bl, color="#8172B3", lw=1.5, ls=":", zorder=3)

        brow = finals[finals["name"] == best_name].iloc[0]
        ax.annotate(
            f"best  {brow['val_loss']:.4f}\nlr={brow['lr']:.2e}  wd={brow['wd']:.2e}",
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e0", ec=EPOCH_COLORS[ep], alpha=0.95),
        )

    for ax in (axes[0], axes[2]):  # y label on the left column only
        ax.set_ylabel("eval/contacts-v1-val/loss")

    handles = [
        Line2D([], [], color="#d3d3d3", lw=1.4, label="other runs"),
        Line2D([], [], color="#333333", lw=2.4, marker="o", ms=4, label="best by final loss (epoch color)"),
    ]
    if bl is not None:
        handles.append(Line2D([], [], color="#8172B3", lw=1.5, ls=":", label=f"1.5B baseline  {bl:.4f}"))
    if (curves["val_loss"] > CURVE_CLIP_HIGH).any():
        handles.append(
            Line2D([], [], color="#888888", marker="^", ls="none", label=f"≥ {CURVE_CLIP_HIGH:.1f} (clipped)")
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        columnspacing=1.1,
        handletextpad=0.4,
    )
    fig.suptitle(f"exp75 {version} — eval loss vs tokens by epoch count", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    outdir = RESULTS_DIR / version
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = outdir / f"loss_curves_by_epoch.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def plot_best_by_epochs(df: pd.DataFrame, baseline: dict | None, version: str) -> None:
    """Best (min) final val loss reached at each epoch count, x on a log2 scale."""
    fin = df[(df["state"] == "finished") & df["val_loss"].notna()]
    best = fin.groupby("epochs")["val_loss"].min().sort_index()
    bl = baseline.get("val_loss") if baseline else None

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "#fbfbfd"})
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#333333")
    ax.grid(False)

    if bl is not None:
        ax.axhline(bl, color="#8172B3", lw=1.5, ls=":", zorder=2)
        ax.text(
            0.012,
            bl,
            f"1.5B baseline  {bl:.4f}",
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#5e4b8b",
        )

    ax.plot(best.index, best.values, color="#999999", lw=1.6, zorder=3)
    for ep, v in best.items():
        ax.scatter([ep], [v], color=EPOCH_COLORS.get(int(ep), "#333333"), s=110, zorder=4)
        ax.annotate(f"{v:.4f}", (ep, v), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=9.5)

    ax.set_xscale("log", base=2)
    ax.set_xticks(list(best.index))
    ax.set_xticklabels([str(int(e)) for e in best.index])
    ax.set_xlabel("epochs")
    ax.set_ylabel("best  eval/contacts-v1-val/loss")
    ax.set_title(f"exp75 {version} — best loss vs epoch count", fontsize=14, fontweight="bold")
    ax.margins(x=0.12, y=0.18)
    fig.tight_layout()

    outdir = RESULTS_DIR / version
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = outdir / f"best_loss_vs_epochs.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def plot_lr_wd_heatmaps(df: pd.DataFrame, version: str) -> None:
    """2x2 LR x WD heatmaps (one per epoch count) of final val loss, plus the long-format CSV.

    Axes use the union of LR/WD values across all epochs so the four grids line up; cells with
    no finished run are left blank. Each facet has its own color scale (loss range narrows a lot
    with more epochs) and outlines its best (lowest-loss) cell in the epoch color.
    """
    fin = df[(df["state"] == "finished") & df["val_loss"].notna()]
    grid = (
        fin[["epochs", "lr", "wd", "val_loss"]].rename(columns={"val_loss": "loss"}).sort_values(["epochs", "lr", "wd"])
    )
    outdir = RESULTS_DIR / version
    outdir.mkdir(parents=True, exist_ok=True)
    grid_csv = outdir / "lr_wd_grid.csv"
    grid.to_csv(grid_csv, index=False)
    print(f"wrote {grid_csv} ({len(grid)} cells)")

    lrs = sorted(fin["lr"].dropna().unique())
    wds = sorted(fin["wd"].dropna().unique())
    epochs = sorted(EPOCH_COLORS)
    cmap = mpl.cm.viridis_r.copy()
    cmap.set_bad("#e6e6e6")  # blank cells

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    for ax, ep in zip(axes, epochs, strict=True):
        ax.set_title(f"{ep} epoch" + ("" if ep == 1 else "s"), fontsize=13, fontweight="bold")
        sub = fin[fin["epochs"] == ep]
        piv = sub.pivot_table(index="lr", columns="wd", values="val_loss", aggfunc="min").reindex(index=lrs, columns=wds)
        matrix = np.ma.masked_invalid(piv.to_numpy(dtype=float))
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="final val loss")

        ax.set_xticks(range(len(wds)))
        ax.set_xticklabels([f"{w:g}" for w in wds], fontsize=8)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.1e}" for lr in lrs], fontsize=8)
        ax.set_xlabel("weight decay")
        ax.set_ylabel("learning rate")

        vmin, vmax = float(matrix.min()), float(matrix.max())
        for i in range(len(lrs)):
            for j in range(len(wds)):
                if matrix.mask[i, j]:
                    continue
                norm = 0.0 if vmax == vmin else (matrix[i, j] - vmin) / (vmax - vmin)
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if norm > 0.55 else "black",
                )
        bi, bj = np.unravel_index(np.argmin(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, edgecolor=EPOCH_COLORS[ep], lw=2.5))

    fig.suptitle(f"exp75 {version} — final val loss over the LR/WD grid by epoch count", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "svg"):
        out = outdir / f"lr_wd_heatmap.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def plot_contact_metrics(version: str) -> None:
    """1x3 scatter: contact AUC and R-precision (all / long) vs loss for the per-epoch winners.

    Points are the CONTACT_METRICS table (one per epoch winner), colored by epoch and
    connected in loss order to highlight the sharp emergence at E8. The loss axis is reversed
    (lower = better, to the right) so the metric-vs-loss trend reads as a negative slope.
    """
    rows = sorted(CONTACT_METRICS, key=lambda r: r["loss"])
    loss = [r["loss"] for r in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white", "axes.facecolor": "#fbfbfd"})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    panels = (
        (axes[0], "auc_long", "contact AUC (long)"),
        (axes[1], "rprec_all", "R-precision (all)"),
        (axes[2], "rprec_long", "R-precision (long)"),
    )

    for ax, key, ylabel in panels:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#333333")
        ax.grid(False)
        y = [r[key] for r in rows]
        ax.plot(loss, y, color="#cccccc", lw=1.4, zorder=2)
        for r in rows:
            ax.scatter(r["loss"], r[key], color=EPOCH_COLORS[r["epochs"]], s=150, zorder=3)
            ax.annotate(
                r["ckpt"], (r["loss"], r[key]), textcoords="offset points", xytext=(7, 6), fontsize=10, fontweight="bold"
            )
        ax.set_xlabel("eval/contacts-v1-val/loss")
        ax.set_ylabel(ylabel)
        ax.margins(x=0.16, y=0.18)
        ax.invert_xaxis()  # lower (better) loss on the right -> downward metric-vs-loss slope

    handles = [
        Line2D([], [], marker="o", ls="none", color=EPOCH_COLORS[ep], label=f"{ep} epoch" + ("" if ep == 1 else "s"))
        for ep in sorted(EPOCH_COLORS)
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"exp75 {version} — contact-prediction metrics vs eval loss", fontsize=15, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    outdir = RESULTS_DIR / version
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        out = outdir / f"contact_metrics_vs_loss.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="exp75 results: W&B -> CSV -> final-loss-vs-completion plot")
    ap.add_argument("--refresh", action="store_true", help="re-fetch runs from W&B (else read cached CSV)")
    ap.add_argument("--version", default="v1", help="sweep version to analyze; partitions the output dir")
    args = ap.parse_args()
    df = load_or_refresh(args.version, args.refresh)
    baseline = load_or_refresh_baseline(args.version, args.refresh)
    plot(df, args.version, baseline)
    curves = load_or_refresh_curves(args.version, args.refresh)
    plot_curves(curves, baseline, args.version)
    plot_best_by_epochs(df, baseline, args.version)
    plot_lr_wd_heatmaps(df, args.version)
    plot_contact_metrics(args.version)


if __name__ == "__main__":
    main()
