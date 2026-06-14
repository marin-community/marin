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
                                 ``eval/contacts-v1-val/loss`` and step (that run is still
                                 training, so the value/step move until it finishes)
    loss_vs_completion.{png,svg} final ``eval/contacts-v1-val/loss`` vs run completion
                                 time, colored by epoch, with the best-so-far frontier
                                 traced and the 1.5B baseline drawn as a horizontal line.
                                 COMPLETED runs only (may loosen later).

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
import pandas as pd

ENTITY = "eric-czech"
PROJECT = "marin"
GROUP = "exp75-contacts-v1-tune"
VAL_LOSS_KEY = "eval/contacts-v1-val/loss"
RESULTS_DIR = Path(__file__).parent / "exp75_results"

# 1.5B unmasked run on the same eval — the baseline exp75 is trying to beat. Lives in a
# different entity/project and is still training, so we record its current step alongside
# the value (the horizontal line is a moving target until that run finishes).
BASELINE_RUN = "open-athena/MarinFold/protein-contacts-1_5b-3.5e-4-contacts-v1-unmasked-3b5cf2"

# Tag keys lifted into their own CSV columns (see exp75_sweep.build_trial tags).
TAG_KEYS = ["epochs", "lr", "wd", "params_exact", "tokens_exact", "tpu", "band"]

# Stable, colorblind-friendly color per epoch rung.
EPOCH_COLORS = {1: "#4C72B0", 2: "#DD8452", 4: "#55A868", 8: "#C44E52"}

# Val-loss values above this are uninteresting (mostly early/in-progress runs); the y-axis
# tops out here and any higher point is pinned to the line and drawn as an up-triangle (^)
# to signal "actually higher", so a few bad runs don't compress the scale for everything else.
CLIP_HIGH = 3.18


def _scatter_clipped(ax, sub, *, size, edgewidth, zorder, alpha=1.0, color, label=None):
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
        edgecolor="white",
        linewidth=edgewidth,
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
            edgecolor="white",
            linewidth=edgewidth,
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


def fetch_baseline() -> dict:
    """Current val loss + step of the 1.5B baseline run (still training, so a snapshot)."""
    import wandb  # noqa: PLC0415 -- lazy: only --refresh touches W&B; plotting reads the JSON

    api = wandb.Api()
    run = api.run(BASELINE_RUN)
    s = run.summary
    return {
        "name": run.name,
        "state": run.state,
        "val_loss": s.get(VAL_LOSS_KEY),
        "step": s.get("_step"),
    }


def load_or_refresh_baseline(version: str, refresh: bool) -> dict | None:
    path = RESULTS_DIR / version / "baseline.json"
    if refresh or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        baseline = fetch_baseline()
        path.write_text(json.dumps(baseline, indent=2))
        print(f"fetched baseline {baseline['name']} loss={baseline['val_loss']} step={baseline['step']} -> {path}")
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
                edgewidth=1.3,
                zorder=3,
                color=EPOCH_COLORS.get(int(ep), "#8c8c8c"),
                label=f"{int(ep)} epoch" + ("" if ep == 1 else "s"),
            )

        # Best-so-far frontier: running minimum of final loss over completion time.
        run_min = done["val_loss"].cummin()
        ax.step(
            done["completed_dt"], run_min, where="post", color="#2b2b2b", lw=1.7, ls="--", zorder=2, label="best so far"
        )
        prev = run_min.shift(1)
        improved = done[prev.isna() | (run_min < prev - 1e-12)]
        ax.scatter(
            improved["completed_dt"],
            improved["val_loss"],
            s=210,
            facecolor="none",
            edgecolor="#b5172f",
            linewidth=1.9,
            zorder=4,
        )

        best = done.loc[done["val_loss"].idxmin()]
        ax.annotate(
            f"best  {best['val_loss']:.4f}\nlr={best['lr']:.2e}\nwd={best['wd']:.2e}\n{int(best['epochs'])} epochs",
            (best["completed_dt"], best["val_loss"]),
            textcoords="offset points",
            xytext=(74, 54),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fff7e0", ec="#b5172f", alpha=0.95),
            arrowprops=dict(arrowstyle="->", color="#b5172f", lw=1.4),
        )

        loc = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc, tz=timezone.utc))

    if not running.empty:
        for ep in sorted(running["epochs"].dropna().unique()):
            sub = running[running["epochs"] == ep]
            _scatter_clipped(
                ax, sub, size=70, edgewidth=0.8, zorder=2.2, alpha=0.22, color=EPOCH_COLORS.get(int(ep), "#8c8c8c")
            )
        # Single grey proxy so the legend explains the faint points without per-epoch duplicates.
        ax.scatter([], [], s=70, color="#8c8c8c", edgecolor="white", alpha=0.45, label=f"in progress ({len(running)})")

    # Clip line + a triangle proxy so the up-triangles read as "value is above the cap".
    if (df["val_loss"] > CLIP_HIGH).any():
        ax.axhline(CLIP_HIGH, color="#cccccc", lw=0.9, zorder=1)
        ax.scatter([], [], marker="^", s=80, color="#8c8c8c", edgecolor="white", label=f"≥ {CLIP_HIGH:.2f} (clipped)")
        ax.set_ylim(top=CLIP_HIGH + 0.012)

    if baseline and baseline.get("val_loss") is not None:
        bl = baseline["val_loss"]
        step = baseline.get("step")
        if step is None:
            step_note = baseline.get("state", "")
        else:
            step_note = f"step {int(step):,}" + (", still running" if baseline.get("state") == "running" else "")
        ax.axhline(bl, color="#555555", lw=1.6, ls=":", zorder=1.5)
        # Detail sits above the line, flush to the y-axis (x in axes coords, y in data coords).
        run_id = BASELINE_RUN.split("/")[-1]
        label_tx = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(
            0.006,
            bl,
            f"1.5B baseline  {bl:.4f}  ({step_note})\n{run_id}",
            transform=label_tx,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color="#333333",
        )

    ax.set_xlabel("run completion time (UTC)")
    ax.set_ylabel("final  eval/contacts-v1-val/loss")
    n_done = 0 if done.empty else len(done)
    ax.set_title(f"exp75 {version} — final val loss by completion time   ({n_done} completed runs)")
    ax.grid(True, which="major", axis="both", alpha=0.3)
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


def main() -> None:
    ap = argparse.ArgumentParser(description="exp75 results: W&B -> CSV -> final-loss-vs-completion plot")
    ap.add_argument("--refresh", action="store_true", help="re-fetch runs from W&B (else read cached CSV)")
    ap.add_argument("--version", default="v1", help="sweep version to analyze; partitions the output dir")
    args = ap.parse_args()
    df = load_or_refresh(args.version, args.refresh)
    baseline = load_or_refresh_baseline(args.version, args.refresh)
    plot(df, args.version, baseline)


if __name__ == "__main__":
    main()
