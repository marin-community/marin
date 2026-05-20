# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze exp11 ``cd``-data-mix sweep results from W&B.

Companion to ``exp11_data_mix_sweep.py``. Pulls ``train/loss`` and every
``eval/<dataset>-(val|test)[-...]/loss`` history series for the runs in a
configured sweep, then renders a heatmap of loss values across mixtures.

Most runs are typically mid-flight, so we never wait for completion. Instead
we compute ``min_step`` = ``min`` over runs of ``max(_step)`` and slice each
run/metric at the largest history step ``<= min_step``. This gives a fair
cross-mixture snapshot at the slowest run's progress.

The W&B fetch result is reduced to a single ``snapshot.csv`` under
``experiments/protein/exp11_data_mix_results/<sweep_id>/``; pass
``--refresh`` to re-pull from W&B. Adding another sweep = one new entry in
``SWEEPS``.

Usage::

    WANDB_API_KEY=... uv run --with matplotlib --with seaborn \\
        python -m experiments.protein.exp11_data_mix_analysis [--refresh]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepConfig:
    """One named sweep we know how to fetch + analyze.

    Attributes:
        id: Slug used as the cache subdir name (``exp11_data_mix_results/<id>/``).
        entity: W&B entity (user/team).
        project: W&B project.
        group: W&B group filter (server-side narrow). Empty string = no filter.
        tag: One required tag (server-side narrow). Empty string = no filter.
        name_regex: Client-side regex applied to ``run.display_name`` to keep
            only this sweep's runs (e.g. version-specific).
        mixture_regex: Regex with a ``mixture`` named group, extracted from
            ``run.display_name`` to identify mixture rows.
    """

    id: str
    entity: str
    project: str
    group: str
    tag: str
    name_regex: str
    mixture_regex: str


SWEEPS: dict[str, SweepConfig] = {
    # exp11 mix-stage v2 (9 mixtures, 100M model, ~4.3B tokens each).
    "run_mix_sweep_v2": SweepConfig(
        id="run_mix_sweep_v2",
        entity="eric-czech",
        project="marin",
        group="exp11-data-mix",
        tag="mix",
        name_regex=r"^prot-exp11-dm-mix-100m-.*-v2-[0-9a-f]+$",
        mixture_regex=r"-(?P<mixture>m\d+)-lr",
    ),
}

DEFAULT_SWEEP = "run_mix_sweep_v2"

# Prefix every plot title with this so the experiment context is unambiguous.
TITLE_PREFIX = "MarinFold Experiment #11"


# Display names per the final mix-sweep table in
# https://github.com/Open-Athena/MarinFold/issues/11#issue-4473272578.
# Used for heatmap row labels and the CSV ``mixture_name`` column.
MIXTURE_NAMES: dict[str, str] = {
    "m1": "m1 (H)",
    "m2": "m2 (M)",
    "m3": "m3 (L)",
    "m4": "m4 (H80/M10/L10)",
    "m5": "m5 (H60/M30/L10)",
    "m6": "m6 (H31/M26/L43)",
    "m7": "m7 (L→H)",
    "m8": "m8 (H→L)",
    "m9": "m9 (L,M → L,M,H → H)",
}


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------


RESULTS_ROOT = Path(__file__).resolve().parent / "exp11_data_mix_results"


def _sweep_dir(sweep_id: str) -> Path:
    return RESULTS_ROOT / sweep_id


def _snapshot_path(sweep_id: str) -> Path:
    return _sweep_dir(sweep_id) / "snapshot.csv"


def _plots_dir(sweep_id: str) -> Path:
    return _sweep_dir(sweep_id) / "plots"


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------

# Keep ``train/loss`` plus any ``eval/<dataset>-(val|test)[-...]/loss`` series.
# This matches both ``eval/protein-docs-high-val/loss`` and the unmasked
# variant ``eval/protein-docs-cd-val-unmasked/loss``; the IID train-carve
# (``...-train/loss``) and the aggregates (``eval/loss``, ``eval/macro_loss``)
# are excluded.
EVAL_METRIC_RE = re.compile(r"^eval/.+-(val|test)(-[^/]*)?/loss$")
TRAIN_METRIC = "train/loss"


def _is_keep_metric(name: str) -> bool:
    return name == TRAIN_METRIC or EVAL_METRIC_RE.match(name) is not None


def _short_metric_label(name: str) -> str:
    """Compact heatmap column label.

    ``eval/protein-docs-high-val/loss`` -> ``high-val``
    ``eval/protein-docs-cd-val-unmasked/loss`` -> ``cd-val-unmasked``
    ``train/loss`` -> ``train``
    """
    if name == TRAIN_METRIC:
        return "train"
    stripped = name.removeprefix("eval/").removesuffix("/loss")
    return stripped.removeprefix("protein-docs-")


# ---------------------------------------------------------------------------
# W&B fetch -> long-form snapshot
# ---------------------------------------------------------------------------


def _wandb_api():
    import wandb

    return wandb.Api()


def _list_runs(sweep: SweepConfig) -> list[tuple[object, str]]:
    """Return ``[(wandb_run, mixture_id), ...]`` for this sweep."""
    api = _wandb_api()
    filters: dict = {}
    if sweep.group:
        filters["group"] = sweep.group
    if sweep.tag:
        filters["tags"] = sweep.tag
    raw = list(api.runs(f"{sweep.entity}/{sweep.project}", filters=filters))
    name_re = re.compile(sweep.name_regex)
    mix_re = re.compile(sweep.mixture_regex)
    matched: list[tuple[object, str]] = []
    for r in raw:
        if not name_re.match(r.display_name):
            continue
        m = mix_re.search(r.display_name)
        if not m:
            logger.warning("Could not extract mixture from %s; skipping", r.display_name)
            continue
        matched.append((r, m.group("mixture")))
    logger.info("Sweep %s: %d runs in scope, %d match name regex", sweep.id, len(raw), len(matched))
    return matched


def _fetch_run_history(run, keep_keys: list[str]) -> pd.DataFrame:
    """Stream ``scan_history`` for the requested keys into a DataFrame."""
    keys = ["_step", *keep_keys]
    rows = list(run.scan_history(keys=keys, page_size=10_000))
    if not rows:
        return pd.DataFrame(columns=keys)
    df = pd.DataFrame(rows)
    df = df.dropna(axis=1, how="all")
    if "_step" not in df.columns:
        return pd.DataFrame(columns=keys)
    df["_step"] = df["_step"].astype(int)
    return df.sort_values("_step").reset_index(drop=True)


def _value_at(df: pd.DataFrame, column: str, ref_step: int) -> tuple[float | None, int | None]:
    """Largest history step ``<= ref_step`` with a non-null value in ``column``."""
    if column not in df.columns:
        return None, None
    sub = df[["_step", column]].dropna()
    sub = sub[sub["_step"] <= ref_step]
    if sub.empty:
        return None, None
    row = sub.sort_values("_step").iloc[-1]
    return float(row[column]), int(row["_step"])


def build_snapshot(sweep: SweepConfig) -> pd.DataFrame:
    """Pull every run, slice at ``min(max(_step))``, return long-form snapshot.

    Returns a frame with columns: ``mixture_id``, ``mixture_name``, ``metric``,
    ``value``, ``step_used``, ``ref_step``, ``run_name``, ``run_state``.
    """
    matched = _list_runs(sweep)
    if not matched:
        raise RuntimeError(f"No runs matched sweep {sweep.id}")

    # Pass 1: fetch history per run.
    per_run: list[tuple[object, str, pd.DataFrame]] = []
    for run, mixture in matched:
        keep_keys = sorted(k for k in run.summary.keys() if _is_keep_metric(k))
        if not keep_keys:
            logger.warning("Run %s has no matching loss metrics; skipping", run.display_name)
            continue
        logger.info("Fetching %s (%d keys, state=%s)", run.display_name, len(keep_keys), run.state)
        df = _fetch_run_history(run, keep_keys)
        if df.empty:
            logger.warning("  empty history; skipping")
            continue
        per_run.append((run, mixture, df))

    if not per_run:
        raise RuntimeError(f"No runs in sweep {sweep.id} produced usable history")

    ref_step = min(int(df["_step"].max()) for _, _, df in per_run)
    logger.info("Reference step (min over runs of max-step) = %d", ref_step)

    # Pass 2: slice each (run, metric) at the largest step <= ref_step.
    rows: list[dict] = []
    for run, mixture, df in per_run:
        metrics = [c for c in df.columns if c != "_step" and _is_keep_metric(c)]
        for metric in metrics:
            value, step = _value_at(df, metric, ref_step)
            if value is None:
                continue
            rows.append(
                {
                    "mixture_id": mixture,
                    "mixture_name": MIXTURE_NAMES.get(mixture, mixture),
                    "metric": _short_metric_label(metric),
                    "value": value,
                    "step_used": step,
                    "ref_step": ref_step,
                    "run_name": run.display_name,
                    "run_state": run.state,
                }
            )
    return pd.DataFrame(rows)


def load_or_build_snapshot(sweep: SweepConfig, *, refresh: bool) -> pd.DataFrame:
    csv_path = _snapshot_path(sweep.id)
    if not refresh and csv_path.exists():
        logger.info("Loading cached snapshot from %s", csv_path)
        return pd.read_csv(csv_path)
    snapshot = build_snapshot(sweep)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(csv_path, index=False)
    logger.info("Wrote %d rows to %s", len(snapshot), csv_path)
    return snapshot


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


# Preferred column order in the heatmap; anything not in this list falls
# through to alphabetical at the right.
PREFERRED_METRIC_ORDER = (
    "train",
    "high-val",
    "high-test",
    "medium-val",
    "medium-test",
    "low-val",
    "low-test",
    "cd-val",
    "cd-val-unmasked",
)


# Mixture groups for the cd-val bar plot. Tracks the issue's three logical
# blocks: pure single-quality, static blends, staged curricula.
MIXTURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Single quality", ("m1", "m2", "m3")),
    ("Static blends", ("m4", "m5", "m6")),
    ("Staged", ("m7", "m8", "m9")),
)


def _ordered_columns(columns: list[str]) -> list[str]:
    in_order = [c for c in PREFERRED_METRIC_ORDER if c in columns]
    rest = sorted(c for c in columns if c not in PREFERRED_METRIC_ORDER)
    return [*in_order, *rest]


def _mixture_sort_key(mixture_id: str) -> tuple[int, str]:
    m = re.search(r"\d+", mixture_id)
    return (int(m.group()) if m else 0, mixture_id)


def render_heatmap(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    ref_step = int(snapshot["ref_step"].iloc[0])

    mixture_order = sorted(snapshot["mixture_id"].unique(), key=_mixture_sort_key)
    # Resolve display labels live (the CSV stores a snapshot at fetch time;
    # this keeps name tweaks in MIXTURE_NAMES from requiring a re-fetch).
    mixture_labels = {m: MIXTURE_NAMES.get(m, m) for m in mixture_order}

    # Heatmap is eval-only; train is a separate scalar plotted elsewhere.
    eval_snapshot = snapshot[snapshot["metric"] != "train"]
    values = eval_snapshot.pivot(index="mixture_id", columns="metric", values="value")
    values = values.reindex(mixture_order)
    cols = _ordered_columns(list(values.columns))
    values = values[cols]

    # Per-column z-score for color; high-magnitude metrics (cd-val-unmasked ~10
    # vs others ~1.4) would otherwise crush the comparison. Annotate with raw
    # values so the absolute loss is still visible.
    z = values.copy().astype(float)
    for c in cols:
        col = values[c].astype(float)
        std = col.std(ddof=0)
        z[c] = (col - col.mean()) / std if std > 0 else 0.0

    nrows, ncols = values.shape
    fig, ax = plt.subplots(figsize=(1.6 + 1.05 * ncols, 0.8 + 0.6 * nrows))

    # Reversed RdBu so lower loss is blue (good) and higher loss is red (bad).
    im = ax.imshow(z.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto", vmin=-2.0, vmax=2.0)

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(cols, rotation=40, ha="right")
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([mixture_labels[m] for m in mixture_order])

    for i in range(nrows):
        for j in range(ncols):
            v = values.iat[i, j]
            zv = z.iat[i, j]
            if pd.isna(v):
                text, color = "—", "black"
            else:
                text = f"{v:.3f}"
                color = "white" if abs(zv) > 1.2 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("per-metric z-score  (blue = lower loss = better)")

    ax.set_title(
        f"{TITLE_PREFIX} — {sweep.id}: loss by mixture @ step={ref_step}\n"
        f"(annotation = raw loss; color = z-score within column)",
        fontsize=10,
    )

    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "heatmap_loss.pdf"
    png_path = out_dir / "heatmap_loss.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


def render_cdval_bars(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
    """1x3 grid of cd-val bar charts, one per ``MIXTURE_GROUPS`` row.

    Y-axis range is fixed across subplots so cross-group comparison is fair.
    """
    import matplotlib.pyplot as plt

    ref_step = int(snapshot["ref_step"].iloc[0])
    cdval = snapshot[snapshot["metric"] == "cd-val"].set_index("mixture_id")
    if cdval.empty:
        logger.warning("No cd-val rows in snapshot; skipping bar plot")
        return

    # Fixed y-limits: pad ~10% above/below the overall cd-val range so every
    # subplot uses the same scale.
    y_all = cdval["value"].to_numpy(dtype=float)
    y_lo, y_hi = float(y_all.min()), float(y_all.max())
    span = max(y_hi - y_lo, 1e-6)
    pad = 0.1 * span
    y_min, y_max = y_lo - pad, y_hi + pad

    fig, axes = plt.subplots(1, len(MIXTURE_GROUPS), figsize=(3.6 * len(MIXTURE_GROUPS), 3.8), sharey=True)
    for ax, (group_label, mixture_ids) in zip(axes, MIXTURE_GROUPS, strict=True):
        values = [float(cdval.loc[mid, "value"]) if mid in cdval.index else float("nan") for mid in mixture_ids]
        labels = [MIXTURE_NAMES.get(mid, mid) for mid in mixture_ids]
        xs = list(range(len(mixture_ids)))
        bars = ax.bar(xs, values, color="#4C78A8", edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(group_label, fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, value in zip(bars, values, strict=True):
            if value != value:  # NaN
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("cd-val loss")
    fig.suptitle(
        f"{TITLE_PREFIX} — {sweep.id}: cd-val loss by mixture @ step={ref_step}",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "cdval_bars.pdf"
    png_path = out_dir / "cdval_bars.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", choices=sorted(SWEEPS), default=DEFAULT_SWEEP)
    parser.add_argument("--refresh", action="store_true", help="Force a fresh W&B fetch.")
    args = parser.parse_args(argv)

    sweep = SWEEPS[args.sweep]
    snapshot = load_or_build_snapshot(sweep, refresh=args.refresh)

    ref_step = int(snapshot["ref_step"].iloc[0])
    wide = snapshot.pivot(index="mixture_name", columns="metric", values="value")
    logger.info("ref_step = %d", ref_step)
    logger.info("\n%s", wide.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "—"))

    plots_dir = _plots_dir(sweep.id)
    render_heatmap(snapshot, sweep, plots_dir)
    render_cdval_bars(snapshot, sweep, plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
