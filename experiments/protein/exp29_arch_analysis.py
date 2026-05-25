# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze exp29 (Llama vs Qwen3 on m11) sweep results from W&B.

Companion to ``exp29_arch_sweep.py``. Pulls ``train/loss`` and every
``eval/<dataset>-(val|test)[-...]/loss`` history series for the two trials
in the sweep, then renders a grouped-bar comparison on the four val splits
we care about: ``cd-val``, ``high-val``, ``medium-val``, ``low-val`` (all
masked, distance-bin only).

Both runs are typically mid-flight while the experiment is being judged, so
we never wait for completion. Instead we compute
``min_step = min(over runs of max(_step))`` and slice each run/metric at the
largest history step ``<= min_step``. This gives a fair cross-trial snapshot
at the slowest run's progress.

The W&B fetch result is reduced to a single ``snapshot.csv`` under
``experiments/protein/exp29_arch_results/``; pass ``--refresh`` to re-pull.

Usage::

    WANDB_API_KEY=... uv run --with matplotlib --with seaborn \\
        python -m experiments.protein.exp29_arch_analysis [--refresh]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

logger = logging.getLogger(__name__)


# --- Sweep identity ----------------------------------------------------------

SWEEP_ID: str = "exp29_arch"
WANDB_ENTITY: str = "eric-czech"
WANDB_PROJECT: str = "marin"
WANDB_GROUP: str = "exp29-arch"
WANDB_TAG: str = "arch"

# Matches run names emitted by exp29_arch_sweep.trial_name(...). Variant id
# (llama|qwen3) is captured into the ``variant`` named group; the trailing
# 6-char executor hash is allowed.
RUN_NAME_RE: re.Pattern = re.compile(
    r"^prot-exp29-100m-.*-m11-(?P<variant>llama|qwen3)-lr.*-v1(?:-[0-9a-f]+)?$"
)

# Prefix every plot title with this so the experiment context is unambiguous.
TITLE_PREFIX: str = "MarinFold Experiment #29"

# Display labels + colors for the two variants. Tableau-10 subset; the Llama
# / Qwen3 colors here match the variant palette used by the exp11 analysis
# script in its pre-retcon form, so plots stay visually continuous.
VARIANT_NAMES: dict[str, str] = {
    "llama": "Llama",
    "qwen3": "Qwen3 (+QK-norm)",
}
VARIANT_COLORS: dict[str, str] = {
    "llama": "#54A24B",
    "qwen3": "#9D7AB8",
}

# --- Filesystem layout -------------------------------------------------------

RESULTS_DIR: Path = Path(__file__).resolve().parent / "exp29_arch_results"
SNAPSHOT_PATH: Path = RESULTS_DIR / "snapshot.csv"
META_PATH: Path = RESULTS_DIR / "meta.json"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

# --- Metric selection --------------------------------------------------------

# Keep ``train/loss`` plus any ``eval/<dataset>-(val|test)[-...]/loss`` series.
EVAL_METRIC_RE: re.Pattern = re.compile(r"^eval/.+-(val|test)(-[^/]*)?/loss$")
TRAIN_METRIC: str = "train/loss"

# Metrics that drive the verdict for an arch comparison: cd-val + the three
# per-quality val splits. All masked. *-test and the cd-val-unmasked metric
# are excluded — they don't change the Llama-vs-Qwen3 verdict.
COMPARISON_METRICS: tuple[str, ...] = ("cd-val", "high-val", "medium-val", "low-val")


def is_keep_metric(name: str) -> bool:
    return name == TRAIN_METRIC or EVAL_METRIC_RE.match(name) is not None


def short_metric_label(name: str) -> str:
    if name == TRAIN_METRIC:
        return "train"
    stripped = name.removeprefix("eval/").removesuffix("/loss")
    return stripped.removeprefix("protein-docs-")


# --- W&B fetch + snapshot construction --------------------------------------


@dataclass(frozen=True)
class EvalMeta:
    """W&B-derived eval config for the sweep; consensus across both runs.

    Pulled from ``run.config`` at build time and validated to be identical
    across runs — figure annotations relying on these numbers are only
    meaningful if every run shares the same eval budget.
    """

    eval_batch_size: int
    eval_seq_len: int
    max_eval_batches: int | None  # None = "no cap" (full-dataset eval)


META_FROM_CONFIG: dict[str, tuple[str, ...]] = {
    "eval_batch_size": ("trainer.train_batch_size",),
    "max_eval_batches": ("trainer.max_eval_batches",),
    "eval_seq_len": ("max_eval_length", "train_seq_len"),
}

MISSING = object()


def get_nested(cfg: object, dotted_key: str) -> object:
    cur: object = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return MISSING
        cur = cur[part]
    return cur


def extract_meta(run) -> dict[str, object]:
    cfg = dict(run.config)
    out: dict[str, object] = {}
    for field, candidates in META_FROM_CONFIG.items():
        for cand in candidates:
            value = get_nested(cfg, cand)
            if value is not MISSING:
                out[field] = value
                break
        else:
            out[field] = None
    return out


def consensus_meta(per_run: list[dict[str, object]]) -> EvalMeta:
    if not per_run:
        raise RuntimeError("No runs to derive meta from")
    consensus: dict[str, object] = {}
    for field in META_FROM_CONFIG:
        values = {m.get(field) for m in per_run}
        if len(values) > 1:
            raise RuntimeError(f"Runs disagree on {field!r}: {sorted(values, key=str)}")
        consensus[field] = values.pop()
    for required in ("eval_batch_size", "eval_seq_len"):
        if consensus[required] is None:
            raise RuntimeError(f"Required meta field {required!r} missing from all runs")
    return EvalMeta(**consensus)  # type: ignore[arg-type]


def save_meta(meta: EvalMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(meta), indent=2) + "\n")


def load_meta(path: Path) -> EvalMeta:
    return EvalMeta(**json.loads(path.read_text()))


def list_runs() -> list[tuple[object, str]]:
    """Return ``[(wandb_run, variant_id), ...]`` for this sweep."""
    api = wandb.Api()
    raw = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"group": WANDB_GROUP, "tags": WANDB_TAG}))
    matched: list[tuple[object, str]] = []
    for r in raw:
        m = RUN_NAME_RE.match(r.display_name)
        if not m:
            continue
        matched.append((r, m.group("variant")))
    logger.info("Sweep %s: %d runs in scope, %d match name regex", SWEEP_ID, len(raw), len(matched))
    return matched


def fetch_run_history(run, keep_keys: list[str]) -> pd.DataFrame:
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


def value_at(df: pd.DataFrame, column: str, ref_step: int) -> tuple[float | None, int | None]:
    if column not in df.columns:
        return None, None
    sub = df[["_step", column]].dropna()
    sub = sub[sub["_step"] <= ref_step]
    if sub.empty:
        return None, None
    row = sub.sort_values("_step").iloc[-1]
    return float(row[column]), int(row["_step"])


def build_snapshot() -> tuple[pd.DataFrame, EvalMeta]:
    """Pull both runs, slice at ``min(max(_step))``, return snapshot + meta.

    Snapshot frame columns: ``variant_id``, ``variant_name``, ``metric``,
    ``value``, ``step_used``, ``ref_step``, ``run_name``, ``run_state``.
    """
    matched = list_runs()
    if not matched:
        raise RuntimeError(f"No runs matched sweep {SWEEP_ID}")
    per_run: list[tuple[object, str, pd.DataFrame]] = []
    metas: list[dict[str, object]] = []
    for run, variant in matched:
        keep_keys = sorted(k for k in run.summary.keys() if is_keep_metric(k))
        if not keep_keys:
            logger.warning("Run %s has no matching loss metrics; skipping", run.display_name)
            continue
        logger.info("Fetching %s (%d keys, state=%s)", run.display_name, len(keep_keys), run.state)
        df = fetch_run_history(run, keep_keys)
        if df.empty:
            logger.warning("  empty history; skipping")
            continue
        per_run.append((run, variant, df))
        metas.append(extract_meta(run))
    if not per_run:
        raise RuntimeError(f"No runs in sweep {SWEEP_ID} produced usable history")

    meta = consensus_meta(metas)
    logger.info("Sweep %s eval meta (consensus across %d runs): %s", SWEEP_ID, len(metas), meta)

    ref_step = min(int(df["_step"].max()) for _, _, df in per_run)
    logger.info("Reference step (min over runs of max-step) = %d", ref_step)

    rows: list[dict] = []
    for run, variant, df in per_run:
        metrics = [c for c in df.columns if c != "_step" and is_keep_metric(c)]
        for metric in metrics:
            value, step = value_at(df, metric, ref_step)
            if value is None:
                continue
            rows.append(
                {
                    "variant_id": variant,
                    "variant_name": VARIANT_NAMES.get(variant, variant),
                    "metric": short_metric_label(metric),
                    "value": value,
                    "step_used": step,
                    "ref_step": ref_step,
                    "run_name": run.display_name,
                    "run_state": run.state,
                }
            )
    return pd.DataFrame(rows), meta


def load_or_build_snapshot(*, refresh: bool) -> tuple[pd.DataFrame, EvalMeta]:
    """Cached fetch: returns (snapshot_df, meta). ``--refresh`` skips the cache."""
    if not refresh and SNAPSHOT_PATH.exists() and META_PATH.exists():
        logger.info("Loading cached snapshot+meta from %s", SNAPSHOT_PATH.parent)
        return pd.read_csv(SNAPSHOT_PATH), load_meta(META_PATH)
    if not refresh and SNAPSHOT_PATH.exists():
        logger.info("Backfilling meta.json from W&B (snapshot already cached)")
        runs = list_runs()
        meta = consensus_meta([extract_meta(r) for r, _ in runs])
        save_meta(meta, META_PATH)
        return pd.read_csv(SNAPSHOT_PATH), meta
    snapshot, meta = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(SNAPSHOT_PATH, index=False)
    save_meta(meta, META_PATH)
    logger.info("Wrote %d rows to %s and meta to %s", len(snapshot), SNAPSHOT_PATH, META_PATH)
    return snapshot, meta


# --- Plot --------------------------------------------------------------------


def variant_sort_key(variant_id: str) -> int:
    keys = list(VARIANT_NAMES.keys())
    return keys.index(variant_id) if variant_id in keys else len(keys)


def render_comparison_bars(snapshot: pd.DataFrame) -> None:
    """Grouped bars: one bar per trial variant per val metric.

    Restricted to ``COMPARISON_METRICS`` (cd-val + the three per-quality val
    splits). Each metric group annotates the absolute loss on each bar plus a
    ``Δ`` line for the difference (qwen3 minus llama) so the direction of the
    effect is immediately readable.
    """
    df = snapshot[snapshot["metric"].isin(COMPARISON_METRICS)]
    if df.empty:
        logger.warning("No target metrics in snapshot for sweep %s; skipping plot", SWEEP_ID)
        return

    ref_step = int(snapshot["ref_step"].iloc[0])
    pivot = df.pivot(index="metric", columns="variant_id", values="value").reindex(list(COMPARISON_METRICS))
    variants = sorted(pivot.columns, key=variant_sort_key)
    pivot = pivot[variants]

    n_variants = len(variants)
    n_metrics = len(COMPARISON_METRICS)
    width = 0.8 / max(n_variants, 1)
    x = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for i, variant in enumerate(variants):
        offset = (i - (n_variants - 1) / 2) * width
        values = pivot[variant].to_numpy(dtype=float)
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=VARIANT_COLORS.get(variant, "#999999"),
            edgecolor="black",
            linewidth=0.5,
            label=VARIANT_NAMES.get(variant, variant),
        )
        for bar, value in zip(bars, values, strict=True):
            if not np.isfinite(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Tighten y-axis around the data so small deltas stay visible while
    # leaving headroom for the Δ annotation box in the upper-right corner.
    y_all = pivot.to_numpy(dtype=float)
    y_finite = y_all[np.isfinite(y_all)]
    if y_finite.size:
        y_lo, y_hi = float(y_finite.min()), float(y_finite.max())
        span = max(y_hi - y_lo, 1e-6)
        ax.set_ylim(y_lo - 0.15 * span, y_hi + 0.55 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(list(COMPARISON_METRICS))
    ax.set_xlabel("Heldout eval split (masked: distance bin only)")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"{TITLE_PREFIX} — {SWEEP_ID}: val loss by trial @ step={ref_step}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    if n_variants == 2:
        a, b = variants
        delta_lines = []
        for metric in COMPARISON_METRICS:
            va, vb = pivot.loc[metric, a], pivot.loc[metric, b]
            if np.isfinite(va) and np.isfinite(vb):
                delta_lines.append(f"{metric}: Δ = {vb - va:+.4f}")
            else:
                delta_lines.append(f"{metric}: Δ = —")
        ax.text(
            0.99,
            0.99,
            f"Δ = {VARIANT_NAMES.get(b, b)} - {VARIANT_NAMES.get(a, a)}\n" + "\n".join(delta_lines),
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9, boxstyle="round,pad=0.3"),
        )

    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = "_vs_".join(variants)
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    png_path = PLOTS_DIR / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


# --- CLI ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Force a fresh W&B fetch.")
    args = parser.parse_args(argv)

    snapshot, meta = load_or_build_snapshot(refresh=args.refresh)
    ref_step = int(snapshot["ref_step"].iloc[0])
    wide = snapshot.pivot(index="variant_name", columns="metric", values="value")
    logger.info("ref_step = %d  meta = %s", ref_step, meta)
    logger.info("\n%s", wide.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "—"))
    render_comparison_bars(snapshot)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
