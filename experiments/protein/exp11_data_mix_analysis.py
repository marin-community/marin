# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze exp11 ``cd``-data-mix sweep results from W&B.

Companion to ``exp11_data_mix_sweep.py``. Pulls ``train/loss`` and every
``eval/<dataset>-(val|test)[-...]/loss`` history series for the runs in a
configured sweep, then renders one of two plot families based on the sweep's
``kind`` field:

* ``kind="mixture"`` (mix/scale sweeps): heatmap of loss across mixtures,
  cd-val bar grid grouped by single-quality / static / staged, and an optional
  training-time-vs-offline-eval scatter when ``EVAL_OF`` maps the sweep to
  its full-dataset companion.
* ``kind="variant"`` (lrsch, arch): a single grouped-bar comparison on the
  four val splits we care about (cd-val + H/M/L val, masked). The mixture is
  fixed (m11), so the per-trial discriminator is the LR schedule or model
  architecture; ``SweepConfig.mixture_regex`` captures the variant id.

Most runs are typically mid-flight, so we never wait for completion. Instead
we compute ``min_step`` = ``min`` over runs of ``max(_step)`` and slice each
run/metric at the largest history step ``<= min_step``. This gives a fair
cross-trial snapshot at the slowest run's progress.

The W&B fetch result is reduced to a single ``snapshot.csv`` under
``experiments/protein/exp11_data_mix_results/<sweep_id>/``; pass
``--refresh`` to re-pull from W&B. Adding another sweep = one new entry in
``SWEEPS``.

Usage::

    WANDB_API_KEY=... uv run --with matplotlib --with seaborn \\
        python -m experiments.protein.exp11_data_mix_analysis \\
            [--sweep <id>] [--refresh]
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
from matplotlib.lines import Line2D

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
            ``run.display_name`` to identify trial rows. For ``kind="variant"``
            sweeps (single mixture, varying schedule/arch), this captures the
            variant id (e.g. ``wsd``/``cosine``) into the ``mixture`` slot.
        kind: ``"mixture"`` (default) for sweeps that vary the data mixture
            (heatmap + cd-val bar grid + optional offline-eval scatter). Set to
            ``"variant"`` for sweeps that fix the mixture and vary another axis
            (LR schedule, arch) — those render via the grouped-bar comparison
            on the val splits only.
    """

    id: str
    entity: str
    project: str
    group: str
    tag: str
    name_regex: str
    mixture_regex: str
    kind: str = "mixture"


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
    # Offline full-dataset eval (no max_eval_batches cap) of the v2 mix
    # checkpoints, produced by ``exp11_data_mix_eval.py`` at EVAL_VERSION=v2.
    # One step (step=0) per run; build_snapshot picks it up via min(max(_step)).
    "run_mix_sweep_v2_eval": SweepConfig(
        id="run_mix_sweep_v2_eval",
        entity="eric-czech",
        project="marin",
        group="exp11-data-mix-eval",
        tag="eval",
        name_regex=r"^prot-exp11-dm-mix-100m-.*-v2-eval-v2$",
        mixture_regex=r"-(?P<mixture>m\d+)-lr",
    ),
    # exp11 lrsch-v1: 2 trials on m11 (size-proportional H/M/L blend), 100M /
    # batch=128 / ~4.3B tokens. WSD (linear, decay=0.2) vs cosine (AdamConfig
    # defaults: cosine + full decay). Single-mixture sweep so ``mixture_regex``
    # captures the variant id instead.
    "run_lrsch_sweep_v1": SweepConfig(
        id="run_lrsch_sweep_v1",
        entity="eric-czech",
        project="marin",
        group="exp11-data-mix",
        tag="lrsch",
        name_regex=r"^prot-exp11-dm-lrsch-100m-.*-m11-(wsd|cosine)-lr.*-v1-[0-9a-f]+$",
        mixture_regex=r"-m11-(?P<mixture>wsd|cosine)-lr",
        kind="variant",
    ),
    # exp11 arch-v1: 2 trials on m11, 100M / batch=128 / ~4.3B tokens. Llama
    # vs equivalent Qwen3 (same dims; Qwen3 defaults add QK-norm).
    "run_arch_sweep_v1": SweepConfig(
        id="run_arch_sweep_v1",
        entity="eric-czech",
        project="marin",
        group="exp11-data-mix",
        tag="arch",
        name_regex=r"^prot-exp11-dm-arch-100m-.*-m11-(llama|qwen3)-lr.*-v1-[0-9a-f]+$",
        mixture_regex=r"-m11-(?P<mixture>llama|qwen3)-lr",
        kind="variant",
    ),
}

DEFAULT_SWEEP = "run_mix_sweep_v2"

# Pair each training sweep with its offline-eval companion (see
# ``render_train_vs_full_eval_scatter``). Missing = no scatter for that sweep.
EVAL_OF: dict[str, str] = {"run_mix_sweep_v2": "run_mix_sweep_v2_eval"}

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


# Display labels for variant-axis sweeps (lrsch, arch). Kept separate from
# MIXTURE_NAMES because these identify the per-trial axis (LR schedule or
# model architecture) rather than the data mixture; sweeps with
# ``SweepConfig.kind="variant"`` look up labels here.
VARIANT_NAMES: dict[str, str] = {
    "wsd": "WSD (linear, decay=0.2)",
    "cosine": "Cosine (full decay)",
    "llama": "Llama",
    "qwen3": "Qwen3 (+QK-norm)",
}
# Stable, color-blind-tolerant categorical palette (Tableau 10 subset).
VARIANT_COLORS: dict[str, str] = {
    "wsd": "#4C78A8",
    "cosine": "#F58518",
    "llama": "#54A24B",
    "qwen3": "#9D7AB8",
}


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------


RESULTS_ROOT = Path(__file__).resolve().parent / "exp11_data_mix_results"


def _sweep_dir(sweep_id: str) -> Path:
    return RESULTS_ROOT / sweep_id


def _snapshot_path(sweep_id: str) -> Path:
    return _sweep_dir(sweep_id) / "snapshot.csv"


def _meta_path(sweep_id: str) -> Path:
    return _sweep_dir(sweep_id) / "meta.json"


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
# W&B fetch -> long-form snapshot + per-sweep eval-config meta
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalMeta:
    """W&B-derived eval config for a sweep; consensus across all matched runs.

    Pulled from ``run.config`` at build time and validated to be identical for
    every run in the sweep — figure annotations relying on these numbers
    (e.g. "training-time eval = N batches x B examples") are only meaningful
    if all runs share the same budget. ``build_snapshot`` raises on any
    disagreement.

    Both training and offline-eval runs expose ``trainer.train_batch_size``
    and ``trainer.max_eval_batches``; the seq-length key differs (``train_seq_len``
    for training, ``max_eval_length`` for offline ``eval_lm`` runs), so the
    extractor prefers ``max_eval_length`` and falls back to ``train_seq_len``.
    """

    eval_batch_size: int
    eval_seq_len: int
    # None = "no cap" (full dataset) — the offline-eval mode.
    max_eval_batches: int | None


# Dotted paths into the nested ``run.config`` dict (W&B stores the levanter
# config as a nested mapping, not dot-flattened). The first candidate whose
# path resolves wins; ``None`` values are preserved (e.g. offline-eval has
# ``max_eval_batches`` explicitly set to None).
_META_FROM_CONFIG: dict[str, tuple[str, ...]] = {
    "eval_batch_size": ("trainer.train_batch_size",),
    "max_eval_batches": ("trainer.max_eval_batches",),
    "eval_seq_len": ("max_eval_length", "train_seq_len"),
}

_MISSING = object()


def _get_nested(cfg: object, dotted_key: str) -> object:
    """Walk a nested mapping by dotted key path; ``_MISSING`` if any segment isn't present."""
    cur: object = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _extract_meta(run) -> dict[str, object]:
    """Pluck the eval-config fields we care about from a single run's config.

    Returns a dict (rather than ``EvalMeta``) so missing values surface as
    ``None`` and ``_consensus_meta`` can render a precise disagreement error.
    """
    cfg = dict(run.config)
    out: dict[str, object] = {}
    for field, candidates in _META_FROM_CONFIG.items():
        for cand in candidates:
            value = _get_nested(cfg, cand)
            if value is not _MISSING:
                out[field] = value
                break
        else:
            out[field] = None
    return out


def _consensus_meta(per_run: list[dict[str, object]], sweep_id: str) -> EvalMeta:
    """Assert every run agrees on each meta field; return the shared values.

    A None value is treated as "absent" — ``max_eval_batches`` can be absent
    (None) for offline-eval runs (uncapped) and still valid; the other two
    fields are required.
    """
    if not per_run:
        raise RuntimeError(f"Sweep {sweep_id}: no runs to derive meta from")
    consensus: dict[str, object] = {}
    for field in _META_FROM_CONFIG:
        values = {m.get(field) for m in per_run}
        if len(values) > 1:
            raise RuntimeError(f"Sweep {sweep_id}: runs disagree on {field!r}: {sorted(values, key=str)}")
        consensus[field] = values.pop()
    for required in ("eval_batch_size", "eval_seq_len"):
        if consensus[required] is None:
            raise RuntimeError(f"Sweep {sweep_id}: required meta field {required!r} missing from all runs")
    return EvalMeta(**consensus)  # type: ignore[arg-type]


def _save_meta(meta: EvalMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(meta), indent=2) + "\n")


def _load_meta(path: Path) -> EvalMeta:
    return EvalMeta(**json.loads(path.read_text()))


def _list_runs(sweep: SweepConfig) -> list[tuple[object, str]]:
    """Return ``[(wandb_run, mixture_id), ...]`` for this sweep."""
    api = wandb.Api()
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


def build_snapshot(sweep: SweepConfig) -> tuple[pd.DataFrame, EvalMeta]:
    """Pull every run, slice at ``min(max(_step))``, return snapshot + meta.

    Snapshot frame columns: ``mixture_id``, ``mixture_name``, ``metric``,
    ``value``, ``step_used``, ``ref_step``, ``run_name``, ``run_state``.
    ``EvalMeta`` is the consensus run-level eval config validated to be equal
    across every matched run.
    """
    matched = _list_runs(sweep)
    if not matched:
        raise RuntimeError(f"No runs matched sweep {sweep.id}")

    # Pass 1: fetch history + extract meta per run.
    per_run: list[tuple[object, str, pd.DataFrame]] = []
    metas: list[dict[str, object]] = []
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
        metas.append(_extract_meta(run))

    if not per_run:
        raise RuntimeError(f"No runs in sweep {sweep.id} produced usable history")

    meta = _consensus_meta(metas, sweep.id)
    logger.info("Sweep %s eval meta (consensus across %d runs): %s", sweep.id, len(metas), meta)

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
    return pd.DataFrame(rows), meta


def load_or_build_snapshot(sweep: SweepConfig, *, refresh: bool) -> tuple[pd.DataFrame, EvalMeta]:
    """Cached fetch: returns (snapshot_df, meta).

    Three cache states:
    - both ``snapshot.csv`` and ``meta.json`` present: load both, no W&B call.
    - only ``snapshot.csv`` present: backfill ``meta.json`` from a cheap
      ``run.config`` fetch (no ``scan_history`` needed).
    - neither present: full ``build_snapshot``.

    ``--refresh`` always falls through to the full rebuild.
    """
    csv_path = _snapshot_path(sweep.id)
    meta_path = _meta_path(sweep.id)
    if not refresh and csv_path.exists() and meta_path.exists():
        logger.info("Loading cached snapshot+meta from %s", csv_path.parent)
        return pd.read_csv(csv_path), _load_meta(meta_path)
    if not refresh and csv_path.exists():
        logger.info("Backfilling meta.json from W&B (snapshot already cached)")
        runs = _list_runs(sweep)
        meta = _consensus_meta([_extract_meta(r) for r, _ in runs], sweep.id)
        _save_meta(meta, meta_path)
        return pd.read_csv(csv_path), meta
    snapshot, meta = build_snapshot(sweep)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(csv_path, index=False)
    _save_meta(meta, meta_path)
    logger.info("Wrote %d rows to %s and meta to %s", len(snapshot), csv_path, meta_path)
    return snapshot, meta


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


# Mixture groups for the cd-val bar plot and the train-vs-full-eval scatter:
# pure single-quality, static blends, staged curricula. Colors are the Tableau
# 10 categorical defaults (stable, color-blind-tolerant).
MIXTURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Single quality", ("m1", "m2", "m3")),
    ("Static blends", ("m4", "m5", "m6")),
    ("Staged", ("m7", "m8", "m9")),
)
GROUP_COLORS: dict[str, str] = {
    "Single quality": "#4C78A8",
    "Static blends": "#54A24B",
    "Staged": "#E45756",
}
GROUP_BY_MIXTURE: dict[str, str] = {mid: label for label, mixtures in MIXTURE_GROUPS for mid in mixtures}

# Full-dataset example count for the offline cd-val eval — packed-sequence
# count in the protein-docs cd-val cache. NOT derivable from W&B (eval runs
# only log per-component loss/bpb, no per-component iter counts) and not
# trivial to derive from the cache either (depends on eval-time packing into
# seq_len=8192 windows). Verified manually by Eric (2026-05-24): 53,699
# packed sequences. Update if the cd-val cache is rebuilt.
CDVAL_FULL_EXAMPLES = 53_699


def _ordered_columns(columns: list[str]) -> list[str]:
    in_order = [c for c in PREFERRED_METRIC_ORDER if c in columns]
    rest = sorted(c for c in columns if c not in PREFERRED_METRIC_ORDER)
    return [*in_order, *rest]


def _mixture_sort_key(mixture_id: str) -> tuple[int, str]:
    m = re.search(r"\d+", mixture_id)
    return (int(m.group()) if m else 0, mixture_id)


def render_heatmap(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
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
    pdf_path = out_dir / "val_test_loss_by_mixture.pdf"
    png_path = out_dir / "val_test_loss_by_mixture.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


def render_cdval_bars(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
    """1x3 grid of cd-val bar charts, one per ``MIXTURE_GROUPS`` row.

    Y-axis range is fixed across subplots so cross-group comparison is fair.
    """
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
    pdf_path = out_dir / "cdval_loss_by_mixture.pdf"
    png_path = out_dir / "cdval_loss_by_mixture.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


# Metrics that drive the variant-axis decision (lrsch / arch): cd-val plus the
# three per-quality val splits. All are masked (distance-bin loss only). The
# *-test, *-train (IID-carve), and cd-val-unmasked metrics are intentionally
# excluded — they don't change the WSD-vs-cosine or Llama-vs-Qwen3 verdict.
VARIANT_COMPARISON_METRICS: tuple[str, ...] = ("cd-val", "high-val", "medium-val", "low-val")


def _variant_sort_key(variant_id: str) -> int:
    """Sort variants in the order they appear in VARIANT_NAMES (definition order)."""
    keys = list(VARIANT_NAMES.keys())
    return keys.index(variant_id) if variant_id in keys else len(keys)


def render_variant_comparison_bars(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
    """Grouped bars: one bar per trial variant per val metric.

    Restricted to ``VARIANT_COMPARISON_METRICS`` (cd-val + the three per-quality
    val splits). Each metric group annotates the absolute loss on each bar plus
    a ``Δ`` line for the difference (second variant minus first) so the
    direction of the effect is immediately readable.
    """
    df = snapshot[snapshot["metric"].isin(VARIANT_COMPARISON_METRICS)]
    if df.empty:
        logger.warning("No target metrics in snapshot for sweep %s; skipping variant comparison", sweep.id)
        return

    ref_step = int(snapshot["ref_step"].iloc[0])
    pivot = df.pivot(index="metric", columns="mixture_id", values="value").reindex(list(VARIANT_COMPARISON_METRICS))
    variants = sorted(pivot.columns, key=_variant_sort_key)
    pivot = pivot[variants]

    n_variants = len(variants)
    n_metrics = len(VARIANT_COMPARISON_METRICS)
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

    # Tighten y-axis around the data so deltas at the 1e-3 scale are visible.
    # Generous top headroom so the Δ box in the upper-right corner clears the
    # tallest bar + its value annotation; padding scales with the data range
    # so small-delta sweeps (lrsch) and large-delta sweeps (arch) both fit.
    y_all = pivot.to_numpy(dtype=float)
    y_finite = y_all[np.isfinite(y_all)]
    if y_finite.size:
        y_lo, y_hi = float(y_finite.min()), float(y_finite.max())
        span = max(y_hi - y_lo, 1e-6)
        ax.set_ylim(y_lo - 0.15 * span, y_hi + 0.55 * span)

    ax.set_xticks(x)
    ax.set_xticklabels(list(VARIANT_COMPARISON_METRICS))
    ax.set_xlabel("Heldout eval split (masked: distance bin only)")
    ax.set_ylabel("Loss")
    ax.set_title(
        f"{TITLE_PREFIX} — {sweep.id}: val loss by trial @ step={ref_step}",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Δ line per metric: only meaningful when exactly two variants are compared
    # (always true for the lrsch/arch sweeps). Renders just under the x-axis
    # tick labels so the bar canvas stays clean.
    if n_variants == 2:
        a, b = variants
        delta_lines = []
        for metric in VARIANT_COMPARISON_METRICS:
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

    # Filename reflects the actual A/B comparison (e.g. "wsd_vs_cosine.png",
    # "llama_vs_qwen3.png") so it's recognizable on disk without opening it.
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "_vs_".join(variants)
    pdf_path = out_dir / f"{stem}.pdf"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


def render_train_vs_full_eval_scatter(
    train_snapshot: pd.DataFrame,
    train_meta: EvalMeta,
    eval_snapshot: pd.DataFrame,
    eval_meta: EvalMeta,
    train_sweep: SweepConfig,
    eval_sweep: SweepConfig,
    out_dir: Path,
) -> None:
    """Scatter: training-time (capped) vs offline (full) cd-val loss per mixture.

    cd-val only — one point per mixture. Other heldout metrics (H/M/L val/test,
    IID-carve, cd-val-unmasked) add visual noise without changing the
    consistency story. Color encodes the mixture group; a least-squares fit
    (formula in legend) shows how well the capped in-loop eval tracks the
    full-dataset truth — slope≈1 + intercept≈0 means the cheap eval is a
    faithful proxy.
    """
    cols = ["mixture_id", "metric"]
    df = (
        train_snapshot.loc[train_snapshot["metric"] == "cd-val", [*cols, "value"]]
        .rename(columns={"value": "train_time"})
        .merge(
            eval_snapshot.loc[eval_snapshot["metric"] == "cd-val", [*cols, "value"]].rename(
                columns={"value": "offline"}
            ),
            on=cols,
        )
        .assign(group=lambda d: d["mixture_id"].map(GROUP_BY_MIXTURE))
        .dropna(subset=["train_time", "offline", "group"])
    )
    if df.empty:
        logger.warning("No overlapping cd-val rows between %s and %s", train_sweep.id, eval_sweep.id)
        return
    if train_meta.eval_seq_len != eval_meta.eval_seq_len:
        raise RuntimeError(
            f"Seq length mismatch: training sweep {train_sweep.id} ran at {train_meta.eval_seq_len}, "
            f"offline-eval sweep {eval_sweep.id} ran at {eval_meta.eval_seq_len}; not comparable."
        )

    x = df["train_time"].to_numpy(dtype=float)
    y = df["offline"].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)

    # Wider canvas so the right-side mixture key has room outside the axes;
    # short on the vertical to keep the figure compact in reports/issues.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, _ in MIXTURE_GROUPS:
        sub = df[df["group"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["train_time"],
            sub["offline"],
            c=GROUP_COLORS[label],
            label="_nolegend_",
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            s=80,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                row["mixture_id"],
                (row["train_time"], row["offline"]),
                xytext=(4, -2),
                textcoords="offset points",
                fontsize=9,
                color=GROUP_COLORS[label],
                fontweight="bold",
                ha="left",
                va="top",
            )

    # Per-axis natural bounds with 8% padding — y=x no longer applies, so the
    # union-bounded square layout (and ``set_aspect("equal")``) just wasted
    # canvas. Each axis now spans only its own data range.
    pad_x = 0.08 * float(np.ptp(x))
    pad_y = 0.08 * float(np.ptp(y))
    x_lo, x_hi = float(x.min()) - pad_x, float(x.max()) + pad_x
    y_lo, y_hi = float(y.min()) - pad_y, float(y.max()) + pad_y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    sign = "+" if intercept >= 0 else "-"
    fit_label = f"y = {slope:.3f}*x {sign} {abs(intercept):.3f}"
    (fit_line,) = ax.plot(
        [x_lo, x_hi],
        [slope * x_lo + intercept, slope * x_hi + intercept],
        "--",
        color="grey",
        linewidth=1.2,
        zorder=1,
        label=fit_label,
    )

    ax.set_xlabel(f"Training-time cd-val loss  (capped: {train_meta.max_eval_batches} batches)")
    ax.set_ylabel("Offline cd-val loss  (full dataset, no cap)")
    ax.set_title(
        f"{TITLE_PREFIX} — {train_sweep.id}: training-time vs offline cd-val loss\n"
        f"Pearson r = {r:.6f}   (1 - r = {1 - r:.2e};   n = {len(df)} mixtures)",
        fontsize=11,
    )
    ax.grid(True, linestyle="--", alpha=0.4)

    # Two legends: fit formula stays compact inside the axes; the mixture key
    # sits outside on the right so all 9 ``MIXTURE_NAMES`` entries fit without
    # crowding the data. Names match the heatmap row labels and bar-chart
    # x-ticks for cross-figure consistency.
    fit_legend = ax.legend(handles=[fit_line], loc="upper left", fontsize=9, framealpha=0.9)
    ax.add_artist(fit_legend)
    mixture_handles: list[Line2D] = []
    for group_label, mids in MIXTURE_GROUPS:
        mixture_handles.append(Line2D([], [], linestyle="none", marker="none", label=group_label))
        for mid in mids:
            mixture_handles.append(
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markerfacecolor=GROUP_COLORS[group_label],
                    markeredgecolor="black",
                    markersize=8,
                    label=f"  {MIXTURE_NAMES.get(mid, mid)}",
                )
            )
    ax.legend(
        handles=mixture_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        framealpha=0.9,
        title="Mixture",
        title_fontsize=9,
    )

    train_examples = train_meta.eval_batch_size * train_meta.max_eval_batches
    train_tokens = train_examples * train_meta.eval_seq_len
    offline_tokens = CDVAL_FULL_EXAMPLES * eval_meta.eval_seq_len
    multiplier = CDVAL_FULL_EXAMPLES / train_examples
    ax.text(
        0.98,
        0.02,
        (
            f"Training-time: capped at {train_meta.max_eval_batches} batches x {train_meta.eval_batch_size} "
            f"= {train_examples:,} examples ({train_tokens / 1e6:.1f}M tokens)\n"
            f"Offline (v2): full cd-val dataset = {CDVAL_FULL_EXAMPLES:,} examples "
            f"({offline_tokens / 1e6:.0f}M tokens, {multiplier:.0f}x larger)"
        ),
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="bottom",
        bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9, boxstyle="round,pad=0.3"),
    )

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "train_vs_full_eval_scatter.pdf"
    png_path = out_dir / "train_vs_full_eval_scatter.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s  (r=%.6f, 1-r=%.2e, n=%d)", pdf_path, png_path, r, 1 - r, len(df))


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
    snapshot, meta = load_or_build_snapshot(sweep, refresh=args.refresh)

    ref_step = int(snapshot["ref_step"].iloc[0])
    wide = snapshot.pivot(index="mixture_name", columns="metric", values="value")
    logger.info("ref_step = %d  meta = %s", ref_step, meta)
    logger.info("\n%s", wide.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "—"))

    plots_dir = _plots_dir(sweep.id)
    if sweep.kind == "variant":
        # lrsch/arch sweeps fix the mixture and vary another axis; the heatmap
        # and per-group cd-val bars assume a 9-mixture cross-section and would
        # render mostly NaN here, so skip them.
        render_variant_comparison_bars(snapshot, sweep, plots_dir)
    else:
        render_heatmap(snapshot, sweep, plots_dir)
        render_cdval_bars(snapshot, sweep, plots_dir)

        eval_sweep_id = EVAL_OF.get(args.sweep)
        if eval_sweep_id is not None:
            eval_sweep = SWEEPS[eval_sweep_id]
            eval_snapshot, eval_meta = load_or_build_snapshot(eval_sweep, refresh=args.refresh)
            render_train_vs_full_eval_scatter(snapshot, meta, eval_snapshot, eval_meta, sweep, eval_sweep, plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
