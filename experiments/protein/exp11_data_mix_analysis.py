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
from matplotlib.ticker import FuncFormatter, NullFormatter

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
    # Human-readable label for figure titles (e.g. ``"Mix-stage sweep v2"``).
    # ``id`` (e.g. ``"run_mix_sweep_v2"``) still drives the cache-dir name and
    # appears in logs; ``display_name`` is what readers see on plots.
    display_name: str
    # Slice strategy for in-progress sweeps. False (default) = use a single
    # ``ref_step = min over runs of max(_step)`` so every run's cell lines up
    # at the same step. True = use each run's latest non-null value for each
    # metric independently — needed when runs are still mid-flight and we want
    # the best-available cd-val per run without waiting for the slowest one.
    per_run_final: bool = False


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
        display_name="Mix-stage sweep v2",
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
        display_name="Mix-stage offline eval v2",
    ),
    # exp11 scale-stage v6 (6 mixtures, 1.5B model, ~21.5B-token target).
    # Runs are still in flight; per_run_final=True takes each run's latest
    # logged cd-val rather than the common min-of-max slice, so the bar chart
    # surfaces best-available numbers even while training continues.
    "run_scale_sweep_v6": SweepConfig(
        id="run_scale_sweep_v6",
        entity="eric-czech",
        project="marin",
        group="exp11-data-mix",
        tag="scale",
        name_regex=r"^prot-exp11-dm-scale-1_5b-.*-v6-[0-9a-f]+$",
        mixture_regex=r"-(?P<mixture>m\d+)-lr",
        display_name="Scale-stage sweep v6",
        per_run_final=True,
    ),
}

DEFAULT_SWEEP = "run_mix_sweep_v2"

# Pair each training sweep with its offline-eval companion (see
# ``render_train_vs_full_eval_scatter``). Missing = no scatter for that sweep.
EVAL_OF: dict[str, str] = {"run_mix_sweep_v2": "run_mix_sweep_v2_eval"}

# Pair each scale sweep with the mix sweep whose recipes it re-runs at larger
# scale (drives ``render_mix_vs_scale_scatter`` in ``main()``). Missing = no
# cross-scale scatter for that sweep.
MIX_OF: dict[str, str] = {"run_scale_sweep_v6": "run_mix_sweep_v2"}

# Recipes that appear in both the mix sweep (100M, ~4.3B tok) and the scale
# sweep (1.5B, ~21.5B tok). Each entry is ``(mix_id, scale_id, label, group)``
# where ``group`` keys into ``GROUP_COLORS``. Mixtures absent here (m4/m5/m8/m9
# in mix; m13 in scale) have no cross-sweep counterpart and are dropped from
# the cross-scale scatter.
SHARED_MIXTURE_PAIRS: tuple[tuple[str, str, str, str], ...] = (
    ("m1", "m10", "H", "Single quality"),
    ("m2", "m14", "M", "Single quality"),
    ("m3", "m15", "L", "Single quality"),
    ("m6", "m11", "H31/M26/L43", "Blends"),
    ("m7", "m12", "L→H", "Blends"),
)

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
    # Scale-stage mixtures (run_scale_sweep, 1.5B).
    "m10": "m10 (H)",
    "m11": "m11 (H31/M26/L43)",
    "m12": "m12 (L→H)",
    "m13": "m13 (L→M→H)",
    "m14": "m14 (M)",
    "m15": "m15 (L)",
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


def _cdval_history_path(sweep_id: str) -> Path:
    return _sweep_dir(sweep_id) / "cdval_history.csv"


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

# Full W&B key for the cd-val eval loss curve (``cd-val`` is its short label).
# Used by the loss-curve fetch in addition to the single-step snapshot.
CDVAL_METRIC = "eval/protein-docs-cd-val/loss"


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

    Pulled from ``run.config`` + ``run.tags`` at build time and validated to
    be identical for every run in the sweep — figure annotations relying on
    these numbers (e.g. "training-time eval = N batches x B examples") are
    only meaningful if all runs share the same budget. ``build_snapshot``
    raises on any disagreement.

    Both training and offline-eval runs expose ``trainer.train_batch_size``
    and ``trainer.max_eval_batches``; the seq-length key differs (``train_seq_len``
    for training, ``max_eval_length`` for offline ``eval_lm`` runs), so the
    extractor prefers ``max_eval_length`` and falls back to ``train_seq_len``.

    ``num_train_steps``, ``params_exact``, and ``tokens_exact`` are best-effort:
    they default to None when unavailable (e.g. older cached ``meta.json``
    files that predate these fields). ``exp11_data_mix_sweep._build_trial``
    bakes the authoritative ``params_exact=<int>`` and ``tokens_exact=<int>``
    into ``run.tags`` — we always prefer those over recomputing from
    ``batch * seq * steps`` so the figure reflects what was actually trained.
    """

    eval_batch_size: int
    eval_seq_len: int
    # None = "no cap" (full dataset) — the offline-eval mode.
    max_eval_batches: int | None
    num_train_steps: int | None = None
    params_exact: int | None = None
    tokens_exact: int | None = None
    # Extrapolated per-run FLOPs at completion. Computed from Levanter's
    # logged ``throughput/total_gflops`` (cumulative GFLOPs through the
    # latest step): rate = total_gflops / latest_step is the constant
    # per-step FLOPs cost; rate * num_train_steps * 1e9 gives the at-completion
    # FLOPs. For finished runs the latest step equals num_train_steps so no
    # extrapolation occurs; the mean across runs is used since rates differ
    # only by step-noise (FLOPs depend on model + step count, not mixture).
    final_flops_per_run: float | None = None


# Dotted paths into the nested ``run.config`` dict (W&B stores the levanter
# config as a nested mapping, not dot-flattened). The first candidate whose
# path resolves wins; ``None`` values are preserved (e.g. offline-eval has
# ``max_eval_batches`` explicitly set to None).
_META_FROM_CONFIG: dict[str, tuple[str, ...]] = {
    "eval_batch_size": ("trainer.train_batch_size",),
    "max_eval_batches": ("trainer.max_eval_batches",),
    "eval_seq_len": ("max_eval_length", "train_seq_len"),
    "num_train_steps": ("trainer.num_train_steps",),
}

# Fields whose absence (None across all runs) is acceptable; everything else
# in EvalMeta must be present after consensus. Used by ``_consensus_meta`` to
# decide whether to raise on a missing-only field.
_OPTIONAL_META_FIELDS: frozenset[str] = frozenset(
    {"max_eval_batches", "num_train_steps", "params_exact", "tokens_exact", "final_flops_per_run"}
)

# Levanter logs cumulative GFLOPs under this summary key; used to derive
# ``final_flops_per_run`` in ``_consensus_meta``.
TOTAL_GFLOPS_KEY = "throughput/total_gflops"

_MISSING = object()


def _get_nested(cfg: object, dotted_key: str) -> object:
    """Walk a nested mapping by dotted key path; ``_MISSING`` if any segment isn't present."""
    cur: object = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _parse_int_tag(tags: object, prefix: str) -> int | None:
    """Parse a ``<prefix>=<int>`` tag from a run's tag list (e.g. ``params_exact=``).

    Tags are the authoritative record of what was trained; the sweep script
    bakes ``params_exact`` and ``tokens_exact`` in at launch. Always prefer
    these over recomputing from config (``batch * seq * steps``) so the figure
    reflects the trained value verbatim.
    """
    if not isinstance(tags, (list, tuple)):
        return None
    for tag in tags:
        if isinstance(tag, str) and tag.startswith(prefix):
            try:
                return int(tag.split("=", 1)[1])
            except ValueError:
                return None
    return None


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
    tags = getattr(run, "tags", None)
    out["params_exact"] = _parse_int_tag(tags, "params_exact=")
    out["tokens_exact"] = _parse_int_tag(tags, "tokens_exact=")
    # Raw inputs for the ``final_flops_per_run`` derivation in
    # ``_consensus_meta``. Per-run (since rate * num_train_steps extrapolates
    # to completion); not declared as EvalMeta fields, so the strict consensus
    # loop ignores them.
    summary = dict(run.summary) if getattr(run, "summary", None) is not None else {}
    out["_latest_total_gflops"] = summary.get(TOTAL_GFLOPS_KEY)
    out["_latest_step"] = summary.get("_step")
    return out


def _derive_final_flops(per_run: list[dict[str, object]], num_train_steps: int | None) -> float | None:
    """Mean (total_gflops / latest_step) across runs * num_train_steps * 1e9.

    Returns None when ``num_train_steps`` is unknown or no run has both
    ``throughput/total_gflops`` and ``_step`` logged. The mean of per-run
    rates is robust to small step-noise differences; rates should be equal
    in principle since FLOPs depend on model + step count, not mixture.
    """
    if num_train_steps is None:
        return None
    rates = []
    for m in per_run:
        gflops = m.get("_latest_total_gflops")
        step = m.get("_latest_step")
        if gflops is None or step is None:
            continue
        step_int = int(step)
        if step_int <= 0:
            continue
        rates.append(float(gflops) / step_int)
    if not rates:
        return None
    return (sum(rates) / len(rates)) * float(num_train_steps) * 1e9


def _consensus_meta(per_run: list[dict[str, object]], sweep_id: str) -> EvalMeta:
    """Assert every run agrees on each meta field; return the shared values.

    A None value is treated as "absent" — fields in ``_OPTIONAL_META_FIELDS``
    can be absent and still valid (e.g. ``max_eval_batches`` is None for
    offline-eval runs); all other fields are required.
    """
    if not per_run:
        raise RuntimeError(f"Sweep {sweep_id}: no runs to derive meta from")
    consensus: dict[str, object] = {}
    derived_fields = {"final_flops_per_run"}
    for field in dataclasses.fields(EvalMeta):
        name = field.name
        if name in derived_fields:
            continue
        values = {m.get(name) for m in per_run}
        if len(values) > 1:
            raise RuntimeError(f"Sweep {sweep_id}: runs disagree on {name!r}: {sorted(values, key=str)}")
        consensus[name] = values.pop()
    for name, value in consensus.items():
        if value is None and name not in _OPTIONAL_META_FIELDS:
            raise RuntimeError(f"Sweep {sweep_id}: required meta field {name!r} missing from all runs")
    consensus["final_flops_per_run"] = _derive_final_flops(per_run, consensus.get("num_train_steps"))  # type: ignore[arg-type]
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

    if sweep.per_run_final:
        # Per-run-final mode: slice each (run, metric) at that run's own
        # largest step. ``ref_step`` is set to ``step_used`` per row so the
        # column stays well-defined; renderers detect non-uniform ref_step
        # and adjust their titles accordingly.
        logger.info("Per-run-final mode: using each run's latest non-null value per metric")
        rows: list[dict] = []
        for run, mixture, df in per_run:
            run_max_step = int(df["_step"].max())
            metrics = [c for c in df.columns if c != "_step" and _is_keep_metric(c)]
            for metric in metrics:
                value, step = _value_at(df, metric, run_max_step)
                if value is None:
                    continue
                rows.append(
                    {
                        "mixture_id": mixture,
                        "mixture_name": MIXTURE_NAMES.get(mixture, mixture),
                        "metric": _short_metric_label(metric),
                        "value": value,
                        "step_used": step,
                        "ref_step": step,
                        "run_name": run.display_name,
                        "run_state": run.state,
                    }
                )
        return pd.DataFrame(rows), meta

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
# cd-val loss curves (full history) + power-law fit
# ---------------------------------------------------------------------------


def _fetch_cdval_history(run, tokens_per_step: int) -> pd.DataFrame:
    """cd-val loss curve for one run, with per-step tokens + cumulative FLOPs.

    ``cd-val`` is logged sparsely (every ``steps_per_eval``) while
    ``throughput/total_gflops`` is logged every step; both are scanned in one
    pass and the dense FLOPs series is interpolated onto the cd-val steps so
    each curve point carries its cumulative training FLOPs. Returns columns
    ``step``, ``tokens``, ``total_flops``, ``cd_val_loss`` (empty frame if the
    run lacks either series).
    """
    rows = list(run.scan_history(keys=["_step", CDVAL_METRIC, TOTAL_GFLOPS_KEY], page_size=10_000))
    cols = ["step", "tokens", "total_flops", "cd_val_loss"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["_step"] = df["_step"].astype(int)
    flops = df[["_step", TOTAL_GFLOPS_KEY]].dropna().sort_values("_step")
    cd = df[["_step", CDVAL_METRIC]].dropna().sort_values("_step")
    if cd.empty or flops.empty:
        return pd.DataFrame(columns=cols)
    steps = cd["_step"].to_numpy()
    total_flops = np.interp(steps, flops["_step"].to_numpy(), flops[TOTAL_GFLOPS_KEY].to_numpy()) * 1e9
    return pd.DataFrame(
        {
            "step": steps,
            "tokens": steps * tokens_per_step,
            "total_flops": total_flops,
            "cd_val_loss": cd[CDVAL_METRIC].to_numpy(dtype=float),
        }
    )


def load_or_build_cdval_history(sweep: SweepConfig, meta: EvalMeta, *, refresh: bool) -> pd.DataFrame:
    """Cached long-form cd-val loss curves for every run in the sweep.

    Columns: ``mixture_id``, ``mixture_name``, ``run_name``, ``step``,
    ``tokens``, ``total_flops``, ``cd_val_loss``. ``tokens = step * (train
    batch * seq)``; ``total_flops`` is the cumulative ``throughput/total_gflops``
    interpolated onto each cd-val step. Cached as ``cdval_history.csv``;
    ``--refresh`` forces a re-pull.
    """
    path = _cdval_history_path(sweep.id)
    if not refresh and path.exists():
        logger.info("Loading cached cd-val history from %s", path)
        return pd.read_csv(path)
    tokens_per_step = meta.eval_batch_size * meta.eval_seq_len
    frames: list[pd.DataFrame] = []
    for run, mixture in _list_runs(sweep):
        history = _fetch_cdval_history(run, tokens_per_step)
        if history.empty:
            logger.warning("No cd-val history for %s; skipping", run.display_name)
            continue
        history.insert(0, "run_name", run.display_name)
        history.insert(0, "mixture_name", MIXTURE_NAMES.get(mixture, mixture))
        history.insert(0, "mixture_id", mixture)
        frames.append(history)
    if not frames:
        raise RuntimeError(f"No cd-val history fetched for sweep {sweep.id}")
    out = pd.concat(frames, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info("Wrote %d cd-val history rows to %s", len(out), path)
    return out


@dataclass(frozen=True)
class PowerLawFit:
    """Pure power law ``L(D) = A * D^(-alpha)`` fit in log-log space.

    ``alpha`` is the per-token decay exponent; ``coef_a`` the scale. ``data_lo``
    / ``data_hi`` bound the ``n_points`` cd-val samples actually fit (i.e. the
    span of in-window data, which may be narrower than the nominal window since
    the floor can fall below the first eval point) — use these to draw the fit
    only over the portion of training it was fit on.
    """

    coef_a: float
    alpha: float
    r_squared: float
    data_lo: float
    data_hi: float
    n_points: int

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        return self.coef_a * np.power(tokens, -self.alpha)


def fit_power_law(tokens: np.ndarray, loss: np.ndarray, t_lo: float, t_hi: float) -> PowerLawFit | None:
    """Least-squares ``log L = log A - alpha * log D`` over ``[t_lo, t_hi]``.

    ``r_squared`` is the coefficient of determination of that line in log-log
    space (1.0 == perfect power law). Returns None when fewer than two in-window
    samples survive (can't fit a line); callers skip overlaying a fit then.
    """
    mask = (tokens >= t_lo) & (tokens <= t_hi) & (tokens > 0) & (loss > 0)
    t, lo = tokens[mask], loss[mask]
    if t.size < 2:
        return None
    log_t, log_l = np.log(t), np.log(lo)
    slope, intercept = np.polyfit(log_t, log_l, 1)
    residuals = log_l - (slope * log_t + intercept)
    ss_tot = float(np.sum((log_l - log_l.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals**2)) / ss_tot if ss_tot > 0 else float("nan")
    return PowerLawFit(
        coef_a=float(np.exp(intercept)),
        alpha=float(-slope),
        r_squared=r_squared,
        data_lo=float(t.min()),
        data_hi=float(t.max()),
        n_points=int(t.size),
    )


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
# Scale stage: only two groups. m10/m14/m15 are the H/M/L single-quality
# baselines; m11 (static blend) and m12/m13 (staged curricula) are bucketed
# together under "Blends" since the scale sweep only re-runs one variant of
# each blend type.
SCALE_MIXTURE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Single quality", ("m10", "m14", "m15")),
    ("Blends", ("m11", "m12", "m13")),
)
# Per-sweep override of MIXTURE_GROUPS for the cd-val bar chart. Anything not
# listed falls back to ``MIXTURE_GROUPS``.
MIXTURE_GROUPS_BY_SWEEP: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "run_scale_sweep_v6": SCALE_MIXTURE_GROUPS,
}
GROUP_COLORS: dict[str, str] = {
    "Single quality": "#4C78A8",
    "Static blends": "#54A24B",
    "Staged": "#E45756",
    "Blends": "#54A24B",
}
GROUP_BY_MIXTURE: dict[str, str] = {mid: label for label, mixtures in MIXTURE_GROUPS for mid in mixtures}

# Staged-curriculum mixtures whose training distribution shifts mid-run
# (L->H, H->L, multi-stage). Their loss curve has stage-transition kinks, so a
# single power law is ill-posed; they're excluded from the projected-vs-actual
# rank-correlation panel (only stationary single-quality + static-blend mixes
# count there). Names per ``MIXTURE_NAMES``.
STAGED_MIXTURES: frozenset[str] = frozenset({"m7", "m8", "m9", "m12", "m13"})

# Power-law fit window as a fraction of num_train_steps. The lower bound (0.1)
# skips the warmup ramp and the early super-fast-descent regime; the upper bound
# (0.8) is the start of the WSD decay phase (warmup=0.1, decay=0.2), so the fit
# covers the stable schedule up to where the LR cooldown begins.
FIT_LOW_STEP_FRAC: float = 0.1
FIT_HIGH_STEP_FRAC: float = 0.8


# Full-dataset example count for the offline cd-val eval — packed-sequence
# count in the protein-docs cd-val cache. NOT derivable from W&B (eval runs
# only log per-component loss/bpb, no per-component iter counts) and not
# trivial to derive from the cache either (depends on eval-time packing into
# seq_len=8192 windows). Manually verified 2026-05-24 at 53,699 packed
# sequences. Update if the cd-val cache is rebuilt.
CDVAL_FULL_EXAMPLES = 53_699


def _ordered_columns(columns: list[str]) -> list[str]:
    in_order = [c for c in PREFERRED_METRIC_ORDER if c in columns]
    rest = sorted(c for c in columns if c not in PREFERRED_METRIC_ORDER)
    return [*in_order, *rest]


def _mixture_sort_key(mixture_id: str) -> tuple[int, str]:
    m = re.search(r"\d+", mixture_id)
    return (int(m.group()) if m else 0, mixture_id)


def _step_label(snapshot: pd.DataFrame) -> str:
    """Title fragment describing the slice step.

    Uniform ref_step => "step=N" (the common-slice case). Non-uniform =>
    "step=A..B per run (latest)" so the title makes clear the values are
    not synchronized across runs.
    """
    unique = sorted({int(x) for x in snapshot["ref_step"].unique()})
    if len(unique) == 1:
        return f"step={unique[0]}"
    return f"step={unique[0]}..{unique[-1]} per run (latest)"


def render_heatmap(snapshot: pd.DataFrame, sweep: SweepConfig, out_dir: Path) -> None:
    step_label = _step_label(snapshot)

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
        f"{TITLE_PREFIX} — {sweep.id}: loss by mixture @ {step_label}\n"
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


def _bar_chart_subtitle(snapshot: pd.DataFrame, sweep: SweepConfig, meta: EvalMeta) -> str | None:
    """Technical subtitle for the cd-val bar chart, or None.

    Returns None if any required EvalMeta field is missing (older cached
    snapshots may predate ``tokens_exact``/``final_flops_per_run``; rerun
    with a fresh meta extraction to populate them).

    Line 1: run shape — params, target steps, target tokens, at-completion
    FLOPs (FLOPs computed from each run's ``throughput/total_gflops`` /
    latest step, averaged, then scaled to ``num_train_steps``).

    Line 2: per-run progress vs target. Omitted when every run displays as
    100% complete; the line would just restate "completion".
    """
    if (
        meta.num_train_steps is None
        or meta.params_exact is None
        or meta.tokens_exact is None
        or meta.final_flops_per_run is None
    ):
        return None
    total = meta.num_train_steps
    # Match the compact "2.95e20" style (no '+' sign) — matplotlib's default
    # ``:.2e`` formatter prints "e+20".
    mantissa, exponent = f"{meta.final_flops_per_run:.2e}".split("e")
    flops_str = f"{mantissa}e{int(exponent)}"
    line1 = (
        f"{meta.params_exact / 1e9:.2f}B params · target {total:,} steps "
        f"= {meta.tokens_exact / 1e9:.3f}B tokens · ~{flops_str} FLOPs/run @ completion"
    )
    ref_lo = int(snapshot["ref_step"].min())
    ref_hi = int(snapshot["ref_step"].max())
    pct_lo, pct_hi = 100 * ref_lo / total, 100 * ref_hi / total
    if round(pct_lo) == 100 and round(pct_hi) == 100:
        return line1
    if ref_lo == ref_hi:
        progress = f"current: step {ref_lo:,} / {total:,} ({pct_lo:.0f}% complete)"
    else:
        progress = f"current: step {ref_lo:,}..{ref_hi:,} / {total:,} (~{pct_lo:.0f}-{pct_hi:.0f}% complete)"
    return f"{line1}\n{progress}"


def render_cdval_bars(snapshot: pd.DataFrame, sweep: SweepConfig, meta: EvalMeta, out_dir: Path) -> None:
    """1xN grid of cd-val bar charts, one per group in the sweep's group list.

    Y-axis range is fixed across subplots so cross-group comparison is fair.
    """
    step_label = _step_label(snapshot)
    groups = MIXTURE_GROUPS_BY_SWEEP.get(sweep.id, MIXTURE_GROUPS)
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

    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups), 3.8), sharey=True)
    # plt.subplots collapses to a single Axes when ncols=1; normalize to a list.
    if len(groups) == 1:
        axes = [axes]
    for ax, (group_label, mixture_ids) in zip(axes, groups, strict=True):
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
    subtitle = _bar_chart_subtitle(snapshot, sweep, meta)
    title = f"{TITLE_PREFIX} — {sweep.display_name}: cd-val loss by mixture"
    if subtitle is None:
        # No extras for this sweep: keep the legacy single-line title with the
        # step slice baked in.
        fig.suptitle(f"{title} @ {step_label}", fontsize=11)
    else:
        # Extras present: drop the step from the title (subtitle covers it
        # via the % complete + step range) and render the technical details
        # on continuation lines.
        fig.suptitle(f"{title}\n{subtitle}", fontsize=10)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "cdval_loss_by_mixture.pdf"
    png_path = out_dir / "cdval_loss_by_mixture.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


def _flops_per_token(history: pd.DataFrame) -> float:
    """Consensus FLOPs/token across runs (for the paired FLOPs x-axis).

    ``total_flops`` accumulates ~linearly in tokens, so FLOPs/token is roughly
    constant; the median is robust to the ``step==0`` row (tokens==0 -> dropped
    as non-finite).
    """
    fpt = (history["total_flops"] / history["tokens"]).replace([np.inf, -np.inf], np.nan).dropna()
    if fpt.empty:
        raise RuntimeError("Cannot derive FLOPs/token: no finite ratios in history")
    return float(fpt.median())


def _rank_ascending(values: dict[str, float]) -> dict[str, int]:
    """1-based rank per key, lowest value == rank 1 (best, for losses)."""
    return {key: r for r, key in enumerate(sorted(values, key=values.__getitem__), start=1)}


def _project_final(curve: pd.DataFrame, lo_tokens: float, hi_tokens: float, end_tokens: float) -> float | None:
    """Power-law projection of cd-val at ``end_tokens`` from a fit over the window."""
    gp = curve[curve["tokens"] > 0]
    fit = fit_power_law(
        gp["tokens"].to_numpy(dtype=float), gp["cd_val_loss"].to_numpy(dtype=float), lo_tokens, hi_tokens
    )
    if fit is None:
        return None
    return float(fit.predict(np.array([end_tokens]))[0])


# High-side fit-bound sweep for the rank-correlation panel (fraction of steps).
# Starts at 0.2 so the [low, hi] window holds >=2 cd-val evals; ends at the WSD
# decay boundary (0.8).
RANK_CORR_HI_FRACS: tuple[float, ...] = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


def _render_rank_corr_panel(
    ax,
    by_mixture: dict[str, pd.DataFrame],
    mixture_ids: list[str],
    actual_final: dict[str, float],
    meta: EvalMeta,
    tokens_per_step: int,
    lo_tokens: float,
    end_tokens: float,
) -> None:
    """Spearman rho(projected, actual final) vs the fit's upper-bound fraction.

    The low bound is fixed; the upper bound sweeps ``RANK_CORR_HI_FRACS``. Only
    ``mixture_ids`` (stationary mixes) are ranked — staged curricula are excluded
    upstream. A vertical guide marks the figure's active ``FIT_HIGH_STEP_FRAC``.
    """
    actual = pd.Series([actual_final[m] for m in mixture_ids]).rank()
    fracs, rhos = [], []
    for frac in RANK_CORR_HI_FRACS:
        hi_tokens = frac * meta.num_train_steps * tokens_per_step
        projected = [_project_final(by_mixture[m], lo_tokens, hi_tokens, end_tokens) for m in mixture_ids]
        if any(p is None for p in projected):
            continue
        rho = float(pd.Series(projected).rank().corr(actual))
        fracs.append(frac * 100)
        rhos.append(rho)

    ax.axvline(FIT_HIGH_STEP_FRAC * 100, color="grey", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.plot(fracs, rhos, marker="o", markersize=4, linewidth=1.3, color="#333333")
    ax.set_xlabel("Fit upper bound (% of steps)", fontsize=9)
    ax.set_ylabel("Spearman $\\rho$\n(proj vs actual)", fontsize=9)
    ax.set_title(
        f"Projected vs actual final-loss rank agreement (n={len(mixture_ids)} stationary mixes: "
        + ", ".join(MIXTURE_NAMES.get(m, m).split(" ", 1)[-1].strip("()") for m in mixture_ids)
        + ")",
        fontsize=9,
    )
    lo = min([*rhos, 0.0]) if rhos else 0.0
    ax.set_ylim(lo - 0.08, 1.08)
    ax.grid(True, linestyle="--", alpha=0.3)


def render_cdval_scale_curves(history: pd.DataFrame, sweep: SweepConfig, meta: EvalMeta, out_dir: Path) -> None:
    """Grid of log-log cd-val loss curves, one cell per run, with power-law fits.

    Columns are the sweep's groups (Single quality / Blends for the scale
    sweep); rows are the runs within each group, so each cell is a single
    mixture. The bottom row carries the shared training-tokens axis and the top
    row a paired training-FLOPs axis (FLOPs is a constant multiple of tokens, so
    the scales align tick-for-tick in log space); inner tick labels are dropped.
    A pure power law ``L = A * D^(-alpha)`` is fit over the step window
    ``[FIT_LOW_STEP_FRAC, FIT_HIGH_STEP_FRAC]`` and drawn dashed with short end
    caps marking the window; alpha/R^2 annotate each cell and final/projected
    losses + ranks sit in the lower-left.

    A full-width panel below the grid (reading as a near-separate figure under
    the grid's x-axis label) shows how the projected-vs-actual final-loss rank
    correlation, over stationary mixes only, varies with the fit's upper bound.
    """
    if meta.num_train_steps is None:
        logger.warning("Sweep %s has no num_train_steps; skipping cd-val curve plot", sweep.id)
        return
    groups = MIXTURE_GROUPS_BY_SWEEP.get(sweep.id, MIXTURE_GROUPS)
    tokens_per_step = meta.eval_batch_size * meta.eval_seq_len
    lo_tokens = FIT_LOW_STEP_FRAC * meta.num_train_steps * tokens_per_step
    hi_tokens = FIT_HIGH_STEP_FRAC * meta.num_train_steps * tokens_per_step
    end_tokens = meta.num_train_steps * tokens_per_step
    flops_per_token = _flops_per_token(history)
    by_mixture = {mid: g.sort_values("tokens") for mid, g in history.groupby("mixture_id")}

    # Per-run power-law fits over the step window, plus two end-of-training
    # numbers per run: the loss the stable-phase trend projects at the full
    # token budget, and the actual final cd-val. Each is ranked across runs
    # (1 = lowest loss = best) to compare the projected vs realized ordering.
    pos = history[history["tokens"] > 0]
    fits: dict[str, PowerLawFit | None] = {}
    actual_final: dict[str, float] = {}
    for mid, g in by_mixture.items():
        gp = g[g["tokens"] > 0].sort_values("tokens")
        if gp.empty:
            continue
        fits[mid] = fit_power_law(
            gp["tokens"].to_numpy(dtype=float), gp["cd_val_loss"].to_numpy(dtype=float), lo_tokens, hi_tokens
        )
        actual_final[mid] = float(gp["cd_val_loss"].iloc[-1])
    projected = {mid: float(f.predict(np.array([end_tokens]))[0]) for mid, f in fits.items() if f is not None}
    rank_proj = _rank_ascending(projected)
    rank_actual = _rank_ascending(actual_final)

    # Shared, tight axis ranges. Token grid is identical across runs; pad x
    # lightly in log space, y just past the data (multiplicative, ~1%), and
    # tick every 0.2 over the covered decade-fraction so cells stay readable.
    tok = pos["tokens"]
    x_lo, x_hi = float(tok.min()) / 1.1, float(tok.max()) * 1.1
    y_min, y_max = float(pos["cd_val_loss"].min()), float(pos["cd_val_loss"].max())
    y_lo, y_hi = y_min * 0.99, y_max * 1.01
    yticks = np.arange(np.floor(y_min * 10) / 10, np.ceil(y_max * 10) / 10 + 1e-9, 0.2)

    ncols = len(groups)
    nrows = max(len(mids) for _, mids in groups)

    # Two stacked subfigures: the per-run grid on top, the rank-correlation
    # panel below it spanning the full width. Width trimmed and height grown
    # versus the bare grid to make room for the panel.
    fig = plt.figure(figsize=(3.2 * ncols, 1.5 * nrows + 2.4), constrained_layout=True)
    sf_grid, sf_panel = fig.subfigures(2, 1, height_ratios=[1.5 * nrows, 1.5])
    axes = np.asarray(sf_grid.subplots(nrows, ncols, sharex=True, sharey=True)).reshape(nrows, ncols)

    for j, (group_label, mixture_ids) in enumerate(groups):
        for i in range(nrows):
            ax = axes[i, j]
            if i >= len(mixture_ids):
                ax.set_visible(False)
                continue
            mid = mixture_ids[i]
            color = GROUP_COLORS.get(group_label, "#4C78A8")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_yticks(yticks)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.grid(True, which="both", linestyle="--", alpha=0.3)

            curve = by_mixture.get(mid)
            if curve is None or curve.empty:
                logger.warning("No cd-val history for mixture %s; skipping curve", mid)
            else:
                # Log axes can't show the step==0 point (tokens==0); drop it.
                pts = curve[curve["tokens"] > 0]
                tokens = pts["tokens"].to_numpy(dtype=float)
                loss = pts["cd_val_loss"].to_numpy(dtype=float)
                ax.plot(tokens, loss, marker="o", markersize=3.0, linewidth=1.0, color=color, alpha=0.9)
                label = MIXTURE_NAMES.get(mid, mid)
                fit = fits.get(mid)
                if fit is not None:
                    grid = np.geomspace(fit.data_lo, fit.data_hi, 100)
                    ax.plot(grid, fit.predict(grid), linestyle="--", linewidth=1.4, color="black", alpha=0.9)
                    # Short vertical end caps mark the fit-window bounds.
                    ends = np.array([fit.data_lo, fit.data_hi])
                    ax.plot(
                        ends,
                        fit.predict(ends),
                        linestyle="none",
                        marker="|",
                        markersize=6,
                        markeredgewidth=1.3,
                        color="black",
                    )
                    label = f"{label}\n$\\alpha$={fit.alpha:.3f}, $R^2$={fit.r_squared:.3f}"
                    # Final/projected losses + cross-run ranks ("#" is compact
                    # rank, lower loss = better) in the empty lower-left corner.
                    ranks = (
                        f"final {actual_final[mid]:.3f} #{rank_actual[mid]}\n"
                        f"proj {projected[mid]:.3f} #{rank_proj[mid]}"
                    )
                    ax.text(0.04, 0.06, ranks, transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
                ax.text(0.96, 0.94, label, transform=ax.transAxes, ha="right", va="top", fontsize=8)

            # FLOPs axis on the top row only; group header above it. FLOPs =
            # flops_per_token * tokens, so the matched log axis aligns
            # tick-for-tick (the constant factor cancels in log space).
            if i == 0:
                flops_ax = ax.twiny()
                flops_ax.set_xscale("log")
                flops_ax.set_xlim(x_lo * flops_per_token, x_hi * flops_per_token)
                # Stacks top-down: group header (title) > FLOPs label > ticks.
                flops_ax.set_xlabel("Training FLOPs (C)", fontsize=8)
                flops_ax.set_title(group_label, fontsize=10, pad=8)

    sf_grid.supxlabel("Training tokens (D)", fontsize=9)
    sf_grid.supylabel("cd-val loss", fontsize=9)

    # Bottom panel: projected-vs-actual rank correlation vs the fit's upper
    # bound, over stationary mixes only (staged curricula excluded).
    corr_mixtures = [m for _, mids in groups for m in mids if m not in STAGED_MIXTURES and m in by_mixture]
    ax_panel = sf_panel.subplots(1, 1)
    _render_rank_corr_panel(
        ax_panel, by_mixture, corr_mixtures, actual_final, meta, tokens_per_step, lo_tokens, end_tokens
    )

    subtitle = (
        f"{meta.params_exact / 1e9:.2f}B params · {meta.tokens_exact / 1e9:.1f}B tokens · "
        f"fit $L = A\\,D^{{-\\alpha}}$ over steps {FIT_LOW_STEP_FRAC:.0%}-{FIT_HIGH_STEP_FRAC:.0%} "
        f"(D ∈ [{lo_tokens / 1e9:.1f}B, {hi_tokens / 1e9:.1f}B] tokens)"
    )
    fig.suptitle(
        f"{TITLE_PREFIX} — {sweep.display_name}: cd-val loss curves + power-law fits\n{subtitle}",
        fontsize=10,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "cdval_loss_curves.pdf"
    png_path = out_dir / "cdval_loss_curves.png"
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


def render_mix_vs_scale_scatter(
    scale_snapshot: pd.DataFrame,
    scale_meta: EvalMeta,
    mix_snapshot: pd.DataFrame,
    mix_meta: EvalMeta,
    scale_sweep: SweepConfig,
    mix_sweep: SweepConfig,
    out_dir: Path,
) -> None:
    """Scatter: cd-val loss for the same recipe at mix scale vs scale-up scale.

    Only the recipes in ``SHARED_MIXTURE_PAIRS`` appear — m4/m5/m8/m9 (mix)
    and m13 (scale) have no cross-sweep counterpart. Pearson r and a
    least-squares fit show how the cheap mix-stage ranking (100M, ~4.3B tok)
    transfers to the more expensive scale-stage ranking (1.5B, ~21.5B tok).
    """
    mix_cdval = mix_snapshot[mix_snapshot["metric"] == "cd-val"].set_index("mixture_id")
    scale_cdval = scale_snapshot[scale_snapshot["metric"] == "cd-val"].set_index("mixture_id")
    rows: list[dict] = []
    for mix_id, scale_id, label, group in SHARED_MIXTURE_PAIRS:
        if mix_id not in mix_cdval.index or scale_id not in scale_cdval.index:
            logger.warning("Skipping pair %s/%s: missing cd-val in one of the sweeps", mix_id, scale_id)
            continue
        rows.append(
            {
                "mix_id": mix_id,
                "scale_id": scale_id,
                "label": label,
                "group": group,
                "mix_value": float(mix_cdval.loc[mix_id, "value"]),
                "scale_value": float(scale_cdval.loc[scale_id, "value"]),
            }
        )
    if not rows:
        logger.warning("No shared mixtures between %s and %s", mix_sweep.id, scale_sweep.id)
        return
    df = pd.DataFrame(rows)
    x = df["mix_value"].to_numpy(dtype=float)
    y = df["scale_value"].to_numpy(dtype=float)
    r = float(np.corrcoef(x, y)[0, 1])
    slope, intercept = np.polyfit(x, y, 1)
    rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    seen_groups: list[str] = []
    for group_label in df["group"].unique():
        sub = df[df["group"] == group_label]
        ax.scatter(
            sub["mix_value"],
            sub["scale_value"],
            c=GROUP_COLORS.get(group_label, "#888888"),
            label=group_label,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
            s=90,
            zorder=3,
        )
        seen_groups.append(group_label)
        for _, row in sub.iterrows():
            ax.annotate(
                row["label"],
                (row["mix_value"], row["scale_value"]),
                xytext=(0, -8),
                textcoords="offset points",
                fontsize=9,
                ha="center",
                va="top",
            )

    pad_x = 0.08 * float(np.ptp(x)) or 0.01
    pad_y = 0.08 * float(np.ptp(y)) or 0.01
    x_lo, x_hi = float(x.min()) - pad_x, float(x.max()) + pad_x
    y_lo, y_hi = float(y.min()) - pad_y, float(y.max()) + pad_y
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    sign = "+" if intercept >= 0 else "-"
    fit_label = f"y = {slope:.3f}*x {sign} {abs(intercept):.3f}"
    ax.plot(
        [x_lo, x_hi],
        [slope * x_lo + intercept, slope * x_hi + intercept],
        "--",
        color="grey",
        linewidth=1.2,
        zorder=1,
        label=fit_label,
    )

    ax.set_xlabel(
        f"Stage 1 cd-val loss  " f"(N={mix_meta.params_exact / 1e9:.2f}B, D={mix_meta.tokens_exact / 1e9:.1f}B)"
    )
    ax.set_ylabel(
        f"Stage 2 cd-val loss  " f"(N={scale_meta.params_exact / 1e9:.2f}B, D={scale_meta.tokens_exact / 1e9:.1f}B)"
    )
    ax.set_title(
        f"{TITLE_PREFIX} — cd-val loss: scale sweep vs mix sweep on shared mixtures\n"
        f"Pearson r = {r:.4f}   Spearman rho = {rho:.4f}   (n = {len(df)} shared mixtures)",
        fontsize=10,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "mix_vs_scale_cdval_scatter.pdf"
    png_path = out_dir / "mix_vs_scale_cdval_scatter.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s  (r=%.4f, rho=%.4f, n=%d)", pdf_path, png_path, r, rho, len(df))


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

    wide = snapshot.pivot(index="mixture_name", columns="metric", values="value")
    logger.info("step slice = %s  meta = %s", _step_label(snapshot), meta)
    logger.info("\n%s", wide.to_string(float_format=lambda v: f"{v:.4f}" if pd.notna(v) else "—"))

    plots_dir = _plots_dir(sweep.id)
    render_heatmap(snapshot, sweep, plots_dir)
    render_cdval_bars(snapshot, sweep, meta, plots_dir)

    cdval_history = load_or_build_cdval_history(sweep, meta, refresh=args.refresh)
    render_cdval_scale_curves(cdval_history, sweep, meta, plots_dir)

    eval_sweep_id = EVAL_OF.get(args.sweep)
    if eval_sweep_id is not None:
        eval_sweep = SWEEPS[eval_sweep_id]
        eval_snapshot, eval_meta = load_or_build_snapshot(eval_sweep, refresh=args.refresh)
        render_train_vs_full_eval_scatter(snapshot, meta, eval_snapshot, eval_meta, sweep, eval_sweep, plots_dir)

    mix_sweep_id = MIX_OF.get(args.sweep)
    if mix_sweep_id is not None:
        mix_sweep = SWEEPS[mix_sweep_id]
        mix_snapshot, mix_meta = load_or_build_snapshot(mix_sweep, refresh=args.refresh)
        render_mix_vs_scale_scatter(snapshot, meta, mix_snapshot, mix_meta, sweep, mix_sweep, plots_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
