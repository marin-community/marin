#!/usr/bin/env python3
"""Final paper plots for synthetic data efficiency experiments.

Stripped-down version of synth_data_efficiency_plots.py containing only the
plot functions and code paths needed for the final paper figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, FuncFormatter, NullFormatter
import numpy as np

from experiments.data_efficiency.synth_data_efficiency_plot_utils import (
    BLACK,
    DATA_STREAM_BAR_STYLE_DICT,
    DATA_STREAM_COLOR_DICT,
    DATA_STREAM_SHORT_NAMES,
    PARAM_STR_COLOR_DICT,
    PARAM_STR_TO_COUNT,
    PRETTY_NAME_DICT,
    PURPLE,
    REAL_DATA_TOKENS,
    RED,
    REGULARIZED_COLOR,
    SYNTH_STREAM_TOKENS,
    WIDE_RECTANGLE_FIGSIZE,
    PowerScalingLaw,
    setup_axes,
    setup_plot_style,
    stream_display_name,
    stream_legend,
)


# ---------------------------------------------------------------------------
# Projects & cache
# ---------------------------------------------------------------------------

PROJECTS = [
    "stanford-mercury/suhas-data-efficiency",
    "stanford-mercury/suhas-eval-data-efficiency",
]
TRAIN_PROJECT = "stanford-mercury/suhas-data-efficiency"
EVAL_PROJECT = "stanford-mercury/suhas-eval-data-efficiency"
CACHE_PATH = "experiments/data_efficiency/cache/synth_data_efficiency.pkl"
DEFAULT_EVAL_RUN_PREFIXES = ("synth-data-best-models-", "synth-data-var-models-")
EvalRunPrefix = str | tuple[str, ...] | list[str]

PLOT_DIR = "experiments/data_efficiency/plots/synth_data_efficiency/"
SCALING_FIGSIZE = (6, 4.7)

PRINT_RUNS = False


def _log_point(plot_name: str, label: str, run_name: str | None, loss: float) -> None:
    if PRINT_RUNS and run_name is not None:
        print(f"  [{plot_name}] {label}: {run_name} -> {loss:.4f}")


# ---------------------------------------------------------------------------
# Metric keys
# ---------------------------------------------------------------------------

METRIC_KEY = "eval/dc_1k_val_normal/loss"
SHORT_CONTEXT_METRIC_KEY = "eval/dc_500_val_short/loss"
LONG_CONTEXT_METRIC_KEY = "eval/dc_500_val_long/loss"
ARXIV_METRIC_KEY = "eval/uncheatable_eval/arxiv_computer_science/loss"
WIKIPEDIA_METRIC_KEY = "eval/uncheatable_eval/wikipedia_english/loss"
GITHUB_PYTHON_METRIC_KEY = "eval/uncheatable_eval/github_python/loss"

REGULARIZED_300M_LOSS = 3.5544
REGULARIZED_ASYMPTOTE = 3.449006944291979


# ---------------------------------------------------------------------------
# Run config parsing
# ---------------------------------------------------------------------------

DATA_STREAM_SPEC_PATTERN = re.compile(r"\+[a-z0-9]+\^[\d.]+")

RUN_CONFIG_BODY_PATTERN = (
    r"(?P<model_size>\d+m\d+k|1_5b\d+k)"
    r"(?P<cda>cda)?"
    r"-203Mx(?P<epochs>\d+)"
    r"-dcr"
    r"(?:\+(?P<data_stream>[a-z0-9]+)\^(?P<mix_ratio>[\d.]+))?"
    r"-cos"
    r"-lr(?P<lr>[\d.]+)"
    r"-wd(?P<wd>[\d.]+)"
    r"(?:-bs(?P<bs>\d+))?"
)

RUN_CONFIG_PATTERN = re.compile(
    rf"^{RUN_CONFIG_BODY_PATTERN}"
    r"(?:-seed(?P<seed>\d+))?$"
)

ENSEMBLE_EVAL_PATTERN = re.compile(
    r"^(?:"
    r"ppl-eval-ensemble-(?P<member_count>\d+)x"
    r"|synth-data-(?:best|var)-models"
    r")-"
    rf"{RUN_CONFIG_BODY_PATTERN}"
    r"(?:-seed(?P<seed>\d+)(?:-(?P<num_steps>\d+))?)?"
    r"-(?P<run_hash>[0-9a-f]+)$"
)


@dataclass(frozen=True)
class RunConfig:
    name: str
    model_size: str
    cda: bool
    epochs: int
    data_stream: str | None
    mix_ratio: float | None
    lr: float
    wd: float
    bs: int | None
    seed: int | None

    @staticmethod
    def from_run_name(name: str) -> RunConfig | None:
        m = RUN_CONFIG_PATTERN.match(name)
        if m is None:
            return None
        return RunConfig(
            name=name,
            model_size=m.group("model_size"),
            cda=m.group("cda") is not None,
            epochs=int(m.group("epochs")),
            data_stream=m.group("data_stream"),
            mix_ratio=float(m.group("mix_ratio")) if m.group("mix_ratio") else None,
            lr=float(m.group("lr")),
            wd=float(m.group("wd")),
            bs=int(m.group("bs")) if m.group("bs") else None,
            seed=int(m.group("seed")) if m.group("seed") else None,
        )

    @property
    def is_seed_run(self) -> bool:
        return self.seed is not None

    @property
    def is_baseline(self) -> bool:
        return self.data_stream is None

    def get_real_tokens(self) -> float:
        base = REAL_DATA_TOKENS * self.epochs
        if self.mix_ratio is None or self.data_stream is None:
            return float(base)
        synth_tokens = SYNTH_STREAM_TOKENS.get(self.data_stream)
        if synth_tokens is None:
            return float(base)
        total_tokens = REAL_DATA_TOKENS * (1 / (1 - self.mix_ratio)) * self.epochs
        synth_epochs = total_tokens / synth_tokens
        return base + REAL_DATA_TOKENS * synth_epochs

    def short_desc(self) -> str:
        parts = [self.model_size]
        if self.cda:
            parts.append("cda")
        parts.append(f"x{self.epochs}")
        if self.data_stream:
            parts.append(f"+{self.data_stream}^{self.mix_ratio}")
        return " ".join(parts)


@dataclass(frozen=True)
class EnsembleEvalConfig:
    name: str
    member_count: int
    model_size: str
    cda: bool
    epochs: int
    data_stream: str | None
    mix_ratio: float | None
    lr: float
    wd: float
    bs: int | None

    @staticmethod
    def from_run_name(name: str) -> EnsembleEvalConfig | None:
        m = ENSEMBLE_EVAL_PATTERN.match(name)
        if m is None:
            return None
        member_count = int(m.group("member_count")) if m.group("member_count") else 1
        return EnsembleEvalConfig(
            name=name,
            member_count=member_count,
            model_size=m.group("model_size"),
            cda=m.group("cda") is not None,
            epochs=int(m.group("epochs")),
            data_stream=m.group("data_stream"),
            mix_ratio=float(m.group("mix_ratio")) if m.group("mix_ratio") else None,
            lr=float(m.group("lr")),
            wd=float(m.group("wd")),
            bs=int(m.group("bs")) if m.group("bs") else None,
        )

    def series_key(self) -> tuple[str, bool, str | None, int, float]:
        return (self.model_size, self.cda, self.data_stream, self.epochs, self.wd)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _load_cache(path: str) -> dict | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _save_cache(path: str, payload: dict) -> None:
    _ensure_parent_dir(path)
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def _make_json_safe(obj: object) -> object:
    """Recursively convert wandb summary/config values to JSON-serializable primitives."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def fetch_matching_runs(project: str) -> list:
    """Fetch all runs whose display name contains x{N}-dcr."""
    import wandb

    api = wandb.Api(timeout=120)
    runs = list(api.runs(project, filters={"display_name": {"$regex": r"x\d+-dcr"}}, per_page=200))
    runs.sort(key=lambda run: run.name or "")
    return runs


def _serialize_run(run: object) -> dict:
    return {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": run.created_at,
        "summary": _make_json_safe(dict(run.summary)),
        "config": _make_json_safe(dict(run.config)),
    }


def _print_cached_runs(project: str, runs: list[dict]) -> None:
    print(f"\n{'=' * 80}")
    print(f"Project: {project}")
    print(f"Matched runs: {len(runs)}")
    print(f"{'=' * 80}")
    for run in runs:
        print(f"  {run['name']:<80s}  state={run['state']}")


def build_cache(cache_path: str) -> dict:
    """Pull all matching runs from both projects and save to cache."""
    payload: dict[str, list[dict]] = {}
    for project in PROJECTS:
        print(f"Fetching runs from {project} ...")
        runs = fetch_matching_runs(project)
        serialized: list[dict] = []
        ignored_multi_stream: list[str] = []
        for run in runs:
            name = run.name or ""
            if _has_multiple_data_streams(name):
                ignored_multi_stream.append(name)
                continue
            serialized.append(_serialize_run(run))
        payload[project] = serialized
        if ignored_multi_stream:
            print(f"  Ignoring {len(ignored_multi_stream)} multi-stream runs from {project}")
            for name in ignored_multi_stream:
                print(f"    {name}")
        _print_cached_runs(project, serialized)

    _save_cache(cache_path, payload)
    print(f"\nCache saved to {cache_path}")
    return payload


def _has_multiple_data_streams(name: str) -> bool:
    return len(DATA_STREAM_SPEC_PATTERN.findall(name)) > 1


def _matches_eval_run_prefix(name: str, eval_run_prefix: EvalRunPrefix) -> bool:
    if isinstance(eval_run_prefix, str):
        return name.startswith(eval_run_prefix)
    return any(name.startswith(prefix) for prefix in eval_run_prefix)


def _filter_finished(payload: dict) -> dict:
    """Keep only runs with state == 'finished', mutating the payload in place."""
    for project, runs in payload.items():
        before = len(runs)
        runs[:] = [r for r in runs if r.get("state") == "finished"]
        dropped = before - len(runs)
        if dropped:
            print(f"  Filtered out {dropped} non-finished runs from {project} ({len(runs)} remaining)")
    return payload


def build_name_to_summary(payload: dict, project: str) -> dict[str, dict]:
    return {run["name"]: run.get("summary", {}) for run in payload.get(project, [])}


def get_loss(summary: dict, metric_key: str = METRIC_KEY) -> float | None:
    val = summary.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_train_runs(payload: dict) -> tuple[list[RunConfig], list[str]]:
    runs = payload.get(TRAIN_PROJECT, [])
    parsed: list[RunConfig] = []
    unmatched: list[str] = []
    for run in runs:
        name = run["name"]
        if _has_multiple_data_streams(name):
            continue
        config = RunConfig.from_run_name(name)
        if config is None:
            unmatched.append(name)
        else:
            parsed.append(config)
    return parsed, unmatched


def parse_ensemble_eval_runs(payload: dict) -> list[EnsembleEvalConfig]:
    runs = payload.get(EVAL_PROJECT, [])
    parsed: list[EnsembleEvalConfig] = []
    for run in runs:
        name = run["name"]
        if _has_multiple_data_streams(name):
            continue
        cfg = EnsembleEvalConfig.from_run_name(name)
        if cfg is not None:
            parsed.append(cfg)
    return parsed


def split_runs(
    configs: list[RunConfig],
) -> tuple[list[RunConfig], list[RunConfig], list[RunConfig]]:
    """Split parsed configs into (baselines, augmented, seed_runs)."""
    seed_runs = [c for c in configs if c.is_seed_run]
    non_seed = [c for c in configs if not c.is_seed_run]
    baselines = [c for c in non_seed if c.is_baseline]
    augmented = [c for c in non_seed if not c.is_baseline]
    return baselines, augmented, seed_runs


def _save_fig(fig: plt.Figure, output_path: str) -> None:
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# Stream lookup helpers
# ---------------------------------------------------------------------------

SHUFFLED_COPY_STREAMS: dict[int, str] = {2: "w2s", 4: "s4", 8: "s8", 16: "s16", 32: "s32"}
SORTED_COPY_STREAMS: dict[int, str] = {2: "w2", 4: "b4", 8: "b8", 16: "b16", 32: "b32"}
LATENT_COPY_STREAMS: dict[int, str] = {2: "z2", 4: "z4", 8: "z8", 16: "z16", 32: "z32"}
COPY_COUNTS = sorted(SHUFFLED_COPY_STREAMS.keys())

MODE_TO_STREAMS: dict[str, dict[int, str]] = {
    "simple": SHUFFLED_COPY_STREAMS,
    "stitched": SORTED_COPY_STREAMS,
    "latent_thoughts": LATENT_COPY_STREAMS,
}

MODE_PRETTY_NAMES: dict[str, str] = {
    "simple": "Simple Rephrasing",
    "stitched": "Stitched Rephrasing",
    "latent_thoughts": "Latent Thoughts",
}

ALL_SERIES: list[tuple[str, dict[int, str]]] = [
    ("Simple", SHUFFLED_COPY_STREAMS),
    ("Stitched", SORTED_COPY_STREAMS),
    ("Latent Thoughts", LATENT_COPY_STREAMS),
]


def _resolve_metric(metric_key: str) -> tuple[str, str, str]:
    """Return (ylabel, title_suffix, file_suffix) for a metric key."""
    table: dict[str, tuple[str, str, str]] = {
        METRIC_KEY: ("IID Loss", "", ""),
        ARXIV_METRIC_KEY: ("Arxiv CS Loss", " (Arxiv CS)", "_arxiv"),
        WIKIPEDIA_METRIC_KEY: ("Wikipedia English Loss", " (Wikipedia English)", "_wikipedia"),
        GITHUB_PYTHON_METRIC_KEY: ("GitHub Python Loss", " (GitHub Python)", "_github_python"),
        SHORT_CONTEXT_METRIC_KEY: ("Short Context Loss", " (Short Context)", "_short_context"),
        LONG_CONTEXT_METRIC_KEY: ("Long Context Loss", " (Long Context)", "_long_context"),
    }
    if metric_key in table:
        return table[metric_key]
    pretty = metric_key.removeprefix("eval/")
    suffix = "_" + re.sub(r"[^a-z0-9]+", "_", metric_key.lower()).strip("_")
    return (metric_key, f" ({pretty})", suffix)


def _best_loss_for_stream(
    runs: list[RunConfig],
    data_stream: str,
    name_to_summary: dict[str, dict],
    model_size: str,
    metric_key: str = METRIC_KEY,
) -> tuple[float | None, str | None, RunConfig | None]:
    best: float | None = None
    best_name: str | None = None
    best_cfg: RunConfig | None = None
    for c in runs:
        if c.data_stream != data_stream or c.model_size != model_size or not c.cda:
            continue
        loss = get_loss(name_to_summary.get(c.name, {}), metric_key=metric_key)
        if loss is not None and (best is None or loss < best):
            best = loss
            best_name = c.name
            best_cfg = c
    return best, best_name, best_cfg


def _best_eval_loss_for_stream(
    eval_runs: list[EnsembleEvalConfig],
    data_stream: str,
    eval_summary: dict[str, dict],
    model_size: str,
    metric_key: str = METRIC_KEY,
    run_prefix: EvalRunPrefix = DEFAULT_EVAL_RUN_PREFIXES,
) -> tuple[float | None, str | None, EnsembleEvalConfig | None]:
    best: float | None = None
    best_name: str | None = None
    best_cfg: EnsembleEvalConfig | None = None
    for run in eval_runs:
        if not _matches_eval_run_prefix(run.name, run_prefix):
            continue
        if run.member_count != 1:
            continue
        if run.data_stream != data_stream or run.model_size != model_size or not run.cda:
            continue
        loss = get_loss(eval_summary.get(run.name, {}), metric_key=metric_key)
        if loss is not None and (best is None or loss < best):
            best = loss
            best_name = run.name
            best_cfg = run
    return best, best_name, best_cfg


def _best_baseline_loss(
    baselines: list[RunConfig],
    name_to_summary: dict[str, dict],
    model_size: str,
    metric_key: str = METRIC_KEY,
) -> tuple[float | None, RunConfig | None]:
    best: float | None = None
    best_cfg: RunConfig | None = None
    for c in baselines:
        if c.is_seed_run or not c.is_baseline or c.model_size != model_size or not c.cda:
            continue
        loss = get_loss(name_to_summary.get(c.name, {}), metric_key=metric_key)
        if loss is not None and (best is None or loss < best):
            best = loss
            best_cfg = c
    return best, best_cfg


def _best_eval_baseline_loss(
    eval_runs: list[EnsembleEvalConfig],
    eval_summary: dict[str, dict],
    model_size: str,
    metric_key: str = METRIC_KEY,
    run_prefix: EvalRunPrefix = DEFAULT_EVAL_RUN_PREFIXES,
) -> tuple[float | None, EnsembleEvalConfig | None]:
    best: float | None = None
    best_cfg: EnsembleEvalConfig | None = None
    for run in eval_runs:
        if not _matches_eval_run_prefix(run.name, run_prefix):
            continue
        if run.member_count != 1 or run.data_stream is not None or run.model_size != model_size or not run.cda:
            continue
        loss = get_loss(eval_summary.get(run.name, {}), metric_key=metric_key)
        if loss is not None and (best is None or loss < best):
            best = loss
            best_cfg = run
    return best, best_cfg


# ---------------------------------------------------------------------------
# 1. Loss vs epochs / mixing ratio
# ---------------------------------------------------------------------------


def plot_loss_vs(
    augmented: list[RunConfig],
    data_stream: str,
    name_to_summary: dict[str, dict],
    *,
    x_attr: str = "epochs",
    color: str = PURPLE,
    output_path: str | None = None,
) -> None:
    """Plot best eval loss vs a RunConfig attribute for a single stream (300M CDA)."""
    xlabel_defaults = {"epochs": "Epochs of Real Data", "mix_ratio": "Mixing Fraction"}

    stream_runs = [
        c for c in augmented
        if c.data_stream == data_stream and c.model_size == "300m4k"
        and not c.is_seed_run and getattr(c, x_attr, None) is not None
    ]
    if not stream_runs:
        return

    x_values = sorted({getattr(c, x_attr) for c in stream_runs})

    all_xs: list[float] = []
    all_losses: list[float] = []
    for c in stream_runs:
        loss = get_loss(name_to_summary.get(c.name, {}))
        if loss is not None and loss < 5.0:
            all_xs.append(getattr(c, x_attr))
            all_losses.append(loss)

    pn = f"loss_vs_{x_attr}_{data_stream}"
    best_per_x: list[tuple[float, float]] = []
    for xv in x_values:
        best_loss: float | None = None
        best_run_name: str | None = None
        for c in stream_runs:
            if getattr(c, x_attr) != xv:
                continue
            loss = get_loss(name_to_summary.get(c.name, {}))
            if loss is not None and (best_loss is None or loss < best_loss):
                best_loss = loss
                best_run_name = c.name
        if best_loss is not None:
            _log_point(pn, f"{x_attr}={xv}", best_run_name, best_loss)
            best_per_x.append((xv, best_loss))

    if not best_per_x:
        return

    xs = [x for x, _ in best_per_x]
    losses = [l for _, l in best_per_x]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    ax.axhline(REGULARIZED_300M_LOSS, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
               label=f"Real Data Only: {REGULARIZED_300M_LOSS:.2f}")
    ax.scatter(all_xs, all_losses, color=color, s=50, alpha=0.2, zorder=2)
    ax.scatter(xs, losses, color=color, s=50, zorder=5)
    ax.plot(xs, losses, "--", color=color, label=stream_legend(data_stream, min(losses)))

    min_loss = min(losses)
    min_x = xs[losses.index(min_loss)]
    ax.scatter(min_x, min_loss, color=RED, s=70, marker="o", edgecolor=RED, linewidth=2, zorder=3)

    ax.set_xscale("log")
    tick_labels = [str(int(v)) if float(v) == int(v) else f"{v:g}" for v in x_values]
    ax.xaxis.set_major_locator(FixedLocator(x_values))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    xlabel = xlabel_defaults.get(x_attr, x_attr.replace("_", " ").title())
    title = f"Loss vs {xlabel} \u2014 {stream_display_name(data_stream)}"
    setup_axes(ax, title=title, xlabel=xlabel, ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        output_path = f"loss_vs_{x_attr}_{data_stream}.png"
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Compare loss vs epochs (multi-stream overlay)
# ---------------------------------------------------------------------------


def plot_compare_loss_vs_epochs(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    data_streams: list[str],
    stream_epochs: dict[str, list[int]] | None = None,
    stream_token_range: dict[str, list[tuple[float, float]]] | None = None,
    use_real_tokens: bool = False,
    output_path: str | None = None,
) -> None:
    """Overlay best-loss-per-epoch curves for multiple data streams."""
    if stream_epochs is None:
        stream_epochs = {}
    if stream_token_range is None:
        stream_token_range = {}

    pn = f"compare_loss_vs_epochs({'_'.join(data_streams)})"
    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    all_x_vals: set[float] = set()
    best_points: list[tuple[float, float, str]] = []

    for stream_idx, ds in enumerate(data_streams):
        color = DATA_STREAM_COLOR_DICT.get(ds, BLACK)
        alpha = 0.45 if stream_idx == 0 else 1.0

        stream_runs = [
            c for c in augmented
            if c.data_stream == ds and c.model_size == "300m4k" and not c.is_seed_run and c.cda
        ]
        token_ranges = stream_token_range.get(ds)
        if token_ranges is not None:
            stream_runs = [
                c for c in stream_runs
                if any(lo <= c.get_real_tokens() <= hi for lo, hi in token_ranges)
            ]
        if not stream_runs:
            continue

        if use_real_tokens:
            by_x: dict[float, list[tuple[float, str]]] = defaultdict(list)
            for c in stream_runs:
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None and loss < 5.0:
                    by_x[c.get_real_tokens()].append((loss, c.name))
            best_per_x: list[tuple[float, float]] = []
            for x in sorted(by_x):
                best_pair = min(by_x[x], key=lambda t: t[0])
                _log_point(pn, f"{ds} tokens={x}", best_pair[1], best_pair[0])
                best_per_x.append((x, best_pair[0]))
        else:
            epoch_counts = sorted({c.epochs for c in stream_runs})
            allowed = stream_epochs.get(ds)
            if allowed is not None:
                epoch_counts = [e for e in epoch_counts if e in allowed]
            best_per_x: list[tuple[float, float]] = []
            for epoch in epoch_counts:
                best_loss: float | None = None
                best_run_name: str | None = None
                for c in stream_runs:
                    if c.epochs != epoch:
                        continue
                    loss = get_loss(name_to_summary.get(c.name, {}))
                    if loss is not None and loss < 5.0 and (best_loss is None or loss < best_loss):
                        best_loss = loss
                        best_run_name = c.name
                if best_loss is not None:
                    _log_point(pn, f"{ds} epochs={epoch}", best_run_name, best_loss)
                    best_per_x.append((float(epoch), best_loss))

        if not best_per_x:
            continue

        xs = [x for x, _ in best_per_x]
        losses = [l for _, l in best_per_x]
        all_x_vals.update(xs)

        ax.set_xscale("log")
        ax.scatter(xs, losses, color=color, s=50, zorder=5, alpha=alpha)
        ax.plot(xs, losses, "--", color=color, alpha=alpha, label=stream_legend(ds, min(losses)))

        overall_best_loss = min(losses)
        overall_best_x = xs[losses.index(overall_best_loss)]
        ax.scatter(overall_best_x, overall_best_loss, color=RED, s=90, marker="o",
                   edgecolor=RED, linewidth=2, zorder=3, facecolors="none")
        best_points.append((overall_best_x, overall_best_loss, ds))

    for i in range(len(best_points) - 1):
        x0, y0, _ = best_points[i]
        x1, y1, _ = best_points[i + 1]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.2, alpha=0.25, shrinkA=8, shrinkB=8),
            zorder=2,
        )

    if use_real_tokens:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e9:.0f}B"))
        xlabel, title = "Real tokens seen", "Scaling Real Tokens"
    else:
        sorted_x = sorted(all_x_vals)
        ax.xaxis.set_major_locator(FixedLocator(sorted_x))
        ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in sorted_x]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        xlabel, title = "Epochs of real data", "Scaling Real Epochs"

    setup_axes(ax, title=title, xlabel=xlabel, ylabel="IID Loss")
    plt.tight_layout()
    if output_path is None:
        suffix = "_real_tokens" if use_real_tokens else ""
        output_path = f"compare_loss_vs_epochs_{'_'.join(data_streams)}{suffix}.png"
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Augmented bar comparison
# ---------------------------------------------------------------------------


def _seed_config_key(c: RunConfig) -> tuple:
    return (c.model_size, c.cda, c.epochs, c.data_stream, c.mix_ratio, c.lr, c.wd, c.bs)


def _eval_config_key(c: EnsembleEvalConfig) -> tuple:
    return (c.member_count, c.model_size, c.cda, c.epochs, c.data_stream, c.mix_ratio, c.lr, c.wd, c.bs)


_SEED_IN_NAME_RE = re.compile(r"-seed\d+")


def plot_augmented_bar(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    data_streams: list[str],
    seed_runs: list[RunConfig] | None = None,
    metric: str = "iid",
    eval_runs: list[EnsembleEvalConfig] | None = None,
    eval_summary: dict[str, dict] | None = None,
    eval_run_prefix: EvalRunPrefix = DEFAULT_EVAL_RUN_PREFIXES,
    require_eval_seed: bool = False,
    title: str = "Synthetic Data Ordering",
    output_path: str | None = None,
) -> None:
    """Bar plot comparing best augmented performance per data stream.

    When *require_eval_seed* is True, only eval runs whose name contains
    ``-seed<N>`` are used.  This filters out single-run (non-seeded) eval
    entries so the bar reflects a proper seed average.
    """
    metric_map = {"iid": (METRIC_KEY, "IID Loss", ""), "arxiv": (ARXIV_METRIC_KEY, "Arxiv CS Loss", "_arxiv")}
    metric_key, ylabel, metric_suffix = metric_map[metric.lower()]
    pn = f"augmented_bar({title}{metric_suffix})"
    use_eval = eval_runs is not None
    source_runs = [*augmented, *(seed_runs or [])]
    source_summary = eval_summary if use_eval else name_to_summary

    labels: list[str] = []
    short_labels: list[str] = []
    losses: list[float] = []
    errors: list[float] = []
    n_seeds: list[int] = []
    colors: list[str] = []
    stream_ids: list[str] = []

    for ds in data_streams:
        grouped: defaultdict[tuple, list[tuple[float, str]]] = defaultdict(list)
        if use_eval:
            assert eval_runs is not None and source_summary is not None
            for c in eval_runs:
                if not _matches_eval_run_prefix(c.name, eval_run_prefix):
                    continue
                if require_eval_seed and not _SEED_IN_NAME_RE.search(c.name):
                    continue
                if c.member_count != 1 or c.data_stream != ds or c.model_size != "300m4k" or not c.cda:
                    continue
                loss = get_loss(source_summary.get(c.name, {}), metric_key=metric_key)
                if loss is not None:
                    grouped[_eval_config_key(c)].append((loss, c.name))
        else:
            seed_groups: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
            non_seed_groups: dict[tuple, list[tuple[float, str]]] = defaultdict(list)
            for c in source_runs:
                if c.data_stream != ds or c.model_size != "300m4k" or not c.cda:
                    continue
                loss = get_loss(source_summary.get(c.name, {}), metric_key=metric_key)
                if loss is None:
                    continue
                key = _seed_config_key(c)
                (seed_groups if c.is_seed_run else non_seed_groups)[key].append((loss, c.name))

            for key, vals in seed_groups.items():
                grouped[key] = vals
            for key, vals in non_seed_groups.items():
                if key not in grouped:
                    grouped[key] = vals

        if not grouped:
            continue

        best_group = min(grouped.values(), key=lambda v: float(np.mean([l for l, _ in v])))
        for loss_val, rname in best_group:
            _log_point(pn, f"stream={ds}", rname, loss_val)
        avg = float(np.mean([l for l, _ in best_group]))
        std = float(np.std([l for l, _ in best_group], ddof=1)) if len(best_group) > 1 else 0.0

        labels.append(stream_display_name(ds))
        short_labels.append(DATA_STREAM_SHORT_NAMES.get(ds, ds))
        losses.append(avg)
        errors.append(std)
        n_seeds.append(len(best_group))
        colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
        stream_ids.append(ds)

    if not losses:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    x_pos = range(len(labels))
    bars = ax.bar(x_pos, losses, color=colors, zorder=5, width=0.6)

    for bar, ds in zip(bars, stream_ids):
        style = DATA_STREAM_BAR_STYLE_DICT.get(ds)
        if style is None:
            continue
        if hatch := style.get("hatch"):
            bar.set_hatch(str(hatch))
        if ec := style.get("edgecolor"):
            bar.set_edgecolor(str(ec))
        if (lw := style.get("linewidth")) is not None:
            bar.set_linewidth(float(lw))
        if (a := style.get("alpha")) is not None:
            bar.set_alpha(float(a))

    if any(n > 1 for n in n_seeds):
        yerr = [e if n > 1 else 0.0 for e, n in zip(errors, n_seeds)]
        ax.errorbar(list(x_pos), losses, yerr=yerr, fmt="none", ecolor=BLACK,
                    elinewidth=1.2, capsize=4, capthick=1.2, zorder=6)

    tops = [l + (e if n > 1 else 0.0) for l, e, n in zip(losses, errors, n_seeds)]
    text_off = max(0.001, 0.01 * (max(tops) - min(losses)))
    for bar, loss, label, n, top in zip(bars, losses, labels, n_seeds, tops):
        ax.text(bar.get_x() + bar.get_width() / 2, top + text_off, f"{loss:.3f}",
                ha="center", va="bottom", fontsize=9, zorder=7)
        bar.set_label(label)

    ax.set_ylim(min(losses) - 0.02, max(tops) + text_off + 0.01)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    setup_axes(ax, title=title, xlabel="", ylabel=ylabel)

    plt.tight_layout()
    if output_path is None:
        output_path = f"augmented_bar_{'_'.join(data_streams)}{metric_suffix}.png"
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 4. CDA bar comparison
# ---------------------------------------------------------------------------

BarQuery = tuple[bool, str | None]


def plot_cda_bar(
    baselines: list[RunConfig],
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    queries: list[BarQuery],
    title: str = "Cross-Document Attention",
    output_path: str | None = None,
) -> None:
    """Bar plot comparing best IID loss for each (cda, data_stream) query."""
    pn = "cda_bar"
    legend_labels: list[str] = []
    tick_labels: list[str] = []
    losses: list[float] = []
    colors: list[str] = []
    cda_flags: list[bool] = []

    for cda, data_stream in queries:
        pool = baselines if data_stream is None else augmented
        best: float | None = None
        best_run_name: str | None = None
        for c in pool:
            if data_stream is None and (c.is_seed_run or not c.is_baseline):
                continue
            if data_stream is not None and (c.data_stream != data_stream or c.is_seed_run):
                continue
            if c.model_size != "300m4k" or c.cda != cda:
                continue
            loss = get_loss(name_to_summary.get(c.name, {}))
            if loss is not None and (best is None or loss < best):
                best = loss
                best_run_name = c.name

        if best is None:
            continue

        cda_tag = "CDA" if cda else "No CDA"
        _log_point(pn, f"cda={cda} stream={data_stream}", best_run_name, best)
        if data_stream is None:
            legend_labels.append(f"Real Data Only ({cda_tag})")
            tick_labels.append(f"Real Data Only ({cda_tag})")
        else:
            legend_labels.append(f"{stream_display_name(data_stream)} ({cda_tag})")
            tick_labels.append(f"{DATA_STREAM_SHORT_NAMES.get(data_stream, data_stream)} ({cda_tag})")
        losses.append(best)
        colors.append(DATA_STREAM_COLOR_DICT.get(data_stream, REGULARIZED_COLOR) if data_stream else REGULARIZED_COLOR)
        cda_flags.append(cda)

    if not losses:
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    bars = ax.bar(range(len(legend_labels)), losses, color=colors, zorder=5, width=0.6)
    for bar, loss, leg_label, is_cda in zip(bars, losses, legend_labels, cda_flags):
        if not is_cda:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_alpha(0.85)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.3f}",
                ha="center", va="bottom", fontsize=9, zorder=6)
        bar.set_label(leg_label)

    ax.set_ylim(min(losses) - 0.02, max(losses) + 0.02)
    ax.set_xticks(list(range(len(tick_labels))))
    ax.set_xticklabels(tick_labels, rotation=20, ha="right")
    setup_axes(ax, title=title, xlabel="", ylabel="IID Loss")

    plt.tight_layout()
    _save_fig(fig, output_path or "cda_bar_comparison.png")


# ---------------------------------------------------------------------------
# 5. Student scaling bar chart
# ---------------------------------------------------------------------------


def plot_student_scaling(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    baselines: list[RunConfig] | None = None,
    output_path: str | None = None,
) -> None:
    """Grouped bar chart: Real Data Only / Simple / Stitched / Latent across 300M and 1.5B."""
    streams = [
        (None, "Real Data Only", REGULARIZED_COLOR),
        ("s32", "Simple Rephrasing (32 Generations)", DATA_STREAM_COLOR_DICT.get("s32", BLACK)),
        ("b32", "Stitched Rephrasing (32 Generations)", DATA_STREAM_COLOR_DICT.get("b32", BLACK)),
        ("z32", "Latent Thoughts (32 Generations)", DATA_STREAM_COLOR_DICT.get("z32", BLACK)),
    ]
    model_sizes = ["300m4k", "1_5b4k"]
    model_labels = {"300m4k": "300M Student", "1_5b4k": "1.5B Student"}

    n_groups = len(model_sizes)
    n_bars = len(streams)
    bar_width = 0.18
    group_gap = 0.20
    group_width = n_bars * bar_width
    group_centers = np.arange(n_groups) * (group_width + group_gap)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    pn = "student_scaling"
    all_losses: list[float] = []
    for bar_idx, (ds, label, color) in enumerate(streams):
        xs, ys = [], []
        for gi, ms in enumerate(model_sizes):
            if ds is None:
                loss, bl_cfg = _best_baseline_loss(baselines or [], name_to_summary, ms)
                if loss is not None:
                    _log_point(pn, f"baseline model={ms}", bl_cfg.name if bl_cfg else None, loss)
            else:
                loss, best_name, _ = _best_loss_for_stream(augmented, ds, name_to_summary, ms)
                if loss is not None:
                    _log_point(pn, f"stream={ds} model={ms}", best_name, loss)
            if loss is not None:
                xs.append(group_centers[gi] + bar_idx * bar_width)
                ys.append(loss)
        if xs:
            all_losses.extend(ys)
            bars = ax.bar(xs, ys, width=bar_width, color=color, label=label, zorder=5)
            for bar, y in zip(bars, ys):
                ax.text(bar.get_x() + bar.get_width() / 2, y + 0.003, f"{y:.3f}",
                        ha="center", va="bottom", fontsize=8, zorder=7)

    if all_losses:
        ax.set_ylim(min(all_losses) - 0.05, max(all_losses) + 0.05)

    ax.set_xticks(group_centers + (n_bars - 1) * bar_width / 2)
    ax.set_xticklabels([model_labels[ms] for ms in model_sizes])
    setup_axes(ax, title="Student Scaling", xlabel="", ylabel="IID Loss")

    # Deduplicate legend
    handles, labs = ax.get_legend_handles_labels()
    seen: set[str] = set()
    ax.legend(
        *zip(*[(h, l) for h, l in zip(handles, labs) if l not in seen and not seen.add(l)])
    )

    plt.tight_layout()
    _save_fig(fig, output_path or "student_scaling.png")


# ---------------------------------------------------------------------------
# 6. IID mixing ablation
# ---------------------------------------------------------------------------

NO_SEED_W2_LOSS = 3.521


def plot_iid_mixing_ablation(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    output_path: str | None = None,
) -> None:
    """Bar chart: seed-only vs stitched w/ seed vs stitched w/o seed."""
    w2_loss, w2_name, _ = _best_loss_for_stream(augmented, "w2", name_to_summary, "300m4k")
    if w2_loss is not None:
        _log_point("iid_mixing_ablation", "stream=w2", w2_name, w2_loss)
    if w2_loss is None:
        print("  No w2 loss found for IID mixing ablation")
        return

    w2_color = DATA_STREAM_COLOR_DICT.get("w2", BLACK)
    bar_labels = [
        "Real Data Only",
        "Stitched Rephrasing\n(2 Generations)",
        "No Real Data,\nStitched Rephrasing\n(2 Generations)",
    ]
    vals = [REGULARIZED_300M_LOSS, w2_loss, NO_SEED_W2_LOSS]
    bar_colors = [REGULARIZED_COLOR, w2_color, w2_color]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    bars = ax.bar(np.arange(3), vals, color=bar_colors, width=0.5, zorder=5)
    bars[2].set_hatch("///")
    bars[2].set_edgecolor("white")

    ax.set_ylim(min(vals) - 0.1, max(vals) + 0.1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, zorder=7)

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(bar_labels)
    setup_axes(ax, title="IID Mixing Ablation", xlabel="", ylabel="IID Loss")

    plt.tight_layout()
    _save_fig(fig, output_path or "iid_mixing_ablation.png")


# ---------------------------------------------------------------------------
# 7. Hyper table
# ---------------------------------------------------------------------------


def get_hyper_table(
    mode: str,
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    output_path: str | None = None,
) -> dict[int, tuple[int, float | None, float, float]]:
    """Return and plot optimal 300M hyperparameters per generation count (by IID loss)."""
    pn = f"hyper_table({mode})"
    stream_map = MODE_TO_STREAMS[mode]
    selected: dict[int, tuple[int, float | None, float, float]] = {}

    for copies, ds in stream_map.items():
        loss, best_name, cfg = _best_loss_for_stream(augmented, ds, name_to_summary, "300m4k")
        if cfg is not None and loss is not None:
            _log_point(pn, f"copies={copies} stream={ds}", best_name, loss)
            selected[copies] = (cfg.epochs, cfg.mix_ratio, cfg.lr, cfg.wd)

    if not selected:
        return selected

    # Build table data
    table_headers = ["Generations"] + [str(c) for c in COPY_COUNTS]
    rows: list[list[str]] = []
    for row_name, idx in [("Real data epochs", 0), ("Mixing ratio", 1), ("Learning rate", 2), ("Weight decay", 3)]:
        row = [row_name]
        for c in COPY_COUNTS:
            vals = selected.get(c)
            if vals is None:
                row.append("-")
            elif idx == 0:
                row.append(str(vals[0]))
            elif idx == 1:
                row.append("0.00" if vals[1] is None else f"{vals[1]:.2f}")
            elif idx == 2:
                row.append(re.sub(r"e([+-])0*(\d+)", r"e\1\2", f"{vals[2]:.0e}"))
            else:
                row.append(f"{vals[3]:.2f}")
        rows.append(row)

    setup_plot_style()
    n_cols = len(table_headers)
    fig_width = max(5.0, 0.75 * n_cols)
    fig_height = 0.35 * (len(rows) + 1) + 0.5
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.set_axis_off()
    ax.set_title(f"Optimal Hyperparameters \u2014 {MODE_PRETTY_NAMES[mode]}",
                 fontsize=12, weight="bold", pad=12)

    table = ax.table(cellText=rows, colLabels=table_headers, cellLoc="center",
                     loc="center", bbox=[0.0, 0.0, 1.0, 1.0])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    left_w = 0.30
    val_w = (1.0 - left_w) / (n_cols - 1)
    for r in range(len(rows) + 1):
        table[(r, 0)].set_width(left_w)
        for ci in range(1, n_cols):
            table[(r, ci)].set_width(val_w)
    for j in range(n_cols):
        table[(0, j)].set_facecolor("#E6E6FA")
        table[(0, j)].set_text_props(weight="bold")
    for r in range(1, len(rows) + 1):
        table[(r, 0)].set_facecolor("#F0F0F0")
        table[(r, 0)].set_text_props(weight="bold")

    plt.tight_layout()
    _save_fig(fig, output_path or f"hyper_table_{mode}.png")
    return selected


# ---------------------------------------------------------------------------
# 8. Generation scaling (copy scaling)
# ---------------------------------------------------------------------------


def plot_generation_scaling(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    metric_key: str = METRIC_KEY,
    series: list[tuple[str, dict[int, str]]] | None = None,
    baselines: list[RunConfig] | None = None,
    eval_runs: list[EnsembleEvalConfig] | None = None,
    eval_summary: dict[str, dict] | None = None,
    eval_run_prefix: EvalRunPrefix = DEFAULT_EVAL_RUN_PREFIXES,
    include_baseline: bool = True,
    include_citation: bool = False,
    output_path: str | None = None,
) -> None:
    """Plot best loss vs generation count for selected stream families."""
    ylabel, title_suffix, metric_suffix = _resolve_metric(metric_key)
    use_eval = eval_runs is not None

    if series is None:
        series = list(ALL_SERIES)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    plot_name = f"generation_scaling({output_path or 'default'})"
    bl_loss: float | None = None
    if include_baseline:
        if use_eval:
            assert eval_runs is not None and eval_summary is not None
            bl_loss, bl_cfg = _best_eval_baseline_loss(eval_runs, eval_summary, "300m4k",
                                                       metric_key=metric_key, run_prefix=eval_run_prefix)
            if bl_loss is not None:
                _log_point(plot_name, "baseline", bl_cfg.name if bl_cfg else None, bl_loss)
        elif baselines is not None:
            bl_loss, bl_cfg = _best_baseline_loss(baselines, name_to_summary, "300m4k", metric_key=metric_key)
            if bl_loss is not None:
                _log_point(plot_name, "baseline", bl_cfg.name if bl_cfg else None, bl_loss)
        if bl_loss is not None:
            bl_label = "Real Data Only (Kim et al., 2025)" if include_citation else "Real Data Only"
            ax.scatter(1, bl_loss, color=REGULARIZED_COLOR, s=50, zorder=5, marker="o",
                       label=f"{bl_label}: {bl_loss:.2f}")

    for label, stream_map in series:
        xs: list[int] = []
        ys: list[float] = []
        colors: list[str] = []
        ds_ids: list[str] = []
        for copies in COPY_COUNTS:
            ds = stream_map.get(copies)
            if ds is None:
                continue
            if use_eval:
                assert eval_runs is not None and eval_summary is not None
                loss, best_name, _ = _best_eval_loss_for_stream(eval_runs, ds, eval_summary, "300m4k",
                                                                metric_key=metric_key, run_prefix=eval_run_prefix)
            else:
                loss, best_name, _ = _best_loss_for_stream(augmented, ds, name_to_summary, "300m4k", metric_key=metric_key)
            if loss is not None:
                _log_point(plot_name, f"{label} copies={copies} stream={ds}", best_name, loss)
                xs.append(1 + copies)
                ys.append(loss)
                colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
                ds_ids.append(ds)

        if not xs:
            continue

        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, color=c, s=50, zorder=5)

        best_ds = ds_ids[ys.index(min(ys))]
        ax.plot(xs, ys, "--", color=colors[0], alpha=0.4, zorder=3,
                label=stream_legend(best_ds, min(ys)))

        if include_baseline and bl_loss is not None:
            ax.plot([1, xs[0]], [bl_loss, ys[0]], ":", color=colors[0], alpha=0.4, zorder=3)

    ax.set_xscale("log")
    tick_locs = ([1] if include_baseline else []) + [1 + c for c in COPY_COUNTS]
    tick_labels = (["0"] if include_baseline else []) + [str(c) for c in COPY_COUNTS]
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    setup_axes(ax, title=f"Scaling Generations{title_suffix}", xlabel="Generations per pretraining doc", ylabel=ylabel)
    plt.tight_layout()

    if output_path is None:
        labels_lower = [l.lower().replace(" ", "_") for l, _ in series]
        output_path = f"generation_scaling_{'_'.join(labels_lower)}{metric_suffix}.png"
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 9. Generation scaling — controlled hyperparameters
# ---------------------------------------------------------------------------


def plot_generation_scaling_controlled(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    baselines: list[RunConfig] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot copy scaling where Stitched and Latent use the Simple-optimal hypers."""
    pn = "generation_scaling_controlled"
    filtered = [c for c in augmented if c.bs == 64]

    def _find(ds: str, epochs: int, mix_ratio: float | None, lr: float, wd: float) -> tuple[float | None, str | None]:
        for c in filtered:
            if c.data_stream == ds and c.model_size == "300m4k" and c.cda:
                if c.epochs == epochs and c.mix_ratio == mix_ratio and c.lr == lr and c.wd == wd:
                    loss = get_loss(name_to_summary.get(c.name, {}))
                    if loss is not None:
                        return loss, c.name
        return None, None

    # Find best Simple per copy count
    simple_hypers: dict[int, tuple[int, float | None, float, float]] = {}
    simple_xs, simple_ys, simple_colors = [], [], []
    for copies in COPY_COUNTS:
        ds = SHUFFLED_COPY_STREAMS.get(copies)
        if ds is None:
            continue
        loss, best_name, cfg = _best_loss_for_stream(filtered, ds, name_to_summary, "300m4k")
        if loss is not None and cfg is not None:
            _log_point(pn, f"Simple copies={copies} stream={ds}", best_name, loss)
            simple_xs.append(1 + copies)
            simple_ys.append(loss)
            simple_colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
            simple_hypers[copies] = (cfg.epochs, cfg.mix_ratio, cfg.lr, cfg.wd)

    # Stitched / Latent with matching hypers
    controlled: dict[str, tuple[list[int], list[float], list[str]]] = {}
    for label, smap in [("Stitched", SORTED_COPY_STREAMS), ("Latent Thoughts", LATENT_COPY_STREAMS)]:
        xs, ys, cols = [], [], []
        for copies in COPY_COUNTS:
            ds = smap.get(copies)
            if ds is None or copies not in simple_hypers:
                continue
            ep, mr, lr, wd = simple_hypers[copies]
            loss, run_name = _find(ds, ep, mr, lr, wd)
            if loss is not None:
                _log_point(pn, f"{label} copies={copies} stream={ds}", run_name, loss)
                xs.append(1 + copies)
                ys.append(loss)
                cols.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
        controlled[label] = (xs, ys, cols)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    bl_loss: float | None = None
    if baselines is not None:
        bl_loss, bl_cfg = _best_baseline_loss(baselines, name_to_summary, "300m4k")
        if bl_loss is not None:
            _log_point(pn, "baseline", bl_cfg.name if bl_cfg else None, bl_loss)
            ax.scatter(1, bl_loss, color=REGULARIZED_COLOR, s=50, zorder=5, marker="o",
                       label=f"Real Data Only: {bl_loss:.2f}")

    all_data = [
        ("Simple", simple_xs, simple_ys, simple_colors, SHUFFLED_COPY_STREAMS),
        ("Stitched", *controlled["Stitched"], SORTED_COPY_STREAMS),
        ("Latent Thoughts", *controlled["Latent Thoughts"], LATENT_COPY_STREAMS),
    ]
    for label, xs, ys, colors, smap in all_data:
        if not xs:
            continue
        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, color=c, s=50, zorder=5)
        best_copies = COPY_COUNTS[[1 + c for c in COPY_COUNTS].index(xs[ys.index(min(ys))])]
        best_ds = smap[best_copies]
        ax.plot(xs, ys, "--", color=colors[0], alpha=0.4, zorder=3,
                label=stream_legend(best_ds, min(ys)))
        if bl_loss is not None:
            ax.plot([1, xs[0]], [bl_loss, ys[0]], ":", color=colors[0], alpha=0.4, zorder=3)

    ax.set_xscale("log")
    tick_locs = [1] + [1 + c for c in COPY_COUNTS]
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.xaxis.set_major_formatter(FixedFormatter(["0"] + [str(c) for c in COPY_COUNTS]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())

    setup_axes(ax, title="Scaling Generations (Controlled Hyperparameters)",
               xlabel="Generations per pretraining doc", ylabel="IID Loss")
    plt.tight_layout()
    _save_fig(fig, output_path or "generation_scaling_controlled.png")


# ---------------------------------------------------------------------------
# 10. Context-length scaling
# ---------------------------------------------------------------------------


def plot_context_length_scaling(
    eval_runs: list[EnsembleEvalConfig],
    eval_summary: dict[str, dict],
) -> None:
    """Plot short- and long-context generation scaling from eval runs."""
    for metric_key, ylabel, title_suffix, file_suffix in [
        (SHORT_CONTEXT_METRIC_KEY, "Short Context Loss", " (Short Context)", "_short_context"),
        (LONG_CONTEXT_METRIC_KEY, "Long Context Loss", " (Long Context)", "_long_context"),
    ]:
        pn = f"context_length_scaling{file_suffix}"
        prefix: EvalRunPrefix = "synth-data-best-models-"

        setup_plot_style()
        fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

        bl_loss, bl_cfg = _best_eval_baseline_loss(eval_runs, eval_summary, "300m4k",
                                                   metric_key=metric_key, run_prefix=prefix)
        if bl_loss is not None:
            _log_point(pn, "baseline", bl_cfg.name if bl_cfg else None, bl_loss)
            ax.scatter(1, bl_loss, color=REGULARIZED_COLOR, s=50, zorder=5, marker="o",
                       label=f"Real Data Only: {bl_loss:.2f}")

        for label, stream_map in ALL_SERIES:
            xs, ys, colors, ds_ids = [], [], [], []
            for copies in COPY_COUNTS:
                ds = stream_map.get(copies)
                if ds is None:
                    continue
                loss, best_name, _ = _best_eval_loss_for_stream(eval_runs, ds, eval_summary, "300m4k",
                                                                metric_key=metric_key, run_prefix=prefix)
                if loss is not None:
                    _log_point(pn, f"{label} copies={copies} stream={ds}", best_name, loss)
                    xs.append(1 + copies)
                    ys.append(loss)
                    colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
                    ds_ids.append(ds)

            if not xs:
                continue

            for x, y, c in zip(xs, ys, colors):
                ax.scatter(x, y, color=c, s=50, zorder=5)

            best_ds = ds_ids[ys.index(min(ys))]
            ax.plot(xs, ys, "--", color=colors[0], alpha=0.4, zorder=3,
                    label=stream_legend(best_ds, min(ys)))
            if bl_loss is not None:
                ax.plot([1, xs[0]], [bl_loss, ys[0]], ":", color=colors[0], alpha=0.4, zorder=3)

        ax.set_xscale("log")
        tick_locs = [1] + [1 + c for c in COPY_COUNTS]
        ax.xaxis.set_major_locator(FixedLocator(tick_locs))
        ax.xaxis.set_major_formatter(FixedFormatter(["0"] + [str(c) for c in COPY_COUNTS]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        setup_axes(ax, title=f"Scaling Generations{title_suffix}",
                   xlabel="Generations per pretraining doc", ylabel=ylabel)
        plt.tight_layout()
        _save_fig(fig, f"generation_scaling{file_suffix}.png")


# ---------------------------------------------------------------------------
# 11. Downstream benchmark scaling
# ---------------------------------------------------------------------------


def plot_downstream_scaling(
    *,
    results_path: str = "synth_data_downstream_benchmark_results.json",
    include_tasks: list[str] | None = None,
    title: str = "Scaling Generations (Downstream Benchmarks)",
    normalize_to_chance: bool = False,
    shuffled_only: bool = False,
    output_path: str = "generation_scaling_downstream_benchmark.png",
) -> None:
    """Plot downstream average benchmark error vs rephrases."""
    if not os.path.exists(results_path):
        print(f"  Downstream results not found at {results_path}; skipping.")
        return

    with open(results_path) as f:
        payload = json.load(f)

    chance_by_task = {"arc_challenge": 0.25, "arc_easy": 0.25, "hellaswag": 0.25,
                      "piqa": 0.50, "sciq": 0.25, "winogrande": 0.50}

    def _avg_error(run_payload: dict) -> float | None:
        seed_payload = run_payload.get("1")
        if not isinstance(seed_payload, dict):
            return None
        scores: list[float] = []
        for task in (include_tasks or list(seed_payload.keys())):
            task_data = seed_payload.get(task)
            if not isinstance(task_data, dict):
                continue
            acc = task_data.get("acc")
            if acc is None:
                continue
            acc_f = float(acc)
            if normalize_to_chance:
                chance = chance_by_task.get(task)
                if chance is None:
                    continue
                scores.append(float(np.clip((acc_f - chance) / (1.0 - chance), 0.0, 1.0)))
            else:
                scores.append(acc_f)
        return 1.0 - float(np.mean(scores)) if scores else None

    series = [("Simple", SHUFFLED_COPY_STREAMS)]
    if not shuffled_only:
        series += [("Stitched", SORTED_COPY_STREAMS), ("Latent Thoughts", LATENT_COPY_STREAMS)]

    pn = f"downstream_scaling({output_path})"
    reg_error: float | None = None
    reg_run_name: str | None = None
    series_errors: dict[str, dict[int, tuple[float, str]]] = {l: {} for l, _ in series}

    for run_name, run_payload in payload.items():
        cfg = RunConfig.from_run_name(run_name)
        if cfg is None or cfg.model_size != "300m4k" or not cfg.cda:
            continue
        err = _avg_error(run_payload)
        if err is None:
            continue
        if cfg.data_stream is None:
            if reg_error is None or err < reg_error:
                reg_error = err
                reg_run_name = run_name
            continue
        for label, smap in series:
            for copies, ds in smap.items():
                if cfg.data_stream == ds:
                    prev = series_errors[label].get(copies)
                    if prev is None or err < prev[0]:
                        series_errors[label][copies] = (err, run_name)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=SCALING_FIGSIZE, dpi=300)

    if reg_error is not None:
        _log_point(pn, "baseline", reg_run_name, reg_error)
        ax.scatter([1], [reg_error], color=REGULARIZED_COLOR, s=60, marker="o", zorder=6,
                   label=f"Real Data Only: {math.floor(reg_error * 100) / 100:.2f}")

    for label, smap in series:
        errs = series_errors[label]
        present = [c for c in COPY_COUNTS if c in errs]
        if not present:
            continue
        for c in present:
            _log_point(pn, f"{label} copies={c}", errs[c][1], errs[c][0])
        xs = [1 + c for c in present]
        ys = [errs[c][0] for c in present]
        color = DATA_STREAM_COLOR_DICT.get(smap[present[0]], BLACK)
        best = min(ys)
        best_copies = present[ys.index(best)]
        best_ds = smap[best_copies]
        floored = math.floor(best * 100) / 100
        ax.scatter(xs, ys, color=color, s=50, zorder=5)
        ax.plot(xs, ys, "--", color=color, alpha=0.8, zorder=4,
                label=stream_legend(best_ds, floored))
        if reg_error is not None:
            ax.plot([1, xs[0]], [reg_error, ys[0]], ":", color=color, alpha=0.5, zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, *[1 + c for c in COPY_COUNTS]]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0", *[str(c) for c in COPY_COUNTS]]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.9, 35)

    setup_axes(ax, title=title, xlabel="Generations per pretraining doc", ylabel="Average benchmark error")
    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# 12. Ensemble scaling
# ---------------------------------------------------------------------------

EnsembleSpec = tuple[str, bool, str | None, int, float] | tuple[str, bool, str | None, int, float, float | None]


def plot_ensemble_scaling(
    ensemble_evals: list[EnsembleEvalConfig],
    eval_summary: dict[str, dict],
    baselines: list[RunConfig],
    baseline_summary: dict[str, dict],
    *,
    ensemble_specs: list[EnsembleSpec],
    title: str = "Synthetic Data Ensembles",
    output_path: str = "ensemble_scaling_cda_augmented_300m.png",
) -> None:
    """Plot ensemble member count vs loss with power-law fits."""
    pn = "ensemble_scaling"
    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)
    plot_max = 10.0

    # Reference: single-model baselines
    opt_bl: dict[str, tuple[RunConfig, float]] = {}
    for cfg in baselines:
        if cfg.is_seed_run or cfg.model_size not in PARAM_STR_TO_COUNT:
            continue
        loss = get_loss(baseline_summary.get(cfg.name, {}))
        if loss is not None:
            ms = cfg.model_size
            if ms not in opt_bl or loss < opt_bl[ms][1]:
                opt_bl[ms] = (cfg, loss)

    if opt_bl:
        sizes = sorted(opt_bl, key=lambda s: PARAM_STR_TO_COUNT[s])
        for s in sizes:
            _log_point(pn, f"baseline model={s}", opt_bl[s][0].name, opt_bl[s][1])
        rx = np.array([PARAM_STR_TO_COUNT[s] for s in sizes])
        ry = np.array([opt_bl[s][1] for s in sizes])
        ax.scatter(rx, ry, color=REGULARIZED_COLOR, s=50, zorder=5)
        try:
            pw = PowerScalingLaw(var_name="N")
            yr = max(float(np.max(ry) - np.min(ry)), 1e-6)
            pw.fit(rx, ry, p0=[yr * np.max(rx)**0.5, 0.5, float(np.min(ry))],
                   bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            xf = np.logspace(np.log10(np.min(rx)), np.log10(np.max(rx)), 200)
            ax.plot(xf, pw.evaluate(xf), "--", color=REGULARIZED_COLOR, alpha=0.8, zorder=4,
                    label=f"Real Data Only (fit: {pw})")
        except RuntimeError:
            ax.plot(rx, ry, "--", color=REGULARIZED_COLOR, alpha=0.8, zorder=4, label="Real Data Only")

    for spec in ensemble_specs:
        ms, cda, ds, ep, wd = spec[:5]
        mr = spec[5] if len(spec) > 5 else None
        color = DATA_STREAM_COLOR_DICT.get(ds, PARAM_STR_COLOR_DICT.get(ms, BLACK)) if ds else PARAM_STR_COLOR_DICT.get(ms, BLACK)

        pretty = PRETTY_NAME_DICT.get(ms, ms)
        slabel = f"{pretty} {stream_display_name(ds)}" if ds else f"{pretty} Real Data Only"

        matching = [
            e for e in ensemble_evals
            if e.model_size == ms and e.cda == cda and e.data_stream == ds
            and e.epochs == ep and e.wd == wd and (mr is None or e.mix_ratio == mr)
        ]
        if not matching:
            continue

        by_count: dict[int, list[tuple[EnsembleEvalConfig, float]]] = defaultdict(list)
        for e in matching:
            loss = get_loss(eval_summary.get(e.name, {}))
            if loss is not None:
                by_count[e.member_count].append((e, loss))

        best_per: list[tuple[int, float]] = []
        for cnt, vs in sorted(by_count.items()):
            best_e, best_l = min(vs, key=lambda t: t[1])
            _log_point(pn, f"{slabel} members={cnt}", best_e.name, best_l)
            best_per.append((cnt, best_l))
        if not best_per:
            continue

        mc = [c for c, _ in best_per]
        ls = [l for _, l in best_per]
        ax.scatter(mc, ls, color=color, s=50, zorder=5)

        try:
            xa = np.array(mc, dtype=float)
            ya = np.array(ls)
            pw = PowerScalingLaw(var_name="K")
            yr = max(float(np.max(ya) - np.min(ya)), 1e-6)
            pw.fit(xa, ya, p0=[yr * np.max(xa)**0.5, 0.5, float(np.min(ya))],
                   bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
            xf = np.logspace(np.log10(float(np.min(xa))), np.log10(plot_max), 200)
            ax.plot(xf, pw.evaluate(xf), "--", color=color, alpha=0.8, zorder=4,
                    label=f"{slabel} (fit: {pw})")
            ax.axhline(pw.asymptote(), color=color, linestyle="--", alpha=0.4, zorder=1)
        except RuntimeError:
            ax.plot(mc, ls, "--", color=color, alpha=0.8, zorder=4, label=slabel)

    ax.set_xscale("log")
    ax.set_xlim(0.9, plot_max)
    ax.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    ax.xaxis.set_major_formatter(FixedFormatter(["1", "2", "3", "4", "5"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    setup_axes(ax, title=title, xlabel="Ensemble member count", ylabel="IID Loss")

    plt.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def main() -> None:
    global PRINT_RUNS
    parser = argparse.ArgumentParser(description="Final paper plots.")
    parser.add_argument("--cache_path", default=CACHE_PATH)
    parser.add_argument("--build_cache", action="store_true",
                        help="Fetch runs from wandb and rebuild the cache, then exit.")
    parser.add_argument("--print-runs", action="store_true",
                        help="Print the run name and loss for every plotted point.")
    args = parser.parse_args()
    PRINT_RUNS = args.print_runs

    if args.build_cache:
        build_cache(args.cache_path)
        return

    payload = _load_cache(args.cache_path)
    if payload is None:
        raise FileNotFoundError(f"No cache at {args.cache_path}. Run with --build_cache first.")
    _filter_finished(payload)

    configs, _ = parse_train_runs(payload)
    baselines, augmented, seed_runs = split_runs(configs)
    name_to_summary = build_name_to_summary(payload, TRAIN_PROJECT)
    ensemble_evals = parse_ensemble_eval_runs(payload)
    eval_summary = build_name_to_summary(payload, EVAL_PROJECT)

    # --- Loss vs epochs / mixing ---
    plot_loss_vs(augmented, "s16", name_to_summary, x_attr="epochs",
                 color=DATA_STREAM_COLOR_DICT["s16"])
    plot_loss_vs(augmented, "s16", name_to_summary, x_attr="mix_ratio",
                 color=DATA_STREAM_COLOR_DICT["s16"])

    # --- Controlled generation scaling ---
    plot_generation_scaling_controlled(augmented, name_to_summary, baselines=baselines)

    # --- Generation scaling (IID, all 3 series) ---
    plot_generation_scaling(augmented, name_to_summary, baselines=baselines,
                           include_citation=True,
                           output_path="generation_scaling_shuffled_sorted_latents.png")

    # --- Generation scaling (IID, simple only) ---
    plot_generation_scaling(augmented, name_to_summary, baselines=baselines,
                           series=[("Simple", SHUFFLED_COPY_STREAMS)],
                           output_path="generation_scaling_shuffled.png")

    # --- Hyper tables ---
    for mode in MODE_TO_STREAMS:
        get_hyper_table(mode, augmented, name_to_summary)

    # --- Student scaling ---
    plot_student_scaling(augmented, name_to_summary, baselines=baselines)

    # --- IID mixing ablation ---
    plot_iid_mixing_ablation(augmented, name_to_summary)

    # --- Compare loss vs epochs ---
    plot_compare_loss_vs_epochs(augmented, name_to_summary,
                                data_streams=["w2s", "s32", "b32"],
                                stream_epochs={"w2s": [2, 4, 8, 16], "s32": [8, 16, 32], "b32": [16, 32, 64]})

    # --- Augmented bar comparisons ---
    plot_augmented_bar(augmented, name_to_summary, data_streams=["s8", "f8", "b8", "z8"],
                       eval_runs=ensemble_evals, eval_summary=eval_summary,
                       eval_run_prefix="synth-data-var-models-", require_eval_seed=True,
                       title="Synthetic Data Ordering")
    plot_augmented_bar(augmented, name_to_summary, data_streams=["s8", "f8", "b8", "z8"],
                       eval_runs=ensemble_evals, eval_summary=eval_summary,
                       eval_run_prefix="synth-data-var-models-", require_eval_seed=True,
                       title="Synthetic Data Ordering (Arxiv CS)", metric="arxiv")
    plot_augmented_bar(augmented, name_to_summary, data_streams=["n8s", "s8", "n8", "b8"],
                       seed_runs=seed_runs, title="Synthetic Data Ordering")

    # --- CDA bar comparison ---
    plot_cda_bar(baselines, augmented, name_to_summary,
                 queries=[(True, None), (False, None), (True, "s16"), (False, "s16"), (True, "b16")],
                 title="Cross-Document Attention")

    # --- Generation scaling (other metrics, eval source) ---
    for mk in [ARXIV_METRIC_KEY, WIKIPEDIA_METRIC_KEY, GITHUB_PYTHON_METRIC_KEY]:
        plot_generation_scaling(augmented, name_to_summary, metric_key=mk,
                                eval_runs=ensemble_evals, eval_summary=eval_summary)

    # --- Context-length scaling ---
    plot_context_length_scaling(ensemble_evals, eval_summary)

    # --- Downstream scaling ---
    plot_downstream_scaling(include_tasks=["arc_easy", "piqa", "sciq"],
                            shuffled_only=True,
                            output_path="generation_scaling_downstream_benchmark_arc_easy_piqa_sciq_shuffled_only.png")
    plot_downstream_scaling(include_tasks=["arc_easy", "piqa", "sciq"],
                            output_path="generation_scaling_downstream_benchmark_arc_easy_piqa_sciq_raw.png")

    # --- Ensemble scaling ---
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=[
            ("300m4k", True, None, 32, 0.80),
            ("300m4k", True, "sdn", 16, 0.40, 0.5),
            ("300m4k", True, "b32", 32, 0.40, 0.75),
            ("300m4k", True, "s32", 32, 0.40, 0.75),
            ("300m4k", True, "z32", 32, 0.40, 0.75),
        ],
    )


if __name__ == "__main__":
    main()
