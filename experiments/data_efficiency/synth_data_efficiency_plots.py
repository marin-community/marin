#!/usr/bin/env python3
"""Pull and cache wandb runs matching x{N}-dcr for synthetic data efficiency analysis."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import wandb

from experiments.data_efficiency.synth_data_efficiency_plot_utils import (
    BLACK,
    DATA_STREAM_COLOR_DICT,
    DATA_STREAM_NAMES,
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
)

PROJECTS = [
    "stanford-mercury/suhas-data-efficiency",
    "stanford-mercury/suhas-eval-data-efficiency",
]
TRAIN_PROJECT = "stanford-mercury/suhas-data-efficiency"
EVAL_PROJECT = "stanford-mercury/suhas-eval-data-efficiency"
CACHE_PATH = "experiments/data_efficiency/cache/synth_data_efficiency.pkl"

# Matches run names containing "x<number>-dcr", e.g. x2-dcr, x16-dcr, x32-dcr
RUN_NAME_PATTERN = re.compile(r"x\d+-dcr")

# Pattern to parse run names like:
#   300m4kcda-203Mx16-dcr+s32^0.75-cos-lr0.0030-wd0.40-bs64
#   150m4k-203Mx16-dcr-cos-lr0.0030-wd0.80-bs64
#   600m4k-203Mx8-dcr-cos-lr0.0010-wd3.20-seed1
RUN_CONFIG_PATTERN = re.compile(
    r"^(?P<model_size>\d+m\d+k|1_5b\d+k)"
    r"(?P<cda>cda)?"
    r"-203Mx(?P<epochs>\d+)"
    r"-dcr"
    r"(?:\+(?P<data_stream>[a-z0-9]+)\^(?P<mix_ratio>[\d.]+))?"
    r"-cos"
    r"-lr(?P<lr>[\d.]+)"
    r"-wd(?P<wd>[\d.]+)"
    r"(?:-bs(?P<bs>\d+))?"
    r"(?:-seed(?P<seed>\d+))?$"
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
        """Compute total real-data tokens seen during training.

        Without a synthetic mix, it's simply REAL_DATA_TOKENS * epochs.
        With a mix, the synthetic stream also contains one copy of the real data,
        so we add the real tokens contributed by epoching through the synthetic
        stream: (total_synthetic_tokens_seen / synth_stream_size) * REAL_DATA_TOKENS.
        """
        base = REAL_DATA_TOKENS * self.epochs
        if self.mix_ratio is None or self.data_stream is None:
            return float(base)
        synth_tokens = SYNTH_STREAM_TOKENS.get(self.data_stream)
        if synth_tokens is None:
            return float(base)
        total_tokens = REAL_DATA_TOKENS * (1 / (1 - self.mix_ratio)) * self.epochs
        synth_epochs = (total_tokens / synth_tokens)
        return base + REAL_DATA_TOKENS * synth_epochs

    def short_desc(self) -> str:
        parts = [self.model_size]
        if self.cda:
            parts.append("cda")
        parts.append(f"x{self.epochs}")
        if self.data_stream:
            parts.append(f"+{self.data_stream}^{self.mix_ratio}")
        return " ".join(parts)


# Pattern for ensemble evaluation runs from the eval project, e.g.:
#   ppl-eval-ensemble-3x-300m4kcda-203Mx16-dcr+b16^0.75-cos-lr0.0030-wd0.40-seed0-49727-ecebca
#   ppl-eval-ensemble-5x-600m4k-203Mx8-dcr-cos-lr0.0010-wd3.20-seed0-6215-e93d37
ENSEMBLE_EVAL_PATTERN = re.compile(
    r"^ppl-eval-ensemble-(?P<member_count>\d+)x"
    r"-(?P<model_size>\d+m\d+k|1_5b\d+k)"
    r"(?P<cda>cda)?"
    r"-203Mx(?P<epochs>\d+)"
    r"-dcr"
    r"(?:\+(?P<data_stream>[a-z0-9]+)\^(?P<mix_ratio>[\d.]+))?"
    r"-cos"
    r"-lr(?P<lr>[\d.]+)"
    r"-wd(?P<wd>[\d.]+)"
    r"(?:-bs(?P<bs>\d+))?"
    r"-seed(?P<seed>\d+)"
    r"-(?P<num_steps>\d+)"
    r"-(?P<run_hash>[0-9a-f]+)$"
)


@dataclass(frozen=True)
class EnsembleEvalConfig:
    """Parsed config for an ensemble evaluation run from the eval project."""

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
        return EnsembleEvalConfig(
            name=name,
            member_count=int(m.group("member_count")),
            model_size=m.group("model_size"),
            cda=m.group("cda") is not None,
            epochs=int(m.group("epochs")),
            data_stream=m.group("data_stream"),
            mix_ratio=float(m.group("mix_ratio")) if m.group("mix_ratio") else None,
            lr=float(m.group("lr")),
            wd=float(m.group("wd")),
            bs=int(m.group("bs")) if m.group("bs") else None,
        )

    @property
    def is_baseline(self) -> bool:
        return self.data_stream is None

    @property
    def total_params(self) -> float:
        return PARAM_STR_TO_COUNT.get(self.model_size, 0) * self.member_count

    def series_key(self) -> tuple[str, bool, str | None, int, float]:
        """Grouping key: (model_size, cda, data_stream, epochs, wd)."""
        return (self.model_size, self.cda, self.data_stream, self.epochs, self.wd)

    def short_label(self) -> str:
        pretty = PRETTY_NAME_DICT.get(self.model_size, self.model_size)
        parts = [pretty]
        if self.cda:
            parts.append("CDA")
        if self.data_stream:
            parts.append(DATA_STREAM_NAMES.get(self.data_stream, self.data_stream))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Cache helpers
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


def fetch_matching_runs(project: str) -> list[wandb.apis.public.Run]:
    """Fetch all runs whose display name contains x{N}-dcr."""
    api = wandb.Api(timeout=120)
    runs = list(api.runs(project, filters={"display_name": {"$regex": r"x\d+-dcr"}}, per_page=200))
    runs.sort(key=lambda run: run.name or "")
    return runs


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


def _serialize_run(run: wandb.apis.public.Run) -> dict:
    return {
        "name": run.name,
        "id": run.id,
        "state": run.state,
        "created_at": run.created_at,
        "summary": _make_json_safe(dict(run.summary)),
        "config": _make_json_safe(dict(run.config)),
    }


def _print_runs(project: str, runs: list[dict]) -> None:
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
        serialized = [_serialize_run(run) for run in runs]
        payload[project] = serialized
        _print_runs(project, serialized)

    _save_cache(cache_path, payload)
    print(f"\nCache saved to {cache_path}")
    return payload


# ---------------------------------------------------------------------------
# Parsing / analysis
# ---------------------------------------------------------------------------

def parse_train_runs(payload: dict) -> tuple[list[RunConfig], list[str]]:
    """Parse all runs under the training project. Returns (parsed, unmatched_names)."""
    runs = payload.get(TRAIN_PROJECT, [])
    parsed: list[RunConfig] = []
    unmatched: list[str] = []
    for run in runs:
        name = run["name"]
        config = RunConfig.from_run_name(name)
        if config is None:
            unmatched.append(name)
        else:
            parsed.append(config)
    return parsed, unmatched


def parse_ensemble_eval_runs(payload: dict) -> list[EnsembleEvalConfig]:
    """Parse ensemble evaluation runs from the eval project."""
    runs = payload.get(EVAL_PROJECT, [])
    parsed: list[EnsembleEvalConfig] = []
    for run in runs:
        cfg = EnsembleEvalConfig.from_run_name(run["name"])
        if cfg is not None:
            parsed.append(cfg)
    print(f"\n  Parsed {len(parsed)} ensemble eval runs from {EVAL_PROJECT}")
    if parsed:
        groups = defaultdict(set)
        for cfg in parsed:
            groups[cfg.series_key()].add(cfg.member_count)
        for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] or "", k[3], k[4])):
            ms, cda, ds, ep, wd = key
            counts = sorted(groups[key])
            label = f"{ms}{'cda' if cda else ''} x{ep} wd{wd}{' +' + ds if ds else ''}"
            print(f"    {label:<55s}  member_counts={counts}")
    return parsed


def print_parsed_summary(configs: list[RunConfig], unmatched: list[str]) -> None:
    seed_runs = [c for c in configs if c.is_seed_run]
    non_seed = [c for c in configs if not c.is_seed_run]
    baselines = [c for c in non_seed if c.is_baseline]
    augmented = [c for c in non_seed if not c.is_baseline]

    print(f"\n{'=' * 80}")
    print("Parsed run summary (stanford-mercury/suhas-data-efficiency)")
    print(f"{'=' * 80}")
    print(f"  Total parsed:    {len(configs)}")
    print(f"  Seed runs:       {len(seed_runs)} (ignored)")
    print(f"  Baselines (dcr only): {len(baselines)}")
    print(f"  Augmented (+stream): {len(augmented)}")
    print(f"  Unmatched:       {len(unmatched)}")

    model_sizes = sorted({c.model_size for c in non_seed})
    epoch_counts = sorted({c.epochs for c in non_seed})
    streams = sorted({c.data_stream for c in augmented if c.data_stream})
    cda_counts = sum(1 for c in non_seed if c.cda)

    print(f"\n  Model sizes:     {model_sizes}")
    print(f"  Epoch counts:    {epoch_counts}")
    print(f"  CDA runs:        {cda_counts}")
    print(f"  Data streams:    {streams}")

    """
    if unmatched:
        print(f"\n  --- Unmatched run names ---")
        for name in unmatched:
            print(f"    {name}")
    """
    print(f"\n  --- Seed run names ---")
    for c in sorted(seed_runs, key=lambda c: (c.model_size, c.cda, c.epochs)):
        print(f"    {c.name}")

    print(f"\n  --- Baselines (non-seed) ---")
    for c in sorted(baselines, key=lambda c: (c.model_size, c.cda, c.epochs)):
        print(f"    {c.name}")

    print(f"\n  --- Augmented runs (non-seed) ---")
    for c in sorted(augmented, key=lambda c: (c.model_size, c.cda, c.epochs, c.data_stream or "", c.mix_ratio or 0)):
        print(f"    {c.short_desc():<40s}  lr={c.lr}  wd={c.wd}  bs={c.bs}  [{c.name}]")

    return baselines, augmented, seed_runs


# ---------------------------------------------------------------------------
# Metric lookup
# ---------------------------------------------------------------------------

METRIC_KEY = "eval/dc_1k_val_normal/loss"


def build_name_to_summary(payload: dict, project: str) -> dict[str, dict]:
    """Map run name -> summary dict for a given project."""
    return {run["name"]: run.get("summary", {}) for run in payload.get(project, [])}


def get_loss(summary: dict, metric_key: str = METRIC_KEY) -> float | None:
    val = summary.get(metric_key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Regularized scaling plot
# ---------------------------------------------------------------------------

PLOT_DIR = "experiments/data_efficiency/plots/synth_data_efficiency/"


def plot_regularized_scaling(
    baselines: list[RunConfig],
    name_to_summary: dict[str, dict],
    output_path: str = "regularized_scaling.png",
    *,
    cda_only: bool = False,
) -> None:
    """For each model size, pick the baseline with the lowest IID loss (separately
    for CDA and non-CDA), then plot parameter count vs loss with power-law fits.

    If *cda_only* is True, only the "With CDA" series is plotted.
    """

    groups: dict[str, list[RunConfig]] = {"no_cda": [], "cda": []}
    for cfg in baselines:
        if cfg.is_seed_run:
            continue
        key = "cda" if cfg.cda else "no_cda"
        groups[key].append(cfg)

    # For each group, for each model_size, find the config with the lowest loss
    optimal: dict[str, dict[str, tuple[RunConfig, float]]] = {}
    for group_name, cfgs in groups.items():
        by_model: dict[str, list[tuple[RunConfig, float]]] = defaultdict(list)
        for cfg in cfgs:
            if cfg.model_size not in PARAM_STR_TO_COUNT:
                continue
            summary = name_to_summary.get(cfg.name, {})
            loss = get_loss(summary)
            if loss is None:
                continue
            by_model[cfg.model_size].append((cfg, loss))

        optimal[group_name] = {}
        for model_size, entries in sorted(by_model.items()):
            best_cfg, best_loss = min(entries, key=lambda t: t[1])
            optimal[group_name][model_size] = (best_cfg, best_loss)

    # Print optimal configs
    for group_name in ("no_cda", "cda"):
        label = "Without CDA" if group_name == "no_cda" else "With CDA"
        print(f"\n  --- Optimal baselines: {label} ---")
        for model_size in sorted(optimal[group_name], key=lambda s: PARAM_STR_TO_COUNT[s]):
            cfg, loss = optimal[group_name][model_size]
            pretty = PRETTY_NAME_DICT.get(model_size, model_size)
            print(f"    {pretty:<8s}  loss={loss:.4f}  lr={cfg.lr}  wd={cfg.wd}  bs={cfg.bs}  x{cfg.epochs}  [{cfg.name}]")

    # Build arrays for plotting
    all_plot_specs = [
        ("cda", "With cross-doc attention", REGULARIZED_COLOR),
        ("no_cda", "Without cross-doc attention", RED),
    ]
    plot_specs = [s for s in all_plot_specs if not cda_only or s[0] == "cda"]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    for group_name, label, color in plot_specs:
        opt = optimal.get(group_name, {})
        if not opt:
            continue
        model_sizes_sorted = sorted(opt, key=lambda s: PARAM_STR_TO_COUNT[s])
        x = np.array([PARAM_STR_TO_COUNT[ms] for ms in model_sizes_sorted])
        y = np.array([opt[ms][1] for ms in model_sizes_sorted])

        # Fit power law
        power_law = PowerScalingLaw(var_name="N")
        y_range = max(float(np.max(y) - np.min(y)), 1e-6)
        a0 = y_range * (np.max(x) ** 0.5)
        power_law.fit(x, y, p0=[a0, 0.5, float(np.min(y))], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

        x_fit = np.logspace(np.log10(np.min(x)), np.log10(np.max(x) * 3), 200)
        y_fit = power_law.evaluate(x_fit)

        asymptote = power_law.asymptote()
        print(f"  {label} fit: {asymptote}")

        fit_label = f"{label} (fit: {power_law})"
        ax.scatter(x, y, color=color, s=40, zorder=5, label=fit_label)
        ax.plot(x_fit, y_fit, "--", color=color, zorder=4, alpha=0.8)
        ax.axhline(asymptote, color=color, linestyle=":", alpha=0.5, zorder=-1)

    # Must set log scale before custom ticks, otherwise set_xscale resets them
    ax.set_xscale("log")

    all_model_sizes = sorted(
        {ms for grp in optimal.values() for ms in grp},
        key=lambda s: PARAM_STR_TO_COUNT[s],
    )
    tick_vals = [PARAM_STR_TO_COUNT[ms] for ms in all_model_sizes]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([PRETTY_NAME_DICT.get(ms, ms) for ms in all_model_sizes])
    ax.tick_params(axis="x", which="minor", bottom=False)

    setup_axes(ax, title="Regularized Scaling", xlabel="Parameters", ylabel="IID Loss")

    plt.tight_layout()
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# Loss vs epochs plots (per data stream)
# ---------------------------------------------------------------------------


REGULARIZED_300M_LOSS = 3.5544
REGULARIZED_ASYMPTOTE = 3.449006944291979


def plot_loss_vs_epochs(
    augmented: list[RunConfig],
    data_stream: str,
    name_to_summary: dict[str, dict],
    *,
    model_size: str = "300m4k",
    color: str = PURPLE,
    output_path: str | None = None,
    include_baselines: bool = False,
) -> None:
    """Plot best eval loss vs number of real-data epochs for a single data stream.

    For each epoch count present in the augmented runs for *data_stream*
    (auto-extracted, not hardcoded), the run with the lowest loss is selected
    and plotted.  All individual runs are also shown at low opacity so the
    spread across hyperparameter choices is visible.
    """
    stream_runs = [
        c for c in augmented
        if c.data_stream == data_stream
        and c.model_size == model_size
        and not c.is_seed_run
    ]
    if not stream_runs:
        print(f"  No augmented runs for stream={data_stream}, model_size={model_size}")
        return

    epoch_counts = sorted({c.epochs for c in stream_runs})

    # Collect (epoch, loss) for every run with a valid loss
    all_epochs: list[int] = []
    all_losses: list[float] = []
    for c in stream_runs:
        loss = get_loss(name_to_summary.get(c.name, {}))
        # NOTE: filters out runs that mess up y-axis
        if loss is not None and loss < 5.0:
            all_epochs.append(c.epochs)
            all_losses.append(loss)

    best_per_epoch: list[tuple[int, RunConfig, float]] = []
    for epoch in epoch_counts:
        epoch_runs = [c for c in stream_runs if c.epochs == epoch]
        scored = []
        for c in epoch_runs:
            loss = get_loss(name_to_summary.get(c.name, {}))
            if loss is not None:
                scored.append((c, loss))
        if not scored:
            continue
        best_cfg, best_loss = min(scored, key=lambda t: t[1])
        best_per_epoch.append((epoch, best_cfg, best_loss))

    if not best_per_epoch:
        print(f"  No valid losses for stream={data_stream}")
        return

    epochs = [e for e, _, _ in best_per_epoch]
    losses = [l for _, _, l in best_per_epoch]

    pretty_stream = DATA_STREAM_NAMES.get(data_stream, data_stream)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    if include_baselines:
        ax.axhline(REGULARIZED_300M_LOSS, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized 300M ({REGULARIZED_300M_LOSS:.2f})")
        ax.axhline(REGULARIZED_ASYMPTOTE, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized asymptote ({REGULARIZED_ASYMPTOTE:.2f})")

    # All runs at low opacity to show spread
    ax.scatter(all_epochs, all_losses, color=color, s=50, alpha=0.2, zorder=2)

    # Best-per-epoch line
    ax.scatter(epochs, losses, color=color, s=50, zorder=5)
    ax.plot(epochs, losses, "--", color=color, label=f"{pretty_stream}: {min(losses):.2f}")

    min_loss = min(losses)
    min_epoch = epochs[losses.index(min_loss)]
    ax.scatter(min_epoch, min_loss, color=RED, s=70, marker="o", edgecolor=RED, linewidth=2, zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(epoch_counts))
    ax.xaxis.set_major_formatter(FixedFormatter([str(int(e)) for e in epoch_counts]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    setup_axes(ax, title=f"Loss vs Epochs \u2014 {pretty_stream}", xlabel="Epochs of real data", ylabel="IID Loss")


    plt.tight_layout()
    if output_path is None:
        output_path = f"loss_vs_epochs_{data_stream}.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


def plot_all_loss_vs_epochs(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    model_size: str = "300m4k",
    include_baselines: bool = False,
) -> None:
    """Generate a loss-vs-epochs plot for every data stream present in *augmented*."""
    streams = sorted({c.data_stream for c in augmented if c.data_stream and not c.is_seed_run})
    print(f"\nGenerating loss-vs-epochs plots for {len(streams)} data streams ...")
    for ds in streams:
        if ds not in DATA_STREAM_COLOR_DICT:
            continue
        print(f"  Plotting loss-vs-epochs for {ds}")
        color = DATA_STREAM_COLOR_DICT.get(ds, BLACK)
        plot_loss_vs_epochs(augmented, ds, name_to_summary, model_size=model_size, color=color,
                            include_baselines=include_baselines)


def compare_loss_vs_epochs(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    data_streams: list[str] = ["w2s", "s32", "b32"],
    stream_epochs: dict[str, list[int]] | None = None,
    stream_token_range: dict[str, list[tuple[float, float]]] | None = None,
    use_real_tokens: bool = False,
    model_size: str = "300m4k",
    include_baselines: bool = False,
    output_path: str | None = None,
) -> None:
    """Overlay best-loss-per-epoch curves for multiple data streams on one plot.

    Highlights the overall best loss for each stream with a red circle and draws
    arrows between consecutive best-loss points to show the progression.

    *stream_epochs*, when provided, maps data-stream codes to the specific epoch
    counts to include, e.g. ``{"w2s": [2, 4, 8, 16], "b32": [16, 32, 64]}``.
    Streams not present in the dict use all available epochs.

    *stream_token_range*, when provided, maps data-stream codes to a list of
    (min, max) real-token bounds.  A run is kept if its ``get_real_tokens()``
    falls within any of the ranges (inclusive), e.g.
    ``{"w2s": [(1e9, 4e9), (6e9, 7.5e9)]}``.

    If *use_real_tokens* is True, the x-axis shows total real tokens seen (via
    ``RunConfig.get_real_tokens()``) instead of epoch count.  Runs are grouped by
    their real-token value per stream.  Cannot be combined with *stream_epochs*.
    """
    if use_real_tokens and stream_epochs:
        raise ValueError("use_real_tokens and stream_epochs cannot both be specified")
    if stream_epochs is None:
        stream_epochs = {}
    if stream_token_range is None:
        stream_token_range = {}

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    if include_baselines:
        ax.axhline(REGULARIZED_300M_LOSS, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized 300M ({REGULARIZED_300M_LOSS:.2f})")
        ax.axhline(REGULARIZED_ASYMPTOTE, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized asymptote ({REGULARIZED_ASYMPTOTE:.2f})")

    all_x_vals: set[float] = set()
    best_points: list[tuple[float, float, str]] = []

    for stream_idx, ds in enumerate(data_streams):
        color = DATA_STREAM_COLOR_DICT.get(ds, BLACK)
        pretty = DATA_STREAM_NAMES.get(ds, ds)
        alpha = 0.45 if stream_idx == 0 else 1.0

        stream_runs = [
            c for c in augmented
            if c.data_stream == ds and c.model_size == model_size and not c.is_seed_run and c.cda
        ]
        token_ranges = stream_token_range.get(ds)
        if token_ranges is not None:
            stream_runs = [
                c for c in stream_runs
                if any(lo <= c.get_real_tokens() <= hi for lo, hi in token_ranges)
            ]
        if not stream_runs:
            print(f"  No augmented runs for stream={ds}, model_size={model_size}")
            continue

        if use_real_tokens:
            # Group runs by real-token value
            by_x: dict[float, list[float]] = defaultdict(list)
            for c in stream_runs:
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None and loss < 5.0:
                    by_x[c.get_real_tokens()].append(loss)
            x_vals = sorted(by_x.keys())
            best_per_x = [(x, min(by_x[x])) for x in x_vals]
        else:
            epoch_counts = sorted({c.epochs for c in stream_runs})
            allowed = stream_epochs.get(ds)
            if allowed is not None:
                epoch_counts = [e for e in epoch_counts if e in allowed]

            best_per_x: list[tuple[float, float]] = []
            for epoch in epoch_counts:
                scored = []
                for c in stream_runs:
                    if c.epochs != epoch:
                        continue
                    loss = get_loss(name_to_summary.get(c.name, {}))
                    if loss is not None and loss < 5.0:
                        scored.append(loss)
                if scored:
                    best_per_x.append((float(epoch), min(scored)))

        if not best_per_x:
            print(f"  No valid losses for stream={ds}")
            continue

        xs = [x for x, _ in best_per_x]
        losses = [l for _, l in best_per_x]
        all_x_vals.update(xs)

        ax.set_xscale("log")

        ax.scatter(xs, losses, color=color, s=50, zorder=5, alpha=alpha)
        ax.plot(xs, losses, "--", color=color, alpha=alpha,
                label=f"{pretty}: {min(losses):.2f}")

        overall_best_loss = min(losses)
        overall_best_x = xs[losses.index(overall_best_loss)]
        ax.scatter(overall_best_x, overall_best_loss, color=RED, s=90, marker="o",
                   edgecolor=RED, linewidth=2, zorder=3, facecolors="none")
        best_points.append((overall_best_x, overall_best_loss, ds))

    # Draw arrows between consecutive best-loss points
    for i in range(len(best_points) - 1):
        x0, y0, _ = best_points[i]
        x1, y1, _ = best_points[i + 1]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color=BLACK, lw=1.2, alpha=0.25,
                shrinkA=8, shrinkB=8,
            ),
            zorder=2,
        )

    if use_real_tokens:
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e9:.0f}B"))
        xlabel = "Real tokens seen"
        title = "Scaling Real Tokens"
    else:
        sorted_x = sorted(all_x_vals)
        ax.xaxis.set_major_locator(FixedLocator(sorted_x))
        ax.xaxis.set_major_formatter(FixedFormatter([str(int(x)) for x in sorted_x]))
        ax.xaxis.set_minor_locator(FixedLocator([]))
        ax.xaxis.set_minor_formatter(NullFormatter())
        xlabel = "Epochs of real data"
        title = "Scaling Real Epochs"

    stream_tag = "_".join(data_streams)
    setup_axes(ax, title=title, xlabel=xlabel, ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        suffix = "_real_tokens" if use_real_tokens else ""
        output_path = f"compare_loss_vs_epochs_{stream_tag}{suffix}.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# Augmented bar comparison
# ---------------------------------------------------------------------------


def _seed_config_key(c: RunConfig) -> tuple:
    """Return a hashable key identifying the config ignoring seed."""
    return (c.model_size, c.cda, c.epochs, c.data_stream, c.mix_ratio, c.lr, c.wd, c.bs)


def plot_augmented_bar_comparison(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    data_streams: list[str],
    model_size: str = "300m4k",
    include_baselines: bool = False,
    use_seed_runs: bool = False,
    seed_runs: list[RunConfig] | None = None,
    output_path: str | None = None,
    title: str = "Synthetic Data Ordering",
) -> None:
    """Bar plot comparing the best augmented loss per data stream.

    When *use_seed_runs* is True, the *seed_runs* list is used instead of
    *augmented*.  Runs sharing the same config (ignoring seed) are grouped,
    their losses averaged, and the config group with the best (lowest) average
    is used for the bar.
    """
    labels: list[str] = []
    short_labels: list[str] = []
    losses: list[float] = []
    colors: list[str] = []

    for ds in data_streams:
        if use_seed_runs:
            source = seed_runs if seed_runs is not None else augmented
            groups: dict[tuple, list[float]] = defaultdict(list)
            for c in source:
                if c.data_stream != ds or c.model_size != model_size or not c.cda:
                    continue
                if not c.is_seed_run:
                    continue
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None:
                    groups[_seed_config_key(c)].append(loss)
            if not groups:
                print(f"  No seed runs for stream={ds}, model_size={model_size}")
                continue
            best_avg = min(np.mean(v) for v in groups.values())
        else:
            best_avg: float | None = None
            for c in augmented:
                if c.data_stream != ds or c.model_size != model_size or c.is_seed_run or not c.cda:
                    continue
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None and (best_avg is None or loss < best_avg):
                    best_avg = loss
            if best_avg is None:
                print(f"  No valid loss for stream={ds}, model_size={model_size}")
                continue

        labels.append(DATA_STREAM_NAMES.get(ds, ds))
        short_labels.append(DATA_STREAM_SHORT_NAMES.get(ds, ds))
        losses.append(float(best_avg))
        colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))

    if not losses:
        print("  No data to plot for augmented bar comparison")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    if include_baselines:
        ax.axhline(REGULARIZED_300M_LOSS, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized 300M ({REGULARIZED_300M_LOSS:.3f})")
        ax.axhline(REGULARIZED_ASYMPTOTE, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=-1,
                   label=f"Regularized asymptote ({REGULARIZED_ASYMPTOTE:.3f})")

    x_pos = range(len(labels))
    bars = ax.bar(x_pos, losses, color=colors, zorder=5, width=0.6)

    for bar, loss, label in zip(bars, losses, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.3f}",
                ha="center", va="bottom", fontsize=9, zorder=6)
        bar.set_label(f"{label}: {loss:.3f}")

    ax.set_ylim(min(losses) - 0.02, max(losses) + 0.02)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    setup_axes(ax, title=title, xlabel="", ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        stream_tag = "_".join(data_streams)
        output_path = f"augmented_bar_{stream_tag}.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# All bar comparison (baselines + augmented, CDA / no-CDA queries)
# ---------------------------------------------------------------------------

BarQuery = tuple[bool, str | None]  # (cda, data_stream | None)


def _bar_query_label(cda: bool, data_stream: str | None) -> str:
    cda_tag = "CDA" if cda else "No CDA"
    if data_stream is None:
        return f"Regularized ({cda_tag})"
    pretty = DATA_STREAM_SHORT_NAMES.get(data_stream, data_stream)
    return f"{pretty} ({cda_tag})"


def _bar_query_color(data_stream: str | None) -> str:
    if data_stream is not None:
        return DATA_STREAM_COLOR_DICT.get(data_stream, BLACK)
    return REGULARIZED_COLOR


def plot_all_bar_comparison(
    baselines: list[RunConfig],
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    queries: list[BarQuery],
    model_size: str = "300m4k",
    output_path: str | None = None,
    title: str = "Best Loss by Configuration",
) -> None:
    """Bar plot comparing best loss for each (cda, data_stream) query.

    Each query is a (cda, data_stream) pair.  When *data_stream* is None the
    search runs over baselines; otherwise over augmented runs.  The CDA flag
    filters consistently in both cases.
    """
    labels: list[str] = []
    losses: list[float] = []
    colors: list[str] = []
    cda_flags: list[bool] = []

    for cda, data_stream in queries:
        if data_stream is None:
            best: float | None = None
            best_name: str | None = None
            for c in baselines:
                if c.is_seed_run or not c.is_baseline or c.model_size != model_size:
                    continue
                if c.cda != cda:
                    continue
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None and (best is None or loss < best):
                    best = loss
                    best_name = c.name
        else:
            best = None
            best_name = None
            for c in augmented:
                if c.data_stream != data_stream or c.model_size != model_size or c.is_seed_run:
                    continue
                if c.cda != cda:
                    continue
                loss = get_loss(name_to_summary.get(c.name, {}))
                if loss is not None and (best is None or loss < best):
                    best = loss
                    best_name = c.name

        label = _bar_query_label(cda, data_stream)
        if best is None:
            print(f"  No valid loss for query cda={cda}, stream={data_stream}")
            continue

        print(f"  {label:<35s}  loss={best:.4f}  [{best_name}]")
        labels.append(label)
        losses.append(best)
        colors.append(_bar_query_color(data_stream))
        cda_flags.append(cda)

    if not losses:
        print("  No data to plot for all bar comparison")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    x_pos = range(len(labels))
    bars = ax.bar(x_pos, losses, color=colors, zorder=5, width=0.6)

    for bar, loss, label, is_cda in zip(bars, losses, labels, cda_flags):
        if not is_cda:
            bar.set_hatch("///")
            bar.set_edgecolor("white")
            bar.set_alpha(0.85)
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{loss:.3f}",
                ha="center", va="bottom", fontsize=9, zorder=6)
        bar.set_label(f"{label}: {loss:.3f}")

    ax.set_ylim(min(losses) - 0.02, max(losses) + 0.02)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    setup_axes(ax, title=title, xlabel="", ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        output_path = "all_bar_comparison.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# Copy scaling plot
# ---------------------------------------------------------------------------

# Maps copy count -> data-stream code for each ordering strategy
SHUFFLED_COPY_STREAMS: dict[int, str] = {2: "w2s", 4: "s4", 8: "s8", 16: "s16", 32: "s32"}
SORTED_COPY_STREAMS: dict[int, str] = {2: "w2", 4: "b4", 8: "b8", 16: "b16", 32: "b32"}


def _best_loss_for_stream(
    augmented: list[RunConfig],
    data_stream: str,
    name_to_summary: dict[str, dict],
    model_size: str,
    *,
    require_cda: bool = False,
) -> float | None:
    """Return the lowest eval loss across all non-seed runs for a given stream."""
    best: float | None = None
    best_name: str | None = None
    for c in augmented:
        if c.data_stream != data_stream or c.model_size != model_size or c.is_seed_run:
            continue
        if require_cda and not c.cda:
            continue
        loss = get_loss(name_to_summary.get(c.name, {}))
        if loss is not None and (best is None or loss < best):
            best = loss
            best_name = c.name
    return best, best_name


def _best_baseline_loss(
    baselines: list[RunConfig],
    name_to_summary: dict[str, dict],
    model_size: str,
    *,
    require_cda: bool = False,
) -> float | None:
    """Return the lowest eval loss among baseline (no data-stream) runs for *model_size*."""
    best: float | None = None
    for c in baselines:
        if c.is_seed_run or not c.is_baseline or c.model_size != model_size:
            continue
        if require_cda and not c.cda:
            continue
        loss = get_loss(name_to_summary.get(c.name, {}))
        if loss is not None and (best is None or loss < best):
            best = loss
    return best


def plot_copy_scaling(
    augmented: list[RunConfig],
    name_to_summary: dict[str, dict],
    *,
    include_sorted: bool = True,
    include_baseline_info: bool = False,
    baselines: list[RunConfig] | None = None,
    model_size: str = "300m4k",
    output_path: str | None = None,
) -> None:
    """Plot best loss vs number of data copies for shuffled (and optionally sorted) streams.

    If *include_baseline_info* is True, a point at x=0 shows the best CDA baseline
    for *model_size*, and a dotted horizontal line marks the CDA regularized asymptote.
    """
    copy_counts = sorted(SHUFFLED_COPY_STREAMS.keys())

    series: list[tuple[str, dict[int, str]]] = [
        ("Shuffled", SHUFFLED_COPY_STREAMS),
    ]
    if include_sorted:
        series.append(("Sorted", SORTED_COPY_STREAMS))

    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    bl_loss: float | None = None
    if include_baseline_info:
        if baselines is not None:
            bl_loss = _best_baseline_loss(baselines, name_to_summary, model_size, require_cda=True)
            if bl_loss is not None:
                ax.scatter(1, bl_loss, color=REGULARIZED_COLOR, s=50, zorder=5, marker="o",
                           label=f"Regularized {PRETTY_NAME_DICT.get(model_size, model_size)}: {bl_loss:.2f}")
        ax.axhline(REGULARIZED_ASYMPTOTE, color=REGULARIZED_COLOR, linestyle="--", alpha=0.4, zorder=2,
                   label=f"Regularized asymptote: {REGULARIZED_ASYMPTOTE:.2f}")

    for label, stream_map in series:
        xs: list[int] = []
        ys: list[float] = []
        colors: list[str] = []
        for copies in copy_counts:
            ds = stream_map.get(copies)
            if ds is None:
                continue
            loss, best_name = _best_loss_for_stream(augmented, ds, name_to_summary, model_size, require_cda=True)
            if loss is not None:
                xs.append(1 + copies)
                ys.append(loss)
                colors.append(DATA_STREAM_COLOR_DICT.get(ds, BLACK))
                print(f"  Best name for {ds} with {copies} copies: {best_name}")

        if not xs:
            print(f"  No data for {label} copy scaling")
            continue

        for x, y, c in zip(xs, ys, colors):
            ax.scatter(x, y, color=c, s=50, zorder=5)
        ax.plot(xs, ys, "--", color=colors[0], alpha=0.4, zorder=3)
        ax.scatter([], [], color=colors[len(colors) // 2], s=50,
                   label=f"{label} WRAP: {min(ys):.2f}")

        if include_baseline_info and bl_loss is not None:
            ax.plot([1, xs[0]], [bl_loss, ys[0]], ":", color=colors[0], alpha=0.4, zorder=3)

        min_loss = min(ys)
        min_copies = xs[ys.index(min_loss)]
        ax.scatter(min_copies, min_loss, color=RED, s=70, marker="o", edgecolor=RED, linewidth=2, zorder=3)

    # x-coordinates are 1 + copies so that log scale handles the 0-rephrase point
    ax.set_xscale("log")
    tick_locs = ([1] if include_baseline_info else []) + [1 + c for c in copy_counts]

    # Bottom x-axis: synthetic tokens in Billions
    synth_labels: list[str] = []
    if include_baseline_info:
        synth_labels.append("0B")
    for copies in copy_counts:
        ds = SHUFFLED_COPY_STREAMS.get(copies)
        if ds and ds in SYNTH_STREAM_TOKENS:
            billions = SYNTH_STREAM_TOKENS[ds] / 1e9
            synth_labels.append(f"{billions:.1f}B")
        else:
            synth_labels.append("")
    ax.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax.xaxis.set_major_formatter(FixedFormatter(synth_labels))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Synthetic tokens")

    # Top x-axis: rephrases per pretraining doc
    ax2 = ax.twiny()
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    rephrase_labels = (["0"] if include_baseline_info else []) + [str(c) for c in copy_counts]
    ax2.xaxis.set_major_locator(FixedLocator(tick_locs))
    ax2.xaxis.set_major_formatter(FixedFormatter(rephrase_labels))
    ax2.xaxis.set_minor_locator(FixedLocator([]))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("Rephrases per pretraining doc")

    title = "Scaling Synthetic Tokens"
    setup_axes(ax, title=title, xlabel="Synthetic tokens", ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        suffix = "_shuffled_sorted" if include_sorted else "_shuffled"
        output_path = f"rephrase_scaling{suffix}.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# Ensemble scaling plot
# ---------------------------------------------------------------------------

# (model_size, cda, data_stream | None, epochs, wd)
EnsembleSpec = tuple[str, bool, str | None, int, float]


def _ensemble_spec_label(spec: EnsembleSpec) -> str:
    model_size, cda, data_stream, epochs, wd = spec
    pretty = PRETTY_NAME_DICT.get(model_size, model_size)
    parts = [pretty]
    """
    if cda:
        parts.append("CDA")
    parts.append(f"x{epochs}")
    parts.append(f"wd{wd}")
    """
    if data_stream:
        parts.append(DATA_STREAM_NAMES.get(data_stream, data_stream))
    return " ".join(parts)


def plot_ensemble_scaling(
    ensemble_evals: list[EnsembleEvalConfig],
    eval_summary: dict[str, dict],
    baselines: list[RunConfig],
    baseline_summary: dict[str, dict],
    *,
    ensemble_specs: list[EnsembleSpec],
    title: str = "Ensemble Scaling",
    output_path: str | None = None,
) -> None:
    """Plot total parameter count vs loss for ensembles of varying size.

    Shows a reference regularized scaling law from single-model baselines,
    then overlays ensemble scaling curves for each requested spec.  Each spec
    is a (model_size, cda, data_stream, epochs, wd) tuple identifying one
    specific group of ensemble evaluation runs.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=WIDE_RECTANGLE_FIGSIZE, dpi=300)

    # Reference: best single-model baseline per model size
    optimal_baselines: dict[str, float] = {}
    for cfg in baselines:
        if cfg.is_seed_run or cfg.model_size not in PARAM_STR_TO_COUNT:
            continue
        loss = get_loss(baseline_summary.get(cfg.name, {}))
        if loss is None:
            continue
        ms = cfg.model_size
        if ms not in optimal_baselines or loss < optimal_baselines[ms]:
            optimal_baselines[ms] = loss

    if optimal_baselines:
        ref_sizes = sorted(optimal_baselines, key=lambda s: PARAM_STR_TO_COUNT[s])
        ref_x = np.array([PARAM_STR_TO_COUNT[ms] for ms in ref_sizes])
        ref_y = np.array([optimal_baselines[ms] for ms in ref_sizes])
        ax.scatter(ref_x, ref_y, color=REGULARIZED_COLOR, s=50, zorder=5)
        try:
            power_law = PowerScalingLaw(var_name="N")
            y_range = max(float(np.max(ref_y) - np.min(ref_y)), 1e-6)
            a0 = y_range * (np.max(ref_x) ** 0.5)
            power_law.fit(
                ref_x, ref_y,
                p0=[a0, 0.5, float(np.min(ref_y))],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            )
            x_fit = np.logspace(np.log10(np.min(ref_x)), np.log10(np.max(ref_x)), 200)
            y_fit = power_law.evaluate(x_fit)
            ax.plot(x_fit, y_fit, "--", color=REGULARIZED_COLOR, alpha=0.8, zorder=4,
                    label=f"Regularized (fit: {power_law})")
        except RuntimeError:
            ax.plot(ref_x, ref_y, "--", color=REGULARIZED_COLOR, alpha=0.8, zorder=4,
                    label="Regularized")

    # Plot each ensemble spec
    for spec in ensemble_specs:
        model_size, cda, data_stream, epochs, wd = spec

        if data_stream is None:
            color = PARAM_STR_COLOR_DICT.get(model_size, BLACK)
        else:
            color = DATA_STREAM_COLOR_DICT.get(data_stream, BLACK)

        series_label = _ensemble_spec_label(spec)

        matching = [
            e for e in ensemble_evals
            if e.model_size == model_size
            and e.cda == cda
            and e.data_stream == data_stream
            and e.epochs == epochs
            and e.wd == wd
        ]
        if not matching:
            print(f"  No ensemble evals for spec {spec}")
            continue

        # For each member_count, pick the eval with the lowest loss
        by_count: dict[int, list[tuple[EnsembleEvalConfig, float]]] = defaultdict(list)
        for e in matching:
            loss = get_loss(eval_summary.get(e.name, {}))
            if loss is not None:
                by_count[e.member_count].append((e, loss))

        best_per_count: list[tuple[int, float]] = []
        for count in sorted(by_count):
            _, best_loss = min(by_count[count], key=lambda t: t[1])
            best_per_count.append((count, best_loss))

        if not best_per_count:
            print(f"  No valid losses for ensemble spec {spec}")
            continue

        total_params = [PARAM_STR_TO_COUNT[model_size] * count for count, _ in best_per_count]
        losses = [loss for _, loss in best_per_count]

        ax.scatter(total_params, losses, color=color, s=50, zorder=5)

        try:
            x_arr = np.array(total_params)
            y_arr = np.array(losses)
            power_law = PowerScalingLaw(var_name="N")
            y_range = max(float(np.max(y_arr) - np.min(y_arr)), 1e-6)
            a0 = y_range * (np.max(x_arr) ** 0.5)
            power_law.fit(
                x_arr, y_arr,
                p0=[a0, 0.5, float(np.min(y_arr))],
                bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
            )
            x_fit = np.logspace(np.log10(np.min(x_arr)), np.log10(np.max(x_arr) * 3), 200)
            y_fit = power_law.evaluate(x_fit)
            ax.plot(x_fit, y_fit, "--", color=color, alpha=0.8, zorder=4,
                    label=f"{series_label} (fit: {power_law})")
        except RuntimeError:
            ax.plot(total_params, losses, "--", color=color, alpha=0.8, zorder=4,
                    label=series_label)

    ax.set_xscale("log")
    base_params = PARAM_STR_TO_COUNT["300m4k"]
    tick_vals = [i * base_params for i in range(1, 6)]
    tick_labels = []
    for i in range(1, 6):
        total_m = i * 300
        if total_m >= 1000:
            tick_labels.append(f"{total_m / 1000:.1f}B")
        else:
            tick_labels.append(f"{total_m}M")
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(axis="x", which="minor", bottom=False)
    setup_axes(ax, title=title, xlabel="Total parameters", ylabel="IID Loss")

    plt.tight_layout()
    if output_path is None:
        specs_str = "_".join(
            f"{ms}{'_cda' if cda else ''}_x{ep}_wd{wd}{'_' + ds if ds else ''}"
            for ms, cda, ds, ep, wd in ensemble_specs
        )
        output_path = f"ensemble_scaling_{specs_str}.png"
    _ensure_parent_dir(PLOT_DIR + output_path)
    fig.savefig(PLOT_DIR + output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {PLOT_DIR + output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic data efficiency plots.")
    parser.add_argument("--cache_path", default=CACHE_PATH)
    parser.add_argument("--build_cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.build_cache:
        payload = build_cache(args.cache_path)
    else:
        payload = _load_cache(args.cache_path)
        if payload is None:
            raise FileNotFoundError(
                f"No cache found at {args.cache_path}. Run with --build_cache first."
            )

    configs, unmatched = parse_train_runs(payload)
    baselines, augmented, seed_runs = print_parsed_summary(configs, unmatched)

    name_to_summary = build_name_to_summary(payload, TRAIN_PROJECT)
    plot_regularized_scaling(baselines, name_to_summary, output_path="regularized_scaling_cda_only.png", cda_only=True)
    plot_regularized_scaling(baselines, name_to_summary, output_path="regularized_scaling.png")
    plot_all_loss_vs_epochs(augmented, name_to_summary, include_baselines=True)
    plot_copy_scaling(augmented, name_to_summary, include_sorted=True, include_baseline_info=True, baselines=baselines)
    plot_copy_scaling(augmented, name_to_summary, include_sorted=False, include_baseline_info=True, baselines=baselines)
    compare_loss_vs_epochs(
        augmented, 
        name_to_summary, 
        data_streams=["w2s", "s32"], 
        stream_epochs={"w2s": [2, 4, 8, 16], "s32": [8, 16, 32]},
    )
    compare_loss_vs_epochs(
        augmented, 
        name_to_summary, 
        data_streams=["w2s", "s32", "b32"], 
        stream_epochs={"w2s": [2, 4, 8, 16], "s32": [8, 16, 32], "b32": [16, 32, 64]},
    )
    compare_loss_vs_epochs(
        augmented, 
        name_to_summary, 
        data_streams=["w2s", "s32", "b32"], 
        use_real_tokens=True,
        stream_token_range={"w2s": [(1.5e9, 4.97e9)], "s32": [(1.8e9, 2e9), (3.7e9, 4e9), (7e9, 8e9)], "b32": [(3.7e9, 4e9), (7e9, 18e9)]},
    )

    plot_augmented_bar_comparison(
        augmented, name_to_summary,
        data_streams=["w2s", "w2", "w2f", ],
        title="Synthetic Data Ordering",
    )

    plot_augmented_bar_comparison(
        augmented, name_to_summary,
        data_streams=["s8", "b8", "n8s", "n8", ],
        title="Synthetic Data Ordering",
    )

    plot_augmented_bar_comparison(
        augmented, name_to_summary,
        data_streams=["s16", "b16", "l16"],
        use_seed_runs=True,
        seed_runs=seed_runs,
        title="Synthetic Data Methods",
    )

    plot_all_bar_comparison(
        baselines, augmented, name_to_summary,
        queries=[
            (True, None),
            (False, None),
            (True, "s16"),
            (False, "s16"),
            (True, "b16"),
        ],
        title="Cross-Document Attention",
    )

    # Ensemble scaling
    ensemble_evals = parse_ensemble_eval_runs(payload)
    eval_summary = build_name_to_summary(payload, EVAL_PROJECT)

    # (model_size, cda, data_stream, epochs, wd)
    base_cda_specs: list[EnsembleSpec] = [
        # ("150m4k",  True,  None, 16, 0.80),
        # ("150m4k",  True,  None, 32, 0.80),
        # ("150m4k",  True,  None, 32, 0.40),
        # ("300m4k",  True,  None, 16, 1.60),
        # ("300m4k",  True,  None, 32, 1.60),
        ("300m4k",  True,  None, 32, 0.80),
        # ("600m4k",  True,  None,  8, 3.20),
        # ("600m4k",  True,  None, 16, 3.20),
        # ("600m4k",  True,  None, 16, 1.60),
        # ("1_5b4k",  True,  None,  8, 3.20),
        # ("1_5b4k",  True,  None, 16, 3.20),
        # ("1_5b4k",  True,  None, 16, 1.60),
    ]
    base_no_cda_specs: list[EnsembleSpec] = [
        # ("150m4k",  False, None, 16, 0.80),
        # ("150m4k",  False, None, 32, 0.80),
        # ("150m4k",  False, None, 32, 0.40),
        # ("300m4k",  False, None, 16, 1.60),
        # ("300m4k",  False, None, 32, 1.60),
        ("300m4k",  False, None, 32, 0.80),
        # ("600m4k",  False, None,  8, 3.20),
        # ("600m4k",  False, None, 16, 3.20),
        # ("600m4k",  False, None, 16, 1.60),
        # ("1_5b4k",  False, None,  8, 3.20),
        # ("1_5b4k",  False, None, 16, 3.20),
        # ("1_5b4k",  False, None, 16, 1.60),
    ]
    sdn_specs: list[EnsembleSpec] = [
        ("300m4k",  True,  "sdn", 16, 0.40),
    ]
    wrap_specs: list[EnsembleSpec] = [
        ("300m4k",  True,  "sdn", 16, 0.40),
        ("300m4k",  True,  "s32", 16, 0.40),
        # ("300m4k",  True,  "b32", 32, 0.40),
    ]
    augmented_specs: list[EnsembleSpec] = [
        ("300m4k",  True,  "sdn", 16, 0.40),
        # ("300m4k",  True,  "b16", 16, 0.40),
        # ("300m4k",  True,  "s16", 16, 0.40),
        ("300m4k",  True,  "l16", 16, 0.40),
        ("300m4k",  True,  "b32", 32, 0.40),
        ("300m4k",  True,  "s32", 16, 0.40),
    ]
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_cda_specs,
        title="Ensemble Scaling (With CDA)",
        output_path="ensemble_scaling_cda_baselines.png",
    )
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_no_cda_specs,
        title="Ensemble Scaling (Without CDA)",
        output_path="ensemble_scaling_no_cda_baselines.png",
    )
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_cda_specs + augmented_specs,
        title="Synthetic Data Ensembles",
        output_path="ensemble_scaling_cda_augmented_300m.png",
    )
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_cda_specs + sdn_specs,
        title="Self-Distill Ensembles",
        output_path="ensemble_scaling_cda_sdn.png",
    )
    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_cda_specs + wrap_specs,
        title="WRAP vs. Self-Distill Ensembles",
        output_path="ensemble_scaling_cda_wrap.png",
    )

    plot_ensemble_scaling(
        ensemble_evals, eval_summary, [], name_to_summary,
        ensemble_specs=base_cda_specs + wrap_specs + [("300m4k",  True,  "b32", 32, 0.40),],
        title="WRAP vs. Self-Distill Ensembles",
        output_path="ensemble_scaling_cda_wrap_b32.png",
    )


if __name__ == "__main__":
    main()
