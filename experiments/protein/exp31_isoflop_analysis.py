# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze exp31 protein iso-FLOP sweep results from W&B.

Companion to ``exp31_isoflop_sweep.py``. Pulls the final
``eval/protein-docs-cd-val/loss`` plus throughput / parameter-count summary
fields for every *finished* run in the sweep, validates that the achieved
training FLOPs (``throughput/total_gflops``) match the planned budget tag
within ``FLOP_VALIDATION_TOL``, and renders iso-FLOP curves of cd-val loss
vs parameter count (one trace per FLOP budget).

The W&B fetch result is reduced to a single ``snapshot.csv`` under
``experiments/protein/exp31_isoflop_results/``; pass ``--refresh`` to re-pull.

Per-step training-loss histories for the regime-comparison plot are cached
separately in ``history.parquet`` (zstd-compressed) and fetched incrementally
on demand.

Usage::

    WANDB_API_KEY=... uv run --with matplotlib \\
        python -m experiments.protein.exp31_isoflop_analysis [--refresh]
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


# --- Sweep identity ----------------------------------------------------------

SWEEP_ID: str = "exp31_isoflop"
VERSION: str = "v2"
WANDB_ENTITY: str = "eric-czech"
WANDB_PROJECT: str = "marin"
WANDB_GROUP: str = "exp31-isoflop"
# All trial-identifying info is in tags; ``isoflop`` selects the sweep and
# ``version=<VERSION>`` discriminates against older revisions of the recipe.
WANDB_TAGS: tuple[str, ...] = ("isoflop", f"version={VERSION}")

TITLE_PREFIX: str = "MarinFold Experiment #31"

# Four FLOP budgets the sweep targets; mirrors ``exp31_isoflop_sweep.BUDGETS``.
BUDGETS: tuple[float, ...] = (3e17, 1e18, 3e18, 1e19)

# Tableau-10 categorical colors, ordered to match BUDGETS.
BUDGET_COLORS: dict[float, str] = {
    3e17: "#4C78A8",
    1e18: "#54A24B",
    3e18: "#E45756",
    1e19: "#9D7AB8",
}

# Runs that did not surpass the critical transition in pairwise-distance-token
# learning (separately characterized in a different session). They are kept in
# the snapshot for traceability but excluded from the iso-FLOP fit line and
# empirical-minimum marker, and drawn at lower opacity on the plot.
REGIME_2_RUNS: tuple[str, ...] = (
    "prot-exp31-iso-F3e17-P28.1M-T644.8M-v2",
    "prot-exp31-iso-F1e18-P4.6M-T10.038B-v2",
    "prot-exp31-iso-F1e18-P28.1M-T2.149B-v2",
    "prot-exp31-iso-F1e18-P190.4M-T445.4M-v2",
)
REGIME_2_ALPHA: float = 0.30

# Regime-1 partners shown alongside REGIME_2_RUNS in the comparison plot.
# Ordering is parallel to ``REGIME_2_RUNS`` so each R1 entry shares a color
# with the R2 entry at the same index.
REGIME_1_RUNS: tuple[str, ...] = (
    "prot-exp31-iso-F3e17-P49.5M-T410.0M-v2",
    "prot-exp31-iso-F1e18-P11.6M-T4.631B-v2",
    "prot-exp31-iso-F1e18-P49.5M-T1.367B-v2",
    "prot-exp31-iso-F1e18-P120.7M-T661.9M-v2",
)

# W&B appends a short hex suffix to display names (e.g. ``...-v2-bfe8f3``);
# strip it so regime/run lookups can use the stable base run name.
RUN_HASH_SUFFIX_RE: re.Pattern[str] = re.compile(r"-[0-9a-f]{4,}$")


def base_run_name(name: str) -> str:
    return RUN_HASH_SUFFIX_RE.sub("", name)


# --- Filesystem layout -------------------------------------------------------

RESULTS_DIR: Path = Path(__file__).resolve().parent / "exp31_isoflop_results"
SNAPSHOT_PATH: Path = RESULTS_DIR / "snapshot.csv"
META_PATH: Path = RESULTS_DIR / "meta.json"
HISTORY_PATH: Path = RESULTS_DIR / "history.parquet"
PLOTS_DIR: Path = RESULTS_DIR / "plots"


# --- Metric / W&B summary keys ----------------------------------------------

EVAL_METRIC: str = "eval/protein-docs-cd-val/loss"

# Levanter throughput-callback summary fields. ``total_gflops`` reports
# training (fwd+bwd) FLOPs in gigaflops -- multiply by 1e9 to compare to the
# planned budget (which is in raw FLOPs).
THROUGHPUT_TOKENS_KEY: str = "throughput/total_tokens"
THROUGHPUT_GFLOPS_KEY: str = "throughput/total_gflops"
PARAMETER_COUNT_KEY: str = "parameter_count"

# Achieved FLOPs may differ slightly from the planned budget due to integer
# step-count rounding (solver tolerance 1%) and the throughput accumulator's
# own rounding; 2% covers both with margin.
FLOP_VALIDATION_TOL: float = 0.02

# Per-step training loss series (used for the regime-comparison plot).
TRAIN_LOSS_KEY: str = "train/loss"
# Smoothing radius in log10(step) units. The window at step s spans
# [s * 10^-r, s * 10^+r], so it's visually uniform on a log-x plot and never
# reaches into the high-loss warmup region from the post-warmup steps. A
# linear-step window can't do both: wide enough to look smooth late in
# training, but small enough early not to pull in warmup loss.
SMOOTHING_LOG_RADIUS: float = 0.04
# Drop the high-loss warmup portion so the post-warmup pattern dominates the
# log-log view; the regime-1 sudden drop happens well after this cutoff.
COMPARISON_MIN_STEP: int = 500


# --- Tag parsing -------------------------------------------------------------


def parse_tag_value(tags: list[str], key: str) -> str | None:
    """Return the value of the first ``key=value`` tag, or None."""
    prefix = f"{key}="
    for t in tags:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return None


def parse_tag_float(tags: list[str], key: str) -> float:
    raw = parse_tag_value(tags, key)
    if raw is None:
        raise KeyError(f"Tag {key!r} missing from {tags!r}")
    return float(raw)


def parse_tag_int(tags: list[str], key: str) -> int:
    raw = parse_tag_value(tags, key)
    if raw is None:
        raise KeyError(f"Tag {key!r} missing from {tags!r}")
    return int(raw)


def fmt_budget_label(budget: float) -> str:
    """Compact label like ``3e17``, ``1e18`` for legend/title use."""
    mantissa, exponent = f"{budget:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


# --- W&B fetch + snapshot ---------------------------------------------------


@dataclass(frozen=True)
class SweepMeta:
    """Lightweight per-sweep meta: provenance + validation outcome."""

    sweep_id: str
    version: str
    wandb_entity: str
    wandb_project: str
    wandb_group: str
    wandb_tags: tuple[str, ...]
    num_finished_runs: int
    flop_validation_tol: float
    max_flop_relative_error: float


def list_runs() -> list[object]:
    """Return all runs in the sweep group narrowed by every WANDB_TAGS entry."""
    api = wandb.Api()
    filters = {"group": WANDB_GROUP, "tags": {"$all": list(WANDB_TAGS)}}
    runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters))
    logger.info("W&B returned %d run(s) in group=%s tags=%s", len(runs), WANDB_GROUP, WANDB_TAGS)
    return runs


def fetch_last_value(run, key: str) -> tuple[float, int] | None:
    """Largest history step with a non-null value for ``key``, or None."""
    rows = list(run.scan_history(keys=["_step", key], page_size=10_000))
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=[key])
    if df.empty:
        return None
    df["_step"] = df["_step"].astype(int)
    row = df.sort_values("_step").iloc[-1]
    return float(row[key]), int(row["_step"])


def build_snapshot() -> tuple[pd.DataFrame, SweepMeta]:
    """Pull every finished run, build a one-row-per-run snapshot + meta."""
    raw = list_runs()
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for r in raw:
        if r.state != "finished":
            skipped.append((r.display_name, f"state={r.state}"))
            continue

        budget = parse_tag_float(r.tags, "budget_exact")
        params_label = parse_tag_value(r.tags, "params")
        if params_label is None:
            raise KeyError(f"Tag 'params' missing from {r.display_name!r}")
        params_planned = parse_tag_int(r.tags, "params_exact")
        tokens_planned = parse_tag_int(r.tags, "tokens_exact")
        batch = parse_tag_int(r.tags, "batch")
        lr = parse_tag_float(r.tags, "lr_exact")
        beta2 = parse_tag_float(r.tags, "beta2")

        summary = r.summary
        total_gflops = summary.get(THROUGHPUT_GFLOPS_KEY)
        total_tokens = summary.get(THROUGHPUT_TOKENS_KEY)
        parameter_count = summary.get(PARAMETER_COUNT_KEY)
        if any(v is None for v in (total_gflops, total_tokens, parameter_count)):
            skipped.append((r.display_name, "missing throughput/parameter_count summary"))
            continue

        eval_hit = fetch_last_value(r, EVAL_METRIC)
        if eval_hit is None:
            skipped.append((r.display_name, f"no {EVAL_METRIC} history"))
            continue
        loss, eval_step = eval_hit

        rows.append(
            {
                "run_name": r.display_name,
                "run_id": r.id,
                "run_state": r.state,
                "budget": budget,
                "budget_label": fmt_budget_label(budget),
                "params_label": params_label,
                "params_planned": params_planned,
                "tokens_planned": tokens_planned,
                "batch": batch,
                "lr": lr,
                "beta2": beta2,
                "parameter_count": int(parameter_count),
                "total_tokens": int(total_tokens),
                "total_gflops": float(total_gflops),
                "achieved_flops": float(total_gflops) * 1e9,
                "cd_val_loss": loss,
                "eval_step": eval_step,
            }
        )

    for name, why in skipped:
        logger.info("Skipped %s: %s", name, why)
    if not rows:
        raise RuntimeError(f"No finished runs in sweep {SWEEP_ID}")

    df = pd.DataFrame(rows).sort_values(["budget", "parameter_count"]).reset_index(drop=True)
    max_err = validate_flops(df, FLOP_VALIDATION_TOL)
    meta = SweepMeta(
        sweep_id=SWEEP_ID,
        version=VERSION,
        wandb_entity=WANDB_ENTITY,
        wandb_project=WANDB_PROJECT,
        wandb_group=WANDB_GROUP,
        wandb_tags=WANDB_TAGS,
        num_finished_runs=len(df),
        flop_validation_tol=FLOP_VALIDATION_TOL,
        max_flop_relative_error=max_err,
    )
    return df, meta


def validate_flops(df: pd.DataFrame, tol: float) -> float:
    """Assert ``|achieved - budget| / budget <= tol`` for every run; return max error."""
    rel_err = (df["achieved_flops"] - df["budget"]).abs() / df["budget"]
    df = df.assign(flop_rel_err=rel_err)
    bad = df[df["flop_rel_err"] > tol]
    if not bad.empty:
        lines = [
            f"  {row.run_name}: budget={row.budget:.3e} achieved={row.achieved_flops:.3e} "
            f"rel_err={row.flop_rel_err:.4f}"
            for row in bad.itertuples()
        ]
        raise RuntimeError(f"{len(bad)}/{len(df)} runs exceed FLOP tolerance {tol:.2%}:\n" + "\n".join(lines))
    return float(rel_err.max())


def save_meta(meta: SweepMeta, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(meta), indent=2) + "\n")


def load_meta(path: Path) -> SweepMeta:
    raw = json.loads(path.read_text())
    raw["wandb_tags"] = tuple(raw["wandb_tags"])
    return SweepMeta(**raw)


def assign_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Annotate each row with ``regime`` (1 = fit, 2 = pre-transition).

    Computed from ``REGIME_2_RUNS`` at read time so toggling the exclusion list
    does not require re-running the W&B fetch.
    """
    base = df["run_name"].map(base_run_name)
    r2 = set(REGIME_2_RUNS)
    unknown = r2 - set(base)
    if unknown:
        raise ValueError(f"REGIME_2_RUNS references runs absent from snapshot: {sorted(unknown)}")
    return df.assign(regime=base.map(lambda n: 2 if n in r2 else 1).astype(int))


def load_or_build_snapshot(*, refresh: bool) -> tuple[pd.DataFrame, SweepMeta]:
    """Cached fetch: returns (snapshot_df, meta). ``--refresh`` skips the cache."""
    if not refresh and SNAPSHOT_PATH.exists() and META_PATH.exists():
        logger.info("Loading cached snapshot+meta from %s", SNAPSHOT_PATH.parent)
        df = pd.read_csv(SNAPSHOT_PATH)
        validate_flops(df, FLOP_VALIDATION_TOL)
        return assign_regime(df), load_meta(META_PATH)
    df, meta = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False)
    save_meta(meta, META_PATH)
    logger.info("Wrote %d rows to %s and meta to %s", len(df), SNAPSHOT_PATH, META_PATH)
    return assign_regime(df), meta


# --- Plot --------------------------------------------------------------------


def render_isoflop_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """One trace per FLOP budget: cd-val loss vs parameter count (log x).

    Regime-1 points are connected in order of increasing params so the U-shaped
    iso-FLOP curve is visually obvious; an open diamond marks the empirical
    minimum within each budget over regime-1 points only. Regime-2 points
    (runs that did not surpass the critical learning transition) are drawn
    semi-transparent with no connecting line.
    """
    if df.empty:
        logger.warning("Empty snapshot; nothing to plot")
        return

    budgets_present = sorted(df["budget"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for budget in budgets_present:
        sub = df[df["budget"] == budget].sort_values("parameter_count")
        r1 = sub[sub["regime"] == 1]
        r2 = sub[sub["regime"] == 2]
        color = BUDGET_COLORS.get(budget, "#999999")
        label = f"C = {fmt_budget_label(budget)} FLOPs ({len(r1)} pts"
        if len(r2):
            label += f", +{len(r2)} R2"
        label += ")"
        if not r1.empty:
            ax.plot(
                r1["parameter_count"],
                r1["cd_val_loss"],
                "-o",
                color=color,
                markersize=7,
                markeredgecolor="black",
                markeredgewidth=0.5,
                linewidth=1.2,
                label=label,
            )
            if len(r1) >= 2:
                i_min = r1["cd_val_loss"].idxmin()
                ax.scatter(
                    r1.loc[i_min, "parameter_count"],
                    r1.loc[i_min, "cd_val_loss"],
                    marker="D",
                    s=70,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.5,
                    zorder=5,
                )
        if not r2.empty:
            ax.scatter(
                r2["parameter_count"],
                r2["cd_val_loss"],
                marker="o",
                s=42,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=REGIME_2_ALPHA,
                zorder=3,
            )

    ax.set_xscale("log")
    # Replace matplotlib's auto-log ticks with one tick per distinct param count,
    # labelled with the abbreviated string from the run's ``params=`` tag.
    tick_df = df[["parameter_count", "params_label"]].drop_duplicates().sort_values("parameter_count")
    ax.set_xticks(tick_df["parameter_count"].tolist())
    ax.set_xticks([], minor=True)
    ax.set_xticklabels(tick_df["params_label"].tolist(), rotation=30, ha="right")
    ax.set_xlabel("Parameter count")
    ax.set_ylabel("cd-val loss  (distance-bin only, masked)")
    n_r1 = int((df["regime"] == 1).sum())
    n_r2 = int((df["regime"] == 2).sum())
    ax.set_title(
        f"{TITLE_PREFIX} - iso-FLOP curves on protein-docs-cd (Qwen3 + WSD, {n_r1} R1 + {n_r2} R2 runs)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    budget_legend = ax.legend(loc="upper right", fontsize=9, framealpha=0.9, title="FLOP budget")
    ax.add_artist(budget_legend)

    regime_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            linestyle="-",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            label="Regime 1 (fit)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            alpha=REGIME_2_ALPHA,
            label="Regime 2 (pre-transition)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            label="Empirical min (R1)",
        ),
    ]
    ax.legend(handles=regime_handles, loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "isoflop_curves.pdf"
    png_path = out_dir / "isoflop_curves.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s and %s", pdf_path, png_path)


# --- Regime comparison: training loss trajectories --------------------------


def fetch_train_history(run, key: str) -> pd.DataFrame:
    """Return per-step ``(_step, train_loss)`` history for a single run, tagged with ``run_name``."""
    rows = list(run.scan_history(keys=["_step", key], page_size=10_000))
    if not rows:
        raise RuntimeError(f"Empty history for {run.display_name!r}")
    df = pd.DataFrame(rows).dropna(subset=[key])
    if df.empty:
        raise RuntimeError(f"No non-null {key!r} samples for {run.display_name!r}")
    df["_step"] = df["_step"].astype(int)
    return df[["_step", key]].rename(columns={key: "train_loss"}).assign(run_name=run.display_name)


def load_or_build_history(needed_names: set[str], *, refresh: bool) -> pd.DataFrame:
    """Cached fetch of per-step training loss for the runs in ``needed_names``.

    Missing runs are fetched and merged into the existing parquet cache; existing
    entries are preserved unless ``refresh`` is True.
    """
    cached: pd.DataFrame | None = None
    if not refresh and HISTORY_PATH.exists():
        cached = pd.read_parquet(HISTORY_PATH)
    present = set(cached["run_name"]) if cached is not None else set()
    missing = needed_names - present
    if not missing:
        assert cached is not None
        return cached[cached["run_name"].isin(needed_names)].reset_index(drop=True)

    raw = list_runs()
    by_name = {r.display_name: r for r in raw}
    pieces: list[pd.DataFrame] = [cached] if cached is not None else []
    for name in sorted(missing):
        if name not in by_name:
            raise KeyError(f"Run {name!r} not present in W&B sweep listing")
        logger.info("Fetching train-loss history for %s", name)
        pieces.append(fetch_train_history(by_name[name], TRAIN_LOSS_KEY))
    hist = pd.concat(pieces, ignore_index=True)
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(HISTORY_PATH, index=False, compression="zstd")
    logger.info("Wrote %d history rows for %d run(s) to %s", len(hist), hist["run_name"].nunique(), HISTORY_PATH)
    return hist[hist["run_name"].isin(needed_names)].reset_index(drop=True)


def smooth_loss(steps: np.ndarray, loss: np.ndarray, log_radius: float) -> np.ndarray:
    """Centered moving average in ``log10(step)`` space.

    For each point ``i``, averages every sample whose ``log10(step)`` falls
    within ``log10(step[i]) +/- log_radius``. Equivalent to a multiplicative
    step window ``[s * 10^-r, s * 10^+r]`` — visually uniform on a log-x plot.
    Steps must be sorted ascending (they are, after the ``sort_values`` upstream).
    """
    log_steps = np.log10(steps.astype(float))
    cs = np.concatenate(([0.0], np.cumsum(loss.astype(float))))
    lo = np.searchsorted(log_steps, log_steps - log_radius, side="left")
    hi = np.searchsorted(log_steps, log_steps + log_radius, side="right")
    return (cs[hi] - cs[lo]) / (hi - lo)


def render_regime_comparison(
    df: pd.DataFrame,
    history: pd.DataFrame,
    r1_bases: tuple[str, ...],
    r2_bases: tuple[str, ...],
    out_dir: Path,
) -> None:
    """1x2 log-log panel: R1 (left) vs R2 (right) smoothed training loss.

    The i-th R1 trace and the i-th R2 trace share a color, so the implicit
    pairing across panels stays visually obvious.
    """
    if len(r1_bases) != len(r2_bases):
        raise ValueError(f"R1 ({len(r1_bases)}) and R2 ({len(r2_bases)}) lists must be the same length")
    base_to_row = {base_run_name(name): row for name, row in zip(df["run_name"], df.to_dict("records"), strict=True)}
    palette = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#9D7AB8", "#B279A2", "#9C755F", "#BAB0AC"]

    fig, (ax_r1, ax_r2) = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)
    for idx, (r1_base, r2_base) in enumerate(zip(r1_bases, r2_bases, strict=True)):
        color = palette[idx % len(palette)]
        for ax, base in ((ax_r1, r1_base), (ax_r2, r2_base)):
            if base not in base_to_row:
                raise KeyError(f"Run {base!r} not present in snapshot")
            row = base_to_row[base]
            run_name = row["run_name"]
            full = history[history["run_name"] == run_name].sort_values("_step")
            if full.empty:
                raise RuntimeError(f"No history rows for run {run_name!r}")
            # Smooth on the full series first so the centered window near the
            # display-clip boundary still has both sides of context. Only after
            # smoothing do we mask to the display range.
            full_steps = full["_step"].to_numpy()
            full_loss = full["train_loss"].to_numpy()
            # Smooth on the full series first (so the centered log-step window
            # at the clip boundary has both sides of context), then mask.
            full_smoothed = smooth_loss(full_steps, full_loss, SMOOTHING_LOG_RADIUS)
            mask = full_steps >= COMPARISON_MIN_STEP
            steps = full_steps[mask]
            ax.plot(steps, full_loss[mask], color=color, linewidth=0.4, alpha=0.18)
            ax.plot(
                steps,
                full_smoothed[mask],
                color=color,
                linewidth=1.4,
                label=f"{row['budget_label']}, {row['params_label']}",
            )

    for ax, title in (
        (ax_r1, "Regime 1 (matched, fit)"),
        (ax_r2, "Regime 2 (pre-transition)"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(left=COMPARISON_MIN_STEP)
        ax.set_ylim(1.3, 2.2)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.9, title="budget, params")

    ax_r1.set_ylabel("Training loss  (centered MA)")
    fig.supxlabel("Training step")
    fig.suptitle(
        f"{TITLE_PREFIX} - Training-loss trajectories: regime-1 vs matched regime-2 runs",
        fontsize=11,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "regime_comparison.pdf"
    png_path = out_dir / "regime_comparison.png"
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

    df, meta = load_or_build_snapshot(refresh=args.refresh)
    logger.info("meta = %s", meta)
    wide_cols = [
        "budget_label",
        "parameter_count",
        "total_tokens",
        "total_gflops",
        "achieved_flops",
        "cd_val_loss",
        "batch",
        "lr",
        "regime",
    ]
    logger.info(
        "\n%s",
        df[wide_cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}" if abs(v) < 1e6 else f"{v:.3e}",
        ),
    )
    render_isoflop_curves(df, PLOTS_DIR)

    base_to_run = dict(zip(df["run_name"].map(base_run_name), df["run_name"], strict=True))
    needed_names = {base_to_run[b] for b in REGIME_1_RUNS + REGIME_2_RUNS}
    history = load_or_build_history(needed_names, refresh=args.refresh)
    render_regime_comparison(df, history, REGIME_1_RUNS, REGIME_2_RUNS, PLOTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
