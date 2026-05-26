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

Usage::

    WANDB_API_KEY=... uv run --with matplotlib \\
        python -m experiments.protein.exp31_isoflop_analysis [--refresh]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

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


# --- Filesystem layout -------------------------------------------------------

RESULTS_DIR: Path = Path(__file__).resolve().parent / "exp31_isoflop_results"
SNAPSHOT_PATH: Path = RESULTS_DIR / "snapshot.csv"
META_PATH: Path = RESULTS_DIR / "meta.json"
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


def load_or_build_snapshot(*, refresh: bool) -> tuple[pd.DataFrame, SweepMeta]:
    """Cached fetch: returns (snapshot_df, meta). ``--refresh`` skips the cache."""
    if not refresh and SNAPSHOT_PATH.exists() and META_PATH.exists():
        logger.info("Loading cached snapshot+meta from %s", SNAPSHOT_PATH.parent)
        df = pd.read_csv(SNAPSHOT_PATH)
        validate_flops(df, FLOP_VALIDATION_TOL)
        return df, load_meta(META_PATH)
    df, meta = build_snapshot()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SNAPSHOT_PATH, index=False)
    save_meta(meta, META_PATH)
    logger.info("Wrote %d rows to %s and meta to %s", len(df), SNAPSHOT_PATH, META_PATH)
    return df, meta


# --- Plot --------------------------------------------------------------------


def render_isoflop_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """One trace per FLOP budget: cd-val loss vs parameter count (log x).

    Points are connected in order of increasing params so the U-shaped
    iso-FLOP curve is visually obvious. An open diamond marks the empirical
    minimum within each budget.
    """
    if df.empty:
        logger.warning("Empty snapshot; nothing to plot")
        return

    budgets_present = sorted(df["budget"].unique())
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for budget in budgets_present:
        sub = df[df["budget"] == budget].sort_values("parameter_count")
        color = BUDGET_COLORS.get(budget, "#999999")
        ax.plot(
            sub["parameter_count"],
            sub["cd_val_loss"],
            "-o",
            color=color,
            markersize=7,
            markeredgecolor="black",
            markeredgewidth=0.5,
            linewidth=1.2,
            label=f"C = {fmt_budget_label(budget)} FLOPs ({len(sub)} pts)",
        )
        # Mark the empirical minimum within each budget.
        if len(sub) >= 2:
            i_min = sub["cd_val_loss"].idxmin()
            ax.scatter(
                sub.loc[i_min, "parameter_count"],
                sub.loc[i_min, "cd_val_loss"],
                marker="D",
                s=70,
                facecolor="white",
                edgecolor=color,
                linewidth=1.5,
                zorder=5,
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
    n_runs = len(df)
    ax.set_title(
        f"{TITLE_PREFIX} - iso-FLOP curves on protein-docs-cd (Qwen3 + WSD, {n_runs} finished runs)",
        fontsize=11,
    )
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9, framealpha=0.9, title="Open diamond = empirical min")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "isoflop_curves.pdf"
    png_path = out_dir / "isoflop_curves.png"
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
    ]
    logger.info(
        "\n%s",
        df[wide_cols].to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}" if abs(v) < 1e6 else f"{v:.3e}",
        ),
    )
    render_isoflop_curves(df, PLOTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
