# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze an exp49 AA-mixture sweep from W&B.

Companion to ``exp49_aamix_sweep.py``. The sweep asks: does adding AA-sequence
pretraining data (``m2`` = 50/50 structure-docs + AA-seq, additive so structure
exposure stays at 500M) help the held-out structure-token task versus
structure-docs alone (``m1`` = 100% structure docs, 500M)? Two mixtures x three
data-ordering seeds (1729/1730/1731), final held-out ``protein-docs-cd-val``
loss (each run's own last logged value via ``run.summary``).

Selects one sweep version via ``--version`` (default ``v2``): only runs whose
name carries that ``-<version>-`` suffix and whose state is ``finished`` are
read, deduped to one run per (mixture, seed). Outputs are version-suffixed so
versions never overwrite each other. One figure (300-dpi PNG + PDF): cd-val loss
per seed for each mixture, with the across-seed mean +/- 1 std.

Usage::

    set -a; source ~/marin.env; set +a   # WANDB_API_KEY / ENTITY / PROJECT
    uv run --with matplotlib python -m experiments.protein.exp49_aamix_analysis [--version v2] [--refresh]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb

logger = logging.getLogger(__name__)

# Sweep identity (see exp49_aamix_sweep.py): logs to eric-czech/marin.
ENTITY = "eric-czech"
PROJECT = "marin"
GROUP = "exp49-aamix"
TAG = "exp49"
CDVAL_METRIC = "eval/protein-docs-cd-val/loss"

MIXTURE_NAMES = {"m1": "m1\n100% docs (500M)", "m2": "m2\n50/50 docs+seq (1B)"}
MIXTURE_ORDER = ("m1", "m2")
SEEDS = (1729, 1730, 1731)
SEED_COLORS = ("#4C78A8", "#F58518", "#54A24B")

RESULTS_ROOT = Path(__file__).resolve().parent / "exp49_aamix_results"


def summary_csv(version: str) -> Path:
    return RESULTS_ROOT / f"summary_runs_{version}.csv"


def fetch_summary(version: str) -> pd.DataFrame:
    """One row per finished trial of ``version``: mixture, seed, final cd-val loss."""
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP, "tags": TAG})
    rows = []
    for run in runs:
        name = run.display_name
        if f"-{version}-" not in name:  # only the requested sweep version
            continue
        if run.state != "finished":  # ignore crashed/failed/killed retries
            continue
        mixture = next((m for m in MIXTURE_ORDER if f"-{m}-" in name), None)
        seed_index = SEEDS.index(run.config["data_seed"]) if run.config.get("data_seed") in SEEDS else None
        summary = dict(run.summary) if run.summary is not None else {}
        rows.append(
            {
                "run_name": name,
                "mixture_id": mixture,
                "seed_index": seed_index,
                "data_seed": run.config.get("data_seed"),
                "state": run.state,
                "step": summary.get("_step"),
                "cdval_loss": summary.get(CDVAL_METRIC),
            }
        )
    if not rows:
        raise RuntimeError(f"No finished exp49 {version} runs in {ENTITY}/{PROJECT} group={GROUP}")
    df = pd.DataFrame(rows)
    # One run per (mixture, seed): keep the furthest-trained if a cell somehow repeats.
    df = (
        df.sort_values(["mixture_id", "seed_index", "step"])
        .drop_duplicates(["mixture_id", "seed_index"], keep="last")
        .reset_index(drop=True)
    )
    logger.info("Fetched %d finished %s run(s)", len(df), version)
    return df


def load_or_fetch_summary(version: str, *, refresh: bool) -> pd.DataFrame:
    csv = summary_csv(version)
    if not refresh and csv.exists():
        logger.info("Loading cached summary from %s", csv)
        return pd.read_csv(csv)
    df = fetch_summary(version)
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False)
    logger.info("Wrote %d run rows to %s", len(df), csv)
    return df


def render_figure(df: pd.DataFrame, out_dir: Path, version: str) -> None:
    """cd-val loss per seed for each mixture, with across-seed mean +/- 1 std."""
    df = df[df["cdval_loss"].notna()]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for x, mix in enumerate(MIXTURE_ORDER):
        sub = df[df["mixture_id"] == mix]
        for _, r in sub.iterrows():
            si = int(r["seed_index"])
            ax.scatter(
                x + (si - 1) * 0.08,
                r["cdval_loss"],
                s=90,
                color=SEED_COLORS[si % len(SEED_COLORS)],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
                label=f"s{si} (seed {SEEDS[si]})" if x == 0 else None,
            )
        vals = sub["cdval_loss"].to_numpy(dtype=float)
        mean, std = float(vals.mean()), float(vals.std(ddof=0))
        ax.errorbar(
            x,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=26,
            capsize=7,
            elinewidth=1.5,
            markeredgewidth=2.2,
            zorder=2,
            label="mean $\\pm$ 1$\\sigma$" if x == 0 else None,
        )
        ax.annotate(
            f"{mean:.4f}\n$\\sigma$={std * 1e3:.1f}e-3",
            (x, mean),
            xytext=(14, 0),
            textcoords="offset points",
            fontsize=8.5,
            va="center",
            color="#333333",
        )

    ax.set_xlim(-0.5, len(MIXTURE_ORDER) - 0.5)
    ax.set_xticks(range(len(MIXTURE_ORDER)))
    ax.set_xticklabels([MIXTURE_NAMES[m] for m in MIXTURE_ORDER])
    ax.set_ylabel("held-out cd-val loss (final)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_title(f"MarinFold Experiment #49 ({version}) — AA-mixture sweep\ncd-val loss by mixture x seed")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"exp49_aamix_cdval_{version}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2", help="Sweep version to analyze (e.g. v1, v2).")
    parser.add_argument("--refresh", action="store_true", help="Force a fresh W&B fetch.")
    args = parser.parse_args(argv)

    df = load_or_fetch_summary(args.version, refresh=args.refresh)
    logger.info(
        "cd-val loss by mixture x seed (%s):\n%s",
        args.version,
        df[["mixture_id", "seed_index", "cdval_loss", "state"]].to_string(index=False),
    )
    render_figure(df, RESULTS_ROOT, args.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
