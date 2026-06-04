# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyze the exp44 1.5B ``cd`` token-scaling sweep from W&B.

Companion to ``exp44_token_scaling_sweep.py``. The sweep asks: how few training
tokens can we spend and still recover the stage-2 mixture ranking / final
``protein-docs-cd-val`` loss? It is a 2-D grid — three token budgets
(``t1``=0.5B, ``t2``=1B, ``t3``=2B) crossed with six ``cd`` mixtures
(``m10``..``m15``) — plus a ``t1``-only data-ordering seed sweep
(``s0``=default 1729 .. ``s4``) that isolates data-permutation noise.

Because the three budgets stop at different step counts (240/480/952), there is
no common reference step that is fair across budgets: every cell uses its **own
run's final** logged value (W&B ``run.summary``), so ``t2``/``t3`` are read at
their full budgets rather than truncated to ``t1``. Most main-grid runs are
finished; in-flight seed runs are included but flagged so the seed figure can
keep its statistics to the comparable (finished) set.

Two figures are produced (300-dpi PNG + PDF):

* ``token_scaling_heatmap`` — the mixture x budget table heatmap of cd-val loss,
  with per-budget ranks and two aggregation columns (mean loss across budgets,
  mean rank across budgets).
* ``t1_seed_spread`` — the ``t1`` (0.5B) seed sweep: cd-val loss per mixture
  across data-ordering seeds, with the across-seed spread contextualized against
  the across-mixture spread. Partial — uses whatever seed runs exist.

All per-run summary data is written to ``summary_runs.csv`` (also the fetch
cache). Pass ``--refresh`` to force a re-pull from W&B.

Usage::

    set -a; source ~/marin.env; set +a   # WANDB_API_KEY
    uv run --with matplotlib --with seaborn \\
        python -m experiments.protein.exp44_token_scaling_analysis [--refresh]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sweep identity (see exp44_token_scaling_sweep.py)
# ---------------------------------------------------------------------------

# NOTE: the sweep logs to project ``marin`` (the ``WANDB_PROJECT=marin-dna`` in
# ~/marin.env is stale and does not resolve via the API).
ENTITY = "eric-czech"
PROJECT = "marin"
GROUP = "exp44-token-scaling"
TAG = "exp44"

# ``prot-exp44-ts-1_5b-<budget>-<tokens>-[s<i>-]<mixture>-lr...-v1-<hash>``.
RUN_NAME_RE = re.compile(r"^prot-exp44-ts-1_5b-(?P<budget>t\d)-[0-9.]+[MB]-(?:(?P<seed>s\d)-)?(?P<mixture>m\d+)-lr")

# Mixture display labels (match the exp44 sweep docstring / exp11 issue table).
MIXTURE_NAMES: dict[str, str] = {
    "m10": "m10 (H)",
    "m11": "m11 (H31/M26/L43)",
    "m12": "m12 (L→H)",
    "m13": "m13 (L→M→H)",
    "m14": "m14 (M)",
    "m15": "m15 (L)",
}
MIXTURE_ORDER: tuple[str, ...] = ("m10", "m11", "m12", "m13", "m14", "m15")

# Token budgets, in plot order, with nominal labels for axes.
BUDGET_ORDER: tuple[str, ...] = ("t1", "t2", "t3")
BUDGET_NOMINAL: dict[str, str] = {"t1": "0.5B", "t2": "1B", "t3": "2B"}

# Data-ordering seeds (exp44 ``DATA_SEEDS``); index 0 is the default (no seed
# token in the run name). Used only to label the seed figure.
DATA_SEEDS: tuple[int, ...] = (1729, 1730, 1731, 1732, 1733)

# Primary metric and the family of per-component eval losses we keep in the CSV.
CDVAL_METRIC = "eval/protein-docs-cd-val/loss"
TRAIN_METRIC = "train/loss"
TOTAL_GFLOPS_KEY = "throughput/total_gflops"
EVAL_LOSS_RE = re.compile(r"^eval/.+-(val|test|train)/loss$")

TITLE_PREFIX = "MarinFold Experiment #44 — 1.5B cd token-scaling sweep"


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

RESULTS_ROOT = Path(__file__).resolve().parent / "exp44_token_scaling_results"
SUMMARY_CSV = RESULTS_ROOT / "summary_runs.csv"
PLOTS_DIR = RESULTS_ROOT / "plots"


# ---------------------------------------------------------------------------
# W&B fetch -> one row per run
# ---------------------------------------------------------------------------


def _get_nested(cfg: dict, dotted_key: str) -> object | None:
    """Walk a nested mapping by dotted key path; None if any segment is absent."""
    cur: object = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _parse_int_tag(tags: list[str], prefix: str) -> int | None:
    """Parse a ``<prefix><int>`` tag (e.g. ``params_exact=1471348736``)."""
    for tag in tags:
        if tag.startswith(prefix):
            try:
                return int(tag.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _short_metric_label(name: str) -> str:
    """``eval/protein-docs-high-val/loss`` -> ``high-val``; ``train/loss`` -> ``train``."""
    if name == TRAIN_METRIC:
        return "train"
    return name.removeprefix("eval/").removesuffix("/loss").removeprefix("protein-docs-")


def _run_row(run) -> dict | None:
    """Flatten one W&B run into a summary row, or None if it isn't an exp44 trial."""
    match = RUN_NAME_RE.match(run.display_name)
    if not match:
        logger.warning("Run %s did not match the exp44 name pattern; skipping", run.display_name)
        return None
    budget = match.group("budget")
    mixture = match.group("mixture")
    seed_tag = match.group("seed") or "s0"
    seed_index = int(seed_tag[1:])

    cfg = dict(run.config)
    summary = dict(run.summary) if run.summary is not None else {}
    tags = list(run.tags or [])

    num_train_steps = _get_nested(cfg, "trainer.num_train_steps")
    batch = _get_nested(cfg, "trainer.train_batch_size")
    seq_len = cfg.get("train_seq_len")
    tokens_exact = _parse_int_tag(tags, "tokens_exact=")
    if tokens_exact is None and None not in (batch, seq_len, num_train_steps):
        tokens_exact = int(batch) * int(seq_len) * int(num_train_steps)
    data_seed = cfg.get("data_seed", DATA_SEEDS[seed_index] if seed_index < len(DATA_SEEDS) else None)

    step = summary.get("_step")
    pct_complete = None
    if step is not None and num_train_steps:
        pct_complete = round(100.0 * (int(step) + 1) / int(num_train_steps), 1)

    row: dict[str, object] = {
        "run_name": run.display_name,
        "budget": budget,
        "budget_nominal": BUDGET_NOMINAL.get(budget, budget),
        "mixture_id": mixture,
        "mixture_name": MIXTURE_NAMES.get(mixture, mixture),
        "seed_tag": seed_tag,
        "seed_index": seed_index,
        "data_seed": data_seed,
        "state": run.state,
        "step": int(step) if step is not None else None,
        "num_train_steps": int(num_train_steps) if num_train_steps else None,
        "pct_complete": pct_complete,
        "tokens_exact": tokens_exact,
        "params_exact": _parse_int_tag(tags, "params_exact="),
        "total_gflops": summary.get(TOTAL_GFLOPS_KEY),
        "cdval_loss": summary.get(CDVAL_METRIC),
        "train_loss": summary.get(TRAIN_METRIC),
    }
    # Every per-component eval loss (sparse across mixtures) under a short label.
    for key in summary:
        if EVAL_LOSS_RE.match(key) and key != CDVAL_METRIC:
            row[_short_metric_label(key)] = summary[key]
    return row


def fetch_summary() -> pd.DataFrame:
    """Pull every exp44 run's final summary into one row-per-run DataFrame."""
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"group": GROUP, "tags": TAG}))
    logger.info("Fetched %d runs from %s/%s group=%s", len(runs), ENTITY, PROJECT, GROUP)
    rows = [r for r in (_run_row(run) for run in runs) if r is not None]
    if not rows:
        raise RuntimeError(f"No exp44 runs matched in {ENTITY}/{PROJECT} group={GROUP}")
    df = pd.DataFrame(rows)
    budget_rank = {b: i for i, b in enumerate(BUDGET_ORDER)}
    mixture_rank = {m: i for i, m in enumerate(MIXTURE_ORDER)}
    df["_b"] = df["budget"].map(lambda b: budget_rank.get(b, 99))
    df["_m"] = df["mixture_id"].map(lambda m: mixture_rank.get(m, 99))
    df = df.sort_values(["_b", "seed_index", "_m"]).drop(columns=["_b", "_m"]).reset_index(drop=True)
    return df


def load_or_fetch_summary(*, refresh: bool) -> pd.DataFrame:
    """Cached fetch: read ``summary_runs.csv`` unless missing or ``--refresh``."""
    if not refresh and SUMMARY_CSV.exists():
        logger.info("Loading cached summary from %s", SUMMARY_CSV)
        return pd.read_csv(SUMMARY_CSV)
    df = fetch_summary()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    logger.info("Wrote %d run rows to %s", len(df), SUMMARY_CSV)
    return df


# ---------------------------------------------------------------------------
# Figure 1 — mixture x budget table heatmap (cd-val loss)
# ---------------------------------------------------------------------------


def _zscore_columns(values: pd.DataFrame) -> pd.DataFrame:
    """Per-column z-score (ddof=0); constant columns map to 0. NaN-safe."""
    z = values.astype(float).copy()
    for col in values.columns:
        c = values[col].astype(float)
        std = c.std(ddof=0)
        z[col] = (c - c.mean()) / std if std and std > 0 else 0.0
    return z


def render_token_scaling_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Mixture (rows) x budget (cols) cd-val heatmap + mean-loss / mean-rank columns.

    Default-seed (``s0``), *finished* runs only — the seed variants get their
    own figure, and an in-flight run sits at an inflated, non-comparable loss
    (its WSD decay hasn't finished), so any incomplete cell is left blank
    ("—") rather than plotted.

    Rows are sorted by mean cd-val loss across budgets (best mixture on top).
    Each budget cell is annotated ``loss`` over ``#rank`` (rank within that
    budget, 1 = lowest loss); the two right-hand columns aggregate across
    budgets. Color is a monotone per-column z-score (darker = higher loss =
    worse) so the table reads at a glance while the annotations carry the raw
    values.
    """
    base = df[(df["seed_index"] == 0) & (df["state"] == "finished")]
    missing = int(((df["seed_index"] == 0) & (df["state"] != "finished")).sum())
    if missing:
        logger.warning("Heatmap: %d default-seed run(s) not finished; leaving those cells blank", missing)
    loss = base.pivot(index="mixture_id", columns="budget", values="cdval_loss")
    loss = loss.reindex(index=MIXTURE_ORDER, columns=BUDGET_ORDER)

    # Rank mixtures within each budget (1 = best); aggregate across budgets.
    # Ranks are row-order-independent, so compute before sorting rows.
    ranks = loss.rank(axis=0, method="min")
    mean_loss = loss.mean(axis=1)
    mean_rank = ranks.mean(axis=1)

    # Sort rows by mean loss (ascending) so the best mixture sits at the top.
    row_order = list(mean_loss.sort_values().index)
    loss = loss.reindex(row_order)
    ranks = ranks.reindex(row_order)
    mean_loss = mean_loss.reindex(row_order)
    mean_rank = mean_rank.reindex(row_order)

    n_budgets = len(BUDGET_ORDER)
    display = loss.copy()
    display["mean_loss"] = mean_loss
    display["mean_rank"] = mean_rank
    cols = [*BUDGET_ORDER, "mean_loss", "mean_rank"]
    display = display[cols]

    z = _zscore_columns(display)
    nrows, ncols = display.shape
    fig, ax = plt.subplots(figsize=(1.8 + 1.25 * ncols, 1.0 + 0.7 * nrows))
    # Monotone Blues over the per-column z-score: lightest = lowest loss (best),
    # darkest = highest loss (worst). vmin/vmax bound the z range symmetrically.
    im = ax.imshow(z.to_numpy(dtype=float), cmap="Blues", aspect="auto", vmin=-2.0, vmax=2.0)

    col_labels = [f"{b}\n({BUDGET_NOMINAL[b]} tok)" for b in BUDGET_ORDER] + ["mean\nloss", "mean\nrank"]
    ax.set_xticks(range(ncols))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([MIXTURE_NAMES[m] for m in row_order])

    for i in range(nrows):
        for j, col in enumerate(cols):
            val = display.iat[i, j]
            zv = z.iat[i, j]
            # White text only on the dark (high-z) end of the monotone ramp.
            color = "white" if zv > 1.0 else "black"
            if pd.isna(val):
                ax.text(j, i, "—", ha="center", va="center", color="black", fontsize=9)
            elif col in BUDGET_ORDER:
                rank = ranks.iat[i, j]
                text = f"{val:.4f}\n#{int(rank)}"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8.5)
            elif col == "mean_rank":
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
            else:
                ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=9)

    # Visual divider between the per-budget block and the aggregation columns.
    ax.axvline(n_budgets - 0.5, color="black", linewidth=2.0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("per-column z-score  (darker = higher loss = worse)")

    ax.set_xticks(np.arange(ncols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(nrows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#9e9e9e", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    tokens = base.set_index("budget")["tokens_exact"].to_dict()
    steps = base.set_index("budget")["num_train_steps"].to_dict()
    shape = "  ".join(f"{b}={tokens.get(b, float('nan')) / 1e9:.3f}B tok / {steps.get(b)} steps" for b in BUDGET_ORDER)
    ax.set_title(
        f"{TITLE_PREFIX}\ncd-val loss by mixture x token budget (default seed, finished runs)\n{shape}",
        fontsize=10,
    )
    fig.tight_layout()
    _save(fig, out_dir, "token_scaling_heatmap")


# ---------------------------------------------------------------------------
# Figure 2 — t1 data-ordering seed spread
# ---------------------------------------------------------------------------

# Tableau-10 categorical palette for seeds s0..s4.
SEED_COLORS = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2")


def render_t1_seed_spread(df: pd.DataFrame, out_dir: Path) -> None:
    """cd-val loss across data-ordering seeds at the ``t1`` (0.5B) budget.

    Only *finished* runs are plotted: at ``t1`` the final cd-val is only
    comparable once the run reaches step 240 (the WSD decay finishes), so an
    in-flight run sits at an inflated, non-comparable loss. The sweep is still
    filling in, so per-mixture seed counts vary; the title reports how many
    seed-runs are still pending. One column per mixture (sorted left-to-right by
    mean cd-val), one marker per seed, with the mean +/- 1 std over the
    available seeds drawn as an error bar.

    A footnote compares the mean across-seed std to the across-mixture range of
    the per-seed-mean losses — the noise-vs-signal ratio that decides whether
    the mixture ranking is recoverable at 0.5B tokens.
    """
    t1 = df[(df["budget"] == "t1") & (df["state"] == "finished") & df["cdval_loss"].notna()].copy()
    if t1.empty:
        logger.warning("No finished t1 cd-val data; skipping seed figure")
        return
    # Sort the x-axis by each mixture's mean cd-val across its finished seeds
    # (ascending, best on the left).
    mixture_mean = t1.groupby("mixture_id")["cdval_loss"].mean()
    mixtures = list(mixture_mean.sort_values().index)
    pending = int(((df["budget"] == "t1") & (df["seed_index"] > 0) & (df["state"] != "finished")).sum())

    fig, ax = plt.subplots(figsize=(2.0 + 1.5 * len(mixtures), 5.2))
    seed_means: dict[str, float] = {}
    multi_seed_stds: list[float] = []
    for x, mix in enumerate(mixtures):
        sub = t1[t1["mixture_id"] == mix]
        for _, r in sub.iterrows():
            si = int(r["seed_index"])
            ax.scatter(
                x + (si - 2) * 0.07,
                r["cdval_loss"],
                s=85,
                color=SEED_COLORS[si % len(SEED_COLORS)],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )
        vals = sub["cdval_loss"].to_numpy(dtype=float)
        mean, std = float(vals.mean()), float(vals.std(ddof=0))
        seed_means[mix] = mean
        if len(vals) >= 2:
            multi_seed_stds.append(std)
        ax.errorbar(
            x,
            mean,
            yerr=std,
            fmt="_",
            color="black",
            markersize=22,
            capsize=6,
            elinewidth=1.4,
            markeredgewidth=2.0,
            zorder=2,
        )
        ax.annotate(
            f"n={len(vals)}\n$\\sigma$={std * 1e3:.1f}e-3",
            (x, mean),
            xytext=(11, 0),
            textcoords="offset points",
            fontsize=7.5,
            va="center",
            color="#333333",
        )

    ax.set_xticks(range(len(mixtures)))
    ax.set_xticklabels([MIXTURE_NAMES[m] for m in mixtures], rotation=20, ha="right")
    ax.set_ylabel("cd-val loss  (t1 = 0.5B tokens, final)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Noise-vs-signal footnote: mean across-seed std (mixtures with >=2 seeds)
    # vs across-mixture range of the per-mixture seed means.
    note = ""
    if multi_seed_stds and len(seed_means) >= 2:
        mean_std = float(np.mean(multi_seed_stds))
        mix_range = max(seed_means.values()) - min(seed_means.values())
        ratio = mix_range / mean_std if mean_std > 0 else float("inf")
        note = (
            f"mean across-seed $\\sigma$ = {mean_std * 1e3:.2f}e-3 "
            f"(over {len(multi_seed_stds)} mixtures w/ >=2 seeds)   |   "
            f"across-mixture range of seed-means = {mix_range * 1e3:.1f}e-3   (signal/noise ~ {ratio:.1f}x)"
        )

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=SEED_COLORS[i],
            markeredgecolor="black",
            label=f"s{i} (seed {DATA_SEEDS[i]})",
        )
        for i in range(len(DATA_SEEDS))
    ]
    legend_handles.append(
        Line2D(
            [],
            [],
            marker="_",
            linestyle="none",
            color="black",
            markersize=12,
            markeredgewidth=2.0,
            label=r"mean $\pm$ 1$\sigma$",
        )
    )
    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)

    n_finished = len(t1)
    title = (
        f"{TITLE_PREFIX}\nt1 (0.5B) data-ordering seed sweep: final cd-val loss across seeds  "
        f"({n_finished} finished seed-runs; {pending} still running/crashed, excluded as incomplete)"
    )
    if note:
        title += f"\n{note}"
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save(fig, out_dir, "t1_seed_spread")


# ---------------------------------------------------------------------------
# Shared save + CLI
# ---------------------------------------------------------------------------


def _save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        logger.info("Saved %s", path)
    plt.close(fig)


def _log_tables(df: pd.DataFrame) -> None:
    base = df[df["seed_index"] == 0]
    wide = base.pivot(index="mixture_name", columns="budget", values="cdval_loss").reindex(columns=BUDGET_ORDER)
    logger.info("cd-val loss (default seed) by mixture x budget:\n%s", wide.to_string(float_format=lambda v: f"{v:.4f}"))
    states = df.groupby("state").size().to_dict()
    logger.info("run states: %s", states)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Force a fresh W&B fetch.")
    args = parser.parse_args(argv)

    df = load_or_fetch_summary(refresh=args.refresh)
    _log_tables(df)
    render_token_scaling_heatmap(df, PLOTS_DIR)
    render_t1_seed_spread(df, PLOTS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
