# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot per-step training loss for the 67B-A2B 10T run (issue #6044) and the June scaling-ladder runs.

Writes two figures:

- ``<out_dir>/67b_a2b_train_loss_tokens.png``: the 67B-A2B lineage (4096-batch start, 8192-batch resume, and the
  three 65k-context cooldowns) against tokens seen.
- ``<out_dir>/67b_a2b_train_loss_vs_scaling_completion.png``: the same lineage plus the d512/d768/d1024 scaling runs
  from the issue #6044 scaling fit, against percent of each run's planned training.

Every logged step is plotted. The W&B API returns at most 10,000 history rows per query, so history is fetched in
step ranges of ``_HISTORY_CHUNK``. Tokens per step are ``train_batch_size * max_seq_len`` from each run's W&B config
(``train_batch_size`` may be a staged schedule), accumulated along the lineage: each run continues from its parent's
token count at the step where it starts.

Run::

    uv run --with matplotlib python experiments/grug/moe/plot_67b_a2b_training_loss.py <out_dir>
"""

import sys
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import wandb

mpl.use("Agg")

WANDB_PROJECT = "marin-community/marin_moe"
LOSS_KEY = "train/loss"
TARGET_TOKENS = 10e12
_HISTORY_CHUNK = 5000
# Lines break across logging gaps longer than this many steps instead of drawing a straight segment.
_MAX_STEP_GAP = 50


@dataclass(frozen=True)
class Segment:
    """One W&B run in the 67B-A2B lineage, plotted from ``start_step`` until ``end_step`` (exclusive) if set."""

    run_id: str
    label: str
    color: str
    parent: str | None
    end_step: int | None = None


LINEAGE = (
    Segment(
        "moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T", "main, bs 4096", "#2a78d6", None, 15_000
    ),
    Segment(
        "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
        "main, bs 8192 (from step 15k)",
        "#2a78d6",
        "moe_67b_a2b_d2560_ep1_rep16_bs4096_seq8192_sw2k_v4_2048_muon_10T",
    ),
    Segment(
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k",
        "65k-ctx cooldown from step 39k",
        "#eb6834",
        "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
    ),
    Segment(
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step102k",
        "65k-ctx cooldown from step 102k",
        "#1baf7a",
        "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
    ),
    Segment(
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step141k",
        "65k-ctx cooldown from step 141k (final)",
        "#8a5cd6",
        "moe_67b_a2b_d2560_ep1_rep8_bs8192_seq8192_sw2k_v4_2048_muon_resume15k_v2_10T",
    ),
)
# (run id, label, color) for the scaling-fit anchors in
# https://github.com/marin-community/marin/issues/6044#issuecomment-4819900707
SCALING_RUNS = (
    ("june_prep_moe_may_d512_ep2_no_long_rope_seq8192_sw2k", "d512 (6 layers)", "#9fb8d8"),
    ("june_prep_moe_may_d768_ep2_no_long_rope_seq8192_sw2k", "d768 (8 layers)", "#5f86b8"),
    ("june_prep_moe_may_d1024_ep2_bs512_no_long_rope_seq8192_sw2k", "d1024 (11 layers)", "#2c4f7c"),
)
# (steps, x values, losses) for one plotted run.
Curve = tuple[np.ndarray, np.ndarray, np.ndarray]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def loss_history(run: wandb.apis.public.Run) -> tuple[np.ndarray, np.ndarray]:
    """(steps, losses) for every logged step; the last value wins where a resumed run re-logged a step.

    Non-positive losses are dropped: the d768 run's final history row (step 262,736) logs a loss of exactly 0.
    """
    by_step: dict[int, float] = {}
    for lo in range(0, run.lastHistoryStep + 1, _HISTORY_CHUNK):
        for row in run.scan_history(keys=["_step", LOSS_KEY], min_step=lo, max_step=lo + _HISTORY_CHUNK):
            if row[LOSS_KEY] > 0:
                by_step[int(row["_step"])] = float(row[LOSS_KEY])
    steps = np.array(sorted(by_step))
    return steps, np.array([by_step[s] for s in steps])


def tokens_per_step(run: wandb.apis.public.Run, steps: np.ndarray) -> np.ndarray:
    """Tokens consumed at each of ``steps``; ``train_batch_size`` is an int or a list of ``{start, value}`` stages."""
    batch = run.config["trainer"]["trainer"]["train_batch_size"]
    stages = [(0, batch)] if isinstance(batch, int) else [(s["start"], s["value"]) for s in batch]
    starts = np.array([start for start, _ in stages])
    values = np.array([value for _, value in stages])
    return values[np.searchsorted(starts, steps, side="right") - 1] * run.config["model"]["max_seq_len"]


def tokens_between(run: wandb.apis.public.Run, start: int, end: int) -> float:
    return float(tokens_per_step(run, np.arange(start, end)).sum())


def lineage_tokens(api: wandb.Api) -> dict[str, Curve]:
    """Steps, tokens seen and loss at every logged step of each lineage segment."""
    runs = {seg.run_id: api.run(f"{WANDB_PROJECT}/{seg.run_id}") for seg in LINEAGE}
    first_step: dict[str, int] = {}
    out = {}
    for seg in LINEAGE:
        steps, loss = loss_history(runs[seg.run_id])
        if seg.end_step is not None:
            keep = steps < seg.end_step
            steps, loss = steps[keep], loss[keep]
        # The root run starts at step 0; a resumed or branched run starts at its first logged step.
        start = 0 if seg.parent is None else int(steps[0])
        first_step[seg.run_id] = start
        # Tokens at this segment's start, accumulated through each ancestor's own batch size.
        base, child_start, parent = 0.0, start, seg.parent
        while parent is not None:
            base += tokens_between(runs[parent], first_step[parent], child_start)
            child_start = first_step[parent]
            parent = next(s.parent for s in LINEAGE if s.run_id == parent)
        # Tokens seen after completing each logged step.
        cumulative = np.cumsum(tokens_per_step(runs[seg.run_id], np.arange(start, int(steps[-1]) + 1)))
        out[seg.run_id] = (steps, base + cumulative[steps - start], loss)
    return out


def _break_gaps(steps: np.ndarray, x: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insert NaN after each logging gap longer than ``_MAX_STEP_GAP`` steps so the line is not drawn across it."""
    cut = np.flatnonzero(np.diff(steps) > _MAX_STEP_GAP) + 1
    return np.insert(x.astype(float), cut, np.nan), np.insert(loss, cut, np.nan)


def _style(ax: plt.Axes) -> None:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_tokens(lineage: dict[str, Curve], path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for seg in LINEAGE:
        steps, tokens, loss = lineage[seg.run_id]
        label = None if seg.run_id == LINEAGE[0].run_id else seg.label
        ax.plot(
            *_break_gaps(steps, tokens / 1e12, loss),
            color=seg.color,
            linewidth=0.35,
            alpha=0.8,
            label=label,
            rasterized=True,
        )
    ax.set_xlabel("tokens seen (T)")
    ax.set_ylabel(LOSS_KEY)
    ax.set_ylim(1.2, 2.0)
    ax.set_xlim(0, 10.2)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    labels[0] = "main run (bs 4096 to step 15k, then bs 8192)"
    ax.legend(handles, labels, frameon=False, loc="upper right")
    for line in ax.get_legend().get_lines():
        line.set_linewidth(2)
        line.set_alpha(1)
    ax.set_title(
        "Marin 67B-A2B (#6044) training loss, every step; y-axis clipped at 2.0 (early steps start near 11.8)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def plot_completion(api: wandb.Api, lineage: dict[str, Curve], path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for run_id, label, color in SCALING_RUNS:
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        steps, loss = loss_history(run)
        total = run.config["trainer"]["trainer"]["num_train_steps"]
        ax.plot(
            *_break_gaps(steps, 100 * steps / total, loss),
            color=color,
            linewidth=0.35,
            alpha=0.8,
            label=label,
            rasterized=True,
        )
    for seg in LINEAGE:
        steps, tokens, loss = lineage[seg.run_id]
        label = None if seg.run_id == LINEAGE[0].run_id else f"67B-A2B {seg.label}"
        ax.plot(
            *_break_gaps(steps, 100 * tokens / TARGET_TOKENS, loss),
            color=seg.color,
            linewidth=0.35,
            alpha=0.8,
            label=label,
            rasterized=True,
        )
    ax.set_xlabel("percent of planned training (scaling runs: step / num_train_steps; 67B-A2B: tokens / 10T)")
    ax.set_ylabel(LOSS_KEY)
    ax.set_ylim(1.2, 3.6)
    ax.set_xlim(0, 101)
    _style(ax)
    handles, labels = ax.get_legend_handles_labels()
    labels[len(SCALING_RUNS)] = "67B-A2B main run"
    ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=9)
    for line in ax.get_legend().get_lines():
        line.set_linewidth(2)
        line.set_alpha(1)
    ax.set_title(
        "Training loss vs percent completion: June scaling runs and Marin 67B-A2B, every step; y-axis clipped at 3.6",
        color=INK,
        fontsize=11,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


if __name__ == "__main__":
    out_dir = sys.argv[1]
    api = wandb.Api(timeout=300)
    lineage = lineage_tokens(api)
    plot_tokens(lineage, f"{out_dir}/67b_a2b_train_loss_tokens.png")
    plot_completion(api, lineage, f"{out_dir}/67b_a2b_train_loss_vs_scaling_completion.png")
