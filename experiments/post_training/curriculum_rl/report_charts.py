# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render round-2 report charts from W&B histories.

Produces success-over-steps and success-over-tokens charts for the in-run
validation sets, curriculum weight/pass-rate trajectories per arm, and a JSON
summary of per-arm token budgets. Runs locally (W&B access only).

    python -m experiments.post_training.curriculum_rl.report_charts --out /tmp/curriculum-r2
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import wandb

logger = logging.getLogger(__name__)

WANDB_PROJECT = "marin-community/marin-curriculum-rl"
VERSION_TAG = "2026.08.29.20"
ARM_ORDER = ("naive", "thompson", "learnability", "grade-adaptive", "grade-prior", "naive-dapo", "thompson-dapo")
VAL_KEYS = ("val-gsm8k", "val-math500", "val-rg-sum", "val-rg-spell", "val-rg-base")
STEP_KEY = "trainer/global_step"
TOKENS_KEY = "generate/avg_num_tokens"
# Responses per generate call: train_batch_size (512) * n_samples_per_prompt (8).
RESPONSES_PER_GENERATE = 512 * 8


def arm_of(run_name: str) -> str | None:
    if VERSION_TAG not in run_name or "smoke" in run_name:
        return None
    return run_name.split("curriculum-rl-")[1].split("-2026")[0]


def pick_runs(api: wandb.Api) -> dict[str, list]:
    """All run entries per arm; retries/reconnects leave several entries per arm."""
    by_arm = defaultdict(list)
    for run in api.runs(WANDB_PROJECT, order="-created_at"):
        arm = arm_of(run.name)
        if arm in ARM_ORDER:
            by_arm[arm].append(run)
    return by_arm


def merged_history(runs: list, keys: list[str], step_key: str = STEP_KEY) -> dict[int, dict[str, float]]:
    """Merge rows across run entries keyed by ``step_key``, later-logged rows winning.

    W&B returns only rows containing every requested key, so callers must group
    keys by the rows they are logged on (token counts carry no trainer step).
    """
    merged: dict[int, dict[str, float]] = {}
    for run in sorted(runs, key=lambda r: r.created_at):
        for row in run.history(keys=[step_key, *keys], samples=5000, pandas=False):
            step = row.get(step_key)
            if step is None:
                continue
            entry = merged.setdefault(int(step), {})
            entry.update({k: v for k, v in row.items() if v is not None})
    return merged


def arm_series(runs: list) -> dict[str, object]:
    """Eval points aligned to cumulative generated tokens across all run entries.

    A resumed run starts a new W&B entry whose ``_step`` restarts at zero, so
    entries are walked chronologically and token spend accumulates across them
    (redone steps after a resume are genuine spend). Within one entry, an eval
    row maps to the cumulative total at the latest preceding token row.
    """
    eval_keys = [STEP_KEY] + [f"eval/{v}/avg_score" for v in VAL_KEYS] + ["eval/all/avg_score"]
    aligned: dict[tuple[int, int], dict[str, float]] = {}
    total = 0.0
    for entry_index, run in enumerate(sorted(runs, key=lambda r: r.created_at)):
        base = total
        cumulative: dict[int, float] = {}
        token_rows = merged_history([run], [TOKENS_KEY], step_key="_step")
        for wandb_step in sorted(token_rows):
            total += token_rows[wandb_step].get(TOKENS_KEY, 0.0) * RESPONSES_PER_GENERATE
            cumulative[wandb_step] = total
        token_steps = sorted(cumulative)
        for wandb_step, row in merged_history([run], eval_keys, step_key="_step").items():
            preceding = [s for s in token_steps if s <= wandb_step]
            aligned[(entry_index, wandb_step)] = {
                **row,
                "cumulative_tokens": cumulative[preceding[-1]] if preceding else base,
            }
    return {"evals": aligned, "total_tokens": total}


def curriculum_series(runs: list, metric: str) -> dict[str, dict[int, float]]:
    """Per-bin trajectories of one curriculum metric (weight or pass_rate)."""
    bin_keys = sorted(
        {
            key
            for run in runs
            for row in run.history(samples=50, pandas=False)
            for key in row
            if key.startswith("curriculum/") and key.endswith(f"/{metric}")
        }
    )
    if not bin_keys:
        return {}
    rows = merged_history(runs, bin_keys)
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for step, row in rows.items():
        for key, value in row.items():
            if key.endswith(f"/{metric}"):
                out[key.split("/")[1]][step] = value
    return out


def plot_lines(series: dict[str, dict], xlabel: str, ylabel: str, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in series.items():
        if not points:
            continue
        xs = sorted(points)
        ax.plot(xs, [points[x] for x in xs], label=label, linewidth=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


@click.command(help=__doc__)
@click.option("--out", type=click.Path(path_type=Path), required=True)
def main(out: Path) -> None:
    logging.basicConfig(level=logging.INFO)
    out.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    by_arm = pick_runs(api)

    data = {arm: arm_series(runs) for arm, runs in by_arm.items()}
    summary = {
        arm: {
            "total_tokens": d["total_tokens"],
            "final_evals": d["evals"][max(d["evals"])] if d["evals"] else {},
            "run_entries": len(by_arm[arm]),
        }
        for arm, d in data.items()
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    for val in ("val-math500", "val-gsm8k", "val-rg-sum", "val-rg-spell", "val-rg-base", "all"):
        key = f"eval/{val}/avg_score"
        by_steps, by_tokens = {}, {}
        for arm, d in data.items():
            by_steps[arm] = {int(r[STEP_KEY]): r[key] for r in d["evals"].values() if key in r and STEP_KEY in r}
            by_tokens[arm] = {r["cumulative_tokens"] / 1e6: r[key] for r in d["evals"].values() if key in r}
        plot_lines(by_steps, "training step", "avg score", f"{val}: score vs steps", out / f"{val}-steps.png")
        plot_lines(by_tokens, "generated tokens (M)", "avg score", f"{val}: score vs tokens", out / f"{val}-tokens.png")

    for arm, runs in by_arm.items():
        for metric in ("weight", "pass_rate"):
            series = curriculum_series(runs, metric)
            if series:
                plot_lines(
                    series, "training step", metric, f"{arm}: bin {metric}", out / f"curriculum-{arm}-{metric}.png"
                )

    logger.info("Wrote charts and summary to %s", out)


if __name__ == "__main__":
    main()
