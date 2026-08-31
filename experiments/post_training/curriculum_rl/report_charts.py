# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render round-3 report charts from W&B histories.

Produces per-validation-bin score charts over steps and generated tokens, the
grade-weighted end metric and frontier-grade headline charts, curriculum
weight/pass-rate trajectories per arm, an empirical learning-velocity curve,
and a JSON summary of per-arm token budgets. Runs locally (W&B access only).

    python -m experiments.post_training.curriculum_rl.report_charts --out /tmp/curriculum-r3
"""

import itertools
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
VERSION_TAG = "2026.08.31"
# Snowball arms were relaunched at micro_train_batch_size_per_gpu=4 under a
# bumped version; the cancelled micro=1 probe lives at plain 2026.08.31 and
# must not merge into the snowball-naive series.
SNOWBALL_VERSION_TAG = "2026.08.31.1"
ARM_ORDER = (
    "naive",
    "naive-dapo",
    "learnability-dapo",
    "grade-prior-dapo",
    "grade-adaptive",
    "grade-prior",
    "snowball-naive",
    "snowball-naive-dapo",
    "snowball-learnability-dapo",
    "snowball-grade-prior-dapo",
)
# Validation bin -> ladder grade (pool.py bins). The end metric weights each
# bin by 1 + grade; the weights are fixed here and never visible to samplers.
VAL_GRADES = {
    "val-rg-sum": 1,
    "val-gsm8k": 3,
    "val-math500": 5,
    "val-amc": 9,
    "val-omni": 11,
    "val-theoremqa": 13,
}
# Lowest per-bin score treated as "solved at this grade" for the frontier chart.
FRONTIER_THRESHOLD = 0.25
STEP_KEY = "trainer/global_step"
TOKENS_KEY = "generate/avg_num_tokens"
# Responses per generate call: train_batch_size * n_samples_per_prompt, from the
# launch.py FULL (512*8) and SNOWBALL_FULL (128*8) presets.
QWEN_RESPONSES_PER_GENERATE = 512 * 8
SNOWBALL_RESPONSES_PER_GENERATE = 128 * 8


def responses_per_generate(arm: str) -> int:
    return SNOWBALL_RESPONSES_PER_GENERATE if arm.startswith("snowball-") else QWEN_RESPONSES_PER_GENERATE


def arm_of(run_name: str) -> str | None:
    if "curriculum-rl-" not in run_name or "smoke" in run_name:
        return None
    arm = run_name.split("curriculum-rl-")[1].split("-2026")[0]
    # Snowball checkpoint names carry the scale label (launch.py only drops the
    # suffix for the Qwen FULL preset): snowball-naive-snowball-full -> snowball-naive.
    arm = arm.removesuffix("-snowball-full")
    tag = SNOWBALL_VERSION_TAG if arm.startswith("snowball-") else VERSION_TAG
    if f"-{tag}-" not in run_name:
        return None
    return arm


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


def arm_series(arm: str, runs: list) -> dict[str, object]:
    """Eval points aligned to cumulative generated tokens across all run entries.

    A resumed run starts a new W&B entry whose ``_step`` restarts at zero, so
    entries are walked chronologically and token spend accumulates across them
    (redone steps after a resume are genuine spend). Within one entry, an eval
    row maps to the cumulative total at the latest preceding token row.
    """
    eval_keys = [STEP_KEY] + [f"eval/{v}/pass_at_1" for v in VAL_GRADES] + ["eval/all/pass_at_1"]
    aligned: dict[tuple[int, int], dict[str, float]] = {}
    total = 0.0
    for entry_index, run in enumerate(sorted(runs, key=lambda r: r.created_at)):
        base = total
        cumulative: dict[int, float] = {}
        token_rows = merged_history([run], [TOKENS_KEY], step_key="_step")
        for wandb_step in sorted(token_rows):
            total += token_rows[wandb_step].get(TOKENS_KEY, 0.0) * responses_per_generate(arm)
            cumulative[wandb_step] = total
        token_steps = sorted(cumulative)
        for wandb_step, row in merged_history([run], eval_keys, step_key="_step").items():
            preceding = [s for s in token_steps if s <= wandb_step]
            aligned[(entry_index, wandb_step)] = {
                **row,
                "cumulative_tokens": cumulative[preceding[-1]] if preceding else base,
            }
    return {"evals": aligned, "total_tokens": total}


def grade_weighted_score(row: dict[str, float]) -> float | None:
    """End metric: per-bin scores averaged with weights proportional to 1 + grade.

    Requires every validation bin so partial eval rows do not skew the metric.
    """
    scores = {v: row.get(f"eval/{v}/pass_at_1") for v in VAL_GRADES}
    if any(s is None for s in scores.values()):
        return None
    weight_total = sum(1 + g for g in VAL_GRADES.values())
    return sum((1 + VAL_GRADES[v]) * s for v, s in scores.items()) / weight_total


def frontier_grade(row: dict[str, float]) -> int | None:
    """Highest validation grade scoring at least FRONTIER_THRESHOLD (0 if none)."""
    scores = {v: row.get(f"eval/{v}/pass_at_1") for v in VAL_GRADES}
    if any(s is None for s in scores.values()):
        return None
    passing = [VAL_GRADES[v] for v, s in scores.items() if s >= FRONTIER_THRESHOLD]
    return max(passing, default=0)


def curriculum_series(runs: list, metric: str) -> dict[str, dict[int, float]]:
    """Per-bin trajectories of one curriculum metric (weight or pass_rate).

    Bin keys come from run summaries; sampling the history for discovery can
    miss per-step curriculum rows among the much denser engine metrics.
    """
    bin_keys = sorted(
        {
            key
            for run in runs
            for key in run.summary.keys()
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


def velocity_samples(pass_rates: dict[str, dict[int, float]]) -> list[tuple[float, float]]:
    """(pass_rate, d pass_rate / d step) pairs from consecutive logged points per bin."""
    samples = []
    for points in pass_rates.values():
        steps = sorted(points)
        for s1, s2 in itertools.pairwise(steps):
            samples.append((points[s1], (points[s2] - points[s1]) / (s2 - s1)))
    return samples


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


def plot_velocity(samples: list[tuple[float, float]], group_size: int, path: Path) -> None:
    """Binned mean learning velocity against pass rate, with sampler reference curves.

    Overlays the pass-variance p(1-p) and group-informative 1 - p^n - (1-p)^n
    weighting curves, each scaled to the peak of the empirical means, so the
    empirical curve can arbitrate between the two weighting assumptions.
    """
    edges = [i / 10 for i in range(11)]
    mids, means = [], []
    for lo, hi in itertools.pairwise(edges):
        bucket = [v for p, v in samples if lo <= p < hi or (hi == 1.0 and p == 1.0)]
        if bucket:
            mids.append((lo + hi) / 2)
            means.append(sum(bucket) / len(bucket))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter([p for p, _ in samples], [v for _, v in samples], s=4, alpha=0.15, label="bin transitions")
    ax.plot(mids, means, "o-", color="black", linewidth=2, label="binned mean")
    if means:
        peak = max(means)
        grid = [i / 100 for i in range(101)]
        pass_var = [p * (1 - p) for p in grid]
        group_inf = [1 - p**group_size - (1 - p) ** group_size for p in grid]
        ax.plot(grid, [v / max(pass_var) * peak for v in pass_var], "--", label="p(1-p) (scaled)")
        ax.plot(
            grid, [v / max(group_inf) * peak for v in group_inf], ":", label=f"group-informative n={group_size} (scaled)"
        )
    ax.set_xlabel("bin pass rate")
    ax.set_ylabel("d pass_rate / d step")
    ax.set_title("Learning velocity vs pass rate (all arms, all bins)")
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

    data = {arm: arm_series(arm, runs) for arm, runs in by_arm.items()}
    summary = {}
    for arm, d in data.items():
        final = d["evals"][max(d["evals"])] if d["evals"] else {}
        summary[arm] = {
            "total_tokens": d["total_tokens"],
            "final_evals": final,
            "final_grade_weighted": grade_weighted_score(final),
            "final_frontier_grade": frontier_grade(final),
            "run_entries": len(by_arm[arm]),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    for val in (*VAL_GRADES, "all"):
        key = f"eval/{val}/pass_at_1"
        by_steps, by_tokens = {}, {}
        for arm, d in data.items():
            by_steps[arm] = {int(r[STEP_KEY]): r[key] for r in d["evals"].values() if key in r and STEP_KEY in r}
            by_tokens[arm] = {r["cumulative_tokens"] / 1e6: r[key] for r in d["evals"].values() if key in r}
        plot_lines(by_steps, "training step", "pass@1", f"{val}: pass@1 vs steps", out / f"{val}-steps.png")
        plot_lines(by_tokens, "generated tokens (M)", "pass@1", f"{val}: pass@1 vs tokens", out / f"{val}-tokens.png")

    weighted_by_tokens, frontier_by_tokens = {}, {}
    for arm, d in data.items():
        weighted_by_tokens[arm] = {}
        frontier_by_tokens[arm] = {}
        for r in d["evals"].values():
            weighted = grade_weighted_score(r)
            if weighted is not None:
                weighted_by_tokens[arm][r["cumulative_tokens"] / 1e6] = weighted
            frontier = frontier_grade(r)
            if frontier is not None:
                frontier_by_tokens[arm][r["cumulative_tokens"] / 1e6] = frontier
    plot_lines(
        weighted_by_tokens,
        "generated tokens (M)",
        "grade-weighted avg score",
        "End metric: grade-weighted validation pass@1 vs tokens",
        out / "grade-weighted-tokens.png",
    )
    plot_lines(
        frontier_by_tokens,
        "generated tokens (M)",
        f"frontier grade (score >= {FRONTIER_THRESHOLD})",
        "Frontier grade vs tokens",
        out / "frontier-grade-tokens.png",
    )

    all_velocity_samples = []
    for arm, runs in by_arm.items():
        pass_rates = curriculum_series(runs, "pass_rate")
        all_velocity_samples.extend(velocity_samples(pass_rates))
        for metric, series in (("weight", curriculum_series(runs, "weight")), ("pass_rate", pass_rates)):
            if series:
                plot_lines(
                    series, "training step", metric, f"{arm}: bin {metric}", out / f"curriculum-{arm}-{metric}.png"
                )
    if all_velocity_samples:
        plot_velocity(all_velocity_samples, group_size=8, path=out / "velocity-vs-pass-rate.png")

    logger.info("Wrote charts and summary to %s", out)


if __name__ == "__main__":
    main()
