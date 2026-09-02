# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reporting for the curriculum-RL experiment.

Subcommands:

- ``charts``: render report charts from W&B histories. Produces, per model
  family (Qwen3-0.6B plus the Snowball 67B-A2B rounds), a grade-weighted
  headline chart, per-grade validation breakouts over steps and tokens, a
  grade-attainment chart (tokens to first reach the frontier threshold at each
  grade), curriculum weight/pass-rate trajectories per arm, an empirical
  learning-velocity curve, and a JSON summary of per-arm token budgets. Raw
  eval points are drawn faintly under an EMA-smoothed line. Runs locally (W&B
  access only); seaborn is not a workspace dependency, so run with an overlay:

      uv run --with seaborn python -m experiments.post_training.curriculum_rl.report charts --out /tmp/curriculum-charts

- ``evals``: print aggregated Evalchemy metrics for every arm from the
  canonical per-run ``record.json``.
- ``pool-stats``: print per-bin row counts and grades for one built pool
  version.
- ``trajectories``: aggregate retained-trajectory statistics (truncation,
  think-token usage, answer-line compliance) for one arm.

``evals``, ``pool-stats``, and ``trajectories`` read the experiment bucket, so
run them as an Iris CPU job on the training cluster (cw-us-east-02a). The
plotting stack is imported module-wide, so every invocation needs the
``--with seaborn`` overlay.
"""

import gzip
import io as std_io
import itertools
import json
import logging
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import click
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import wandb
from marin.evaluation.records import CW_RECORDS_PREFIX, RunStatus, list_records
from marin.training.training import temporary_storage_base_path
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.curriculum_rl.launch import EXPERIMENT_NAME

logger = logging.getLogger(__name__)

WANDB_PROJECT = "marin-community/marin-curriculum-rl"
POOL_ROOT = "s3://marin-us-east-02a/marin/documents/curriculum-rl-pool"

# Trajectory aggregation (the ``trajectories`` subcommand).
MAX_ARCHIVES = 80
STEP_BUCKET = 20
# Graded answer lines and \boxed{} both live at the very end of a response.
TAIL_CHARS = 400
THINK_TOKENS = ("<|start_think|>", "<|end_think|>", "<think>", "</think>")
ANSWER_LINE = re.compile(r"(####\s*\S+|Answer:\s*\S+)")
BIN_TOKENS = ("gsm8k", "svamp", "asdiv", "math", "numina", "omni", "aime", "theoremqa", "hardmath", "rg-sum", "putnam")
VERSION_TAG = "2026.08.31"
# Snowball arms were relaunched at micro_train_batch_size_per_gpu=4 under a
# bumped version; the cancelled micro=1 probe lives at plain 2026.08.31 and
# must not merge into the snowball-naive series.
SNOWBALL_VERSION_TAG = "2026.08.31.1"
# Round 4: MuonH at 1e-5, 120 steps x 64 prompts, system-prompted pool,
# reversion_mass=2. Charted as its own family so round-3 series stay intact.
SNOWBALL_R4_VERSION_TAG = "2026.09.01.1"
# Round 5: round-4 recipe with an 8192-token response budget and the
# tightened system prompt.
SNOWBALL_R5_VERSION_TAG = "2026.09.02.1"
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
    "snowball-r4-naive",
    "snowball-r4-grade-prior-dapo",
    "snowball-r5-naive",
    "snowball-r5-grade-prior-dapo",
)
# Sampler identity shared across families: the snowball-* arms reuse the Qwen
# sampler configurations, so cross-family charts keep one color per sampler.
SAMPLER_ORDER = (
    "naive",
    "naive-dapo",
    "learnability-dapo",
    "grade-prior-dapo",
    "grade-adaptive",
    "grade-prior",
)
FAMILY_LABELS = {
    "qwen": "Qwen3-0.6B",
    "snowball": "Snowball 67B-A2B SFT",
    "snowball-r4": "Snowball 67B-A2B SFT (round 4: MuonH + system prompt)",
    "snowball-r5": "Snowball 67B-A2B SFT (round 5: 8192-token budget)",
}
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
BIN_ORDER = sorted(VAL_GRADES, key=VAL_GRADES.__getitem__)
# Lowest per-bin score treated as "solved at this grade" for grade attainment.
FRONTIER_THRESHOLD = 0.25
# EMA coefficients: eval series are sparse (one point per eval interval) and
# get light softening; per-step curriculum series are dense and get more.
EVAL_ALPHA = 0.5
CURRICULUM_ALPHA = 0.3
STEP_KEY = "trainer/global_step"
TOKENS_KEY = "generate/avg_num_tokens"
# Responses per generate call: train_batch_size * n_samples_per_prompt, from the
# launch.py FULL (512*8) and SNOWBALL_FULL (128*8) presets.
QWEN_RESPONSES_PER_GENERATE = 512 * 8
SNOWBALL_RESPONSES_PER_GENERATE = 128 * 8
SNOWBALL_R4_RESPONSES_PER_GENERATE = 64 * 8
SNOWBALL_R5_RESPONSES_PER_GENERATE = 64 * 8


def family_of(arm: str) -> str:
    if arm.startswith("snowball-r5-"):
        return "snowball-r5"
    if arm.startswith("snowball-r4-"):
        return "snowball-r4"
    return "snowball" if arm.startswith("snowball-") else "qwen"


def sampler_of(arm: str) -> str:
    return arm.removeprefix("snowball-r5-").removeprefix("snowball-r4-").removeprefix("snowball-")


def responses_per_generate(arm: str) -> int:
    if family_of(arm) == "snowball-r5":
        return SNOWBALL_R5_RESPONSES_PER_GENERATE
    if family_of(arm) == "snowball-r4":
        return SNOWBALL_R4_RESPONSES_PER_GENERATE
    return SNOWBALL_RESPONSES_PER_GENERATE if family_of(arm) == "snowball" else QWEN_RESPONSES_PER_GENERATE


FAMILY_TAGS = {
    "qwen": VERSION_TAG,
    "snowball": SNOWBALL_VERSION_TAG,
    "snowball-r4": SNOWBALL_R4_VERSION_TAG,
    "snowball-r5": SNOWBALL_R5_VERSION_TAG,
}


def arm_of(run_name: str) -> str | None:
    if "curriculum-rl-" not in run_name or "smoke" in run_name:
        return None
    arm = run_name.split("curriculum-rl-")[1].split("-2026")[0]
    # Snowball checkpoint names carry the scale label (launch.py only drops the
    # suffix for the Qwen FULL preset): snowball-naive-snowball-full -> snowball-naive,
    # snowball-naive-snowball-full-r4 -> snowball-r4-naive.
    if arm.endswith("-snowball-full-r5"):
        arm = "snowball-r5-" + arm.removeprefix("snowball-").removesuffix("-snowball-full-r5")
    if arm.endswith("-snowball-full-r4"):
        arm = "snowball-r4-" + arm.removeprefix("snowball-").removesuffix("-snowball-full-r4")
    arm = arm.removesuffix("-snowball-full")
    if f"-{FAMILY_TAGS[family_of(arm)]}-" not in run_name:
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


def eval_frame(data: dict[str, dict]) -> pd.DataFrame:
    """Tidy per-eval-point rows: one row per (arm, bin) score, plus grade-weighted.

    ``smoothed`` is an EMA over each (arm, bin) series in token order; raw
    scores stay in ``score`` so charts can show both.
    """
    rows = []
    for arm, d in data.items():
        for r in d["evals"].values():
            base = {
                "arm": arm,
                "family": family_of(arm),
                "sampler": sampler_of(arm),
                "step": r.get(STEP_KEY),
                "tokens_m": r["cumulative_tokens"] / 1e6,
            }
            for v, grade in VAL_GRADES.items():
                score = r.get(f"eval/{v}/pass_at_1")
                if score is not None:
                    rows.append({**base, "bin": v, "grade": grade, "score": score})
            weighted = grade_weighted_score(r)
            if weighted is not None:
                rows.append({**base, "bin": "grade-weighted", "grade": math.nan, "score": weighted})
    frame = pd.DataFrame(rows).sort_values("tokens_m")
    frame["smoothed"] = frame.groupby(["arm", "bin"])["score"].transform(lambda s: s.ewm(alpha=EVAL_ALPHA).mean())
    return frame


def present_samplers(frame: pd.DataFrame) -> list[str]:
    return [s for s in SAMPLER_ORDER if s in set(frame["sampler"])]


def sampler_palette() -> dict[str, tuple]:
    return dict(zip(SAMPLER_ORDER, sns.color_palette("colorblind", len(SAMPLER_ORDER)), strict=True))


def draw_arm_lines(ax, frame: pd.DataFrame, x: str, palette: dict[str, tuple]) -> None:
    """Raw series faintly under the EMA line, one color per sampler."""
    for sampler in present_samplers(frame):
        g = frame[frame["sampler"] == sampler]
        ax.plot(g[x], g["score"], color=palette[sampler], alpha=0.2, linewidth=1)
        ax.plot(g[x], g["smoothed"], color=palette[sampler], linewidth=2.2, label=sampler)


def plot_headline(frame: pd.DataFrame, family: str, path: Path) -> None:
    sub = frame[(frame["family"] == family) & (frame["bin"] == "grade-weighted")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    draw_arm_lines(ax, sub, "tokens_m", sampler_palette())
    ax.set_xlabel("generated tokens (M)")
    ax.set_ylabel("grade-weighted pass@1")
    ax.set_title(f"{FAMILY_LABELS[family]}: grade-weighted validation pass@1")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_grade_breakout(frame: pd.DataFrame, family: str, x: str, xlabel: str, path: Path) -> None:
    """One panel per validation bin, ordered by grade, shared axes.

    Shared y makes plateaus at the easy grades and floors at the hard grades
    directly comparable across panels.
    """
    sub = frame[(frame["family"] == family) & frame["grade"].notna()]
    palette = sampler_palette()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    for ax, bin_name in zip(axes.flat, BIN_ORDER, strict=True):
        draw_arm_lines(ax, sub[sub["bin"] == bin_name], x, palette)
        ax.set_title(f"grade {VAL_GRADES[bin_name]}: {bin_name}", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("pass@1")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"{FAMILY_LABELS[family]}: validation pass@1 by grade", y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sustained_crossing(g: pd.DataFrame) -> float:
    """Tokens at the first raw score >= FRONTIER_THRESHOLD held for two
    consecutive evals (a final-eval crossing has no successor and counts), so a
    single noisy point is not attainment yet the final frontier grade agrees
    with the summary. NaN when never crossed."""
    scores = g["score"].tolist()
    tokens = g["tokens_m"].tolist()
    for i, score in enumerate(scores):
        if score >= FRONTIER_THRESHOLD and (i == len(scores) - 1 or scores[i + 1] >= FRONTIER_THRESHOLD):
            return tokens[i]
    return math.nan


def attainment_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """First sustained FRONTIER_THRESHOLD crossing per (arm, bin), in tokens.

    Never-crossing combinations keep a NaN, which barplot renders as an
    absent bar.
    """
    rows = []
    graded = frame[frame["grade"].notna()]
    for (family, sampler, bin_name), g in graded.groupby(["family", "sampler", "bin"]):
        rows.append(
            {
                "family": family,
                "sampler": sampler,
                "bin": bin_name,
                "tokens_m": sustained_crossing(g.sort_values("tokens_m")),
            }
        )
    return pd.DataFrame(rows)


def plot_attainment(attainment: pd.DataFrame, family: str, path: Path) -> None:
    sub = attainment[attainment["family"] == family]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.barplot(
        data=sub,
        x="bin",
        y="tokens_m",
        hue="sampler",
        order=BIN_ORDER,
        hue_order=present_samplers(sub),
        palette=sampler_palette(),
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("generated tokens (M) to attain grade")
    ax.set_title(f"{FAMILY_LABELS[family]}: tokens to reach pass@1 >= {FRONTIER_THRESHOLD} (no bar: never reached)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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


def smoothed_points(points: dict[int, float], alpha: float) -> tuple[list[int], list[float]]:
    xs = sorted(points)
    ys: list[float] = []
    for x in xs:
        ys.append(points[x] if not ys else alpha * points[x] + (1 - alpha) * ys[-1])
    return xs, ys


def plot_curriculum(series: dict[str, dict[int, float]], arm: str, metric: str, path: Path) -> None:
    """Per-train-bin trajectories colored dark-to-light by grade order.

    Weights use a log scale: the interesting failure mode is starvation at the
    renormalization floor, invisible on a linear axis next to dominant bins.
    """
    bins = sorted(series)
    colors = sns.color_palette("viridis", len(bins))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for color, bin_name in zip(colors, bins, strict=True):
        xs, ys = smoothed_points(series[bin_name], CURRICULUM_ALPHA)
        ax.plot(xs, ys, color=color, linewidth=1.8, label=bin_name)
    if metric == "weight":
        ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(metric)
    ax.set_title(f"{arm}: curriculum bin {metric}")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def velocity_samples(pass_rates: dict[str, dict[int, float]]) -> list[tuple[float, float]]:
    """(pass_rate, d pass_rate / d step) pairs from consecutive logged points per bin."""
    samples = []
    for points in pass_rates.values():
        steps = sorted(points)
        for s1, s2 in itertools.pairwise(steps):
            samples.append((points[s1], (points[s2] - points[s1]) / (s2 - s1)))
    return samples


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
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter([p for p, _ in samples], [v for _, v in samples], s=5, alpha=0.15, label="bin transitions")
    ax.plot(mids, means, "o-", color="black", linewidth=2, label="binned mean")
    if means:
        # Rare large per-step jumps stretch the raw scatter over ~20x the
        # binned-mean range; clip the axis so the mean-vs-reference comparison
        # stays legible.
        peak = max(means)
        ax.set_ylim(-peak, 3 * peak)
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
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


@click.group(help=__doc__)
def main() -> None:
    logging.basicConfig(level=logging.INFO)


@main.command()
@click.option("--out", type=click.Path(path_type=Path), required=True)
def charts(out: Path) -> None:
    """Render report charts and a JSON summary from W&B histories."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
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

    frame = eval_frame(data)
    attainment = attainment_frame(frame)
    for family in FAMILY_LABELS:
        if frame[frame["family"] == family].empty:
            continue
        plot_headline(frame, family, out / f"{family}-grade-weighted-tokens.png")
        plot_grade_breakout(frame, family, "tokens_m", "generated tokens (M)", out / f"{family}-grades-tokens.png")
        plot_grade_breakout(frame, family, "step", "training step", out / f"{family}-grades-steps.png")
        plot_attainment(attainment, family, out / f"{family}-attainment.png")

    all_velocity_samples = []
    for arm, runs in by_arm.items():
        pass_rates = curriculum_series(runs, "pass_rate")
        all_velocity_samples.extend(velocity_samples(pass_rates))
        for metric, series in (("weight", curriculum_series(runs, "weight")), ("pass_rate", pass_rates)):
            if series:
                plot_curriculum(series, arm, metric, out / f"curriculum-{arm}-{metric}.png")
    if all_velocity_samples:
        plot_velocity(all_velocity_samples, group_size=8, path=out / "velocity-vs-pass-rate.png")

    logger.info("Wrote charts and summary to %s", out)


@main.command()
def evals() -> None:
    """Print aggregated Evalchemy metrics for every curriculum-RL arm."""
    for record in list_records(CW_RECORDS_PREFIX):
        # Model names either start with the experiment name or are
        # owner-prefixed (e.g. "power-curriculum-rl-naive").
        if f"{EXPERIMENT_NAME}-" not in record.model.name:
            continue
        if record.status is not RunStatus.SUCCEEDED:
            print(f"== {record.model.name} {record.run_id} {record.status}")
            continue
        for task, metrics in record.metrics.items():
            print(f"== {record.model.name} {task} {json.dumps(metrics)}")


@main.command(name="pool-stats")
@click.option("--version", required=True)
def pool_stats(version: str) -> None:
    """Print per-bin row counts and grades for one built pool version."""
    for filename in ("train.parquet", "validation.parquet"):
        path = prefix_join(prefix_join(POOL_ROOT, version), filename)
        with StoragePath(path).open("rb") as handle:
            table = pq.read_table(handle, columns=["extra_info"])
        counts: Counter[tuple[int, str]] = Counter()
        for info in table.column("extra_info").to_pylist():
            counts[(int(info["grade"]), str(info["data_source"]))] += 1
        print(f"== {filename}")
        for (grade, source), n in sorted(counts.items()):
            print(json.dumps({"grade": grade, "bin": source, "rows": n}))


def record_stats(rec: dict, reasons: tuple[str, ...]) -> dict:
    resp = rec.get("response", {})
    text = resp.get("text") or ""
    extras = json.dumps(rec.get("trajectory", {}).get("environment_extras"))
    bin_token = next((t for t in BIN_TOKENS if t in extras), "other")
    return {
        "step": rec.get("global_step") or 0,
        "bin": bin_token,
        # Failures and truncations are retained mandatorily, so the
        # non-mandatory remainder is a hash sample of successful terminating
        # rollouts only. None when the record id is missing from the ledger.
        "sampled": None if not reasons else "mandatory" not in reasons,
        "truncated": resp.get("stop_reason") == "length",
        "think": any(t in text for t in THINK_TOKENS),
        "answer_line": bool(ANSWER_LINE.search(text[-TAIL_CHARS:])),
        "boxed": "\\boxed{" in text[-TAIL_CHARS:],
        "chars": len(text),
    }


@main.command()
@click.argument("arm")
@click.argument("version")
def trajectories(arm: str, version: str) -> None:
    """Aggregate retained-trajectory statistics for one arm.

    Reports, per 20-step bucket and per bin: truncation rate
    (stop_reason=length), canonical think-token usage, graded answer-line
    compliance, and \\boxed{} usage.
    """
    out = f"s3://marin-us-east-02a/marin/users/power/checkpoints/curriculum-rl/{arm}/{version}"
    root = temporary_storage_base_path(out, ttl_days=14, category="skyrl")
    traj_root = prefix_join(prefix_join(root, "attempts"), "trajectories")
    ledger = json.loads(StoragePath(prefix_join(traj_root, "_retention_ledger.json")).read_text())
    reasons_by_id = {rid: tuple(entry.get("reasons", ())) for rid, entry in ledger.get("records", {}).items()}
    archives = sorted(ledger.get("archives", {}))
    if len(archives) > MAX_ARCHIVES:
        stride = len(archives) / MAX_ARCHIVES
        sampled = [archives[int(i * stride)] for i in range(MAX_ARCHIVES)]
        print(f"sampling {MAX_ARCHIVES} of {len(archives)} archives (even stride)")
    else:
        sampled = archives
        print(f"reading all {len(archives)} archives")

    rows = []
    for apath in sampled:
        payload = StoragePath(prefix_join(traj_root, apath)).read_bytes()
        try:
            zf = zipfile.ZipFile(std_io.BytesIO(payload))
        except zipfile.BadZipFile as exc:
            print("archive", apath, "unreadable:", exc)
            continue
        for name in zf.namelist():
            if not name.endswith(".json.gz"):
                continue
            record_id = name.rsplit("/", 1)[-1].removesuffix(".json.gz")
            reasons = reasons_by_id.get(record_id, ())
            rows.append(record_stats(json.loads(gzip.decompress(zf.read(name))), reasons))
    matched = sum(1 for r in rows if r["sampled"] is not None)
    print(f"records: {len(rows)} (ledger reason matched for {matched})")

    def bucket(rows_subset: list[dict], label: str) -> None:
        n = len(rows_subset)
        if not n:
            return
        trunc = sum(r["truncated"] for r in rows_subset) / n
        think = sum(r["think"] for r in rows_subset) / n
        ans = sum(r["answer_line"] for r in rows_subset) / n
        boxed = sum(r["boxed"] for r in rows_subset) / n
        chars = sum(r["chars"] for r in rows_subset) / n
        print(
            f"{label:>16} n={n:<6} trunc={trunc:.3f} think={think:.3f} "
            f"answer_line={ans:.3f} boxed={boxed:.3f} avg_chars={chars:,.0f}"
        )

    for label, subset in (
        ("success stream (hash-sampled)", [r for r in rows if r["sampled"] is True]),
        ("failure/trunc stream (mandatory)", [r for r in rows if r["sampled"] is False]),
    ):
        print(f"\n==== {label} ====")
        by_step = defaultdict(list)
        for r in subset:
            by_step[r["step"] // STEP_BUCKET * STEP_BUCKET].append(r)
        for k in sorted(by_step):
            bucket(by_step[k], f"steps {k}-{k + STEP_BUCKET - 1}")
        by_bin = defaultdict(list)
        for r in subset:
            by_bin[r["bin"]].append(r)
        for k in sorted(by_bin):
            bucket(by_bin[k], k)


if __name__ == "__main__":
    main()
