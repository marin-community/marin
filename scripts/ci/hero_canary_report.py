# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a hero GB200 canary run from W&B and compare it with the previous run.

The canary restores a pinned hero checkpoint and trains a fixed number of updates.
This reads the run's per-step history, drops the warmup steps, and reports median
step time, MFU and throughput, peak HBM, drop fraction, final train loss and the
dropless eval loss. It compares them with the most recent earlier canary restored
from the same checkpoint, so a re-pinned checkpoint starts a new baseline.

Exits non-zero only when the run is incomplete (missing steps or eval). A step-time
change beyond STEP_TIME_WARN_FRACTION is reported as a GitHub warning, not a failure,
until repeated same-commit runs calibrate the noise.
"""

import dataclasses
import os
import statistics
import sys

import click
import wandb

RUN_ID_PREFIX = "hero-canary-"
WARMUP_STEPS = 4
# One run's 30-step median is within about ±0.5%; a difference of two runs is noisier.
STEP_TIME_WARN_FRACTION = 0.015
HISTORY_KEYS = (
    "_step",
    "throughput/duration",
    "throughput/mfu",
    "throughput/tokens_per_second",
    "memory/peak_gib",
    "moe/drop_fraction",
    "train/loss",
)
EVAL_LOSS_KEY = "eval_dropless/macro_loss"
CHECKPOINT_CONFIG_PATH = ("trainer", "trainer", "load_checkpoint_path")


@dataclasses.dataclass(frozen=True)
class CanaryScore:
    run_id: str
    git_commit: str | None
    steps_logged: int
    step_time: float
    mfu: float
    tokens_per_second: float
    peak_hbm_gib: float
    drop_fraction: float
    train_loss: float
    eval_loss: float | None


def score_run(run_id: str, git_commit: str | None, rows: list[dict], summary: dict, start_step: int) -> CanaryScore:
    """Score one run from its W&B history rows and summary.

    W&B steps are zero-based: the first update after restoring step N logs at _step N.
    """
    by_step = {row["_step"]: row for row in rows if row.get("throughput/duration") is not None}
    ordered = [by_step[step] for step in sorted(by_step)]
    steady = [row for row in ordered if row["_step"] >= start_step + WARMUP_STEPS]
    if not steady:
        raise ValueError(f"{run_id} logged no steps past warmup")
    return CanaryScore(
        run_id=run_id,
        git_commit=git_commit,
        steps_logged=len(ordered),
        step_time=statistics.median(row["throughput/duration"] for row in steady),
        mfu=statistics.median(row["throughput/mfu"] for row in steady),
        tokens_per_second=statistics.median(row["throughput/tokens_per_second"] for row in steady),
        peak_hbm_gib=max(row["memory/peak_gib"] for row in ordered),
        drop_fraction=statistics.mean(row["moe/drop_fraction"] for row in steady),
        train_loss=ordered[-1]["train/loss"],
        eval_loss=summary.get(EVAL_LOSS_KEY),
    )


def incomplete_reasons(score: CanaryScore, expected_steps: int) -> list[str]:
    reasons = []
    if score.steps_logged < expected_steps:
        reasons.append(f"logged {score.steps_logged} of {expected_steps} steps")
    if score.eval_loss is None:
        reasons.append("the dropless eval did not report")
    return reasons


def _change(current: float, previous: float) -> str:
    return f"{(current - previous) / previous * 100:+.2f}"


def report_markdown(current: CanaryScore, previous: CanaryScore | None) -> str:
    rows = [
        ("Median step time (s)", "step_time", "{:.3f}"),
        ("Median MFU (%)", "mfu", "{:.2f}"),
        ("Median tokens/s", "tokens_per_second", "{:,.0f}"),
        ("Peak HBM (GiB)", "peak_hbm_gib", "{:.1f}"),
        ("Mean drop fraction", "drop_fraction", "{:.2e}"),
        ("Final train loss", "train_loss", "{:.4f}"),
        ("Dropless eval macro loss", "eval_loss", "{:.4f}"),
    ]
    prev_label = f"`{previous.run_id}` (`{previous.git_commit}`)" if previous else "none"
    lines = [
        "### Hero GB200 canary",
        "",
        f"This run: `{current.run_id}` (`{current.git_commit}`). Previous run: {prev_label}.",
        "",
        "| Metric | This run | Previous | Change (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, field, fmt in rows:
        value = getattr(current, field)
        prev_value = getattr(previous, field) if previous else None
        shown = fmt.format(value) if value is not None else "missing"
        prev_shown = fmt.format(prev_value) if prev_value is not None else "-"
        change = _change(value, prev_value) if value is not None and prev_value else "-"
        lines.append(f"| {label} | {shown} | {prev_shown} | {change} |")
    return "\n".join(lines) + "\n"


def _config_value(config: dict, path: tuple[str, ...]):
    value = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _load_score(run, start_step: int) -> CanaryScore:
    rows = list(run.scan_history(keys=list(HISTORY_KEYS)))
    return score_run(run.id, run.config.get("git_commit"), rows, dict(run.summary), start_step)


def _previous_run(api: wandb.Api, entity: str, project: str, run_id: str, checkpoint: str):
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": f"^{RUN_ID_PREFIX}"}, "state": "finished"},
        order="-created_at",
        per_page=20,
    )
    for run in runs:
        if run.id != run_id and _config_value(run.config, CHECKPOINT_CONFIG_PATH) == checkpoint:
            return run
    return None


@click.command()
@click.option("--run-id", required=True)
@click.option("--entity", required=True)
@click.option("--project", required=True)
@click.option("--start-step", type=int, required=True, help="Step of the restored checkpoint.")
@click.option("--steps", type=int, required=True, help="Updates the canary trains after the restore.")
@click.option("--checkpoint", required=True, help="Restored checkpoint path; only runs from it are compared.")
def main(run_id: str, entity: str, project: str, start_step: int, steps: int, checkpoint: str) -> None:
    api = wandb.Api(timeout=60)
    current = _load_score(api.run(f"{entity}/{project}/{run_id}"), start_step)
    previous_run = _previous_run(api, entity, project, run_id, checkpoint)
    previous = _load_score(previous_run, start_step) if previous_run else None

    markdown = report_markdown(current, previous)
    print(markdown)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(markdown)

    if previous and abs(current.step_time - previous.step_time) / previous.step_time > STEP_TIME_WARN_FRACTION:
        print(
            f"::warning title=Hero canary step time::{_change(current.step_time, previous.step_time)}% "
            f"against {previous.run_id}"
        )
    if reasons := incomplete_reasons(current, steps):
        print(f"Hero canary incomplete: {'; '.join(reasons)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
