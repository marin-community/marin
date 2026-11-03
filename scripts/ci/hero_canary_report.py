# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score a hero GB200 canary against recent scheduled runs from its checkpoint.

Missing steps or eval, or a loss above its ceiling, fail the report. A step-time change beyond
STEP_TIME_WARN_FRACTION emits a GitHub warning and a regression output for Slack.
"""

import dataclasses
import os
import re
import statistics
import sys

import click
import wandb

# Scheduled runs are hero-canary-<run>-<attempt>; manual dispatches use hero-canary-manual-.
SCHEDULED_RUN_ID = re.compile(r"^hero-canary-\d+-\d+$")
BASELINE_RUNS = 5
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


def failure_reasons(score: CanaryScore, expected_steps: int, max_train_loss: float, max_eval_loss: float) -> list[str]:
    reasons = []
    if score.steps_logged < expected_steps:
        reasons.append(f"logged {score.steps_logged} of {expected_steps} steps")
    if score.train_loss > max_train_loss:
        reasons.append(f"final train loss {score.train_loss:.4f} is above {max_train_loss}")
    if score.eval_loss is None:
        reasons.append("the dropless eval did not report")
    elif score.eval_loss > max_eval_loss:
        reasons.append(f"dropless eval macro loss {score.eval_loss:.4f} is above {max_eval_loss}")
    return reasons


def _change(current: float, previous: float) -> str:
    return f"{(current - previous) / previous * 100:+.2f}"


def baseline_score(scores: list[CanaryScore]) -> CanaryScore | None:
    """The per-metric median of earlier runs, or None when there are none."""
    if not scores:
        return None
    eval_losses = [score.eval_loss for score in scores if score.eval_loss is not None]
    return CanaryScore(
        run_id=f"median of {len(scores)} runs",
        git_commit=None,
        steps_logged=min(score.steps_logged for score in scores),
        step_time=statistics.median(score.step_time for score in scores),
        mfu=statistics.median(score.mfu for score in scores),
        tokens_per_second=statistics.median(score.tokens_per_second for score in scores),
        peak_hbm_gib=statistics.median(score.peak_hbm_gib for score in scores),
        drop_fraction=statistics.median(score.drop_fraction for score in scores),
        train_loss=statistics.median(score.train_loss for score in scores),
        eval_loss=statistics.median(eval_losses) if eval_losses else None,
    )


def step_time_regression(current: CanaryScore, baseline: CanaryScore | None) -> str | None:
    """A description when median step time moved more than STEP_TIME_WARN_FRACTION from the baseline."""
    if baseline is None:
        return None
    if abs(current.step_time - baseline.step_time) / baseline.step_time <= STEP_TIME_WARN_FRACTION:
        return None
    return f"{_change(current.step_time, baseline.step_time)}% against the {baseline.run_id}"


def report_markdown(current: CanaryScore, baseline: CanaryScore | None) -> str:
    rows = [
        ("Median step time (s)", "step_time", "{:.3f}"),
        ("Median MFU (%)", "mfu", "{:.2f}"),
        ("Median tokens/s", "tokens_per_second", "{:,.0f}"),
        ("Peak HBM (GiB)", "peak_hbm_gib", "{:.1f}"),
        ("Mean drop fraction", "drop_fraction", "{:.2e}"),
        ("Final train loss", "train_loss", "{:.4f}"),
        ("Dropless eval macro loss", "eval_loss", "{:.4f}"),
    ]
    baseline_label = f"the {baseline.run_id} from this checkpoint" if baseline else "none yet"
    lines = [
        "### Hero GB200 canary",
        "",
        f"This run: `{current.run_id}` (`{current.git_commit}`). Baseline: {baseline_label}.",
        "",
        "| Metric | This run | Baseline | Change (%) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, field, fmt in rows:
        value = getattr(current, field)
        prev_value = getattr(baseline, field) if baseline else None
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


def _baseline_runs(api: wandb.Api, entity: str, project: str, run_id: str, checkpoint: str) -> list:
    runs = api.runs(
        f"{entity}/{project}",
        filters={"display_name": {"$regex": SCHEDULED_RUN_ID.pattern}, "state": "finished"},
        order="-created_at",
        per_page=50,
    )
    matching = []
    for run in runs:
        if run.id == run_id or not SCHEDULED_RUN_ID.match(run.id):
            continue
        if _config_value(run.config, CHECKPOINT_CONFIG_PATH) != checkpoint:
            continue
        matching.append(run)
        if len(matching) == BASELINE_RUNS:
            break
    return matching


@click.command()
@click.option("--run-id", required=True)
@click.option("--entity", required=True)
@click.option("--project", required=True)
@click.option("--start-step", type=int, required=True, help="Step of the restored checkpoint.")
@click.option("--steps", type=int, required=True, help="Updates the canary trains after the restore.")
@click.option("--checkpoint", required=True, help="Restored checkpoint path; only runs from it are compared.")
@click.option("--max-train-loss", type=float, required=True, help="Fail when the final train loss is higher.")
@click.option("--max-eval-loss", type=float, required=True, help="Fail when the dropless eval macro loss is higher.")
@click.option("--github-output", is_flag=True, help="Write regression and its description to $GITHUB_OUTPUT.")
def main(
    run_id: str,
    entity: str,
    project: str,
    start_step: int,
    steps: int,
    checkpoint: str,
    max_train_loss: float,
    max_eval_loss: float,
    github_output: bool,
) -> None:
    api = wandb.Api(timeout=60)
    current = _load_score(api.run(f"{entity}/{project}/{run_id}"), start_step)
    baseline = baseline_score(
        [_load_score(run, start_step) for run in _baseline_runs(api, entity, project, run_id, checkpoint)]
    )

    markdown = report_markdown(current, baseline)
    print(markdown)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(markdown)

    regression = step_time_regression(current, baseline)
    if regression:
        print(f"::warning title=Hero canary step time::{regression}")
    if github_output:
        path = os.environ.get("GITHUB_OUTPUT")
        if not path:
            raise click.UsageError("--github-output needs $GITHUB_OUTPUT to be set")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"regression={'true' if regression else 'false'}\nregression_detail={regression or ''}\n")
    if reasons := failure_reasons(current, steps, max_train_loss, max_eval_loss):
        print(f"Hero canary failed: {'; '.join(reasons)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
