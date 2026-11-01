# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An experiment's final metric from W&B, and each variant's verdict against its same-seed control.

The verdict is a fixed rule per Track (``config.TRACKS``): Δ against the same-seed, same-commit control, with
Δ ≤ -noise a win, Δ ≥ +noise a loss and anything between noise.
"""

from dataclasses import dataclass
from enum import StrEnum
from statistics import median

import wandb

from experiments.belay.config import CONFIG, TRACKS, Baseline, Rule
from experiments.belay.submit import ExperimentSpec, run_name


class Verdict(StrEnum):
    WIN = "win"
    NOISE = "noise"
    LOSS = "loss"


@dataclass(frozen=True)
class RunResult:
    name: str
    variant: str
    seed: int
    control: bool
    step: int
    loss: float
    delta: float | None
    """Variant minus its same-seed control; None for a control."""
    verdict: Verdict | None
    bpb: float | None = None


def change(variant: float, control: float, baseline: Baseline) -> float:
    """Δ of ``variant`` against ``control`` under the baseline's rule (lower is better)."""
    if baseline.rule is Rule.RELATIVE:
        return (variant - control) / control
    return variant - control


def verdict(delta: float, baseline: Baseline) -> Verdict:
    if delta <= -baseline.noise:
        return Verdict.WIN
    if delta >= baseline.noise:
        return Verdict.LOSS
    return Verdict.NOISE


def overall_verdict(verdicts: list[Verdict]) -> Verdict:
    """One verdict for the experiment: a win if any variant won, a loss if all lost, otherwise noise."""
    if Verdict.WIN in verdicts:
        return Verdict.WIN
    if verdicts and all(v is Verdict.LOSS for v in verdicts):
        return Verdict.LOSS
    return Verdict.NOISE


@dataclass(frozen=True)
class Measure:
    """One finished run as the comment's table shows it."""

    name: str
    steps: int
    loss: float
    s_step: float
    tokens: float | None
    bpb: float | None


STEP_TIME = "throughput/duration"
BPB = "eval/paloma/macro_bpb"
SKIP_STEPS = 5  # compile and warm-up steps left out of a run's median s/step


def measure(name: str, metric: str, skip: int = SKIP_STEPS) -> Measure:
    """Steps, final ``metric``, median s/step and tokens of the finished W&B run ``name``; refuses an unfinished one."""
    matches = list(wandb.Api().runs(CONFIG.wandb_project, filters={"display_name": name}, order="-created_at"))
    if not matches or matches[0].state != "finished":
        raise SystemExit(f"refusing: {name} has no finished W&B run")
    run = matches[0]
    durations = [row[STEP_TIME] for row in run.scan_history(keys=["_step", STEP_TIME]) if row["_step"] >= skip]
    if not durations:
        raise SystemExit(f"refusing: {name} logged no {STEP_TIME} after step {skip}")
    loss = run.summary.get(metric)
    return Measure(
        name=name,
        steps=int(run.summary["_step"]) + 1,
        loss=float("nan") if loss is None else float(loss),
        s_step=median(durations),
        tokens=run.summary.get("throughput/total_tokens"),
        bpb=run.summary.get(BPB),
    )


def compared(control: Measure, variant: Measure, baseline: Baseline) -> tuple[float, Verdict]:
    delta = round(change(variant.loss, control.loss, baseline), 4)
    return delta, verdict(delta, baseline)


@dataclass(frozen=True)
class AfterData:
    """What follows an equal-data stage: whether the equal-time stage runs, the final status if not, and the
    line the comment shows under stage 1."""

    run_equal_time: bool
    final_status: str | None
    line: str


def after_equal_data(time_change: float, stage_1: Verdict, tolerance: float) -> AfterData:
    """The fixed rule, from Δt = variant s/step / control s/step - 1.

    Within ``tolerance``, equal data is already equal time, so stage 1 is final. A slower variant only gets fewer
    steps at equal time, so it runs stage 2 only after a stage-1 win. A faster variant gets more steps, so it always
    runs stage 2.
    """
    percent = f"{abs(time_change):.0%}"
    if abs(time_change) <= tolerance:
        return AfterData(
            False,
            f"{stage_1} at equal time (step time within {tolerance:.0%})",
            f"Variant step time within {tolerance:.0%} of the control → equal data is equal time; final.",
        )
    if time_change > 0 and stage_1 is not Verdict.WIN:
        return AfterData(
            False,
            f"{stage_1} at equal data; slower, equal time not run",
            f"Variant {percent} slower and did not win at equal data → equal time not run.",
        )
    if time_change > 0:
        return AfterData(True, None, f"Variant {percent} slower and won at equal data → equal-time run.")
    return AfterData(True, None, f"Variant {percent} faster → equal-time run (more steps at equal time).")


def experiment_results(spec: ExperimentSpec) -> list[RunResult]:
    """Final loss of every run in ``spec`` from W&B, with deltas and verdicts; refuses if any run lacks one."""
    baseline = TRACKS[spec.track]
    api = wandb.Api()
    finals = {}
    for run in spec.runs.values():
        name = run_name(spec.experiment, run.name)
        matches = list(api.runs(CONFIG.wandb_project, filters={"display_name": name}, order="-created_at"))
        if not matches or matches[0].state != "finished" or matches[0].summary.get(baseline.metric) is None:
            raise SystemExit(f"refusing: {name} has no finished W&B run with {baseline.metric}")
        finals[run.name] = (int(matches[0].summary["_step"]), float(matches[0].summary[baseline.metric]))
    controls = {r.seed: finals[r.name][1] for r in spec.runs.values() if r.control}
    results = []
    for run in spec.runs.values():
        step, loss = finals[run.name]
        if run.seed not in controls:
            raise SystemExit(f"refusing: {spec.experiment} has no control at seed {run.seed}")
        delta = None if run.control else round(change(loss, controls[run.seed], baseline), 4)
        results.append(
            RunResult(
                name=run_name(spec.experiment, run.name),
                variant=run.variant,
                seed=run.seed,
                control=run.control,
                step=step,
                loss=loss,
                delta=delta,
                verdict=None if delta is None else verdict(delta, baseline),
            )
        )
    return results
