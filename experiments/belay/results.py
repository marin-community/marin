# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An experiment's final metric from W&B, and each variant's verdict against its same-seed control.

The verdict is a fixed rule per Track (``config.TRACKS``): Δ against the same-seed, same-commit control, with
Δ ≤ -noise a win, Δ ≥ +noise a loss and anything between noise.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import median

import wandb

from experiments.belay.config import CONFIG, Baseline, ProjectionBasis, Rule


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


# The hero-mfu Track's metric: wall-clock seconds per step over steps [6, 24) of its 30-step run, as
# belay_hero_mfu.py --accept measures it. No run logs it; it is computed from each step's log time.
WALL_STEP_TIME = "s/step"
WALL_STEPS = (6, 24)


def wall_step_time(timestamps: dict[int, float]) -> float:
    """Seconds per step between the first and last logged steps inside ``WALL_STEPS`` (step -> log time)."""
    steps = sorted(s for s in timestamps if WALL_STEPS[0] <= s < WALL_STEPS[1])
    if len(steps) < 2:
        raise SystemExit(f"refusing: fewer than two logged steps in {WALL_STEPS}")
    return (timestamps[steps[-1]] - timestamps[steps[0]]) / (steps[-1] - steps[0])


@dataclass(frozen=True)
class Projection:
    """A complete ladder carried to hero compute (``ladder_projection``)."""

    delta: float
    """The variant's projected Paloma macro loss minus the recipe's, at hero compute."""
    multiplier: float
    """The compute the recipe would need to reach the variant's projected loss, over the hero compute."""
    alpha: float
    """The variant's fitted exponent (the recipe's is the basis ``alpha``)."""
    extrapolation: float
    """Hero compute over the largest rung's."""


def ladder_projection(deltas: dict[str, float], basis: ProjectionBasis) -> Projection:
    """Fit the variant's rungs (the recipe's fitted loss plus each rung's Paloma macro Δ, at the rung's FLOPs) with the
    recipe's form, floor + A·C^-alpha, by least squares in log space, and compare the two curves at hero compute."""
    if set(deltas) != set(basis.rung_flops):
        raise ValueError(f"a projection needs every rung {sorted(basis.rung_flops)}, got {sorted(deltas)}")

    def recipe(flops: float) -> float:
        return basis.floor + basis.a * flops**-basis.alpha

    xs = [math.log(basis.rung_flops[r]) for r in deltas]
    ys = [math.log(recipe(basis.rung_flops[r]) + d - basis.floor) for r, d in deltas.items()]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sum((x - mx) ** 2 for x in xs)
    variant = basis.floor + math.exp(my - slope * mx) * basis.hero_flops**slope
    multiplier = ((variant - basis.floor) / basis.a) ** (-1 / basis.alpha) / basis.hero_flops
    return Projection(
        delta=variant - recipe(basis.hero_flops),
        multiplier=multiplier,
        alpha=-slope,
        extrapolation=basis.hero_flops / max(basis.rung_flops.values()),
    )


def measure(name: str, metric: str, skip: int = SKIP_STEPS) -> Measure:
    """Steps, final ``metric``, median s/step and tokens of the finished W&B run ``name``; refuses an unfinished one."""
    matches = list(wandb.Api().runs(CONFIG.wandb_project, filters={"display_name": name}, order="-created_at"))
    if not matches or matches[0].state != "finished":
        raise SystemExit(f"refusing: {name} has no finished W&B run")
    run = matches[0]
    durations = [row[STEP_TIME] for row in run.scan_history(keys=["_step", STEP_TIME]) if row["_step"] >= skip]
    if not durations:
        raise SystemExit(f"refusing: {name} logged no {STEP_TIME} after step {skip}")
    if metric == WALL_STEP_TIME:
        rows = run.scan_history(keys=["global_step", "_timestamp"])
        loss = wall_step_time({int(row["global_step"]): row["_timestamp"] for row in rows})
    else:
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
