# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""An experiment's final ``CONFIG.metric`` from W&B, and each variant's verdict against its same-seed control.

The verdict is a fixed rule on Δ = variant - control (same seed, same commit): Δ ≤ -``CONFIG.noise`` is a win,
Δ ≥ +``CONFIG.noise`` a loss, anything between is noise.
"""

from dataclasses import dataclass
from enum import StrEnum

import wandb

from experiments.belay.config import CONFIG
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


def verdict(delta: float) -> Verdict:
    if delta <= -CONFIG.noise:
        return Verdict.WIN
    if delta >= CONFIG.noise:
        return Verdict.LOSS
    return Verdict.NOISE


def overall_verdict(verdicts: list[Verdict]) -> Verdict:
    """One verdict for the experiment: a win if any variant won, a loss if all lost, otherwise noise."""
    if Verdict.WIN in verdicts:
        return Verdict.WIN
    if verdicts and all(v is Verdict.LOSS for v in verdicts):
        return Verdict.LOSS
    return Verdict.NOISE


def experiment_results(spec: ExperimentSpec) -> list[RunResult]:
    """Final loss of every run in ``spec`` from W&B, with deltas and verdicts; refuses if any run lacks one."""
    api = wandb.Api()
    finals = {}
    for run in spec.runs.values():
        name = run_name(spec.experiment, run.name)
        matches = list(api.runs(CONFIG.wandb_project, filters={"display_name": name}, order="-created_at"))
        if not matches or matches[0].state != "finished" or matches[0].summary.get(CONFIG.metric) is None:
            raise SystemExit(f"refusing: {name} has no finished W&B run with {CONFIG.metric}")
        finals[run.name] = (int(matches[0].summary["_step"]), float(matches[0].summary[CONFIG.metric]))
    controls = {r.seed: finals[r.name][1] for r in spec.runs.values() if r.control}
    results = []
    for run in spec.runs.values():
        step, loss = finals[run.name]
        if run.seed not in controls:
            raise SystemExit(f"refusing: {spec.experiment} has no control at seed {run.seed}")
        delta = None if run.control else round(loss - controls[run.seed], 4)
        results.append(
            RunResult(
                name=run_name(spec.experiment, run.name),
                variant=run.variant,
                seed=run.seed,
                control=run.control,
                step=step,
                loss=loss,
                delta=delta,
                verdict=None if delta is None else verdict(delta),
            )
        )
    return results
