# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay builder controls through RolloutEngine and compare each grade with its expectation.

A control runs as an ordinary trial whose model is a ``ScriptedModel``: it answers each request
with the control's next assistant turn. A workspace control is delivered the way an agent would
leave it: one scripted shell call per file (``spec.controls.workspace_turn``: decode the bytes into
the path and set its mode, run by the shell tool as the task machine's ``MachineRuntimeSpec.user``,
after the setup commands), then a final reply (``WORKSPACE_REPLY``). The task's own grader then runs on that
machine state. Grading is the engine's, unchanged.

A control replays under the engine's messages trace, as every trial does (``lowering.lower``):
a scripted turn carries no token ids, the engine skips its prefix, evidence and log-probability
checks, and the attempt records no token evidence, which a control never needs.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from rolloutengine.models import TaskRolloutSpec
from taskcompendium.models import TaskSpec

from taskforge.ledger.records import Ledger
from taskforge.spec.controls import (
    Control,
    ScriptedModel,
    Workspace,
    candidate_turns,
    context_assistant_turns,
    validate_controls,
    wire_message,
)
from taskforge.stage50_validate.outcome import Graded, Outcome, TrialKind
from taskforge.stage50_validate.trials import EngineSettings, RetryBackoff, TrialPlan, run_trial

WORKSPACE_REPLY = "The workspace is ready for grading."


class ControlVerdict(StrEnum):
    MET = "met"
    VIOLATED = "violated"
    UNGRADED = "ungraded"
    """The replay produced no grade, so the control says nothing about the grader."""


@dataclass(frozen=True)
class ControlOutcome:
    control: Control
    outcome: Outcome
    verdict: ControlVerdict


@dataclass(frozen=True)
class ControlPlan:
    """Which workpiece the controls belong to, their deadlines, how they retry, and where they are recorded.

    Each control is one ``CONTROL`` trial named by its id. ``first_attempts`` maps a control id to
    the number of its first attempt file and ledger step, as ``TrialPlan.first_attempt`` does for a
    trial: a control re-entered after earlier attempts passes the count already on disk. A control
    it does not name starts at 0.
    """

    workpiece_id: str
    round: int
    max_retries: int
    retry_backoff: RetryBackoff
    evidence_dir: Path
    ledger: Ledger
    first_attempts: Mapping[str, int]

    def __post_init__(self) -> None:
        if any(attempt < 0 for attempt in self.first_attempts.values()):
            raise ValueError("A control plan needs non-negative first attempts")

    def trial_plan(self, control_id: str) -> TrialPlan:
        """The ``CONTROL`` trial plan of the control named ``control_id``."""
        return TrialPlan(
            workpiece_id=self.workpiece_id,
            round=self.round,
            kind=TrialKind.CONTROL,
            k=1,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
            evidence_dir=self.evidence_dir,
            ledger=self.ledger,
            first_attempt=self.first_attempts.get(control_id, 0),
        )


def control_turns(control: Control) -> tuple[dict[str, Any], ...]:
    """The assistant turns a control replays."""
    if isinstance(control.payload, Workspace):
        return candidate_turns(WORKSPACE_REPLY, control.payload.files)
    return tuple(wire_message(turn) for turn in control.payload.turns)


def check_replayable(task: TaskSpec, controls: Sequence[Control], max_turns: int | None) -> None:
    """Refuse a control set that cannot be replayed against ``task`` within ``max_turns`` turns.

    Raises:
        ValueError: ``controls`` is not a valid control set for ``task`` (``validate_controls``) or a
            control needs more than ``max_turns`` turns.
    """
    validate_controls(task, controls)
    too_long = [control.id for control in controls if max_turns is not None and len(control_turns(control)) > max_turns]
    if too_long:
        raise ValueError(f"Controls {too_long} need more than max_turns={max_turns} turns")


async def replay(
    spec: TaskRolloutSpec,
    controls: Sequence[Control],
    plan: ControlPlan,
    settings: EngineSettings,
) -> list[ControlOutcome]:
    """Replay every control concurrently as a ``CONTROL`` trial of ``spec``, named by its id.

    Raises:
        ValueError: ``check_replayable`` refuses ``controls``, or ``plan.first_attempts`` names a control not
            in ``controls``.
    """
    task = spec.task
    check_replayable(task, controls, settings.limits.max_turns)
    unknown = sorted(set(plan.first_attempts) - {control.id for control in controls})
    if unknown:
        raise ValueError(f"First attempts name controls {unknown} that are not replayed")
    async with asyncio.TaskGroup() as group:
        runs = [
            group.create_task(replay_control(spec, control, plan.trial_plan(control.id), settings))
            for control in controls
        ]
    return [run.result() for run in runs]


async def replay_control(
    spec: TaskRolloutSpec, control: Control, plan: TrialPlan, settings: EngineSettings
) -> ControlOutcome:
    """Replay one control of a set ``check_replayable`` accepted as the ``CONTROL`` trial of ``plan``, named by
    its id."""
    model = ScriptedModel(control_turns(control), context_assistant_turns(spec.task))
    return control_outcome(control, await run_trial(spec, plan, settings, model, control.id))


def control_outcome(control: Control, outcome: Outcome) -> ControlOutcome:
    """``outcome`` judged against ``control.expect``."""
    if not isinstance(outcome, Graded):
        return ControlOutcome(control, outcome, ControlVerdict.UNGRADED)
    met = control.expect.met_by(outcome.grade)
    return ControlOutcome(control, outcome, ControlVerdict.MET if met else ControlVerdict.VIOLATED)
