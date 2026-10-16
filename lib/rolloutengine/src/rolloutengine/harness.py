# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The harness contract: what the model sees and does in one task attempt.

The engine holds harnesses by name; ``TaskRolloutSpec.harness`` selects one. For each attempt the engine
calls ``check`` before any machine starts and ``open`` when the task machine is ready. It then drives
the episode that ``open`` returns: it sends the episode's prompt to the model and passes each model turn
to ``advance``, until the episode is done or a limit ends the attempt. A task whose grader is
``SessionGrader`` is graded by the harness's ``grade``; the engine grades every other task with the
task's grader.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from shellbox.machine import Machine
from taskcompendium.grading_result import GradeResult

from rolloutengine.contracts import EventSink, ModelTurn, Prompt, Rollout, Transition
from rolloutengine.spec import TaskRolloutSpec


@dataclass(frozen=True)
class HarnessContext:
    """One attempt's inputs for a harness.

    ``machine`` is the prepared task machine, or ``None`` when the spec selects none. A harness may
    report its own events, such as tool calls, to ``events``. ``run_id`` identifies the attempt.
    """

    spec: TaskRolloutSpec
    machine: Machine | None
    events: EventSink
    run_id: str


class Episode(Protocol):
    """One attempt of a harness: the first prompt, then the response to each model turn."""

    async def prepare(self) -> Prompt: ...

    async def advance(self, turn: ModelTurn) -> Transition: ...

    async def close(self) -> None: ...


def accept(spec: TaskRolloutSpec) -> None:
    """A check that accepts every spec."""


@dataclass(frozen=True)
class Harness[E: Episode]:
    """How the model side of a task runs.

    ``open`` starts one episode per attempt, for example an episode class. ``check`` rejects a spec
    before any machine starts. ``grade`` grades an episode's record for a ``SessionGrader`` task, for
    example an unbound episode method such as ``MathEpisode.grade``; a harness without it cannot run
    such tasks.
    """

    open: Callable[[HarnessContext], E]
    check: Callable[[TaskRolloutSpec], None] = accept
    grade: Callable[[E, Rollout], Awaitable[GradeResult]] | None = None
