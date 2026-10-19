# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The harness contract: the agent loop of one task attempt, and the recorder that it calls the model through.

A harness runs the loop: it sends prompts, runs tools, and decides when the attempt ends. It calls the model
through its ``RolloutRecorder``, which records the exact tokens and applies the turn and token limits.
The harness does not grade: the engine grades the rollout with the task's grader after the harness ends.
"""

from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from shellbox.machine import Machine
from taskcompendium.models import TaskSpec

from rolloutengine.models import EventSink, ModelTurn, Prompt, RolloutLimits


class HarnessCleanup(Protocol):
    """The attempt's cleanup scope: it closes resources after the record is committed and graded."""

    def push(self, name: str, close: Callable[[], Coroutine[Any, Any, None]]) -> None:
        """Run ``close`` when the attempt ends, within ``RolloutLimits.cleanup_timeout``.

        A failure does not change the recorded turns or the grade. ``Rollout.metrics`` counts it, and
        ``run_task`` also adds it to the grade diagnostics.
        """
        ...


@dataclass(frozen=True)
class HarnessContext:
    """One attempt's inputs for a harness.

    ``machine`` is the prepared task machine, or ``None`` when the attempt has none. A harness may report
    its own events, such as tool calls, to ``events``, and registers the close of its resources with
    ``cleanup``. ``run_id`` identifies the attempt.
    """

    task: TaskSpec
    machine: Machine | None
    events: EventSink
    cleanup: HarnessCleanup
    run_id: str


class RolloutRecorder(Protocol):
    """The model, as the harness of one attempt calls it. It records the exact tokens and applies the limits.

    One request can be open at a time. A request that is cancelled while the model generates changes
    nothing, so the harness can send it again. If the harness ends before its next request, the record
    keeps the cancelled request without an answer. When a limit ends the attempt, the next ``start`` or
    ``reply`` raises ``LimitReached``, and the harness lets it propagate. A failed model request raises
    ``RolloutEnded`` with the model's error as its cause. A harness that catches a refusal and asks again
    runs until a turn or attempt deadline. Each request after the end of the attempt yields before it
    raises, so a harness that asks again in a loop does not stop other attempts.
    """

    @property
    def limits(self) -> RolloutLimits:
        """The attempt's limits, for example to tell the model that one turn is left."""
        ...

    async def start(self, prompt: Prompt) -> ModelTurn:
        """Start a new conversation with ``prompt`` and return the model's turn.

        A later ``start`` replaces the chat, for example after the harness makes its history shorter or
        changes its tools. The rollout keeps the earlier conversations.
        """
        ...

    async def reply(
        self, observations: tuple[dict[str, Any], ...], *, token_ids: tuple[int, ...] | None = None
    ) -> ModelTurn:
        """Send ``observations`` after the model's last message, and return the next turn.

        ``token_ids`` pins the exact tokens of the next prompt. They must start with the tokens served so far:
        under ``TraceMode.EXACT_TOKENS``, other tokens break the token contract before the request is sent.
        """
        ...

    def record(
        self,
        *,
        observations: tuple[dict[str, Any], ...] = (),
        data: object = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Add to the step of the last turn: observations that the model never receives, data, and metrics.

        ``data`` is for the trainer, for example a reward, and the engine does not read it. A turn with
        recorded observations ends the conversation's turns: ``reply`` refuses it. After ``LimitReached`` or a
        failed model request, the harness can still add data and metrics, for example a final reward, to the
        step of the last turn until the record is final. If no turn exists, ``record`` raises the refusal again.
        """
        ...


class Harness(Protocol):
    """The agent loop for a task: what the model sees, which tools run, and when the attempt ends."""

    def check(self, task: TaskSpec, limits: RolloutLimits) -> None:
        """Raise an error if this harness cannot run ``task`` under ``limits``, before any machine starts."""
        ...

    async def run(self, context: HarnessContext, recorder: RolloutRecorder) -> None:
        """Run one attempt. Return when the task is done, and let ``RolloutEnded`` and ``LimitReached`` propagate."""
        ...
