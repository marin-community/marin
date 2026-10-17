# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ManagedAgent: many agent attempts through one RolloutEngine (#10165).

A managed agent is a registered harness, which is the agent program, plus a model endpoint and limits. The engine
runs each attempt: it checks the spec, applies the limits through the recorder, records every conversation, grades
with the task's grader, and closes the attempt's resources. ``ManagedAgent`` adds only admission: it runs at most
``concurrency`` attempts at a time and returns one outcome for each spec, so that a failed attempt does not stop the
others. Work across processes, such as a Zephyr pipeline, gives each worker its own ``ManagedAgent``.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass

from rolloutengine.contracts import GradedRolloutInterrupted, Model, Rollout
from rolloutengine.engine import RolloutEngine
from rolloutengine.spec import RolloutLimits, TaskRolloutSpec, TraceMode
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)

AGENT_SOURCE = Source(dataset="managed-agent", revision="1", row="0", importer_revision="1")
"""A ``TaskSpec`` records where it comes from. An agent's work has no dataset, so all of it shares this source."""


def agent_task(task_id: str, instructions: str) -> TaskSpec:
    """An ungraded text task whose public context holds the instructions, for an agent that brings its own tools."""
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=instructions),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="A managed agent attempt has no grader"),
        source=AGENT_SOURCE,
    )


def agent_spec(task: TaskSpec, harness: str, limits: RolloutLimits) -> TaskRolloutSpec:
    """The engine spec of one attempt without machines. A chat endpoint renders the tokens, so the trace is messages."""
    return TaskRolloutSpec(
        task=task,
        harness=harness,
        task_machine=None,
        verifier_machine=None,
        limits=limits,
        trace=TraceMode.MESSAGES,
    )


@dataclass(frozen=True)
class AgentOutcome:
    """The result of one attempt. ``rollout`` and ``grade`` are ``None`` only if the engine raised before a record."""

    id: str
    rollout: Rollout | None
    grade: GradeResult | None
    failure: str | None = None

    @property
    def answer(self) -> str:
        """The last assistant message without tool calls, or ``""`` if the agent gave none."""
        conversations = () if self.rollout is None else self.rollout.conversations
        for conversation in reversed(conversations):
            for message in reversed(conversation.messages):
                if message.get("role") == "assistant" and not message.get("tool_calls") and message.get("content"):
                    return str(message["content"])
        return ""


class ManagedAgent:
    """Run attempts through one engine, at most ``concurrency`` at a time, and yield one outcome for each spec.

    Each attempt uses its task ID as its run ID, so its events and its rollout carry that ID.
    """

    def __init__(self, engine: RolloutEngine, model: Model, *, concurrency: int):
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self.engine = engine
        self.model = model
        self.concurrency = concurrency

    async def run(self, specs: Iterable[TaskRolloutSpec]) -> AsyncIterator[AgentOutcome]:
        """Each spec's outcome, in completion order. Stopping the iteration cancels the attempts that still run."""
        specs = list(specs)
        # A spec that the engine cannot run fails the batch before any attempt starts.
        for spec in specs:
            self.engine.check(spec)
        admission = asyncio.Semaphore(self.concurrency)
        attempts = [asyncio.create_task(self._attempt(spec, admission)) for spec in specs]
        try:
            for attempt in asyncio.as_completed(attempts):
                yield await attempt
        finally:
            for attempt in attempts:
                attempt.cancel()
            await asyncio.gather(*attempts, return_exceptions=True)

    async def _attempt(self, spec: TaskRolloutSpec, admission: asyncio.Semaphore) -> AgentOutcome:
        async with admission:
            try:
                result = await self.engine.run_task(spec, self.model, run_id=spec.task.id)
            except GradedRolloutInterrupted as interrupted:
                failure = f"{interrupted.operation.value}: {interrupted.__cause__!r}"
                return AgentOutcome(spec.task.id, interrupted.rollout, interrupted.grade, failure)
            except Exception as error:
                # The outcome reports the failure, so that one attempt cannot stop the others.
                return AgentOutcome(spec.task.id, None, None, repr(error))
        return AgentOutcome(spec.task.id, result.rollout, result.grade)
