# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The RSI loop with scripted boxes: failed rollouts become ideas, ideas become tasks, and training gets them."""

import pytest

# The loop runs in the marin-taskforge environment; other suites skip it.
pytest.importorskip("taskforge")

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

from rolloutengine.models import GradedRollout, Rollout
from taskcompendium.grading_result import GradeResult, Outcome
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
from taskforge.capability import CapabilityIdea

from experiments.post_training.rsi.feedback import FailedRun, synthesized
from experiments.post_training.rsi.loop import Boxes, LoopConfig, rsi_loop
from experiments.post_training.rsi.rollouts import TaskRollouts
from experiments.post_training.rsi.streams import batched, from_iterable

TASKS_PER_STEP = 2
STEPS = 8


def task(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=task_id),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="test"),
        source=Source(dataset="test", revision="1", row=task_id, importer_revision="1"),
    )


def graded(run_id: str, reward: float) -> GradedRollout:
    return GradedRollout(Rollout(run_id, (), "stop"), GradeResult(Outcome.GRADED, reward, None))


def idea(capability_id: str) -> CapabilityIdea:
    return CapabilityIdea(
        capability_id=capability_id,
        subject_id="failed-runs",
        subject_name="Failed runs",
        catalog_version="test",
        capability={"id": capability_id},
        prerequisite_edges=(),
        capability_hash="0",
    )


async def train(tasks: AsyncIterable[TaskSpec]) -> AsyncIterator[TaskRollouts]:
    """Fails every seed task and passes every synthesized task; stops after ``STEPS`` steps."""
    step = 0
    async for batch in batched(tasks, TASKS_PER_STEP):
        step += 1
        for item in batch:
            reward = 1.0 if "baby-rsi.synthesized" in item.tags else 0.0
            yield TaskRollouts(item, (graded(f"{item.id}-a", reward), graded(f"{item.id}-b", reward)), f"step-{step}")
        await asyncio.sleep(0.01)  # The feedback boxes run while the trainer updates.
        if step == STEPS:
            return


async def analyze_failed_runs(batches: AsyncIterable[list[FailedRun]]) -> AsyncIterator[CapabilityIdea]:
    analysis = 0
    async for batch in batches:
        analysis += 1
        assert all(graded.grade.reward == 0.0 for _, graded in batch)
        yield idea(f"idea-{analysis}")


async def generate_tasks(ideas: AsyncIterable[CapabilityIdea]) -> AsyncIterator[TaskSpec]:
    async for each in ideas:
        for k in range(2):
            yield synthesized(task(f"task-{k}"), each)


async def run() -> list[TaskRollouts]:
    seed = from_iterable(task(f"seed-{k}") for k in range(100))
    boxes = Boxes(train=train, analyze_failed_runs=analyze_failed_runs, generate_tasks=generate_tasks)
    config = LoopConfig(max_pass_rate=0.25, failures_per_analysis=4, synthesized_share=0.5, seed=0)
    return [group async for group in rsi_loop(seed, boxes, config)]


def test_the_loop_trains_on_tasks_made_from_its_own_failures():
    groups = asyncio.run(run())

    assert len(groups) == TASKS_PER_STEP * STEPS
    ids = [group.task.id for group in groups]
    first_synthesized = next(index for index, task_id in enumerate(ids) if task_id.startswith("idea-"))
    # The first analysis needs 4 failed seed tasks, so the first synthesized task comes after them.
    assert first_synthesized >= 4
    assert all(task_id.startswith("seed-") for task_id in ids[:first_synthesized])
    assert {task_id for task_id in ids if task_id.startswith("idea-1.")} == {"idea-1.task-0", "idea-1.task-1"}
