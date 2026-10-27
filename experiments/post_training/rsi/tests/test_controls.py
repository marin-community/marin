# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The controls of gate 8 feed back what the policy failed, without synthesis."""

import pytest

# The controls run in the marin-taskforge environment; other suites skip them.
pytest.importorskip("taskforge")

import asyncio

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

from experiments.post_training.rsi.controls import Retriever, replay_failures
from experiments.post_training.rsi.rollouts import TaskRollouts
from experiments.post_training.rsi.seed import PoolRow
from experiments.post_training.rsi.streams import from_iterable


def task(task_id: str, text: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=text),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="test"),
        source=Source(dataset="test", revision="1", row=task_id, importer_revision="1"),
    )


def group(item: TaskSpec, reward: float) -> TaskRollouts:
    attempt = GradedRollout(Rollout(f"{item.id}-a", (), "stop"), GradeResult(Outcome.GRADED, reward, None))
    return TaskRollouts(item, (attempt, attempt), "skyrl-step-1")


def test_replay_gives_back_only_the_failed_seed_tasks_under_new_ids():
    groups = [group(task("easy", "add two numbers"), 1.0), group(task("hard", "sort these words"), 0.0)]

    async def run() -> list[TaskSpec]:
        return [replayed async for replayed in replay_failures(from_iterable(groups), max_pass_rate=0.25)]

    (replayed,) = asyncio.run(run())
    assert (replayed.id, replayed.context) == ("hard.replay-1", groups[1].task.context)


def test_retrieval_returns_the_most_similar_unused_task_once():
    texts = {"a": "sort these words by character code", "b": "sort the numbers", "c": "sort these words"}
    retriever = Retriever([PoolRow("s", task(k, v).model_dump_json()) for k, v in texts.items()])
    assert retriever.claim("c")  # the seed stream took c

    first = retriever.nearest("please sort these words in code order")
    second = retriever.nearest("please sort these words in code order")

    # a is the closest; c is closer than b, but the seed already claimed it, so b comes next; then none is left.
    assert [first, second] == [0, 1]
    assert retriever.nearest("sort words") is None
