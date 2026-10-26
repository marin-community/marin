# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaluation box of the RSI loop: ``eval_scores, rollouts = eval_policy(model)`` of "Post-training: Boxes/Systems".

The box follows the training stream only to know the policy step. At step 0 and then every ``every`` steps it runs the
live policy on a fixed evaluation set and yields the graded groups of that evaluation; the failed-run analysis reads
these groups, not the training rollouts. Each evaluation also appends its pass rates (the ``eval_scores``) to
``evaluations.jsonl``. The training stream has unbounded queues, so an evaluation never holds training back.
"""

import asyncio
import json
import logging
from collections import defaultdict
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.rsi.rollouts import PolicyRunner, TaskRollouts
from experiments.post_training.rsi.skyrl_box import SKYRL_MODEL_PREFIX

logger = logging.getLogger(__name__)

EVALUATIONS_FILE = "evaluations.jsonl"

Evaluate = Callable[[int], Awaitable[Sequence[TaskRollouts]]]


def policy_step(group: TaskRollouts) -> int:
    """The policy step of a training group of the SkyRL train box."""
    return int(group.model.removeprefix(SKYRL_MODEL_PREFIX))


async def eval_policy(
    groups: AsyncIterable[TaskRollouts], evaluate: Evaluate, every: int, step_of: Callable[[TaskRollouts], int]
) -> AsyncIterator[TaskRollouts]:
    """The groups of an evaluation at the first training group of step 0, and then at every ``every`` steps."""
    if every < 1:
        raise ValueError(f"every must be at least 1, got {every}")
    next_step = 0
    async for group in groups:
        step = step_of(group)
        if step < next_step:
            continue
        for evaluated in await evaluate(step):
            yield evaluated
        next_step = (step // every + 1) * every


@dataclass(frozen=True)
class EvalSet:
    """The tasks of the evaluation box and the report label of each, such as its family or its gap type."""

    tasks: Sequence[TaskSpec]
    labels: Mapping[str, str]


def evaluator(runner: PolicyRunner, eval_set: EvalSet, samples: int, output: StoragePath) -> Evaluate:
    """``samples`` attempts of each task with the live policy; appends the pass rate of each label to the record."""
    path = output / EVALUATIONS_FILE

    async def evaluate(step: int) -> Sequence[TaskRollouts]:
        logger.info("Evaluating the live policy at step %d on %d tasks", step, len(eval_set.tasks))
        groups = await runner.run_all(eval_set.tasks, samples)
        record = {"step": step, "pass_rates": label_pass_rates(groups, eval_set.labels)}
        logger.info("Evaluation at step %d: %s", step, record["pass_rates"])
        lines = path.read_text() if await asyncio.to_thread(path.exists) else ""
        await asyncio.to_thread(path.write_text, lines + json.dumps(record) + "\n")
        return groups

    return evaluate


def label_pass_rates(groups: Sequence[TaskRollouts], labels: Mapping[str, str]) -> dict[str, float]:
    """The mean pass rate of the tasks of each label, and of all tasks under ``*``; ungraded tasks are left out."""
    rates: defaultdict[str, list[float]] = defaultdict(list)
    for group in groups:
        if group.pass_rate is None:
            continue
        rates[labels[group.task.id]].append(group.pass_rate)
        rates["*"].append(group.pass_rate)
    return {label: sum(values) / len(values) for label, values in sorted(rates.items())}
