# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue a restarted loop from its synthesized tasks.

The SkyRL job keeps the training state, and the feed keeps the tasks that the loop wrote (see ``skyrl_box``). A
restarted loop gives the training instances of the accepted tasks that are not in the feed back to the loop. The
ideas of the finished synthesis sessions are known, so the analysis does not yield them again.
"""

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec
from taskforge.agentic.gate import Evidence
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import AcceptedTask, TaskReview
from taskforge.capability import CapabilityIdea

from experiments.post_training.baby_rsi.feedback import (
    INSTANCE_TIMEOUT,
    synthesized,
    synthesized_id,
    training_ids,
)


def known_ideas(output: StoragePath) -> frozenset[str]:
    """The capability IDs of the synthesis sessions that finished."""
    return frozenset(
        CapabilityIdea.model_validate_json(path.read_text()).capability_id
        for path in (output / "ideas" / "*" / "idea.json").glob()
    )


async def untrained_instances(
    output: StoragePath, trained: Iterable[str], instances_per_task: int, factory: MachineFactory
) -> AsyncIterator[TaskSpec]:
    """The accepted tasks of the finished sessions and their training instances that the run did not train on."""
    done = set(trained)
    for saved in accepted_tasks(output):
        idea, accepted = saved.idea, saved.accepted
        untrained = [
            task_id
            for task_id in training_ids(accepted, instances_per_task)
            if synthesized_id(idea, task_id) not in done
        ]
        instances = await task_instances(accepted, untrained, factory, timeout=INSTANCE_TIMEOUT)
        if synthesized_id(idea, accepted.task.id) not in done:
            instances.insert(0, accepted.task)
        for task in instances:
            yield synthesized(task, idea)


@dataclass(frozen=True)
class SavedTask:
    """An accepted task of a finished session, its idea, and the directory of its files."""

    idea: CapabilityIdea
    accepted: AcceptedTask
    directory: StoragePath


def accepted_tasks(output: StoragePath) -> Iterator[SavedTask]:
    """The accepted tasks of the finished sessions in ``output``.

    A session writes ``idea.json`` last and a task ``task.json`` last, so only complete records count.
    """
    for idea_path in sorted((output / "ideas" / "*" / "idea.json").glob(), key=str):
        idea = CapabilityIdea.model_validate_json(idea_path.read_text())
        for path in sorted((idea_path.parent / "tasks" / "*" / "task.json").glob(), key=str):
            directory = path.parent
            accepted = AcceptedTask(
                task=TaskSpec.model_validate_json(path.read_text()),
                evidence=Evidence(**json.loads((directory / "evidence.json").read_text())),
                review=TaskReview.model_validate_json((directory / "review.json").read_text()),
                program=(directory / "program.txt").read_text(),
                workspace=(directory / "workspace.tgz").read_bytes(),
            )
            yield SavedTask(idea, accepted, directory)
