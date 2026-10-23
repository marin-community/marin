# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue a restarted loop from its synthesized tasks.

The SkyRL job keeps the training state, and the feed keeps the tasks that the loop wrote (see ``skyrl_box``). A
restarted loop gives the training instances of the accepted tasks that are not in the feed back to the loop. The
ideas of the finished synthesis sessions are known, so the analysis does not yield them again.
"""

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import AcceptedTask
from taskforge.capability import CapabilityIdea

from experiments.post_training.rsi.feedback import (
    IDEAS_DIRECTORY,
    INSTANCE_TIMEOUT,
    SESSION_FILE,
    TASKS_DIRECTORY,
    WORKSPACE_FILE,
    SessionRecord,
    synthesized,
    synthesized_id,
    training_ids,
)


def sessions(output: StoragePath) -> Iterator[tuple[StoragePath, SessionRecord]]:
    """The records of the synthesis sessions that finished in ``output``, each with its directory."""
    for path in sorted((output / IDEAS_DIRECTORY / "*" / SESSION_FILE).glob(), key=str):
        yield path.parent, SessionRecord.model_validate_json(path.read_text())


def known_ideas(output: StoragePath) -> frozenset[str]:
    """The capability IDs of the synthesis sessions that finished."""
    return frozenset(record.idea.capability_id for _, record in sessions(output))


async def untrained_instances(
    output: StoragePath, trained: Iterable[str], instances_per_task: int, factory: MachineFactory
) -> AsyncIterator[TaskSpec]:
    """The accepted tasks of the finished sessions and their training instances that the run did not train on; the
    tasks of a held-out idea are never trained on."""
    done = set(trained)
    for saved in accepted_tasks(output):
        if saved.held_out:
            continue
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
    """An accepted task of a finished session, its idea, and the directory of its workspace and probes."""

    idea: CapabilityIdea
    accepted: AcceptedTask
    directory: StoragePath
    held_out: bool
    """Whether the idea is held out: its tasks are only probes."""


def accepted_tasks(output: StoragePath) -> Iterator[SavedTask]:
    """The accepted tasks of the synthesis sessions that finished in ``output``."""
    for directory, record in sessions(output):
        for saved in record.tasks:
            task_directory = directory / TASKS_DIRECTORY / saved.task.id
            accepted = AcceptedTask(
                task=saved.task,
                evidence=saved.evidence,
                review=saved.review,
                program=saved.program,
                workspace=(task_directory / WORKSPACE_FILE).read_bytes(),
            )
            yield SavedTask(record.idea, accepted, task_directory, record.held_out)
