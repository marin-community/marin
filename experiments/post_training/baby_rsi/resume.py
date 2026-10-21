# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue a preempted run from what it completed: its last uploaded adapter, its groups, and its synthesized tasks.

A run uploads, every few steps, the adapter with its optimizer state and the records of the groups that it trained.
Step ``k`` samples with the adapter of step ``k - 1`` (``base`` for the first step), so the steps before adapter
``step-N`` are the groups whose ``model`` is ``base`` or an earlier adapter. A resumed run loads ``step-N``, skips the
pool tasks that those steps trained, and gives the training instances of the accepted tasks that it did not train
on yet back to the loop. The ideas of the finished synthesis sessions are known, so the analysis does not yield them
again.
"""

import json
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import MachineFactory
from taskcompendium.models import TaskSpec
from taskforge.agentic.gate import Evidence
from taskforge.agentic.instances import task_instances
from taskforge.agentic.tools import AcceptedTask, TaskReview
from taskforge.capability import CapabilityIdea

from experiments.post_training.baby_rsi.feedback import (
    INSTANCE_TIMEOUT,
    SYNTHESIZED_LABEL_PREFIX,
    synthesized,
    synthesized_id,
    training_ids,
)

OPTIMIZER_FILE = "optimizer.pt"
"""The last file of an uploaded adapter: an adapter directory without it is incomplete."""


ADAPTER_PREFIX = "step-"


def adapter_name(step: int) -> str:
    return f"{ADAPTER_PREFIX}{step:04d}"


def adapter_step(name: str) -> int:
    """The step of an adapter name, 0 for the base policy."""
    return int(name.removeprefix(ADAPTER_PREFIX)) if name.startswith(ADAPTER_PREFIX) else 0


@dataclass(frozen=True)
class Progress:
    """The completed steps of a run: the last uploaded adapter, and the records of the groups that came before it."""

    step: int
    """The step of the adapter to load, or 0 for a run that starts from the base policy."""
    records: list[dict[str, Any]]

    @property
    def pool_tasks(self) -> int:
        return sum(not record["label"].startswith(SYNTHESIZED_LABEL_PREFIX) for record in self.records)

    @property
    def trained_ids(self) -> frozenset[str]:
        return frozenset(record["task_id"] for record in self.records)


def read_progress(output: StoragePath, adapter_dir: Path) -> Progress:
    """The progress of the run in ``output``; downloads its last complete adapter into ``adapter_dir``."""
    steps = [
        adapter_step(path.name)
        for path in (output / "adapters" / f"{ADAPTER_PREFIX}*").glob()
        if (path / OPTIMIZER_FILE).exists()
    ]
    if not steps:
        return Progress(0, [])
    step = max(steps)
    local = adapter_dir / adapter_name(step)
    local.mkdir(parents=True, exist_ok=True)
    for path in (output / "adapters" / adapter_name(step) / "*").glob():
        (local / path.name).write_bytes(path.read_bytes())
    trained = {"base", *(adapter_name(earlier) for earlier in range(1, step))}
    lines = (output / "groups.jsonl").read_text().splitlines()
    return Progress(step, [record for line in lines if (record := json.loads(line))["model"] in trained])


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
