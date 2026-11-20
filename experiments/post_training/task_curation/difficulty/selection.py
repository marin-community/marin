# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a prepared selection to its planned population and original task bytes."""

from experiments.post_training.task_curation.difficulty.importer import population_revision
from experiments.post_training.task_curation.difficulty.planner import SourceAttempts
from experiments.post_training.task_curation.difficulty_prepared import PreparedSource, PreparedTask


def selected_tasks(source: PreparedSource, requested: SourceAttempts) -> list[PreparedTask]:
    if requested.registry_revision != population_revision(source.sample.model_dump(mode="json")):
        raise ValueError("Planned population changed")
    hashes = {task.task_id: task.task_sha256 for task in source.sample.tasks}
    for task_id, expected, count in requested.tasks:
        if hashes[task_id] != expected or count != 1:
            raise ValueError("Planned task bytes or coverage policy changed")
    wanted = {task_id for task_id, _, _ in requested.tasks}
    tasks = [task for task in source.tasks if task.task_id in wanted]
    if {task.task_id for task in tasks} != wanted:
        raise ValueError("Prepared tasks do not cover the planned selection")
    return tasks
