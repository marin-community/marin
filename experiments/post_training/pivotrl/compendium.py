# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Selected pivots as lowered tasks, for MarinSkyRL to train on beside other task-compendium domains.

A task-curation campaign turns each candidate set into a curated source,
``data/rl/pivotrl-<set>-<hash>`` (``experiments/post_training/task_curation/datasets/pivotrl``),
whose admitted tasks carry their candidate's ``source_id`` as a :data:`CANDIDATE_TAG_PREFIX` tag.
:func:`pivot_tasks` keeps the admitted tasks of the pivots a selection artifact chose and writes
them to ``tasks.parquet`` in the ``lowered_task_spec`` layout of MarinSkyRL's ``TaskDataset``. A
pivot is one model turn graded in process, so it needs no task or verifier machine.
"""

import json
from dataclasses import dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.lowering import SHELLBOX_SESSION, lower_task
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec, TaskSessionSpec
from taskcompendium.models import TaskSpec
from zephyr.writers import write_parquet_file

from experiments.post_training.pivotrl.selection import TRAIN_FILENAME

CANDIDATE_TAG_PREFIX = "pivotrl-candidate:"
"""Prefixes a curated task's tag holding its candidate's ``extra_info.source_id``."""
LOWERED_TASK_COLUMN = "lowered_task_spec"
"""The column MarinSkyRL's ``skyrl_train.dataset.tasks.TaskDataset`` reads."""
TASKS_FILENAME = "tasks.parquet"
CURATED_FINAL_GLOB = "final/part-*.parquet"
TASK_SCHEMA = pa.schema([pa.field(LOWERED_TASK_COLUMN, pa.string(), nullable=False)])


@dataclass(frozen=True)
class PivotSession:
    """Deadlines for one pivot attempt: a single model turn, graded in process. ``None`` sets no deadline."""

    model_turn_timeout: float | None
    cleanup_timeout: float


def candidate_id(task: TaskSpec) -> str:
    """The ``source_id`` of the candidate a curated pivot task was converted from."""
    (tag,) = (tag for tag in task.tags if tag.startswith(CANDIDATE_TAG_PREFIX))
    return tag.removeprefix(CANDIDATE_TAG_PREFIX)


def lowered_pivot(task: TaskSpec, session: PivotSession) -> LoweredTaskSpec:
    """A pivot task in MarinSkyRL's built-in Shellbox session, with no machines."""
    return lower_task(
        task,
        TaskRuntimeSpec(task_machine=None, verifier_machine=None),
        TaskSessionSpec(
            task_session=SHELLBOX_SESSION,
            max_turns=1,
            model_turn_timeout=session.model_turn_timeout,
            command_timeout=None,
            tool_turn_timeout=None,
            total_turn_timeout=None,
            attempt_timeout=None,
            verifier_timeout=None,
            cleanup_timeout=session.cleanup_timeout,
        ),
        factories={},
        sessions={},
        providers={},
    )


def selected_candidates(pivots_path: str) -> set[str]:
    with (StoragePath(pivots_path) / TRAIN_FILENAME).open("rb") as source:
        rows = pq.read_table(source, columns=["extra_info"]).column("extra_info").to_pylist()
    return {row["source_id"] for row in rows}


def curated_tasks(curated_path: str) -> list[TaskSpec]:
    """Every admitted task of a curated source."""
    tasks = []
    for path in sorted((StoragePath(curated_path) / CURATED_FINAL_GLOB).glob(), key=str):
        with path.open("rb") as source:
            rows = pq.read_table(source, columns=["task_json"]).column("task_json").to_pylist()
        tasks.extend(TaskSpec.model_validate_json(row) for row in rows)
    return tasks


def pivot_tasks(*, curated_path: str, pivots_path: str, output_path: str, session: PivotSession) -> None:
    """Write the curated tasks of the selected pivots as lowered tasks, plus counts.

    Args:
        curated_path: A curated PivotRL source of the same candidate set the pivots were selected from.
        pivots_path: A selection artifact; its ``train.parquet`` names the selected candidates.
        output_path: Where ``tasks.parquet`` and ``manifest.json`` go.
        session: Deadlines for each attempt.

    A selected pivot that curation did not admit, such as one whose expert turn its grader scores
    below 1, is counted in the manifest and left out.
    """
    selected = selected_candidates(pivots_path)
    tasks = curated_tasks(curated_path)
    kept = [task for task in tasks if candidate_id(task) in selected]
    if not kept:
        raise ValueError("no selected pivot is an admitted curated task")
    rows = ({LOWERED_TASK_COLUMN: lowered_pivot(task, session).model_dump_json()} for task in kept)
    output = StoragePath(output_path)
    write_parquet_file(rows, str(output / TASKS_FILENAME), schema=TASK_SCHEMA)
    manifest = {
        "selected_pivots": len(selected),
        "admitted_tasks": len(tasks),
        "tasks": len(kept),
        "selected_not_admitted": len(selected - {candidate_id(task) for task in kept}),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
