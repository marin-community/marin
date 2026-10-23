# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed of the RSI loop: tasks sampled from the existing RL task pool, with the pool's strata weights.

The pool is the training file of the Snowball Smartie RL run that trained Marin-Snowball-67B-A2B-v0.1. Each row
holds a lowered TaskSpec and its stratum (``teacher_route``): structural closure, competitive coding, Reasoning Gym,
STEM multiple choice, and calendar. A sample keeps the share of each stratum in the pool.

Two conversions make the rows run on any policy and without Daytona:

- A structural-closure row holds prompt token IDs of the Snowball tokenizer. ``structure_closure_task`` converts
  it to a conversation that ends with the unfinished assistant text, which a policy continues with its own tokenizer.
- The script graders of the other strata name a Docker image with Python, ``reasoning_gym``, ``verifyit``, numpy and
  pandas. The seed tasks grade with the host's Python environment instead, which has the same packages, in a
  bubblewrap sandbox (see ``rollouts.machine_factories``).
"""

import json
import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import islice

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import PackageSpec
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.models import SCHEMA_VERSION, ScriptGrader, Source, TaskSpec
from taskcompendium.pipeline.models import ImportRejection, RawRow

POOL_URL = "hf://datasets/open-athena/Snowball-Smartie-RL-2026.10.15.32-Data@cb728a7636caad3fc6fa6b6ba85a49194d9d69a3"
"""The task files of the Snowball Smartie RL run 2026.10.15.32, pinned to a revision."""
TRAIN_FILE = "train.parquet"
VALIDATION_FILE = "validation.parquet"
STRATUM_COLUMN = "teacher_route"
TASK_COLUMN = "lowered_task_spec"
STRUCTURAL = "structural"
POOL_SCHEMA = "0.27"
"""The TaskSpec schema of the pool rows. Schema 0.28 only adds the optional ``FinalAction.normal_submission``, so
a row of the pool is also a TaskSpec of the current schema."""


@dataclass(frozen=True)
class PoolTask:
    """A task of the pool and its stratum."""

    task: TaskSpec
    stratum: str


class TaskPool:
    """The rows of one pool file, read once. Tasks convert when they are sampled."""

    def __init__(self, rows: Sequence[tuple[str, str]]):
        self._rows = rows
        """``(stratum, lowered task JSON)`` for each row, in file order."""

    @classmethod
    def read(cls, url: str) -> "TaskPool":
        with StoragePath(url).open("rb") as handle:
            table = pq.read_table(handle, columns=[STRATUM_COLUMN, TASK_COLUMN])
        return cls(list(zip(table[STRATUM_COLUMN].to_pylist(), table[TASK_COLUMN].to_pylist(), strict=True)))

    @cached_property
    def strata(self) -> dict[str, list[int]]:
        """The row indices of each stratum."""
        strata: dict[str, list[int]] = {}
        for index, (stratum, _) in enumerate(self._rows):
            strata.setdefault(stratum, []).append(index)
        return strata

    @property
    def weights(self) -> dict[str, float]:
        """The share of each stratum in the pool."""
        return {stratum: len(rows) / len(self._rows) for stratum, rows in self.strata.items()}

    def stream(self, seed: int, *, exclude: frozenset[str] = frozenset()) -> Iterator[PoolTask]:
        """Each task of the pool once, in an order whose every prefix keeps the strata weights.

        The order shuffles each stratum with ``seed`` and then draws the stratum that is furthest below its weight.
        Tasks whose ID is in ``exclude`` are skipped, for example a held-out evaluation set.
        """
        rng = random.Random(seed)
        queues = {stratum: rng.sample(rows, len(rows)) for stratum, rows in sorted(self.strata.items())}
        weights = self.weights
        drawn: Counter[str] = Counter()
        while queues:
            total = sum(drawn.values()) + 1
            stratum = max(queues, key=lambda name: weights[name] * total - drawn[name])
            index = queues[stratum].pop()
            if not queues[stratum]:
                del queues[stratum]
            drawn[stratum] += 1
            task = self.task(index)
            if task is not None and task.task.id not in exclude:
                yield task

    def sample(self, n: int, seed: int, *, exclude: frozenset[str] = frozenset()) -> list[PoolTask]:
        """``n`` tasks of the pool with its strata weights: the first ``n`` tasks of ``stream``."""
        return list(islice(self.stream(seed, exclude=exclude), n))

    def task(self, index: int) -> PoolTask | None:
        """The task of one row, or ``None`` when the row does not convert."""
        stratum, lowered = self._rows[index]
        task = pool_task(json.loads(lowered), stratum)
        return None if task is None else PoolTask(task, stratum)


def pool_task(lowered: Mapping[str, object], stratum: str) -> TaskSpec | None:
    """The TaskSpec of one lowered pool row, ready for any policy and a host grader; ``None`` if it does not convert."""
    if stratum == STRUCTURAL:
        native = lowered["task"]
        assert isinstance(native, Mapping)
        source = Source.model_validate(native["source"])
        converted = structure_closure_task(RawRow(str(native["id"]), source, {TASK_COLUMN: json.dumps(lowered)}))
        return None if isinstance(converted, ImportRejection) else converted.task
    native = lowered["task"]
    assert isinstance(native, Mapping) and native["schema_version"] == POOL_SCHEMA
    return host_graded(TaskSpec.model_validate({**native, "schema_version": SCHEMA_VERSION}))


def host_graded(task: TaskSpec) -> TaskSpec:
    """``task`` with its script grader's Docker image replaced by the host's Python environment."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment.docker_image is None:
        return task
    environment = grader.environment.model_copy(update={"docker_image": None, "package_spec": PackageSpec()})
    return task.model_copy(update={"grader": grader.model_copy(update={"environment": environment})})
