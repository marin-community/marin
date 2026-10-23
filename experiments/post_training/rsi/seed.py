# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed of the RSI loop: tasks of the RL data catalog's Snowball curriculum, with its family weights.

The curriculum (``rl-data/curricula/snowball-v0``, made by ``experiments/post_training/task_curation``) holds
100,000 tasks of five families: calendar, competitive coding, Reasoning Gym, STEM and structural. Most rows hold a
TaskSpec. The structural closure rows hold the native lowered spec of the Snowball Smartie RL run, with prompt token
IDs of the Snowball tokenizer; ``structure_closure_task`` converts such a row to a conversation that ends with the
unfinished assistant text, which a policy continues with its own tokenizer. The held-out set is the 910 Snowball
development tasks, which the curriculum excludes, in the same native format.

The script graders name a packages lock or a Docker image with Python, ``reasoning_gym``, ``verifyit``, numpy and
pandas. They grade with the host's Python environment, which has the same packages, in a bubblewrap sandbox (see
``rollouts.machine_factories``).
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

CURRICULUM_URL = "s3://marin-us-east-02a/marin/rl-data/curricula/snowball-v0/2026.10.19.2/train.parquet"
HELDOUT_URL = "s3://marin-us-east-02a/marin/rl-data/inputs/snowball-heldout/2026.10.23.1/validation.parquet"
JUDGED_SOURCES = frozenset({"Task Trove:laion__glaive-code-assistant-sandboxes-verified-v2"})
"""Curriculum sources that a judge model grades. The SkyRL job has no judge, so the seed leaves them out."""
FAMILY_COLUMN = "family"
SOURCE_COLUMN = "source_id"
CONTRACT_COLUMN = "contract_kind"
TASK_JSON_COLUMN = "task_json"
NATIVE_CONTRACT = "native-structure_closure"
STRATUM_COLUMN = "teacher_route"
LOWERED_COLUMN = "lowered_task_spec"
STRUCTURAL = "structural"
POOL_SCHEMA = "0.27"
"""The TaskSpec schema of the rows. Schema 0.28 only adds the optional ``FinalAction.normal_submission``, so a row is
also a TaskSpec of the current schema."""


@dataclass(frozen=True)
class PoolRow:
    stratum: str
    payload: str
    """A TaskSpec, or with ``native`` a lowered spec of the Smartie run, as JSON."""
    native: bool


@dataclass(frozen=True)
class PoolTask:
    """A task of the pool and its stratum."""

    task: TaskSpec
    stratum: str


class TaskPool:
    """The rows of one pool file, read once. Tasks convert when they are sampled."""

    def __init__(self, rows: Sequence[PoolRow]):
        self._rows = rows

    @classmethod
    def read(cls, url: str) -> "TaskPool":
        """The rows of the curriculum (``CURRICULUM_URL``) or of a native file (``HELDOUT_URL``), in file order."""
        with StoragePath(url).open("rb") as handle:
            parquet = pq.ParquetFile(handle)
            if FAMILY_COLUMN not in parquet.schema_arrow.names:
                table = parquet.read(columns=[STRATUM_COLUMN, LOWERED_COLUMN])
                rows = zip(table[STRATUM_COLUMN].to_pylist(), table[LOWERED_COLUMN].to_pylist(), strict=True)
                return cls([PoolRow(stratum, lowered, native=True) for stratum, lowered in rows])
            columns = [FAMILY_COLUMN, SOURCE_COLUMN, CONTRACT_COLUMN, TASK_JSON_COLUMN, LOWERED_COLUMN]
            table = parquet.read(columns=columns)
        values = (table[column].to_pylist() for column in columns)
        return cls(
            [
                PoolRow(family, lowered, native=True) if contract == NATIVE_CONTRACT else PoolRow(family, task, False)
                for family, source, contract, task, lowered in zip(*values, strict=True)
                if source not in JUDGED_SOURCES
            ]
        )

    @cached_property
    def strata(self) -> dict[str, list[int]]:
        """The row indices of each stratum."""
        strata: dict[str, list[int]] = {}
        for index, row in enumerate(self._rows):
            strata.setdefault(row.stratum, []).append(index)
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
        row = self._rows[index]
        task = pool_task(row)
        return None if task is None else PoolTask(task, row.stratum)


def pool_task(row: PoolRow) -> TaskSpec | None:
    """The TaskSpec of one pool row, ready for any policy and a host grader; ``None`` if it does not convert."""
    if not row.native:
        return host_graded(current_schema(json.loads(row.payload)))
    lowered = json.loads(row.payload)
    native = lowered["task"]
    assert isinstance(native, Mapping)
    if row.stratum != STRUCTURAL:
        return host_graded(current_schema(native))
    source = Source.model_validate(native["source"])
    converted = structure_closure_task(RawRow(str(native["id"]), source, {LOWERED_COLUMN: row.payload}))
    return None if isinstance(converted, ImportRejection) else converted.task


def current_schema(task: Mapping[str, object]) -> TaskSpec:
    assert task["schema_version"] == POOL_SCHEMA
    return TaskSpec.model_validate({**task, "schema_version": SCHEMA_VERSION})


def host_graded(task: TaskSpec) -> TaskSpec:
    """``task`` with its script grader's Docker image replaced by the host's Python environment."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment.docker_image is None:
        return task
    environment = grader.environment.model_copy(update={"docker_image": None, "package_spec": PackageSpec()})
    return task.model_copy(update={"grader": grader.model_copy(update={"environment": environment})})
