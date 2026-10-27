# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The seed of the RSI loop: tasks of the Snowball 100k selection, with its family weights.

The selection (``rl-data/task-selection/snowball-100k``, made by ``experiments/post_training/task_curation``) holds
100,000 training tasks of five families: calendar, competitive coding, Reasoning Gym, STEM and structural. Its
``validation.parquet`` is a fixed panel of 100 tasks, 20 for each family, and ``full_validation.parquet`` holds the 910
Snowball development tasks, which no training task repeats. Every row is a TaskSpec as JSON. A structural-closure
task ends with an unfinished assistant message, which a policy continues.

The script graders name a packages lock or a Docker image with Python, ``reasoning_gym``, ``verifyit``, numpy and
pandas. They grade with the host's Python environment, which has the same packages, in a bubblewrap sandbox (see
``rollouts.machine_factories``).
"""

import random
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from itertools import islice

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import PackageSpec
from taskcompendium.models import ScriptGrader, TaskSpec

SELECTION_URL = "s3://marin-us-east-02a/marin/rl-data/task-selection/snowball-100k/2026.10.23.2"
TRAIN_URL = f"{SELECTION_URL}/train.parquet"
VALIDATION_URL = f"{SELECTION_URL}/validation.parquet"
FULL_VALIDATION_URL = f"{SELECTION_URL}/full_validation.parquet"
TASK_COLUMN = "task_spec"
ID_PATTERN = re.compile(r'\{"id":\s*"([^"]+)"')
FAMILY_BY_SOURCE = {
    "eurus2_code": "competitive_coding",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_calendar_v2": "calendar",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_comp_coding": "competitive_coding",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_reasoning_gym": "reasoning_gym",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_stem_mcqa": "stem",
    "nemotron_ultra_rlvr1_ultra_sft_step3200_structured_outputs_v2": "structural",
    "openscience": "stem",
    "reasoning_gym_generated": "reasoning_gym",
    "snowball-v01-structure-closure-train": "structural",
    "structure-closure": "structural",
    "tasktrove-calendar": "calendar",
    "tasktrove-glaive_code": "competitive_coding",
    "tasktrove-if_calendar": "calendar",
    "tasktrove-structured": "structural",
}
"""The family of each source, which the task ID names: ``<source>-<hash>``, or ``<source>:<hash>``. The rows do not
carry their family."""


@dataclass(frozen=True)
class PoolRow:
    stratum: str
    payload: str
    """A TaskSpec as JSON."""


@dataclass(frozen=True)
class PoolTask:
    """A task of the pool and its stratum (its family)."""

    task: TaskSpec
    stratum: str


def family(task_id: str) -> str:
    """The family of a selection task, from the source that its ID names."""
    source = task_id.split(":")[0] if ":" in task_id else task_id.rsplit("-", 1)[0]
    if source not in FAMILY_BY_SOURCE:
        raise ValueError(f"Task {task_id!r} is from the source {source!r}, which has no family")
    return FAMILY_BY_SOURCE[source]


class TaskPool:
    """The rows of one pool file, read once. Tasks parse when they are sampled."""

    def __init__(self, rows: Sequence[PoolRow]):
        self._rows = rows

    @classmethod
    def read(cls, url: str) -> "TaskPool":
        """The rows of a selection file (``TRAIN_URL``, ``VALIDATION_URL`` or ``FULL_VALIDATION_URL``), in file order."""
        with StoragePath(url).open("rb") as handle:
            payloads = pq.ParquetFile(handle).read(columns=[TASK_COLUMN])[TASK_COLUMN].to_pylist()
        return cls([PoolRow(family(row_id(payload)), payload) for payload in payloads])

    @property
    def rows(self) -> Sequence[PoolRow]:
        return self._rows

    def filtered(self, keep: Callable[[PoolRow], bool]) -> "TaskPool":
        """The pool without the rows for which ``keep`` is false."""
        return TaskPool([row for row in self._rows if keep(row)])

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
            if task.task.id not in exclude:
                yield task

    def sample(self, n: int, seed: int, *, exclude: frozenset[str] = frozenset()) -> list[PoolTask]:
        """``n`` tasks of the pool with its strata weights: the first ``n`` tasks of ``stream``."""
        return list(islice(self.stream(seed, exclude=exclude), n))

    def task(self, index: int) -> PoolTask:
        """The task of one row, ready for any policy and a host grader."""
        row = self._rows[index]
        return PoolTask(host_graded(TaskSpec.model_validate_json(row.payload)), row.stratum)


def row_id(payload: str) -> str:
    """The ID of a task row, which a TaskSpec dump writes first."""
    match = ID_PATTERN.match(payload)
    if match is None:
        raise ValueError(f"A task row does not start with its ID: {payload[:80]!r}")
    return match.group(1)


def host_graded(task: TaskSpec) -> TaskSpec:
    """``task`` with its script grader's Docker image replaced by the host's Python environment."""
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or grader.environment.docker_image is None:
        return task
    environment = grader.environment.model_copy(update={"docker_image": None, "package_spec": PackageSpec()})
    return task.model_copy(update={"grader": grader.model_copy(update={"environment": environment})})
