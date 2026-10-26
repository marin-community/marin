# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capability-gap tasks of the controlled capability-gap gate.

The Reasoning Gym rows of the seed come from one generated population: for each of 104 task types, 1000 entries of a
dataset with a fixed seed, of which the seed keeps about 200. A gap type is a type whose rows the seed drops. Its
fresh instances are other entries of the same population, written as the seed rows are: a seed row of the type is
the template, and the entry, the generation and the recorded controls in the grader's contract change. The seed
grader regenerates the entry, so the generator must run under ``PYTHONHASHSEED=0``, as the pinned generator did.

Held-out instances of the gap types (entries with an index divisible by ``EVAL_MODULUS``) never train; the training
instances are the other entries that the seed does not use.
"""

import base64
import copy
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import click
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import TaskSpec

from experiments.post_training.rsi.seed import TRAIN_URL, TaskPool, host_graded
from experiments.post_training.task_curation.datasets.reasoning_gym.generate import (
    ROWS_PER_TASK,
    generated_rows,
    task_names,
)
from experiments.post_training.task_curation.datasets.reasoning_gym.tasks import (
    EXCLUDED_GENERATORS,
    GENERATOR_ARCHIVE,
    GENERATOR_VERSION,
    PYTHON_HASH_SEED,
)

logger = logging.getLogger(__name__)

GENERATED_SOURCE = "reasoning_gym_generated"
CONFIG_PATH = "config.json"
EVAL_MODULUS = 5
SYNTHESIZED_TAG = "baby-rsi.synthesized"
GAP_CAPABILITY_PREFIX = "capability:gap-"
HELDOUT_NAME_PREFIX = "heldout-"
HELDOUT_CAPABILITY_PREFIX = f"capability:{HELDOUT_NAME_PREFIX}"
READ_BATCH_ROWS = 2000


def is_generated(payload: str) -> bool:
    """Whether a task row is a generated Reasoning Gym row; its ID, near the start, says so."""
    return GENERATED_SOURCE in payload[:80]


def without_generators(pool: TaskPool, names: frozenset[str]) -> TaskPool:
    """``pool`` without the rows of the Reasoning Gym task types ``names``."""
    return pool.filtered(lambda row: generator_of(row.payload) not in names)


def contract_resource(task: dict) -> dict:
    return next(file for file in task["resources"]["verifier"] if file["path"] == CONFIG_PATH)


def contract(task: dict) -> dict:
    """The grader contract of a generated row: its entry, the generation that made it, and its recorded controls."""
    return json.loads(base64.b64decode(contract_resource(task)["source"]["content_base64"]))["contract"]


def generator_of(payload: str) -> str | None:
    """The Reasoning Gym task type of a seed row, or ``None`` for any other row."""
    return contract(json.loads(payload))["task"] if is_generated(payload) else None


@dataclass(frozen=True)
class TypeRows:
    """The seed's rows of one task type: one of them as a template, and the entry indices that the seed uses."""

    template: str
    used: frozenset[int]


def scan_types(train_url: str = TRAIN_URL) -> dict[str, TypeRows]:
    """The generated task types of the seed, with a template row and the used entry indices of each."""
    templates: dict[str, str] = {}
    used: defaultdict[str, set[int]] = defaultdict(set)
    with StoragePath(train_url).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=READ_BATCH_ROWS, columns=["task_spec"]):
            for payload in batch.column(0).to_pylist():
                if not is_generated(payload):
                    continue
                generation = contract(json.loads(payload))["generation"]
                templates.setdefault(generation["task"], payload)
                used[generation["task"]].add(generation["index"])
    return {name: TypeRows(templates[name], frozenset(used[name])) for name in templates}


def training_indices(rows: TypeRows) -> list[int]:
    """Entries that never train in the seed and are not held out."""
    return [index for index in range(ROWS_PER_TASK) if index % EVAL_MODULUS != 0 and index not in rows.used]


def heldout_indices(count: int) -> list[int]:
    return [index for index in range(ROWS_PER_TASK) if index % EVAL_MODULUS == 0][:count]


def fresh_tasks(
    types: Mapping[str, TypeRows], indices: Mapping[str, Sequence[int]], capability_prefix: str
) -> dict[str, list[TaskSpec]]:
    """The tasks of the entries ``indices`` of each type, tagged as synthesized for ``capability_prefix + type``."""
    if os.environ.get("PYTHONHASHSEED") != str(PYTHON_HASH_SEED):
        raise ValueError(f"Fresh instances need PYTHONHASHSEED={PYTHON_HASH_SEED}, as the grader regenerates them")
    excluded = dict(EXCLUDED_GENERATORS)
    names = task_names(excluded)
    rows_by_global = {
        name: {index * len(names) + names.index(name): index for index in indices[name]} for name in indices
    }
    wanted = frozenset(row for rows in rows_by_global.values() for row in rows)
    generated = dict(generated_rows(GENERATOR_VERSION, excluded, PYTHON_HASH_SEED, 0, 1, wanted))
    tasks: dict[str, list[TaskSpec]] = {name: [] for name in indices}
    for name in indices:
        for global_index in sorted(rows_by_global[name]):
            row = generated[global_index]
            if row["reproducible"] is not True:
                logger.warning("%s entry %d does not regenerate; dropped", name, rows_by_global[name][global_index])
                continue
            tasks[name].append(fresh_task(types[name].template, row, name, global_index, capability_prefix + name))
    return tasks


def fresh_task(template: str, row: Mapping, name: str, global_index: int, capability: str) -> TaskSpec:
    """The task of one generated row, in the form of the seed row ``template`` of the same type."""
    task = json.loads(template)
    resource = contract_resource(task)
    config = json.loads(base64.b64decode(resource["source"]["content_base64"]))
    fresh = copy.deepcopy(config)
    fresh["contract"].update(
        entry=row["entry"],
        generation=row["generation"],
        recorded_pinned_generator_controls=row["recorded_pinned_generator_controls"],
    )
    resource["source"]["content_base64"] = base64.b64encode(json.dumps(fresh).encode()).decode()
    task["context"]["events"][0]["content"] = row["entry"]["question"]
    task["id"] = f"{GENERATED_SOURCE}-gap-{name}-{row['generation']['index']}"
    task["source"]["row"] = f"{GENERATOR_ARCHIVE}:{global_index}"
    task["tags"] = [SYNTHESIZED_TAG, capability]
    return host_graded(TaskSpec.model_validate(task))


def interleave(tasks: Mapping[str, Sequence[TaskSpec]]) -> Iterator[TaskSpec]:
    """The tasks of every type in turn, so that any prefix keeps the types in equal shares."""
    queues = [list(reversed(queue)) for queue in tasks.values()]
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.pop()


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO)


@main.command("heldout")
@click.option("--types", "type_names", required=True, help="Comma-separated Reasoning Gym task types.")
@click.option("--per-type", type=int, default=100, show_default=True)
@click.option("--output", required=True, help="A JSONL file of the held-out gap tasks.")
def heldout(type_names: str, per_type: int, output: str) -> None:
    """Write held-out instances of the gap types."""
    names = type_names.split(",")
    rows = scan_types()
    tasks = fresh_tasks(rows, {name: heldout_indices(per_type) for name in names}, HELDOUT_CAPABILITY_PREFIX)
    lines = [task.model_dump_json() + "\n" for each in tasks.values() for task in each]
    StoragePath(output).write_text("".join(lines))
    logger.info("Wrote %d held-out tasks of %s", len(lines), names)


if __name__ == "__main__":
    main()
