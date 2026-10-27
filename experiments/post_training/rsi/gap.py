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
import dataclasses
import json
import logging
import os
import random
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import click
import pyarrow.parquet as pq
from reasoning_gym.factory import DATASETS
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
DIAGNOSTIC_CAPABILITY_PREFIX = "capability:diagnostic-"
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


def diagnostic_indices(rows: TypeRows, count: int) -> list[int]:
    """Entries for the loop's evaluation box: the last training entries, so that they differ from the first ones that
    an oracle trains on, and never held out."""
    return training_indices(rows)[-count:]


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


GATE5_GRADE = Path(__file__).with_name("gate5_grade.py")


def entry_task(template: str, name: str, task_id: str, question: str, entry: Mapping, capability: str) -> TaskSpec:
    """A task of type ``name`` with an entry that is not from the generated population, graded as the seed tasks are
    (the text after the last ``Answer:``, by the dataset's own scorer) but against the entry itself, which
    ``gate5_grade.py`` reads instead of regenerating it."""
    task = json.loads(template)
    config = {"task": name, "entry": dict(entry)}
    task["resources"]["verifier"] = [
        {"path": "grade.py", "source": {"kind": "inline_file", "content_base64": _b64(GATE5_GRADE.read_text())}},
        {"path": CONFIG_PATH, "source": {"kind": "inline_file", "content_base64": _b64(json.dumps(config))}},
    ]
    task["grader"]["env"] = {}
    task["context"]["events"][0]["content"] = question
    task["id"] = task_id
    task["tags"] = [SYNTHESIZED_TAG, capability]
    return host_graded(TaskSpec.model_validate(task))


def _b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def interleave(tasks: Mapping[str, Sequence[TaskSpec]]) -> Iterator[TaskSpec]:
    """The tasks of every type in turn, so that any prefix keeps the types in equal shares."""
    queues = [list(reversed(queue)) for queue in tasks.values()]
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.pop()


@dataclass(frozen=True)
class GapChoice:
    gap: list[str]
    placebo: str
    candidates: list[str]
    """The screened types inside the pass-rate window, from which the choice was drawn."""


def module_group(task: str) -> str:
    """The Reasoning Gym package group of a task type, such as ``arithmetic`` or ``games``."""
    return DATASETS[task][0].__module__.split(".")[1]


def choose(screen: Sequence[Mapping], low: float, high: float, gaps: int, seed: int) -> GapChoice:
    """Draw ``gaps`` gap types and one placebo type from the screened types whose pass rate is in [low, high].

    Each drawn type comes from a different package group, so that two variants of one skill are never both drawn.
    """
    candidates = sorted(result["task"] for result in screen if low <= result["pass_rate"] <= high)
    rng = random.Random(seed)
    drawn: list[str] = []
    for task in rng.sample(candidates, len(candidates)):
        if module_group(task) not in {module_group(other) for other in drawn}:
            drawn.append(task)
        if len(drawn) == gaps + 1:
            break
    if len(drawn) < gaps + 1:
        raise ValueError(f"Only {len(drawn)} package groups among {len(candidates)} candidates")
    return GapChoice(gap=drawn[:gaps], placebo=drawn[gaps], candidates=candidates)


@click.group()
def main() -> None:
    logging.basicConfig(level=logging.INFO)


@main.command("heldout")
@click.option("--types", "type_names", required=True, help="Comma-separated Reasoning Gym task types.")
@click.option("--per-type", type=int, default=100, show_default=True)
@click.option("--output", required=True, help="A JSONL file of the held-out gap tasks.")
@click.option("--train", "train_url", default=TRAIN_URL, show_default=True, help="The seed's training tasks.")
def heldout(type_names: str, per_type: int, output: str, train_url: str) -> None:
    """Write held-out instances of the gap types."""
    names = type_names.split(",")
    rows = scan_types(train_url)
    tasks = fresh_tasks(rows, {name: heldout_indices(per_type) for name in names}, HELDOUT_CAPABILITY_PREFIX)
    lines = [task.model_dump_json() + "\n" for each in tasks.values() for task in each]
    StoragePath(output).write_text("".join(lines))
    logger.info("Wrote %d held-out tasks of %s", len(lines), names)


@main.command("choose")
@click.option("--screen", "screen_file", required=True, help="The screen.json of screen_gap.py.")
@click.option("--low", type=float, default=0.10, show_default=True)
@click.option("--high", type=float, default=0.40, show_default=True)
@click.option("--gaps", type=int, default=3, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--output", required=True, help="A JSON file of the choice.")
def choose_command(screen_file: str, low: float, high: float, gaps: int, seed: int, output: str) -> None:
    """Choose the gap types and the placebo type from a screen."""
    choice = choose(json.loads(StoragePath(screen_file).read_text()), low, high, gaps, seed)
    StoragePath(output).write_text(json.dumps(dataclasses.asdict(choice), indent=2))
    logger.info("Gap types %s, placebo %s, from %d candidates", choice.gap, choice.placebo, len(choice.candidates))


if __name__ == "__main__":
    main()
