# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print deterministic Reasoning Gym rows as JSONL from the ``reasoning_gym`` on ``PYTHONPATH``.

Usage, with ``PYTHONHASHSEED`` set, since some generators iterate over sets; rows record the seed so
the grader can regenerate with it:

- ``generate.py size EXCLUDED_GENERATORS_JSON`` prints the number of rows of a whole run.
- ``generate.py rows GENERATOR_VERSION EXCLUDED_GENERATORS_JSON PART PARTS [--indices JSON]`` prints
  part ``PART`` of ``PARTS``, restricted to the row indices in the JSON list when one is given.

Rows cycle the sorted task registry, ``ROWS_PER_TASK`` entries per task with a stable per-task seed,
and record the scorer's reward for the task's known answer and for a fixed wrong answer. Each row also
records whether a fresh dataset regenerates its entry, as the grader does before scoring.

Part ``PART`` of ``PARTS`` generates every ``PARTS``-th task of the registry, each from its own dataset
read in order, and prints each row as ``{"index": ..., "row": ...}`` with its index in the whole cycle,
so the parts together print the rows of a single run. Restricting a part to some indices prints the
same rows when entries depend only on their task's seed and index. Entries that depend on earlier
reads or on process state can differ between runs; the grader cannot reproduce them either, which is
what the fresh-dataset check records.

The grader imports this module from ``/tests`` for the seeds and the JSON encoding.
"""

import argparse
import contextlib
import dataclasses
import json
import operator
import os
import sys
from collections.abc import Callable, Collection, Iterator
from datetime import date, datetime, time
from fractions import Fraction
from numbers import Integral
from typing import Any, cast

import numpy as np

# The generator library prints during import as well as generation; keep stdout for JSONL rows.
with contextlib.redirect_stdout(sys.stderr):
    import reasoning_gym
    from reasoning_gym.factory import DATASETS

ROWS_PER_TASK = 1000
GENERATION_SEED = 42
NEGATIVE_CANDIDATE = "definitely wrong"

type Scorer = Callable[[str, dict[str, Any]], float]


def json_value(value: object) -> object:
    """The JSON form of generated values JSON cannot represent directly."""
    if isinstance(value, Integral):
        return operator.index(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date | datetime | time):
        return {"python_type": "datetime." + type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Fraction):
        return {"python_type": "fractions.Fraction", "numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


def encoded(value: object) -> Any:
    """``value`` as it reads back from a JSONL row."""
    return json.loads(json.dumps(value, default=json_value))


def task_seed(name: str) -> int:
    """The seed of a task's dataset; registry positions fix seeds even for excluded tasks."""
    return GENERATION_SEED + sorted(DATASETS).index(name)


def negative_control(scorer: Scorer, entry: dict[str, Any]) -> dict[str, Any]:
    """Record the scorer's reward for a fixed wrong answer, or the error it raised."""
    try:
        return {"candidate": NEGATIVE_CANDIDATE, "reward": float(scorer(NEGATIVE_CANDIDATE, entry))}
    except Exception as error:
        return {
            "candidate": NEGATIVE_CANDIDATE,
            "reward": None,
            "scoring_error": {"type": type(error).__name__, "message": str(error)},
        }


def positive_candidate(name: str, entry: dict[str, Any]) -> str | None:
    """The entry's answer, or the documented solution example of tasks whose answer is unset."""
    if isinstance(entry["answer"], str):
        return entry["answer"]
    metadata = entry["metadata"]
    if name == "graph_color":
        example = metadata["possible_answer"]
        return json.dumps(example) if example is not None else None
    if name == "propositional_logic":
        return metadata["example_answer"] or None
    if name == "rubiks_cube":
        return metadata["example_correct_answer"] or None
    return None


def task_names(excluded_generators: Collection[str]) -> list[str]:
    """The generated tasks, in the order rows cycle them."""
    return [name for name in sorted(DATASETS) if name not in excluded_generators]


def row_count(excluded_generators: Collection[str]) -> int:
    return len(task_names(excluded_generators)) * ROWS_PER_TASK


def generated_rows(
    generator_version: str,
    excluded_generators: dict[str, str],
    python_hash_seed: int,
    part: int,
    parts: int,
    indices: frozenset[int] | None,
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one part's rows with their indices in the cycle over every task, only ``indices`` when given."""
    names = task_names(excluded_generators)
    # (entry index, registry position) pairs in cycle order; a row's index is divmod's inverse.
    if indices is None:
        entries = [(index, position) for index in range(ROWS_PER_TASK) for position in range(part, len(names), parts)]
    else:
        if not all(0 <= row < len(names) * ROWS_PER_TASK for row in indices):
            raise ValueError(f"Row indices must be below {len(names) * ROWS_PER_TASK}")
        entries = sorted(divmod(row, len(names)) for row in indices if row % len(names) % parts == part)
    positions = sorted({position for _, position in entries})
    datasets = {
        position: reasoning_gym.create_dataset(names[position], size=ROWS_PER_TASK, seed=task_seed(names[position]))
        for position in positions
    }
    scorers = {position: cast(Scorer, reasoning_gym.get_score_answer_fn(names[position])) for position in positions}
    for index, position in entries:
        name, dataset = names[position], datasets[position]
        entry = dataset[index]
        # The grader builds a fresh dataset and reads one index, so a generator whose entries
        # depend on earlier reads or on process state cannot be graded.
        fresh = reasoning_gym.create_dataset(name, size=ROWS_PER_TASK, seed=task_seed(name))[index]
        scorer = scorers[position]
        answer = positive_candidate(name, entry)
        # Scorers may compare tuple-valued metadata, so score the entry before its JSON round trip.
        positive = {"candidate": answer, "reward": float(scorer(answer, entry)) if answer is not None else None}
        yield (
            index * len(names) + position,
            {
                "entry": encoded(entry),
                "reproducible": encoded(fresh) == encoded(entry),
                "generation": {
                    "task": name,
                    "seed": task_seed(name),
                    "index": index,
                    "config": dataclasses.asdict(dataset.config),
                    "python_hash_seed": python_hash_seed,
                },
                "recorded_pinned_generator_controls": {
                    "generator_version": generator_version,
                    "positive": positive,
                    "negative": negative_control(scorer, entry),
                    "execution": "Pinned reasoning-gym scorer",
                },
            },
        )


def print_rows(
    generator_version: str, excluded_generators_json: str, part: int, parts: int, indices: frozenset[int] | None
) -> None:
    if not 0 <= part < parts:
        raise ValueError(f"Part {part} is outside 0..{parts - 1}")
    python_hash_seed = int(os.environ["PYTHONHASHSEED"])
    excluded = json.loads(excluded_generators_json)
    rows = iter(generated_rows(generator_version, excluded, python_hash_seed, part, parts, indices))
    while True:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                index, row = next(rows)
            except StopIteration:
                return
        print(json.dumps({"index": index, "row": row}, ensure_ascii=False, default=json_value), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    size = commands.add_parser("size")
    size.add_argument("excluded_generators")
    rows = commands.add_parser("rows")
    rows.add_argument("generator_version")
    rows.add_argument("excluded_generators")
    rows.add_argument("part", type=int)
    rows.add_argument("parts", type=int)
    rows.add_argument("--indices", type=lambda value: frozenset(json.loads(value)))
    arguments = parser.parse_args()
    if arguments.command == "size":
        print(row_count(json.loads(arguments.excluded_generators)))
        return
    print_rows(
        arguments.generator_version, arguments.excluded_generators, arguments.part, arguments.parts, arguments.indices
    )


if __name__ == "__main__":
    main()
