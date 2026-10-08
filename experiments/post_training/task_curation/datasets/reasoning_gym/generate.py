# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print deterministic Reasoning Gym rows as JSONL from the ``reasoning_gym`` on ``PYTHONPATH``.

Usage: ``generate.py GENERATOR_VERSION EXCLUDED_GENERATORS_JSON``, with ``PYTHONHASHSEED`` set, since
some generators iterate over sets; rows record the seed so the grader can regenerate with it.
Rows cycle the sorted task registry, ``ROWS_PER_TASK`` entries per task with a stable per-task seed,
and record the scorer's reward for the task's known answer and for a fixed wrong answer. Each row also
records whether a fresh dataset regenerates its entry, as the grader does before scoring.

The grader imports this module from ``/tests`` for the seeds and the JSON encoding.
"""

import contextlib
import dataclasses
import json
import operator
import os
import sys
from collections.abc import Callable, Iterator
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


def generated_rows(
    generator_version: str, excluded_generators: dict[str, str], python_hash_seed: int
) -> Iterator[dict[str, Any]]:
    names = [name for name in sorted(DATASETS) if name not in excluded_generators]
    datasets = {name: reasoning_gym.create_dataset(name, size=ROWS_PER_TASK, seed=task_seed(name)) for name in names}
    scorers = {name: cast(Scorer, reasoning_gym.get_score_answer_fn(name)) for name in names}
    for index in range(ROWS_PER_TASK):
        for name in names:
            dataset = datasets[name]
            entry = dataset[index]
            # The grader builds a fresh dataset and reads one index, so a generator whose entries
            # depend on earlier reads or on process state cannot be graded.
            fresh = reasoning_gym.create_dataset(name, size=ROWS_PER_TASK, seed=task_seed(name))[index]
            scorer = scorers[name]
            answer = positive_candidate(name, entry)
            # Scorers may compare tuple-valued metadata, so score the entry before its JSON round trip.
            positive = {"candidate": answer, "reward": float(scorer(answer, entry)) if answer is not None else None}
            yield {
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
            }


def main(generator_version: str, excluded_generators_json: str) -> None:
    python_hash_seed = int(os.environ["PYTHONHASHSEED"])
    rows = iter(generated_rows(generator_version, json.loads(excluded_generators_json), python_hash_seed))
    while True:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                row = next(rows)
            except StopIteration:
                return
        print(json.dumps(row, ensure_ascii=False, default=json_value), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
