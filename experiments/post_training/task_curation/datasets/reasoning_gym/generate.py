# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print deterministic Reasoning Gym rows as JSONL from the generator checkout on ``PYTHONPATH``.

Usage: ``generate.py GENERATOR_REVISION EXCLUDED_GENERATORS_JSON PYTHON_HASH_SEED``. Rows cycle the
sorted task registry, ``ROWS_PER_TASK`` entries per task with a stable per-task seed, and record the
scorer's reward for the task's known answer and for a fixed wrong answer.
"""

import contextlib
import dataclasses
import json
import operator
import sys
from collections.abc import Callable
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
    if isinstance(value, Integral):
        return operator.index(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date | datetime | time):
        return {"python_type": "datetime." + type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Fraction):
        return {"python_type": "fractions.Fraction", "numerator": value.numerator, "denominator": value.denominator}
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


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


def generated_rows(generator_revision: str, excluded_generators: dict[str, str], python_hash_seed: int):
    names = sorted(DATASETS)
    datasets = {}
    scorers: dict[str, Scorer] = {}
    for index in range(ROWS_PER_TASK):
        for task_index, name in enumerate(names):
            # Registry positions determine seeds even for excluded tasks.
            if name in excluded_generators:
                continue
            if name not in datasets:
                datasets[name] = reasoning_gym.create_dataset(
                    name, size=ROWS_PER_TASK, seed=GENERATION_SEED + task_index
                )
                scorers[name] = cast(Scorer, reasoning_gym.get_score_answer_fn(name))
            dataset = datasets[name]
            entry = dataset[index]
            scorer = scorers[name]
            answer = positive_candidate(name, entry)
            # Scorers may compare tuple-valued metadata, so score the entry before its JSON round trip.
            positive = {"candidate": answer, "reward": float(scorer(answer, entry)) if answer is not None else None}
            yield {
                "entry": json.loads(json.dumps(entry, default=json_value)),
                "generation": {
                    "task": name,
                    "seed": GENERATION_SEED + task_index,
                    "index": index,
                    "config": dataclasses.asdict(dataset.config),
                    "python_hash_seed": python_hash_seed,
                },
                "recorded_pinned_generator_controls": {
                    "generator_revision": generator_revision,
                    "positive": positive,
                    "negative": negative_control(scorer, entry),
                    "execution": "Pinned reasoning-gym scorer",
                },
            }


def main(generator_revision: str, excluded_generators_json: str, python_hash_seed: int) -> None:
    rows = iter(generated_rows(generator_revision, json.loads(excluded_generators_json), python_hash_seed))
    while True:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                row = next(rows)
            except StopIteration:
                return
        print(json.dumps(row, ensure_ascii=False, default=json_value), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
