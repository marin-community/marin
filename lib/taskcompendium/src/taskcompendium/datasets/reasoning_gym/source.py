# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit deterministic rows from a pinned reasoning-gym source checkout."""

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
import reasoning_gym
from reasoning_gym.factory import DATASETS

ROWS_PER_TASK = 1000
GENERATION_SEED = 42


def _json_value(value: object) -> object:
    if isinstance(value, Integral):
        return operator.index(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, date | datetime | time):
        return {"python_type": "datetime." + type(value).__name__, "isoformat": value.isoformat()}
    if isinstance(value, Fraction):
        return {
            "python_type": "fractions.Fraction",
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


def negative_control(scorer: Callable[[str, dict[str, Any]], float], entry: dict[str, Any]) -> dict[str, Any]:
    """Record native failures on a synthetic control without assigning a reward."""
    candidate = "definitely wrong"
    try:
        return {"candidate": candidate, "reward": float(scorer(candidate, entry))}
    except Exception as error:
        return {
            "candidate": candidate,
            "reward": None,
            "scoring_error": {"type": type(error).__name__, "message": str(error)},
        }


def positive_candidate(name: str, entry: dict[str, Any]) -> str | None:
    """Use only the native answer or the family's documented solution witness."""
    if isinstance(entry["answer"], str):
        return entry["answer"]
    metadata = entry["metadata"]
    if name == "graph_color":
        witness = metadata["possible_answer"]
        return json.dumps(witness) if witness is not None else None
    if name == "propositional_logic":
        return metadata["example_answer"] or None
    if name == "rubiks_cube":
        return metadata["example_correct_answer"] or None
    return None


def generated_rows(generator_revision: str, excluded_generators: dict[str, str], python_hash_seed: int):
    """Cycle the sorted task registry with stable per-task seeds and native score evidence."""
    names = sorted(DATASETS)
    datasets = {}
    scorers: dict[str, Callable[[str, dict[str, Any]], float]] = {}
    for index in range(ROWS_PER_TASK):
        for task_index, name in enumerate(names):
            # Registry positions determine seeds even for explicit exclusions.
            if name in excluded_generators:
                continue
            if name not in datasets:
                datasets[name] = reasoning_gym.create_dataset(
                    name, size=ROWS_PER_TASK, seed=GENERATION_SEED + task_index
                )
                scorers[name] = cast(Callable[[str, dict[str, Any]], float], reasoning_gym.get_score_answer_fn(name))
            dataset = datasets[name]
            native_entry = dataset[index]
            scorer = scorers[name]
            answer = positive_candidate(name, native_entry)
            # Native scorers may compare tuple-valued metadata. JSON is the
            # transport boundary, not the input type of acquisition controls.
            positive = {
                "candidate": answer,
                "reward": float(scorer(answer, native_entry)) if answer is not None else None,
            }
            negative = negative_control(scorer, native_entry)
            entry = json.loads(json.dumps(native_entry, default=_json_value))
            yield {
                "entry": entry,
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
                    "negative": negative,
                    "execution": "Pinned reasoning-gym native scorer",
                },
            }


def main(generator_revision: str, excluded_generators_json: str, python_hash_seed: int) -> None:
    """Keep native generator diagnostics out of the JSONL transport."""
    rows = iter(generated_rows(generator_revision, json.loads(excluded_generators_json), python_hash_seed))
    while True:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                row = next(rows)
            except StopIteration:
                return
        print(json.dumps(row, ensure_ascii=False, default=_json_value), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
