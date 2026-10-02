# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit deterministic rows from a pinned reasoning-gym source checkout."""

import dataclasses
import json
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import reasoning_gym
from reasoning_gym.factory import DATASETS

GENERATOR_REVISION = "49b07130b3fcd12f2d064bba7c43869543a0e7e7"
ROWS_PER_TASK = 1000
GENERATION_SEED = 42


def _json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Generated value of type {type(value).__name__} is not JSON serializable")


def generated_rows():
    """Cycle the sorted task registry with stable per-task seeds and native score evidence."""
    names = sorted(DATASETS)
    datasets = {}
    scorers: dict[str, Callable[[str, dict[str, Any]], float]] = {}
    for index in range(ROWS_PER_TASK):
        for task_index, name in enumerate(names):
            if name not in datasets:
                datasets[name] = reasoning_gym.create_dataset(
                    name, size=ROWS_PER_TASK, seed=GENERATION_SEED + task_index
                )
                scorers[name] = cast(Callable[[str, dict[str, Any]], float], reasoning_gym.get_score_answer_fn(name))
            dataset = datasets[name]
            entry = json.loads(json.dumps(dataset[index], default=_json_value))
            scorer = scorers[name]
            answer = entry["answer"]
            yield {
                "entry": entry,
                "generation": {
                    "task": name,
                    "seed": GENERATION_SEED + task_index,
                    "index": index,
                    "config": dataclasses.asdict(dataset.config),
                },
                "recorded_pinned_generator_controls": {
                    "generator_revision": GENERATOR_REVISION,
                    "positive": {"candidate": answer, "reward": float(scorer(answer, entry))},
                    "negative": {"candidate": "definitely wrong", "reward": float(scorer("definitely wrong", entry))},
                    "execution": "Pinned reasoning-gym native scorer",
                },
            }


if __name__ == "__main__":
    for row in generated_rows():
        print(json.dumps(row, ensure_ascii=False, default=_json_value), flush=True)
