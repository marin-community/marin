# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Encoded Reasoning Gym ``time_intervals`` entries, as the generator writes them and as recorded on another day."""

import reasoning_gym

from experiments.post_training.task_curation.datasets.reasoning_gym import generate


def time_intervals_entry(index: int, day: str | None = None) -> dict:
    """An encoded ``time_intervals`` entry, with its start and end moved to ``day`` when one is given."""
    seed = generate.task_seed("time_intervals")
    entry = generate.encoded(
        reasoning_gym.create_dataset("time_intervals", size=generate.ROWS_PER_TASK, seed=seed)[index]
    )
    if day is None:
        return entry
    metadata = entry["metadata"]
    start, end = (f"{day} {metadata[key].split()[1]}" for key in ("start_time", "end_time"))
    return {**entry, "metadata": {**metadata, "start_time": start, "end_time": end}}
