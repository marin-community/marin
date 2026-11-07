"""Deterministic, versioned per-model batch files for the effective20 panel."""

from typing import TypeVar

T = TypeVar("T")


def full_batches(tasks: list[T], model: str, prefix: str, sizes: dict[str, int]) -> list[tuple[str, list[T]]]:
    """Partition each model once; retries only read the same immutable files."""
    if model not in sizes or not isinstance(sizes[model], int) or not 1 <= sizes[model] <= 218:
        raise ValueError(f"Invalid frozen batch size: {model}")
    if len(tasks) != 218:
        raise ValueError(f"Expected exactly 218 non-pilot tasks for {model}")
    size = sizes[model]
    if size == 218:
        return [(prefix + "results.jsonl", tasks)]
    return [
        (prefix + f"results-part{index + 1}.jsonl", tasks[start : start + size])
        for index, start in enumerate(range(0, len(tasks), size))
    ]
