"""Task-owned reductions of graded examples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def outcome_report(outcomes: Sequence[Mapping[str, Any]], population: Sequence[str]) -> dict[str, Any]:
    """Collect native outcome partitions against the source's full population."""
    ids = {name: set() for name in outcomes[0] if name != "instance_id"}
    for row in outcomes:
        for name in ids:
            if row[name]:
                ids[name].add(row["instance_id"])
    ids["incomplete"] = set(population) - ids["submitted"]
    return {
        **{f"{name}_ids": sorted(values) for name, values in ids.items()},
        **{f"{name}_instances": len(values) for name, values in ids.items() if name != "incomplete"},
    }
