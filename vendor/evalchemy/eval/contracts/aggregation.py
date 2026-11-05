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


def complete_macro(
    counts: Mapping[str, tuple[float, float]],
    expected: Mapping[str, int],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, float | bool]:
    """Equal-cell macros with full-population gates and separate partial diagnostics."""
    metrics = {}
    cells = {}
    total = sum(counts[cell][1] for cell in expected)
    solved = sum(counts[cell][0] for cell in expected)
    metrics.update(num_total=total, num_solved=solved, expected_count=sum(expected.values()))
    for cell, population in expected.items():
        correct, graded = counts[cell]
        if not 0 <= correct <= graded <= population:
            raise ValueError(f"{cell}: invalid correct/graded/population counts")
        metrics.update({f"{cell}/expected": population, f"{cell}/graded": graded,
                        f"{cell}/correct": correct, f"{cell}/missing": population - graded})
        if graded:
            metrics[f"{cell}/partial_accuracy"] = correct / graded
        if graded == population:
            cells[cell] = correct / population
            metrics[cell] = cells[cell]
    for name, members in groups.items():
        if all(cell in cells for cell in members):
            metrics[name] = sum(cells[cell] for cell in members) / len(members)
    if total:
        metrics["accuracy"] = solved / total
    metrics["complete"] = len(cells) == len(expected)
    return metrics
