# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Observed-grid minima and basin sharpness for one arm of the arXiv TPP-10 sweep."""

from __future__ import annotations

import math

EXCLUDED_COLUMNS = frozenset(
    {
        "run_name",
        "domain",
        "arm",
        "percent",
        "trainer_seed",
        "subset_seed",
        "fingerprint",
        "checkpoint_sha256",
        "plan_sha256",
    }
)


def analyze(rows: list[dict], allocations: dict[int, dict], grid: list[int], arm: str) -> dict:
    """Per-evaluation minima on the measured grid, with the excess loss at every other grid point."""
    selected = [row for row in rows if row["arm"] == arm]
    if {row["percent"] for row in selected} != set(grid) or len(selected) != len(grid):
        raise ValueError(f"Missing or duplicate measured coordinates for arm {arm}")
    epochs_key = "matched_epochs" if arm == "matched" else "target_epochs"
    metrics = sorted(key for key in selected[0] if key not in EXCLUDED_COLUMNS)
    result: dict[str, dict] = {}
    for metric in metrics:
        values = {row["percent"]: float(row[metric]) for row in selected}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Nonfinite response: {metric}")
        minimum = min(grid, key=lambda p: (values[p], p))
        index = grid.index(minimum)
        result[metric] = {
            "grid_percent": grid,
            "epochs": [allocations[p][epochs_key] for p in grid],
            "loss": [values[p] for p in grid],
            "excess_percent": [100 * (values[p] - values[minimum]) / values[minimum] for p in grid],
            "minimum_percent": minimum,
            "minimum_epochs": allocations[minimum][epochs_key],
            "minimum_loss": values[minimum],
            "boundary_minimum": index in (0, len(grid) - 1),
            "neighbor_excess_percent": {
                str(grid[i]): 100 * (values[grid[i]] - values[minimum]) / values[minimum]
                for i in (index - 1, index + 1)
                if 0 <= i < len(grid)
            },
        }
    return {
        "arm": arm,
        "metrics": result,
        "scope": "One trainer seed and one matched subset; observed minima on the recorded grid.",
    }
