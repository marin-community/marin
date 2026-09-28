# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Observed-grid selection and transfer summaries for the two-domain survey."""

from __future__ import annotations

import math


def analyze(rows: list[dict], controls: dict[str, float], allocations: dict[int, dict], grid: list[int]) -> dict:
    """Compare each proxy choice against the target's minimum on the same grid."""
    result: dict[str, dict] = {}
    for domain in sorted({row["domain"] for row in rows}):
        curves: dict[str, dict] = {}
        for arm in ("matched", "target"):
            selected = [row for row in rows if row["domain"] == domain and row["arm"] == arm]
            values = {row["percent"]: row["macro_bpb"] for row in selected}
            if len(values) != len(selected) or set(values) != set(grid) - {0}:
                raise ValueError(f"Missing or duplicate measured coordinates: {domain}/{arm}")
            values[0] = controls[arm]
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Nonfinite response: {domain}/{arm}")
            minimum = min(grid, key=lambda p: (values[p], p))
            index = grid.index(minimum)
            epochs_key = "matched_epochs" if arm == "matched" else "target_epochs"
            curves[arm] = {
                "grid_percent": grid,
                "loss": [values[p] for p in grid],
                "epochs": [allocations[p][epochs_key] for p in grid],
                "minimum_percent": minimum,
                "minimum_epochs": allocations[minimum][epochs_key],
                "minimum_loss": values[minimum],
                "boundary_minimum": index in (0, len(grid) - 1),
                "neighbor_losses": {str(grid[i]): values[grid[i]] for i in (index - 1, index + 1) if 0 <= i < len(grid)},
            }
        choice = curves["matched"]["minimum_percent"]
        target_at_choice = curves["target"]["loss"][grid.index(choice)]
        result[domain] = {
            "curves": curves,
            "target_loss_at_proxy_choice": target_at_choice,
            "target_grid_regret": target_at_choice - curves["target"]["minimum_loss"],
        }
    return {
        "metric": "equal_mean_seven_uncheatable_bpb",
        "domains": result,
        "scope": "One trainer seed and one matched subset; observed minima on the recorded grid.",
    }
