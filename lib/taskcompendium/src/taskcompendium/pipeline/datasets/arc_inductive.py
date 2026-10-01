# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned arc inductive source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_arc_injection import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="arc_inductive-answerability",
    version="1",
    criteria=(
        "The requested Python transform must be grounded in complete public input-output examples. "
        "A small held-out set does not by itself prove the puzzle is incoherent or unsolvable.",
        "Compare hidden test cases with a transformation supported by all examples where feasible. "
        "Missing oracle code or this prototype's unbound isolated runtime is readiness, not a "
        "content defect.",
        "Inspect the source Dockerfile against the public dependency promises. The wrapper lists "
        "numpy/scipy but the embedded source additionally promises torch; distinguish that actual "
        "missing source dependency from the prototype's current runtime binding.",
        "The source grader executes transform(grid), coerces returned cells with int(), and "
        "compares every row to held-out outputs. It extracts solution.py first and answer.txt as a "
        "fallback; this is code evaluation, not an exact text match against an oracle program.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("arc_inductive", snapshot, rubric=RUBRIC)
