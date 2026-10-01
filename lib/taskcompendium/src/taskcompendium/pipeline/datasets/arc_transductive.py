# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned arc transductive source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_arc_injection import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="arc_transductive-answerability",
    version="1",
    criteria=(
        "The public examples and test grid must be complete and readable. Judge the common "
        "transformation rule, not whether the review model can fully solve a difficult ARC puzzle.",
        "Compare the private expected grid against the examples and test input when a concrete rule"
        " can be established. Do not invent an alternative key from superficial pattern matching.",
        "The preserved source parser compares grid rows and cells, accepts bare digits, JSON or "
        "boxed grids, and ignores nonnumeric prose lines. The wrapper requests plain "
        "space-separated rows while its quoted source asks for a boxed output; record this format "
        "conflict rather than silently rewriting it.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("arc_transductive", snapshot, rubric=RUBRIC)
