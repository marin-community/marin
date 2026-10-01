# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned advanced calculations source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_math_qa import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="advanced_calculations-answerability",
    version="1",
    criteria=(
        "The wrapper grades only the final requested expression. Preserve that scope, but reject "
        "contradictory requests for multiple answers or methods requiring tools absent from the "
        "public task.",
        "Independently compute the requested final quantity when feasible. Check radians versus "
        "degrees, units, domain errors, precision, rounding, and the declared absolute/relative "
        "tolerance.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("advanced_calculations", snapshot, rubric=RUBRIC)
