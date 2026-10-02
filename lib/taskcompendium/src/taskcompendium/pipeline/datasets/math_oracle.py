# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_oracle task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.tasktrove_math import MATH_CRITERIA
from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "SankalpKJ__nemotron-math-oracle-filtered-v2"
RUBRIC = ReviewRubric(
    id="math_oracle-answerability",
    version="1",
    criteria=(
        *MATH_CRITERIA,
        "Check that oracle-filtered references solve the public problem; source oracle existence is evidence of "
        "grader compatibility, not proof of mathematical correctness.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_oracle", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
