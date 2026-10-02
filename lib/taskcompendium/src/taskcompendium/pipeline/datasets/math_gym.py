# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_gym task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.tasktrove_math import MATH_CRITERIA
from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__nemotron-gym-math-v5"
RUBRIC = ReviewRubric(
    id="math_gym-answerability",
    version="1",
    criteria=(
        *MATH_CRITERIA,
        "Check complete contest statements and exact final-answer format; independently verify feasible "
        "calculations and flag private references answering a different quantity.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_gym", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
