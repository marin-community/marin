# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_stack task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.tasktrove_math import MATH_CRITERIA
from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__nemotron-gym-math-stack-overflow-v3"
RUBRIC = ReviewRubric(
    id="math_stack-answerability",
    version="1",
    criteria=(
        *MATH_CRITERIA,
        "Check mathematical questions for missing prior context, definitions, diagrams, or truncated expressions; "
        "a plausible private answer cannot fill absent public premises.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_stack", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
