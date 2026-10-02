# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_prism task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.tasktrove_math import MATH_CRITERIA
from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__nemo-prism-math-v3"
RUBRIC = ReviewRubric(
    id="math_prism-answerability",
    version="1",
    criteria=(
        *MATH_CRITERIA,
        "Check symbolic olympiad statements, quantifiers, strict versus attained extrema, and whether escaped "
        "LaTeX keys express the requested quantity.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_prism", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
