# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_prism task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__nemo-prism-math-v3"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="math_prism-answerability",
    version="1",
    criteria=(
        "Require a complete mathematical problem, supplied givens, notation, units, and requested result.",
        "Check private reference consistency; difficulty alone is not a defect and a failed control does not prove "
        "the problem is bad.",
        "Original source grader code and data remain private. Cleanup comparator parity is unsupported; distinguish "
        "content quality from grading readiness.",
        "Check symbolic olympiad statements, quantifiers, strict versus attained extrema, and whether escaped "
        "LaTeX keys express the requested quantity.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_prism", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
