# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math_oracle task source and mathematical quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.tasktrove_math import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "SankalpKJ__nemotron-math-oracle-filtered-v2"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="math_oracle-answerability",
    version="1",
    criteria=(
        "Require a complete mathematical problem, supplied givens, notation, units, and requested result.",
        "Check private reference consistency; difficulty alone is not a defect and a failed control does not prove "
        "the problem is bad.",
        "Original source grader code and data remain private. Cleanup comparator parity is unsupported; distinguish "
        "content quality from grading readiness.",
        "Check that oracle-filtered references solve the public problem; source oracle existence is evidence of "
        "grader compatibility, not proof of mathematical correctness.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned sample to typed math normalization and controls."""
    return family_recipe("math_oracle", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
