# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned codenet source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_code import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="codenet-answerability",
    version="2",
    criteria=(
        "An example or private input contradicting explicit public bounds is a task defect even if the "
        "main algorithm is clear and the oracle passes. Do not treat that contradiction as a minor issue.",
        "Check that the Python stdin/stdout instruction agrees with private inputs, outputs, and "
        "oracle. The grader compares whitespace-separated tokens and requires at least two cases.",
        "Check every visible private input against the public domain: extra values, too few values,"
        " and violated size bounds can penalize a correct program even when the supplied oracle "
        "passes.",
        "Check whether rewritten statements preserve the original algorithmic problem. Flag "
        "invented behavior, incorrect examples, missing definitions, and inconsistent reference "
        "outputs.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe("codenet", snapshot, image, rubric=RUBRIC, timeout=timeout, memory_mb=memory_mb)
