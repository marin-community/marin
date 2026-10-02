# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned curriculum_easy source and Python task review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.python_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_curriculum-easy"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="curriculum_easy-answerability",
    version="1",
    criteria=(
        "Check that the public Python API, output filenames, return values, and exceptions agree with private tests.",
        "Flag contradictory examples, unstated behavior, missing fixtures, and unavailable dependencies.",
        "Oracle solutions and private tests must remain hidden; explicitly public setup tests are part of the contract.",
        "A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.",
        "Check the Python entry point and each stated beginner-level rule against test cases, including empty "
        "input and boundaries.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe(
        "curriculum_easy",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
