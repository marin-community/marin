# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned e2egit source and Python task review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.python_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_e2egit-v2"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="e2egit-answerability",
    version="1",
    criteria=(
        "Check that the public Python API, output filenames, return values, and exceptions agree with private tests.",
        "Flag contradictory examples, unstated behavior, missing fixtures, and unavailable dependencies.",
        "Oracle solutions and private tests must remain hidden; explicitly public setup tests are part of the contract.",
        "A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.",
        "Check calculator, banking, inventory, and library APIs against tests; inspect exact error messages and "
        "whether filename normalization leaves any unstated behavior.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe(
        "e2egit",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
