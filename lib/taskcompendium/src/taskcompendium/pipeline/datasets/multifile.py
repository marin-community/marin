# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned multifile source and Python task review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.python_tasks import PUBLIC_FIXTURE_CRITERION
from taskcompendium.pipeline.datasets.python_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_multifile-v3"
RUBRIC = ReviewRubric(
    id="multifile-answerability",
    version="1",
    criteria=(
        "Check that the public Python API, output filenames, return values, and exceptions agree with private tests.",
        "The repair note explicitly exposes setup tests as API evidence; assess the request together with these "
        "fixtures and flag contradictions between them.",
        PUBLIC_FIXTURE_CRITERION,
        "A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.",
        "Check that every required file and import is specified and captured by the grading contract; flag tests "
        "that require unavailable sibling modules.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe(
        "multifile",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
