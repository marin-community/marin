# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned unitsyn large Python test source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import python_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_unitsyn-python-large-v2"
RUBRIC = ReviewRubric(
    id="unitsyn_large-answerability",
    version="1",
    criteria=(
        "Check that the public Python API, filenames, return values, and exception behavior "
        "agree with the private tests.",
        "Check parameter meanings and essential transition rules against the tests and oracle; an algorithmic "
        "constraint missing from the public request is a defect even when the oracle passes.",
        "Do not equate a quantity, rate, time limit, and lower bound; identify concrete parameter mismatches.",
        "Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.",
        "Private tests and oracle solutions are review evidence and must remain hidden from the solving actor.",
        "A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    return python_tasks.recipe(
        "unitsyn_large",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
