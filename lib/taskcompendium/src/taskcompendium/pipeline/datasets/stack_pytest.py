# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned stack pytest Python test source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import python_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_stack-pytest-v2"
RUBRIC = ReviewRubric(
    id="stack_pytest-answerability",
    version="1",
    criteria=(
        "Check that the adapted Stack Overflow request defines the tested API and supplies all relevant context.",
        "Check that the named modules and package files in the public request are captured by the runtime; "
        "a solution.py-only submission cannot implement a different named package.",
        "Missing oracle controls imply verification uncertainty, not an automatically bad problem.",
        "Flag undefined behavior, missing fixtures, unavailable dependencies, and contradictory examples.",
        "Private tests and oracle solutions are review evidence and must remain hidden from the solving actor.",
        "A passing oracle shows test compatibility, not specification coverage; cite a concrete defect when rejecting.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    return python_tasks.recipe(
        "stack_pytest",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
