# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned pymethods Python test source and review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets import python_tasks
from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_pymethods2test-v3"
RUBRIC = ReviewRubric(
    id="pymethods-answerability",
    version="2",
    criteria=(
        "For partitioning and scheduling problems, check whether contiguity, order, indivisibility, and "
        "coverage restrictions are explicitly supplied. Construct a better valid solution under the "
        "public rules before accepting a narrower private optimum.",
        "Check tests against the stated input domain, including zero values and allowed worker counts. "
        "Reject a contradiction in expected behavior; distinguish explicitly described edge cases from a "
        "merely abbreviated constraints list.",
        "Check method signatures, class context, return values, and exceptions against the private tests.",
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
        "pymethods",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
