# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned curriculum_medium source and Python task review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.python_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "DCAgent__exp_rpt_curriculum-medium-v2"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="curriculum_medium-answerability",
    version="3",
    criteria=(
        "For every boolean membership assertion in disclosed setup tests, derive the expected value from "
        "its literal group fixture and the public parsing rule before deciding quality. Including "
        "assertions in the contract does not excuse a contradiction with an explicit prose rule. Cite "
        "the fixture membership and contradictory assertion when one exists.",
        "Public setup tests may specify missing API details, but they do not override an explicit prose "
        "rule unless the task states a precedence rule. A membership fixture that marks a listed member "
        "false contradicts a rule that all listed members are true; cite the literal values.",
        "Check that the public Python API, output filenames, return values, and exceptions agree with private tests.",
        "The repair note explicitly exposes setup tests as API evidence; assess the request together with these "
        "fixtures and flag contradictions between them.",
        "Oracle solutions and private tests must remain hidden; explicitly public setup tests are part of the contract.",
        "A passing oracle shows compatibility with tests; assess whether those tests cover the public specification.",
        "Check every stated algorithmic rule, mutation requirement, and boundary against the private tests; "
        "difficulty alone is not a defect.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe(
        "curriculum_medium",
        snapshot,
        image,
        config=CONFIG,
        revision=REVISION,
        rubric=RUBRIC,
        timeout=timeout,
        memory_mb=memory_mb,
    )
