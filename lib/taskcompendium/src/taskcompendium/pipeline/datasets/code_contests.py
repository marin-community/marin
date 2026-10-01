# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned code contests source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_code import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="code_contests-answerability",
    version="1",
    criteria=(
        "Check the full stdin/stdout problem, constraints, examples, and private cases for "
        "agreement. Absent diagrams, interactive protocols without an interactor, and contradictory"
        " outputs are defects.",
        "Inspect numerical error clauses and special-output semantics. Exact line comparison cannot"
        " grade an arbitrary valid construction unless the task specifies a canonical output.",
        "The source supplies no oracle solution. Assess static coherence from the problem and cases"
        " anyway; missing executable positive controls and inability to solve quickly are not "
        "quality defects.",
    ),
)


def recipe(snapshot: Path, image: str, *, timeout: float, memory_mb: int) -> DatasetRecipe:
    """Bind a converted snapshot and explicit grading limits."""
    return family_recipe("code_contests", snapshot, image, rubric=RUBRIC, timeout=timeout, memory_mb=memory_mb)
