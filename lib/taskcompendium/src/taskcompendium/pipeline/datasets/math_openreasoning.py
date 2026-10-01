# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math openreasoning source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_math_qa import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="math_openreasoning-answerability",
    version="1",
    criteria=(
        "Check the full mathematical problem, givens, notation, units, diagrams, and requested "
        "result. Reject absent diagrams, contradictory assumptions, or a private key inconsistent "
        "with a demonstrated solution.",
        "Independently verify short calculations. For long proofs, assess whether the problem is "
        "well posed; difficulty and inability to solve immediately are not defects. Do not invent a"
        " reference conflict.",
        "The private typed math key is graded by the cleanup math-verify comparator. Original SymPy"
        " comparator parity has not been established. Scalar, equation, interval, set, and ordered "
        "sequence distinctions matter.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("math_openreasoning", snapshot, rubric=RUBRIC)
