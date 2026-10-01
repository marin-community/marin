# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned knowledge mcqa source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_math_qa import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="knowledge_mcqa-answerability",
    version="1",
    criteria=(
        "Require a complete question and all labeled options. Check whether exactly one option is "
        "defensible from the stated context, and whether the reference selects it. Overlapping "
        "answers or unstated assumptions behind strongest/best claims are concrete defects.",
        "Specialized medical or scientific knowledge is allowed. Unsupported specificity, "
        "contradictory premises, and fabricated distinctions between near-identical options are "
        "defects; unfamiliarity alone is not.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("knowledge_mcqa", snapshot, rubric=RUBRIC)
