# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned web search mcqa source binding and quality rubric."""

from pathlib import Path

from taskcompendium.pipeline.datasets.atlas_math_qa import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

RUBRIC = ReviewRubric(
    id="web_search_mcqa-answerability",
    version="1",
    criteria=(
        "Require a complete question, labeled options, and one defensible answer. The dataset name "
        "does not provide a browser, search results, or citations. Reject questions that require "
        "missing live evidence.",
        "Check overlapping options and unsupported strongest/best claims. A question answerable "
        "from stable knowledge does not require a browsing tool merely because of its dataset name.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to this source's normalization and rubric."""
    return family_recipe("web_search_mcqa", snapshot, rubric=RUBRIC)
