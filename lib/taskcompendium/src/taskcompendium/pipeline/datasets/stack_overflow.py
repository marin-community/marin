# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned stack_overflow checklist source and answerability review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.rubric_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__stackexchange-overflow-sandboxes-verified-v2"
REVISION = "02923004846e4e73862c20962f823a6d05100e7a"
RUBRIC = ReviewRubric(
    id="stack_overflow-answerability",
    version="1",
    criteria=(
        "Require a complete public request and any code, prior turns, or external passages needed to answer it.",
        "Check each private criterion against the public request; flag invented constraints or incorrect premises.",
        "The source uses one holistic numeric judge over four criteria and has no gold answer; its runtime judge "
        "is unbound.",
        "Check that error reports include relevant code, versions, input, and observed behavior; distinguish "
        "plausible advice from an answer justified by supplied context.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to its original private judge and source review rubric."""
    return family_recipe("stack_overflow", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
