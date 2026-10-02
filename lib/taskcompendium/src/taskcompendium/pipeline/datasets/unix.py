# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned unix checklist source and answerability review criteria."""

from pathlib import Path

from taskcompendium.pipeline.datasets.instruction_following import REVISION
from taskcompendium.pipeline.datasets.rubric_tasks import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

CONFIG = "laion__stackexchange-unix-sandboxes-verified-v2"
RUBRIC = ReviewRubric(
    id="unix-answerability",
    version="1",
    criteria=(
        "Require a complete public request and any code, prior turns, or external passages needed to answer it.",
        "Check each private criterion against the public request; flag invented constraints or incorrect premises.",
        "The source uses one holistic numeric judge over four criteria and has no gold answer; its runtime judge "
        "is unbound.",
        "Check shell, distribution, filesystem, quoting, permissions, and tool assumptions; different valid "
        "commands must not be excluded by an arbitrary checklist.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    """Bind the pinned snapshot to its original private judge and source review rubric."""
    return family_recipe("unix", snapshot, config=CONFIG, revision=REVISION, rubric=RUBRIC)
