# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned HH helpful-rejection-sampled preference collection."""

from pathlib import Path

from taskcompendium.pipeline.datasets import preference_tasks
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

DATASET = "Anthropic/hh-rlhf"
REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
CONFIG = "helpful-rejection-sampled"
RUBRIC = ReviewRubric(
    id="hh_helpful_rejection_sampled-answerability",
    version="1",
    criteria=(
        (
            "Assess the underlying public task independently of the rejection-sampled candidate ranking and "
            "any candidate errors."
        ),
        *preference_tasks.PREFERENCE_CRITERIA,
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return preference_tasks.recipe(
        "hh_helpful_rejection_sampled",
        snapshot,
        dataset=DATASET,
        revision=REVISION,
        config=CONFIG,
        rubric=RUBRIC,
        normalize=preference_tasks.normalize_hh,
    )
