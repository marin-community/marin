# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned HH harmless-base preference collection."""

from pathlib import Path

from taskcompendium.pipeline.datasets import preference_tasks
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

DATASET = "Anthropic/hh-rlhf"
REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
CONFIG = "harmless-base"
RUBRIC = ReviewRubric(
    id="hh_harmless_base-answerability",
    version="1",
    criteria=(
        (
            "Assess whether the public request is intelligible and whether a safe response is possible; "
            "harmful requests can have meaningful refusal responses."
        ),
        *preference_tasks.PREFERENCE_CRITERIA,
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return preference_tasks.recipe(
        "hh_harmless_base",
        snapshot,
        dataset=DATASET,
        revision=REVISION,
        config=CONFIG,
        rubric=RUBRIC,
        normalize=preference_tasks.normalize_hh,
    )
