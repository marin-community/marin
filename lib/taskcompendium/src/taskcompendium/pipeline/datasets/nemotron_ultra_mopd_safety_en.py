# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mopd GenRM selection: safety_en."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "mopd"
SELECTOR = "safety_en"
COMPONENT = SELECTOR
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
FAMILY = "preference"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_mopd/safety_en"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_mopd_safety_en-answerability",
    version="1",
    criteria=(
        (
            "Compare the request with its safety principle: a benign craft request mentioning a gun may"
            " call for a glue gun and helpful guidance."
        ),
        (
            "These are generation prompts with a GenRM principle, not stored chosen/rejected pairs; do "
            "not invent pair labels."
        ),
        (
            "The original principle and agent settings remain private grader evidence; no exact-answer "
            "key is supplied."
        ),
        (
            "Reject missing context or contradictory requirements, separating those defects from an "
            "unbound GenRM evaluator."
        ),
        "Do not assume records in another blend with the same selector are byte-identical or a verified alias.",
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe("nemotron_ultra_mopd_safety_en", snapshot, BLEND, SELECTOR, FAMILY, RUBRIC, component=COMPONENT)
