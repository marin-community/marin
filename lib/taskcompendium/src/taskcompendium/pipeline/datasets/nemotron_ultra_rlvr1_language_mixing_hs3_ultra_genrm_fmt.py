# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr1 GenRM selection: language_mixing_hs3_ultra_genrm_fmt."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "rlvr1"
SELECTOR = "language_mixing_hs3_ultra_genrm_fmt"
COMPONENT = SELECTOR
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
FAMILY = "preference"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_rlvr1/language_mixing_hs3_ultra_genrm_fmt"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_rlvr1_language_mixing_hs3_ultra_genrm_fmt-answerability",
    version="1",
    criteria=(
        (
            "Check the actual language instructions against the full multilingual history and private "
            "evaluation principle."
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
    return family_recipe(
        "nemotron_ultra_rlvr1_language_mixing_hs3_ultra_genrm_fmt",
        snapshot,
        BLEND,
        SELECTOR,
        FAMILY,
        RUBRIC,
        component=COMPONENT,
    )
