# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mopd GenRM selection: hs3_en."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "mopd"
SELECTOR = "hs3_en"
COMPONENT = SELECTOR
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RLHF-GenRM-v1"
FAMILY = "preference"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_mopd/hs3_en"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_mopd_hs3_en-answerability",
    version="1",
    criteria=(
        (
            "Assess the full English conversation and private GenRM principle; earlier assistant errors"
            " are context, not a new gold answer."
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
    return family_recipe("nemotron_ultra_mopd_hs3_en", snapshot, BLEND, SELECTOR, FAMILY, RUBRIC, component=COMPONENT)
