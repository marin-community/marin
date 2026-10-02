# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mopd blend selection: ultra_sft_step3200_instruction_following."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.datasets.nemotron_ultra_rubrics import FAMILY_CRITERIA
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "mopd"
SELECTOR = "ultra_sft_step3200_instruction_following"
FAMILY = "instruction-following"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_mopd/ultra_sft_step3200_instruction_following"
COMPONENT = "ultra_sft_step3200_instruction_following"
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_mopd_ultra_sft_step3200_instruction_following-quality",
    version="1",
    criteria=FAMILY_CRITERIA[FAMILY]
    + (
        (
            "This is the ultra_sft_step3200_instruction_following selection from the mopd training "
            "blend, originating at "
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-instruction_following. Judge its actual"
            " retained source fields and agent reward contract; do not infer byte equivalence with "
            "another blend."
        ),
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe(
        "nemotron_ultra_mopd_ultra_sft_step3200_instruction_following",
        snapshot,
        BLEND,
        SELECTOR,
        FAMILY,
        RUBRIC,
        component=COMPONENT,
    )
