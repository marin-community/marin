# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr2 blend selection: ultra_sft_step3200_multichallenge_len40k."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.datasets.nemotron_ultra_rubrics import FAMILY_CRITERIA
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "rlvr2"
SELECTOR = "ultra_sft_step3200_multichallenge_len40k"
FAMILY = "instruction-following"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_rlvr2/ultra_sft_step3200_multichallenge_len40k"
COMPONENT = "ultra_sft_step3200_multichallenge_len40k"
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_rlvr2_ultra_sft_step3200_multichallenge_len40k-quality",
    version="1",
    criteria=FAMILY_CRITERIA[FAMILY]
    + (
        (
            "This is the ultra_sft_step3200_multichallenge_len40k selection from the rlvr2 training "
            "blend, originating at "
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Instruction-Following-MultiTurnChat-v1."
            " Judge its actual retained source fields and agent reward contract; do not infer byte "
            "equivalence with another blend."
        ),
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe(
        "nemotron_ultra_rlvr2_ultra_sft_step3200_multichallenge_len40k",
        snapshot,
        BLEND,
        SELECTOR,
        FAMILY,
        RUBRIC,
        component=COMPONENT,
    )
