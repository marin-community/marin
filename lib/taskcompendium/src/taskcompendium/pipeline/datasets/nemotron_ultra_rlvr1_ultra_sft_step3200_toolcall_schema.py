# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr1 blend selection: ultra_sft_step3200_toolcall_schema."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.datasets.nemotron_ultra_rubrics import FAMILY_CRITERIA
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "rlvr1"
SELECTOR = "ultra_sft_step3200_toolcall_schema"
FAMILY = "tool-use"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_rlvr1/ultra_sft_step3200_toolcall_schema"
COMPONENT = "ultra_sft_step3200_toolcall_schema"
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_rlvr1_ultra_sft_step3200_toolcall_schema-quality",
    version="1",
    criteria=FAMILY_CRITERIA[FAMILY]
    + (
        (
            "This is the ultra_sft_step3200_toolcall_schema selection from the rlvr1 training blend, "
            "originating at "
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1. "
            "Judge its actual retained source fields and agent reward contract; do not infer byte "
            "equivalence with another blend."
        ),
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe(
        "nemotron_ultra_rlvr1_ultra_sft_step3200_toolcall_schema",
        snapshot,
        BLEND,
        SELECTOR,
        FAMILY,
        RUBRIC,
        component=COMPONENT,
    )
