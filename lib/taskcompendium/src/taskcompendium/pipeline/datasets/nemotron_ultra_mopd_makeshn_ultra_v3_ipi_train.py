# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mopd blend selection: makeshn_ultra_v3_ipi_train."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.datasets.nemotron_ultra_rubrics import FAMILY_CRITERIA
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "mopd"
SELECTOR = "makeshn_ultra_v3_ipi_train"
FAMILY = "agentic-safety"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_mopd/makeshn_ultra_v3_ipi_train"
COMPONENT = "makeshn_ultra_v3_ipi_train"
UPSTREAM = "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train-quality",
    version="1",
    criteria=FAMILY_CRITERIA[FAMILY]
    + (
        (
            "This is the makeshn_ultra_v3_ipi_train selection from the mopd training blend, originating"
            " at "
            "https://huggingface.co/datasets/nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1. "
            "Judge its actual retained source fields and agent reward contract; do not infer byte "
            "equivalence with another blend."
        ),
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe(
        "nemotron_ultra_mopd_makeshn_ultra_v3_ipi_train", snapshot, BLEND, SELECTOR, FAMILY, RUBRIC, component=COMPONENT
    )
