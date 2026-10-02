# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned mopd blend selection: swe_pivot_len40k/nebius/SWE-rebench-V2."""

from pathlib import Path

from taskcompendium.pipeline.datasets.nemotron_ultra import recipe as family_recipe
from taskcompendium.pipeline.datasets.nemotron_ultra_rubrics import FAMILY_CRITERIA
from taskcompendium.pipeline.models import DatasetRecipe, ReviewRubric

BLEND = "mopd"
SELECTOR = "swe_pivot_len40k"
FAMILY = "swe-repo"
ATLAS_ID = "MarinSkyRL:nemotron_ultra_mopd/swe_pivot_len40k/nebius/SWE-rebench-V2"
COMPONENT = "swe_pivot_len40k/nebius/SWE-rebench-V2"
UPSTREAM = "https://huggingface.co/datasets/nebius/SWE-rebench-V2"
RUBRIC = ReviewRubric(
    id="nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2-quality",
    version="1",
    criteria=FAMILY_CRITERIA[FAMILY]
    + (
        (
            "This is the swe_pivot_len40k/nebius/SWE-rebench-V2 selection from the mopd training blend,"
            " originating at https://huggingface.co/datasets/nebius/SWE-rebench-V2. Judge its actual "
            "retained source fields and agent reward contract; do not infer byte equivalence with "
            "another blend."
        ),
    ),
)


def recipe(snapshot: Path) -> DatasetRecipe:
    return family_recipe(
        "nemotron_ultra_mopd_swe_pivot_len40k_nebius_swe_rebench_v2",
        snapshot,
        BLEND,
        SELECTOR,
        FAMILY,
        RUBRIC,
        component=COMPONENT,
    )
