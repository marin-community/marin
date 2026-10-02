# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned direct rlvr_ifeval instruction-following source."""

from pathlib import Path

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import direct_instruction
from taskcompendium.pipeline.models import DatasetRecipe, ImportRejection, RawRow, ReviewRubric

DATASET = "allenai/RLVR-IFeval"
REVISION = "47c03c73621c4aab2b824b7818681117d662770e"
SPLIT = "train"
RUBRIC = ReviewRubric(
    id="rlvr_ifeval-answerability",
    version="1",
    criteria=direct_instruction.CRITERIA,
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return direct_instruction.normalize(row, "messages", "ground_truth")


def recipe(snapshot: Path) -> DatasetRecipe:
    return direct_instruction.recipe(
        "rlvr_ifeval",
        snapshot,
        dataset=DATASET,
        revision=REVISION,
        split=SPLIT,
        rubric=RUBRIC,
        normalize_row=normalize,
    )
