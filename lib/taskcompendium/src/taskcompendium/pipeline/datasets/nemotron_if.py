# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned direct nemotron_if instruction-following source."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import direct_instruction
from taskcompendium.pipeline.models import DatasetRecipe, ImportRejection, RawRow, ReviewRubric

DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"
REVISION = "ab2a40d258a6a4d9d4c277d702aeea445081766c"
SPLIT = "instruction_following"
RUBRIC = ReviewRubric(
    id="nemotron_if-answerability",
    version="1",
    criteria=direct_instruction.CRITERIA,
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return direct_instruction.normalize(row, "input", "args")


def recipe() -> DatasetRecipe:
    return direct_instruction.recipe(
        "nemotron_if",
        dataset=DATASET,
        revision=REVISION,
        split=SPLIT,
        rubric=RUBRIC,
        normalize_row=normalize,
    )
