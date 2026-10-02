# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned direct nemotron_if instruction-following source."""

from pathlib import Path

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets import direct_instruction
from taskcompendium.pipeline.models import DatasetRecipe, ImportRejection, RawRow, ReviewRubric

DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"
REVISION = "ab2a40d258a6a4d9d4c277d702aeea445081766c"
SPLIT = "instruction_following"
RUBRIC = ReviewRubric(
    id="nemotron_if-answerability",
    version="1",
    criteria=(
        (
            "Identify every public content request and requirement across the complete conversation; do not "
            "discard earlier requests."
        ),
        (
            "Check the private canonical constraint configuration against the public wording and identify "
            "missing, additional, or contradictory constraints."
        ),
        (
            "Formal constraint rewards do not establish factual correctness or useful content; judge the "
            "underlying request separately."
        ),
        (
            "The canonical source rewards the fraction of constraints satisfied; TaskTrove similarly named "
            "checks can differ in counting and punctuation semantics."
        ),
        (
            "Missing requested documents or inputs are defects; an unbound canonical evaluator alone is not a "
            "content defect."
        ),
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return direct_instruction.normalize(row, "input", "args")


def recipe(snapshot: Path) -> DatasetRecipe:
    return direct_instruction.recipe(
        "nemotron_if",
        snapshot,
        dataset=DATASET,
        revision=REVISION,
        split=SPLIT,
        rubric=RUBRIC,
        normalize_row=normalize,
    )
