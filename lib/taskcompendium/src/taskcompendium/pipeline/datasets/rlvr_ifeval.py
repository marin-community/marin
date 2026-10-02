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
