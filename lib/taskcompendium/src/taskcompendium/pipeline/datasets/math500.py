# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math500 source recipe."""

from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.datasets.direct_math import field_math_task, math_recipe
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)

DATASET = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
CONFIG = "default"
SPLIT = "test"
SOURCE_FILE = "test.jsonl"
SOURCE_FORMAT = "jsonl"

RUBRIC = ReviewRubric(
    id="math500-quality",
    version="1",
    criteria=(
        "Preserve tuple order, intervals, units, and mathematical domains. This official test subset is "
        "evaluation data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "problem", "answer", ("answer", "solution", "subject", "level", "unique_id"))


def recipe() -> DatasetRecipe:
    return math_recipe("math500", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.EVAL, RUBRIC)
