# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned aime_1983_2024 source recipe."""

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

DATASET = "di-zhang-fdu/AIME_1983_2024"
REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "AIME_Dataset_1983_2024.csv"
SOURCE_FORMAT = "csv"

RUBRIC = ReviewRubric(
    id="aime_1983_2024-quality",
    version="1",
    criteria=(
        "Historical AIME answers are integers; preserve contest year and problem number privately and reserve "
        "this benchmark for evaluation.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    return field_math_task(row, "Question", "Answer", ("Answer", "ID", "Year", "Problem Number", "Part"))


def recipe() -> DatasetRecipe:
    return math_recipe("aime_1983_2024", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.EVAL, RUBRIC)
