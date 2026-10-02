# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned numina_math source recipe."""

from verifyit.modes.extract import extract_boxed

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_math import math_recipe, math_task
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    HFSource,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
)

DATASET = "AI-MO/NuminaMath-CoT"
REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "data/train-*.parquet"
SOURCE_FORMAT = "parquet"

RUBRIC = ReviewRubric(
    id="numina_math-quality",
    version="1",
    criteria=(
        "The boxed solution conclusion is a generated reference; assess its derivation rather than assuming "
        "its correctness.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    problem, solution = row.data.get("problem"), row.data.get("solution")
    if not isinstance(problem, str) or not isinstance(solution, str):
        return ImportRejection(reason="missing_prompt_or_reference", detail="problem and solution strings are required")
    expected = extract_boxed(solution)
    if not expected:
        return ImportRejection(reason="missing_final_answer", detail="Solution lacks a boxed conclusion")
    return math_task(
        row, (TextMessage(role="user", content=problem),), expected, {"solution": solution, "source": row.data["source"]}
    )


def recipe() -> DatasetRecipe:
    return math_recipe("numina_math", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.TRAIN, RUBRIC)
