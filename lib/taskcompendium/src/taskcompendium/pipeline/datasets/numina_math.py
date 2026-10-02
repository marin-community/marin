# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned numina_math source recipe."""

from pathlib import Path

from verifyit.modes.extract import extract_boxed

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_math import math_task
from taskcompendium.pipeline.datasets.hf_math import math_controls
from taskcompendium.pipeline.models import (
    CheckSuite,
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)

DATASET = "AI-MO/NuminaMath-CoT"
REVISION = "9d8d210c9f6a36c8f3cd84045668c9b7800ef517"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "data/train-00000-of-00005.parquet"
SOURCE_FORMAT = "parquet"
ACQUISITION = "file"
VIEWER_OFFSET = 0

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


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="numina_math",
        version="numina_math-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="numina_math-controls", revision="1", parameters={"comparator": "cleanup-math-verify"}, run=math_controls
        ),
    )
