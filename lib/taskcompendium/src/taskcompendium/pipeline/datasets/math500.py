# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned math500 source recipe."""

from pathlib import Path

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

DATASET = "HuggingFaceH4/MATH-500"
REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
CONFIG = "default"
SPLIT = "test"
SOURCE_FILE = "test.jsonl"
SOURCE_FORMAT = "jsonl"
ACQUISITION = "file"
VIEWER_OFFSET = 0

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
    problem, expected = row.data.get("problem"), row.data.get("answer")
    if not isinstance(problem, str) or not isinstance(expected, str):
        return ImportRejection(reason="missing_prompt_or_reference", detail="problem and answer strings are required")
    evidence = {key: row.data[key] for key in ("answer", "solution", "subject", "level", "unique_id")}
    return math_task(row, (TextMessage(role="user", content=problem),), expected, evidence)


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="math500",
        version="math500-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.EVAL,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="math500-controls", revision="1", parameters={"comparator": "cleanup-math-verify"}, run=math_controls
        ),
    )
