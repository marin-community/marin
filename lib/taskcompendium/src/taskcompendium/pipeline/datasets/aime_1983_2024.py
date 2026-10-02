# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned aime_1983_2024 source recipe."""

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

DATASET = "di-zhang-fdu/AIME_1983_2024"
REVISION = "3e2cc86390666c5c756622afc0eeb9e6194496bc"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "AIME_Dataset_1983_2024.csv"
SOURCE_FORMAT = "csv"
ACQUISITION = "file"
VIEWER_OFFSET = 0

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
    problem, expected = row.data.get("Question"), row.data.get("Answer")
    if not isinstance(problem, str) or not isinstance(expected, str):
        return ImportRejection(reason="missing_prompt_or_reference", detail="Question and Answer strings are required")
    evidence = {key: row.data[key] for key in ("Answer", "ID", "Year", "Problem Number", "Part")}
    return math_task(row, (TextMessage(role="user", content=problem),), expected, evidence)


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="aime_1983_2024",
        version="aime_1983_2024-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.EVAL,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="aime_1983_2024-controls",
            revision="1",
            parameters={"comparator": "cleanup-math-verify"},
            run=math_controls,
        ),
    )
