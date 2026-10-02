# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned asdiv source recipe."""

import re
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

DATASET = "chaochun/nlu-asdiv-dataset"
REVISION = "883f90a9a65bf00304ba8f37423910fe743abc47"
CONFIG = "original-xml"
SPLIT = "train"
SOURCE_FILE = "https://raw.githubusercontent.com/chaochun/nlu-asdiv-dataset/883f90a9a65bf00304ba8f37423910fe743abc47/dataset/ASDiv.xml"
SOURCE_FORMAT = "xml"
ACQUISITION = "file"
VIEWER_OFFSET = 0

RUBRIC = ReviewRubric(
    id="asdiv-quality",
    version="1",
    criteria=(
        "The Body and Question jointly specify the problem. Answer parentheses contain units, which must "
        "agree with the public quantity.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    body, question, answer = (row.data.get(key) for key in ("Body", "Question", "Answer"))
    if not all(isinstance(value, str) and value.strip() for value in (body, question, answer)):
        return ImportRejection(reason="missing_prompt_or_reference", detail="Body, Question and Answer are required")
    expected = re.sub(r"\s*\([^)]*\)\s*$", "", str(answer)).strip()
    evidence = {key: row.data[key] for key in ("Answer", "Formula", "Solution-Type", "Source", "Grade", "ID")}
    return math_task(row, (TextMessage(role="user", content=f"{body}\n\n{question}"),), expected, evidence)


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="asdiv",
        version="asdiv-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
        check_suite=CheckSuite(
            id="asdiv-controls", revision="1", parameters={"comparator": "cleanup-math-verify"}, run=math_controls
        ),
    )
