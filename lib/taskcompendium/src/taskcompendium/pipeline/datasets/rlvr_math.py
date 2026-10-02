# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned rlvr_math source recipe."""

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

DATASET = "allenai/RLVR-MATH"
REVISION = "bd2a93551b503a395fadd1a740d957559cfe6f3c"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "data/train-00000-of-00001.parquet"
SOURCE_FORMAT = "parquet"

RUBRIC = ReviewRubric(
    id="rlvr_math-quality",
    version="1",
    criteria=(
        "Retain public few-shot worked examples and distinguish them from the final question. This selected split is "
        "training data.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    messages, expected = row.data.get("messages"), row.data.get("ground_truth")
    if not isinstance(messages, list) or not messages or not isinstance(expected, str):
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="messages and ground_truth strings are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    evidence = {key: row.data[key] for key in ("ground_truth", "dataset", "constraint_type", "constraint")}
    return math_task(row, events, expected, evidence)


def recipe() -> DatasetRecipe:
    return math_recipe("rlvr_math", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.TRAIN, RUBRIC)
