# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned gsm8k source recipe."""

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

DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
CONFIG = "main"
SPLIT = "train"
SOURCE_FILE = "main/train-00000-of-00001.parquet"
SOURCE_FORMAT = "parquet"

RUBRIC = ReviewRubric(
    id="gsm8k-quality",
    version="1",
    criteria=(
        "The final #### answer is the numeric key; retain the worked derivation privately and check its "
        "arithmetic against the question.",
        "Check that the private answer solves the complete public problem. Difficulty alone is not a quality defect.",
        "The cleanup typed math comparator is used; upstream reward-scorer parity is unverified.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    problem, answer = row.data.get("question"), row.data.get("answer")
    if not isinstance(problem, str) or not isinstance(answer, str) or "####" not in answer:
        return ImportRejection(
            reason="missing_prompt_or_reference", detail="question and answer with #### final separator are required"
        )
    expected = answer.rsplit("####", 1)[-1].strip()
    return math_task(row, (TextMessage(role="user", content=problem),), expected, {"answer": answer})


def recipe() -> DatasetRecipe:
    return math_recipe("gsm8k", HFSource(DATASET, REVISION, CONFIG, SPLIT), normalize, IntendedUse.TRAIN, RUBRIC)
