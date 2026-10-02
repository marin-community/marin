# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned verifiable_code source and its private evaluator contract."""

from pathlib import Path

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_contracts import contract_task
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportRejection,
    IntendedUse,
    RawRow,
    ReviewRubric,
    SnapshotSource,
)

DATASET = "open-r1/verifiable-coding-problems-python"
REVISION = "b761a24a95fa03289a231d2d31c183636ffb9833"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "data/train-00000-of-00011.parquet"
SOURCE_FORMAT = "parquet"
ACQUISITION = "file"
VIEWER_OFFSET = 0

RUBRIC = ReviewRubric(
    id="verifiable_code-quality",
    version="1",
    criteria=(
        "Check the public problem_statement against every private verification_info test. The "
        "gold_standard_solution is private evidence. Multiple valid outputs and",
        "permissive source comparisons must not be replaced by exact text matching.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    problem, verification = row.data.get("problem_statement"), row.data.get("verification_info")
    if not isinstance(problem, str) or not isinstance(verification, dict) or not verification.get("test_cases"):
        return ImportRejection(
            reason="missing_prompt_or_tests", detail="problem_statement and verification_info test_cases are required"
        )
    contract = {
        key: row.data[key]
        for key in (
            "verification_info",
            "gold_standard_solution",
            "metadata",
            "source",
            "task_type",
            "problem_id",
            "in_source_id",
        )
    }
    return contract_task(
        row,
        (TextMessage(role="user", content=problem),),
        "verifiable_code",
        contract,
        ("Open-R1 source test runner and comparator",),
    )


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="verifiable_code",
        version="verifiable_code-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
    )
