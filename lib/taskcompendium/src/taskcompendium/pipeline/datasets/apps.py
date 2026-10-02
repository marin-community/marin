# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned apps source and its private evaluator contract."""

import json
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

DATASET = "codeparrot/apps"
REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
CONFIG = "default"
SPLIT = "train"
SOURCE_FILE = "train.jsonl"
SOURCE_FORMAT = "jsonl"
ACQUISITION = "file"
VIEWER_OFFSET = 0

RUBRIC = ReviewRubric(
    id="apps-quality",
    version="1",
    criteria=(
        "Check the public problem and starter code against every private test. Multiple valid outputs and "
        "permissive source comparisons must not be replaced by exact text matching.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def normalize(row: RawRow) -> TaskSpec | ImportRejection:
    question, encoded = row.data.get("question"), row.data.get("input_output")
    if not isinstance(question, str) or not isinstance(encoded, str):
        return ImportRejection(reason="missing_prompt_or_tests", detail="question and input_output JSON are required")
    tests = json.loads(encoded)
    if not tests.get("inputs") or len(tests["inputs"]) != len(tests.get("outputs", [])):
        return ImportRejection(reason="invalid_test_contract", detail="Paired nonempty inputs and outputs are required")
    starter = row.data.get("starter_code", "")
    prompt = question + ("\n\nStarter code:\n" + starter if starter else "")
    contract = {key: row.data[key] for key in ("input_output", "solutions", "difficulty", "url", "id")}
    return contract_task(
        row,
        (TextMessage(role="user", content=prompt),),
        "apps",
        contract,
        ("upstream APPS function-call/stdin harness and output comparator",),
    )


def recipe(snapshot: Path) -> DatasetRecipe:
    return DatasetRecipe(
        name="apps",
        version="apps-v1",
        source=SnapshotSource(DATASET, REVISION, CONFIG, SPLIT, str(snapshot)),
        normalize=normalize,
        intended_use=IntendedUse.TRAIN,
        rubric=RUBRIC,
    )
