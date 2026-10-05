# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalization and review policies for directly imported code tasks."""

import json
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.datasets.direct_contracts import contract_task
from taskcompendium.pipeline.models import (
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPipeline,
)

APPS_RUBRIC = ReviewRubric(
    id="apps-quality",
    version="1",
    criteria=(
        "Check the public problem and starter code against every private test. Multiple valid outputs and "
        "permissive source comparisons must not be replaced by exact text matching.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def normalize_apps(row: RawRow) -> TaskSpec | ImportRejection:
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


EURUS2_CODE_RUBRIC = ReviewRubric(
    id="eurus2_code-quality",
    version="1",
    criteria=(
        "Require ability=code. Preserve source reward_model ground truth and every prompt message; private "
        "function tests and source evaluator requirements remain private.",
        "Missing runtime binding is a readiness limitation, not a task quality defect. Identify missing "
        "public context separately from implementation difficulty.",
    ),
)


def normalize_eurus2_code(row: RawRow) -> TaskSpec | ImportRejection:
    if row.data.get("ability") != "code":
        return ImportRejection(reason="source_selector_mismatch", detail="Eurus code requires ability=code")
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or not reward.get("ground_truth"):
        return ImportRejection(
            reason="missing_prompt_or_tests", detail="prompt and reward_model ground_truth are required"
        )
    events = tuple(TextMessage(role=message["role"], content=message["content"]) for message in messages)
    contract = {key: row.data[key] for key in ("reward_model", "extra_info", "data_source", "ability")}
    return contract_task(
        row,
        events,
        "eurus2_code",
        contract,
        ("PRIME code evaluator, function-call/stdin harness and source comparator",),
    )


VERIFIABLE_CODE_RUBRIC = ReviewRubric(
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


def normalize_verifiable_code(row: RawRow) -> TaskSpec | ImportRejection:
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


def select_eurus_code(row: dict[str, Any], staged_root: StoragePath) -> bool:
    del staged_root
    return row["ability"] == "code"


def apps_pipeline() -> TaskPipeline:
    return TaskPipeline(normalize=normalize_apps, rubric=APPS_RUBRIC)


def eurus2_code_pipeline() -> TaskPipeline:
    return TaskPipeline(normalize=normalize_eurus2_code, rubric=EURUS2_CODE_RUBRIC)


def verifiable_code_pipeline() -> TaskPipeline:
    return TaskPipeline(normalize=normalize_verifiable_code, rubric=VERIFIABLE_CODE_RUBRIC)
