# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Normalization and review policies for directly imported code tasks."""

import json
import re
from collections.abc import Mapping
from typing import Any

from rigging.filesystem.storage_path import StoragePath

from taskcompendium.datasets.direct_contracts import contract_task
from taskcompendium.models import TaskSpec, TextMessage
from taskcompendium.pipeline.models import (
    ImportFailureKind,
    ImportRejection,
    RawRow,
    ReviewRubric,
    TaskPolicy,
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


def validate_code_cases(ground_truth: str | Mapping | list) -> None:
    """Reject source test layouts the pinned LCB scorer cannot normalize."""
    cases = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if isinstance(cases, Mapping) and "test_cases" in cases:
        if cases.get("language") not in (None, "python"):
            raise ValueError("LCB supports only Python verification specs")
        cases = cases["test_cases"]
    elif isinstance(cases, Mapping):
        inputs, outputs = cases.get("inputs"), cases.get("outputs")
        if not isinstance(inputs, list) or not isinstance(outputs, list) or not inputs or len(inputs) != len(outputs):
            raise ValueError("LCB requires aligned nonempty input and output lists")
        kind = "functional" if cases.get("fn_name") is not None else "stdin"
        cases = [
            {"input": item, "output": result, "testtype": kind, "fn_name": cases.get("fn_name")}
            for item, result in zip(inputs, outputs, strict=True)
        ]
    if not isinstance(cases, list) or not cases or not all(isinstance(case, Mapping) for case in cases):
        raise ValueError("LCB requires a nonempty list of test cases")
    modes = set()
    names = set()
    for case in cases:
        kind = case.get("testtype", case.get("type"))
        if kind in ("stdin", "stdin_stdout"):
            if not isinstance(case.get("input"), str) or not isinstance(case.get("output"), str):
                raise ValueError("Standard-input LCB cases require string input and output")
            modes.add("stdin")
        elif kind in ("functional", "call_based"):
            metadata = case.get("metadata")
            name = case.get("fn_name") or (metadata.get("func_name") if isinstance(metadata, Mapping) else None)
            if not isinstance(name, str) or not name:
                raise ValueError("Functional LCB cases require a function name")
            if not isinstance(case.get("input"), str | list):
                raise ValueError("Functional LCB input must be a list or encoded string")
            modes.add("functional")
            names.add(name)
        else:
            raise ValueError("Unsupported LCB test-case type")
    if len(modes) != 1 or len(names) > 1:
        raise ValueError("LCB cases must share one execution mode and function name")


def has_code_block(response: str) -> bool:
    """Match the original scorer's fenced-code extraction boundary."""
    return re.search(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL) is not None


def normalize_apps(row: RawRow) -> TaskSpec | ImportRejection:
    question, encoded = row.data.get("question"), row.data.get("input_output")
    if not isinstance(question, str) or not isinstance(encoded, str):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_tests",
            detail="question and input_output JSON are required",
        )
    try:
        tests = json.loads(encoded)
    except json.JSONDecodeError:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_test_contract",
            detail="input_output is not valid JSON",
        )
    if not isinstance(tests, dict):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_test_contract",
            detail="input_output must contain a test object",
        )
    inputs, outputs = tests.get("inputs"), tests.get("outputs")
    if not isinstance(inputs, list) or not inputs or not isinstance(outputs, list) or len(inputs) != len(outputs):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_test_contract",
            detail="Paired nonempty inputs and outputs are required",
        )
    starter = row.data.get("starter_code", "")
    prompt = question + ("\n\nStarter code:\n" + starter if starter else "")
    contract = {
        key: row.data[key]
        for key in ("input_output", "solutions", "difficulty", "url", "id", "problem_id")
        if key in row.data
    }
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
        return ImportRejection(
            kind=ImportFailureKind.CONVERTER_ERROR,
            reason="source_selector_mismatch",
            detail="Eurus code requires ability=code",
        )
    messages, reward = row.data.get("prompt"), row.data.get("reward_model")
    if not isinstance(messages, list) or not messages or not isinstance(reward, dict) or not reward.get("ground_truth"):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_tests",
            detail="prompt and reward_model ground_truth are required",
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
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="missing_prompt_or_tests",
            detail="problem_statement and verification_info test_cases are required",
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


def select_eurus_code(row: dict[str, Any], _staged_root: StoragePath) -> bool:
    return row["ability"] == "code"


def apps_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_apps, rubric=APPS_RUBRIC)


def eurus2_code_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_eurus2_code, rubric=EURUS2_CODE_RUBRIC)


def verifiable_code_policy() -> TaskPolicy:
    return TaskPolicy(normalize=normalize_verifiable_code, rubric=VERIFIABLE_CODE_RUBRIC)
