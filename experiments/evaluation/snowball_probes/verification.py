# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Predeclared checks for the shared opening task, separate from GLM judgments."""

import dataclasses
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from jsonschema.exceptions import SchemaError
from jsonschema.validators import validator_for
from marin.inference.shell_workspace import ShellWorkspaceRequest, ShellWorkspaceTerminated, shell_command_result
from pydantic import BaseModel, Field
from referencing import Registry
from referencing.exceptions import Unresolvable


class JsonSchemaCheck(BaseModel):
    kind: Literal["json_schema"]
    name: str
    schema_text: str


class ExactTextCheck(BaseModel):
    kind: Literal["exact_text"]
    name: str
    expected: str


class ShellCheck(BaseModel):
    kind: Literal["shell"]
    name: str
    command: str


TaskCheck = Annotated[JsonSchemaCheck | ExactTextCheck | ShellCheck, Field(discriminator="kind")]


class CheckStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class CheckResult:
    name: str
    kind: str
    status: CheckStatus
    detail: str
    output: dict | None = None


def reject_json_constant(value: str) -> None:
    raise ValueError(f"{value} is not a JSON value")


def check_schema(check: JsonSchemaCheck) -> dict:
    schema = json.loads(check.schema_text, parse_constant=reject_json_constant)
    if not isinstance(schema, dict):
        raise ValueError("Verification schema must be a JSON object")
    validator_for(schema).check_schema(schema)
    return schema


def validate_checks(checks: list[TaskCheck], mode: str) -> None:
    for check in checks:
        if isinstance(check, JsonSchemaCheck):
            check_schema(check)
        if isinstance(check, ShellCheck) and mode != "agentic":
            raise ValueError("A chat-only task cannot require shell verification")


def run_checks(
    checks: list[TaskCheck],
    answer: str,
    files: dict[str, str],
    commands: list[str],
    cpu_limit: int,
    limit: str | None,
) -> dict:
    """Check the opening answer and workspace without changing the candidate's workspace."""
    results = []
    for check in checks:
        if limit is not None:
            results.append(CheckResult(check.name, check.kind, CheckStatus.INCONCLUSIVE, f"Harness limit: {limit}"))
            continue
        if isinstance(check, ExactTextCheck):
            passed = answer.strip() == check.expected.strip()
            results.append(
                CheckResult(
                    check.name,
                    check.kind,
                    CheckStatus.PASSED if passed else CheckStatus.FAILED,
                    "Outer whitespace ignored",
                )
            )
            continue
        if isinstance(check, JsonSchemaCheck):
            try:
                value = json.loads(answer, parse_constant=reject_json_constant)
            except ValueError as error:
                results.append(CheckResult(check.name, check.kind, CheckStatus.FAILED, str(error)))
                continue
            try:
                schema = check_schema(check)
                validator = validator_for(schema)(schema, registry=Registry())
                errors = [error.message for error in validator.iter_errors(value)]
            except (ValueError, SchemaError, Unresolvable) as error:
                results.append(CheckResult(check.name, check.kind, CheckStatus.INCONCLUSIVE, str(error)))
                continue
            results.append(
                CheckResult(
                    check.name,
                    check.kind,
                    CheckStatus.FAILED if errors else CheckStatus.PASSED,
                    "; ".join(errors)[:2000],
                )
            )
            continue
        try:
            request = ShellWorkspaceRequest.from_payload(
                {"files": files, "commits": [], "history": commands, "command": check.command}
            )
            result = shell_command_result(request, cpu_limit=cpu_limit)
        except (ValueError, ShellWorkspaceTerminated) as error:
            results.append(CheckResult(check.name, check.kind, CheckStatus.INCONCLUSIVE, str(error)))
            continue
        output = dataclasses.asdict(result)
        for field in ("stdout", "stderr"):
            output[field] = output[field][:4000]
        unsupported = bool(result.unsupported) or result.stderr.startswith("python: unsupported")
        if result.stop_reason or unsupported:
            status = CheckStatus.INCONCLUSIVE
        else:
            status = CheckStatus.PASSED if result.exit_code == 0 else CheckStatus.FAILED
        results.append(CheckResult(check.name, check.kind, status, result.stop_reason or "", output))
    score = None
    if results and all(result.status != CheckStatus.INCONCLUSIVE for result in results):
        score = float(all(result.status == CheckStatus.PASSED for result in results))
    return {
        "phase": "shared_opening",
        "score": score,
        "checks": [dataclasses.asdict(result) for result in results],
    }
