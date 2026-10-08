# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind original Ultra text/schema and typed tool-argument validation."""

import asyncio
import json
from collections.abc import Callable
from functools import partial

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.grading_result import Outcome
from taskcompendium.models import AnswerType, AssistantToolCalls, ConversationToolCall, TaskSpec, TextMessage
from taskcompendium.pipeline.execution_binding import bind_grader_recipe
from taskcompendium.pipeline.models import (
    CheckResult,
    CheckStatus,
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)

from experiments.post_training.task_curation.datasets.nemotron_ultra.grading.binding import (
    ANSWER_EXTRACTOR,
    invocation_bytes,
    normalize_terminal_grader,
    score_package,
)
from experiments.post_training.task_curation.datasets.shared import grade_final_message

CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.structured_outputs:grade_structured_output",
    "args": ["answer", "contract", "terminal_message"],
    "answer_extractor": ANSWER_EXTRACTOR,
    "input_format": "text",
}


def grader_package(config: dict, image: str) -> GraderPackage:
    return score_package(
        config,
        invocation={
            **CALL,
            "input_format": "event" if config["contract"].get("response_mode", "text") == "tool_call" else "text",
        },
        timeout=60,
        image=image,
    )


def normalize_isolated(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    mode = row.data.get("response_mode", "text")
    if mode not in {"text", "tool_call"}:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_structured_response_mode",
            detail=str(mode),
        )
    schema_type = str(row.data.get("schema_type", "json")).lower()
    if mode == "text" and schema_type not in {"json", "yaml", "toml", "xml", "csv"}:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_structured_schema_type",
            detail=schema_type,
        )
    result = normalize_terminal_grader(
        row,
        image=image,
        normalize_task=normalize_task,
        allowed_agents=("structured_outputs_simple_agent", "structured_outputs_v3_simple_agent"),
        package=grader_package,
        answer_type=AnswerType.NATIVE_ACTION if mode == "tool_call" else AnswerType.TEXT,
    )
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    if mode == "tool_call" and not task.final_tools:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="missing_structured_tool_schema",
            detail="Typed terminal tool grading requires the original public tool declaration",
        )
    try:
        schema = json.loads(row.data["schema_str"])
        # Original OpenAPI validate defaults to OAS31Validator, which checks
        # the fixed 2020-12 metaschema even when the row declares another draft.
        Draft202012Validator.check_schema(schema)
    except (json.JSONDecodeError, SchemaError) as error:
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="invalid_original_structured_schema",
            detail=str(error),
        )
    return result


async def isolated_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    """Probe the original schema runtime without constructing a schema-valid witness."""
    contract = grader_config(task)["contract"]
    if contract.get("response_mode", "text") == "tool_call":
        payload_key = contract.get("tool_payload_key")
        arguments = {payload_key: {}} if payload_key else {}
        call = ConversationToolCall(
            call_id="diagnostic", name=contract.get("tool_name") or task.final_tools[0].name, arguments=arguments
        )
        diagnostic: TextMessage | AssistantToolCalls = AssistantToolCalls(calls=(call,))
    else:
        candidates = {
            "json": "{}",
            "yaml": "{}",
            "toml": "diagnostic = 1",
            "xml": "<diagnostic />",
            "csv": "diagnostic\n1\n",
        }
        diagnostic = TextMessage(role="assistant", content=candidates[str(contract.get("schema_type", "json")).lower()])
    result = await grade_final_message(task, diagnostic, factory, machine_spec, timeout)
    details = result.detail or {}
    broken_schema = details.get("error_type") == "schema_error" or (
        details.get("error_type") == "validation_error"
        and str(details.get("error_message", "")).startswith("SchemaError:")
    )
    if broken_schema:
        status = CheckStatus.FAIL
    elif result.status == Outcome.GRADED and result.reward is not None and 0 <= result.reward <= 1:
        status = CheckStatus.PASS
    elif result.status == Outcome.INFRA_ERROR:
        status = CheckStatus.INFRA_ERROR
    else:
        status = CheckStatus.FAIL
    return VerificationReport(
        checks=[
            CheckResult(
                check="native_runtime",
                status=status,
                detail=result.error or f"Diagnostic status={result.status}; reward={result.reward}; not a witness",
            ),
            CheckResult(
                check="positive_witness",
                status=CheckStatus.SKIPPED,
                detail="The source supplies a schema without a passing response witness",
            ),
        ]
    )


def verification_report(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(isolated_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind original structured evaluators without executing declared terminal tools."""
    return bind_grader_recipe(
        recipe,
        normalize=partial(normalize_isolated, image=image, normalize_task=recipe.policy.normalize),
        verification=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-structured-output-runtime",
        grader_bytes=invocation_bytes(CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )
