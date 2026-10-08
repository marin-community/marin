# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind source scorers to isolated execution without changing their reward rules."""

import asyncio
import json
import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

from shellbox.machine import MachineFactory, MachineSpec
from taskcompendium.datasets.nemotron_ultra.normalization import VERIFIER_REVISION
from taskcompendium.grader import GraderPackage, grader_config
from taskcompendium.models import (
    AnswerType,
    AssistantToolCalls,
    ConversationToolCall,
    FileReward,
    FinalAction,
    PlainText,
    RewardFile,
    RewardFileFormat,
    ScriptGrader,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.execution_binding import bind_grader_recipe, bound_grader_task, grading_environment
from taskcompendium.pipeline.models import (
    DatasetRecipe,
    ImportFailureKind,
    ImportRejection,
    NormalizedTask,
    RawRow,
    VerificationReport,
)
from taskcompendium.pipeline.verification import control_result
from taskcompendium.runtime.resources import inline_resource
from verifyit.execution import source_callable

from experiments.post_training.task_curation.datasets.shared import grade_final_message

SOURCE_CALLABLE = Path(source_callable.__file__)
ANSWER_EXTRACTOR = "skyrl_gym.envs.nemotron_ultra.answer_extraction:final_answer_text"
RDKIT_CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.rdkit_chemistry:grade_rdkit_chemistry",
    "args": ["answer", "contract"],
    "answer_extractor": ANSWER_EXTRACTOR,
}
TOOL_CALL = {
    "function": "skyrl_gym.envs.nemotron_ultra.tool_call:grade_expected_action",
    "args": ["contract.expected_action", "terminal_message"],
    "answer_extractor": ANSWER_EXTRACTOR,
    "input_format": "event",
}
ANSWER_PATH = "/app/answer.txt"
SCORE_PATH = "/logs/verifier/score.json"


def invocation_bytes(invocation: dict) -> bytes:
    return json.dumps(invocation, sort_keys=True, allow_nan=False).encode() + SOURCE_CALLABLE.read_bytes()


def score_package(
    config: dict, *, invocation: dict, timeout: float, image: str, state_path: str | None = None
) -> GraderPackage:
    argv = (
        "python3",
        "/tests/source_callable.py",
        "/tests/invocation.json",
        "/tests/config.json",
        ANSWER_PATH,
        SCORE_PATH,
    )
    if state_path is not None:
        argv = (*argv, state_path)
    return GraderPackage(
        ScriptGrader(
            argv=argv,
            cwd="/",
            environment=grading_environment(image),
            answer_path=ANSWER_PATH,
            reward=FileReward(files=(RewardFile(path=SCORE_PATH, format=RewardFileFormat.JSON),)),
            timeout=timeout,
        ),
        (
            inline_resource("source_callable.py", SOURCE_CALLABLE.read_bytes()),
            inline_resource("invocation.json", json.dumps(invocation, allow_nan=False).encode()),
            inline_resource("config.json", json.dumps(config, allow_nan=False).encode()),
        ),
    )


def normalize_terminal_grader(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
    allowed_agents: tuple[str, ...],
    package: Callable[[dict, str], GraderPackage],
    answer_type: AnswerType,
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Replace a stateless terminal evaluator while preserving the source contract."""
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    config = grader_config(task)
    if config["source_revision"] != VERIFIER_REVISION or config["contract"]["agent_ref"]["name"] not in allowed_agents:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_text_agent",
            detail=f"Only the pinned {allowed_agents} evaluators are bound",
        )
    if answer_type == AnswerType.TEXT and (task.interaction_tools or task.final_tools):
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_text_tool_request",
            detail="This original stateless evaluator accepts terminal text only",
        )
    answer_format = FinalAction() if answer_type == AnswerType.NATIVE_ACTION else PlainText()
    task = bound_grader_task(
        task.model_copy(update={"answer_type": answer_type, "answer_format": answer_format}), package(config, image)
    )
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


def normalize_rdkit(
    row: RawRow,
    *,
    image: str,
    normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection],
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Preserve the public request and source row, replacing only the unbound scorer."""
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    config = grader_config(task)
    contract = config["contract"]
    if config["source_revision"] != VERIFIER_REVISION or contract["agent_ref"]["name"] != "rdkit_chemistry_agent":
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_chemistry_agent",
            detail="Only the pinned rdkit_chemistry_agent evaluator is bound",
        )
    if contract["property_type"] not in {"count", "bool", "presence", "fragment"}:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_chemistry_property",
            detail=str(contract["property_type"]),
        )
    if not math.isfinite(float(contract["expected_answer"])):
        return ImportRejection(
            kind=ImportFailureKind.SOURCE_DEFECT,
            reason="nonfinite_chemistry_target",
            detail="The native rounded comparator requires a finite target",
        )
    task = bound_grader_task(task, score_package(config, invocation=RDKIT_CALL, timeout=60, image=image))
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


async def rdkit_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    contract = grader_config(task)["contract"]
    expected = round(float(contract["expected_answer"]))
    wrapper = r"\boxed{%s}" if contract.get("use_box_format", False) else "((%s))"
    checks = []
    for name, answer, reward in (
        ("empty", "", 0.0),
        ("reference", wrapper % expected, 1.0),
        ("perturbed", wrapper % (expected + 1), 0.0),
        ("unwrapped", str(expected), 0.0),
    ):
        grade = await grade_final_message(
            task, TextMessage(role="assistant", content=answer), factory, machine_spec, timeout
        )
        checks.append(control_result(grade, name, reward))
    return VerificationReport(checks)


def verification_report(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(rdkit_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind_rdkit(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Use the original scorer in a fresh isolated machine for each control."""
    return bind_grader_recipe(
        recipe,
        normalize=partial(normalize_rdkit, image=image, normalize_task=recipe.policy.normalize),
        verification=partial(verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-rdkit-controls",
        grader_bytes=invocation_bytes(RDKIT_CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )


def normalize_tool_action(
    row: RawRow, *, image: str, normalize_task: Callable[[RawRow], TaskSpec | NormalizedTask | ImportRejection]
) -> TaskSpec | NormalizedTask | ImportRejection:
    """Retain single-step action prediction without executing the proposed tool."""
    result = normalize_task(row)
    if isinstance(result, ImportRejection):
        return result
    task = result.task if isinstance(result, NormalizedTask) else result
    config = grader_config(task)
    contract = config["contract"]
    if config["source_revision"] != VERIFIER_REVISION or contract["agent_ref"]["name"] not in {
        "single_step_tool_use_with_argument_comparison_agent",
        "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
        "toolcall_schema_single_step_tool_use_with_argument_comparison_agent",
    }:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED,
            reason="unsupported_native_action_agent",
            detail="Only the pinned single-step action comparison agents are bound",
        )
    expected = contract["expected_action"]
    if expected["type"] not in {"message", "function_call"}:
        return ImportRejection(
            kind=ImportFailureKind.UNSUPPORTED, reason="unsupported_native_action_type", detail=expected["type"]
        )
    task = bound_grader_task(
        task.model_copy(update={"answer_type": AnswerType.NATIVE_ACTION, "answer_format": FinalAction()}),
        score_package(config, invocation=TOOL_CALL, timeout=60, image=image),
    )
    return replace(result, task=task) if isinstance(result, NormalizedTask) else task


async def tool_action_checks(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    expected = grader_config(task)["contract"]["expected_action"]
    empty = TextMessage(role="assistant", content="")
    if expected["type"] == "function_call":
        call = ConversationToolCall(
            call_id="control", name=expected["name"], arguments=json.loads(expected["arguments"])
        )
        controls = (
            ("empty", empty, 0.0),
            ("reference", AssistantToolCalls(calls=(call,)), 1.0),
            ("wrong_tool", AssistantToolCalls(calls=(call.model_copy(update={"name": "__wrong_tool__"}),)), 0.0),
            ("extra_call", AssistantToolCalls(calls=(call, call.model_copy(update={"call_id": "extra"}))), 0.0),
        )
    else:
        controls = (
            ("empty", empty, 0.0),
            ("reference", TextMessage(role="assistant", content="A nonliteral response."), 1.0),
            (
                "unexpected_call",
                AssistantToolCalls(
                    calls=(ConversationToolCall(call_id="control", name="__wrong_tool__", arguments={}),)
                ),
                0.0,
            ),
        )
    checks = []
    for name, event, reward in controls:
        grade = await grade_final_message(task, event, factory, machine_spec, timeout)
        checks.append(control_result(grade, name, reward))
    return VerificationReport(checks)


def tool_verification_report(
    task: TaskSpec, *, factory: MachineFactory, machine_spec: MachineSpec, timeout: float
) -> VerificationReport:
    return asyncio.run(tool_action_checks(task, factory=factory, machine_spec=machine_spec, timeout=timeout))


def bind_tool_action(
    recipe: DatasetRecipe,
    *,
    image: str,
    factory: MachineFactory,
    machine_spec: MachineSpec,
    worker_image: str | None,
    timeout: float,
) -> DatasetRecipe:
    """Bind original action comparison; terminal event delivery is required by the runner."""
    return bind_grader_recipe(
        recipe,
        normalize=partial(normalize_tool_action, image=image, normalize_task=recipe.policy.normalize),
        verification=partial(tool_verification_report, factory=factory, machine_spec=machine_spec, timeout=timeout),
        suite_id="original-ultra-tool-action-controls",
        grader_bytes=invocation_bytes(TOOL_CALL),
        image=image,
        factory=factory,
        machine_spec=machine_spec,
        worker_image=worker_image,
        timeout=timeout,
        verifier_revision=VERIFIER_REVISION,
    )
