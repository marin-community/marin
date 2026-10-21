"""Durable single-turn answers and bounded grading without another model request."""

import asyncio
import hashlib
import json
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, ConfigDict
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.engine import RolloutEngine
from rolloutengine.harness.prefill import PrefillHarness
from rolloutengine.models import GradedRolloutInterrupted, ModelRequest, ModelTurn, RolloutOperation, TaskRolloutSpec
from shellbox.machine import MachineFactory
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import (
    CONVERSATION_ANSWERS,
    ScriptGrader,
    VerifyitGrader,
    grades_with_judge,
    verifyit_spec,
)
from verifyit.spec import StructureClosureSpec
from skyrl_gym.rollout_specs import is_shell_harness, shell_harness


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def spec_digest(spec: TaskRolloutSpec) -> str:
    return hashlib.sha256(json_bytes(spec.model_dump(mode="json"))).hexdigest()


def require_regradable(spec: TaskRolloutSpec) -> None:
    """Limit saved-answer recovery to complete single-turn conversation submissions."""
    grader = spec.task.grader
    if (
        spec.limits.max_turns != 1
        or spec.task_machine is not None
        or not (spec.harness == "prefill" or is_shell_harness(spec.harness))
        or spec.task.interaction_tools
        or spec.task.tool_bindings
        or spec.task.environment_requirements.tool_providers
        or spec.task.output_paths
        or spec.task.answer_type not in CONVERSATION_ANSWERS
        or not isinstance(grader, ScriptGrader | VerifyitGrader)
    ):
        raise ValueError("Saved-answer grading requires one turn and no captured output files")
    if isinstance(grader, VerifyitGrader) and (
        grades_with_judge(grader)
        or (grader.environment is None and not isinstance(verifyit_spec(grader), StructureClosureSpec))
    ):
        raise ValueError("Saved-answer recovery supports sandbox graders and host structure_closure only")
    if isinstance(grader, ScriptGrader) and (grader.collect or grader.artifacts):
        raise ValueError("Saved-answer grading cannot recapture the original agent machine")


class SavedAnswer(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spec: TaskRolloutSpec
    spec_sha256: str
    instance_id: str
    repetition_id: int
    messages: tuple[dict[str, Any], ...]
    request_options: dict[str, Any]
    response_text: str
    response_logprobs: tuple[float, ...] | None
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    stop_reason: str


class AnswerAdmission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    answer_sha256: str
    eligible: bool
    failure_operation: str | None
    failure_type: str | None
    grade_status: str


def answer_digest(record: SavedAnswer) -> str:
    return hashlib.sha256(json_bytes(record.model_dump(mode="json"))).hexdigest()


def answer_uri(root: str, instance_id: str, repetition_id: int = 0) -> str:
    key = hashlib.sha256(json_bytes([instance_id, repetition_id])).hexdigest()
    return str(StoragePath(root) / f"{key}.json")


def save_answer(
    root: str,
    spec: TaskRolloutSpec,
    instance_id: str,
    repetition_id: int,
    request: ModelRequest,
    turn: ModelTurn,
) -> None:
    require_regradable(spec)
    record = SavedAnswer(
        spec=spec,
        spec_sha256=spec_digest(spec),
        instance_id=instance_id,
        repetition_id=repetition_id,
        messages=(*request.messages, turn.message),
        request_options=request.options,
        response_text=turn.text,
        response_logprobs=turn.logprobs,
        prompt_token_ids=turn.prompt_token_ids,
        response_token_ids=turn.response_token_ids,
        stop_reason=turn.stop_reason,
    )
    conditional_object(answer_uri(root, instance_id, repetition_id)).write(
        json_bytes(record.model_dump(mode="json")), expected_version=None
    )


def admit_answer(
    root: str,
    instance_id: str,
    repetition_id: int,
    messages: tuple[dict[str, Any], ...],
    failure_operation: str | None,
    failure_type: str | None,
    grade_status: str,
) -> None:
    """Keep invalid transport/harness results ineligible even if their raw answer was saved."""
    uri = answer_uri(root, instance_id, repetition_id)
    if not StoragePath(uri).exists():
        return  # A failed generation produced no completed answer to grade.
    record = SavedAnswer.model_validate_json(StoragePath(uri).read_bytes())
    admission = AnswerAdmission(
        answer_sha256=answer_digest(record),
        eligible=failure_operation in {None, "grade"} and record.messages == messages,
        failure_operation=failure_operation,
        failure_type=failure_type,
        grade_status=grade_status,
    )
    conditional_object(uri + ".admission.json").write(json_bytes(admission.model_dump()), expected_version=None)


async def regrade_answer(
    record: SavedAnswer,
    original_spec: TaskRolloutSpec,
    factories: dict[str, MachineFactory],
    *,
    timeout: float,
) -> GradeResult:
    """Run the original grader against the exact messages; never construct a model client."""
    require_regradable(original_spec)
    if record.spec_sha256 != spec_digest(original_spec) or spec_digest(record.spec) != record.spec_sha256:
        raise ValueError("Saved answer differs from the original complete task rollout specification")
    calls = 0

    async def saved_turn(request: ModelRequest) -> ModelTurn:
        nonlocal calls
        calls += 1
        if calls != 1 or request.messages != record.messages[:-1] or request.options != record.request_options:
            raise ValueError("Original harness requested a different conversation or another model turn")
        return ModelTurn(
            record.messages[-1],
            record.prompt_token_ids,
            record.response_token_ids,
            record.response_logprobs,
            record.stop_reason,
            text=record.response_text,
        )

    harness = PrefillHarness() if original_spec.harness == "prefill" else shell_harness(original_spec.harness)
    engine = RolloutEngine(factories, {original_spec.harness: harness})
    try:
        async with asyncio.timeout(timeout):
            result = await engine.run_task(original_spec, saved_turn)
    except GradedRolloutInterrupted as error:
        if error.operation != RolloutOperation.GRADE:
            return GradeResult(Outcome.INFRA_ERROR, None, f"Saved-answer replay failed during {error.operation}")
        return GradeResult(Outcome.INFRA_ERROR, None, type(error.__cause__).__name__)
    if (
        calls != 1
        or len(result.rollout.conversations) != 1
        or result.rollout.conversations[0].messages != record.messages
    ):
        raise ValueError("Original harness replay did not preserve the one saved conversation")
    return result.grade


async def regrade_once(
    record: SavedAnswer,
    admission: AnswerAdmission,
    original_spec: TaskRolloutSpec,
    factories: dict[str, MachineFactory],
    receipt_root: str,
    *,
    timeout: float,
) -> dict:
    """Reserve one immutable attempt before grading; an interrupted reservation is never retried."""
    if admission.answer_sha256 != answer_digest(record) or not admission.eligible:
        raise ValueError("Saved answer was not admitted by the original rollout harness")
    if admission.failure_operation != "grade" and admission.grade_status != Outcome.INFRA_ERROR:
        raise ValueError("Only an original grader infrastructure failure can consume the saved-answer attempt")
    if original_spec.harness == "prefill" and any(
        record.messages[-1].get(channel) for channel in ("reasoning_content", "reasoning", "tool_calls")
    ):
        raise ValueError("A parsed prefill channel cannot become a plain-text saved answer")
    root = (
        StoragePath(receipt_root) / hashlib.sha256(json_bytes([record.instance_id, record.repetition_id])).hexdigest()
    )
    started = str(root / "started.json")
    result_uri = str(root / "result.json")
    if record.spec_sha256 != spec_digest(original_spec) or spec_digest(record.spec) != record.spec_sha256:
        raise ValueError("Saved answer differs from the original complete task rollout specification")
    proof = {
        "answer_sha256": answer_digest(record),
        "spec_sha256": spec_digest(original_spec),
        "stop_reason": record.stop_reason,
        "timeout": timeout,
        "regenerated_completions": 0,
    }
    if StoragePath(started).exists():
        if json.loads(StoragePath(started).read_text()) != proof:
            raise ValueError("Saved-answer grading reservation has different provenance or bounds")
        if StoragePath(result_uri).exists():
            result = json.loads(StoragePath(result_uri).read_text())
            if any(result.get(key) != value for key, value in proof.items()):
                raise ValueError("Saved-answer grading receipt has different provenance or bounds")
            return result
        return {"status": Outcome.UNAVAILABLE, "reward": None, "error": "Previous saved-answer attempt is incomplete"}
    conditional_object(started).write(json_bytes(proof), expected_version=None)
    try:
        grade = await regrade_answer(record, original_spec, factories, timeout=timeout)
    except (TimeoutError, OSError, RuntimeError) as error:
        # A bounded infrastructure failure remains an unscored attempt in the receipt.
        grade = GradeResult(Outcome.INFRA_ERROR, None, type(error).__name__)
    receipt = {**proof, **asdict(grade)}
    conditional_object(result_uri).write(json_bytes(receipt), expected_version=None)
    return receipt
