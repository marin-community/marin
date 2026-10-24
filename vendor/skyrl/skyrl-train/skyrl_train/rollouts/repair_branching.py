"""Teacher-repair branch construction for mixed-policy identity GRPO groups.

Given a student rollout and one verified attribution error, build the paired
branch: rewind to the clean prefix before the claim's clause,
sample a teacher continuation for the remainder of that assistant turn, then
resume the current student on subsequent turns using each branch's own history.
The original token sequence is retained; paired loss masks exclude the shared prefix. Teacher sampling parameters are
recorded verbatim in the branch metrics; failures reject the branch for audit
rather than resampling until success.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol
from uuid import uuid4

from loguru import logger
from rolloutengine.models import ModelTurn, RolloutContractError
from skyrl_gym.identity_repair.detector import AttributionDetection
from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_train.inference_engines.base import ChatContinuation, InferenceEngineInput
from skyrl_train.rollouts.training_record import TrainingRollout, TrainingStep, TrainingTransition
from skyrl_train.trajectory_runners.types import BEHAVIOR_SOURCES_KEY, TokenProvenance

STUDENT_SOURCE = 0
TEACHER_SOURCE = 1


@dataclass(frozen=True)
class TurnContinuation:
    """One model-generated remainder of an assistant turn.

    ``token_ids`` and ``logprobs`` cover only the generated continuation; the
    served prefix is excluded. ``assistant_message`` is the full merged assistant
    message (prefix content plus continuation) as the engine rendered it.
    """

    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    text: str
    stop_reason: str
    assistant_message: Mapping[str, Any]


class BranchChatClient(Protocol):
    """Chat generation against one served model, resumable from an exact prefix."""

    async def continue_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        prefix_token_ids: tuple[int, ...],
        continue_final_message: bool,
        assistant_message_index: int,
        sampling_params: Mapping[str, Any],
        max_context_length: int,
        max_prompt_length: int | None,
        chat_template_kwargs: Mapping[str, Any],
        session_id: str,
    ) -> TurnContinuation: ...


@dataclass(frozen=True)
class TeacherRepairSpec:
    """Recorded sampling distributions for the two generators and context limits."""

    teacher_sampling_params: Mapping[str, Any]
    student_sampling_params: Mapping[str, Any]
    max_context_length: int
    max_prompt_length: int | None = None
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    max_resume_turns: int = 8


@dataclass(frozen=True)
class RepairBranch:
    rollout: TrainingRollout
    pair_id: str
    boundary_token_index: int
    teacher_token_count: int
    behavior_sources: tuple[int, ...]
    teacher_stop_reason: str


@dataclass(frozen=True)
class RepairRejection:
    reason: str
    detail: str


def _decode_prefix(tokenizer, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=False)


def _claim_turn_index(original: TrainingRollout, detection: AttributionDetection, tokenizer) -> int:
    claim = detection.claim
    indices = range(len(original.steps)) if detection.turn_index is None else (detection.turn_index,)
    for index in indices:
        step = original.steps[index]
        text = (
            _decode_prefix(tokenizer, step.turn.response_token_ids)
            if detection.sampled_text is not None
            else step.turn.text or ""
        )
        if detection.sampled_text is not None and text != detection.sampled_text:
            continue
        if text[claim.char_start : claim.char_end] == claim.text:
            return index
    raise RolloutContractError(f"Claim span {claim.char_start}:{claim.char_end} does not match any assistant turn text")


def _clause_token_index(tokenizer, turn: ModelTurn, detection: AttributionDetection) -> int:
    decoded = _decode_prefix(tokenizer, turn.response_token_ids)
    # Displayed text may omit native reasoning delimiters. Locate the exact claim
    # in the sampled stream rather than treating display offsets as token offsets.
    visible = turn.text or ""
    if detection.sampled_text is not None:
        if detection.sampled_text != decoded:
            raise RolloutContractError("Detection text differs from the sampled response")
        clause_start = detection.clause_start_char
    elif decoded == visible:
        clause_start = detection.clause_start_char
    else:
        if not visible or decoded.count(visible) != 1:
            raise RolloutContractError("Cannot align displayed claim with sampled response")
        clause_start = decoded.index(visible) + detection.clause_start_char
    boundary = 0
    for index in range(len(turn.response_token_ids) + 1):
        prefix = _decode_prefix(tokenizer, turn.response_token_ids[:index])
        if len(prefix) > clause_start:
            break
        if decoded.startswith(prefix):
            boundary = index
    return boundary


async def build_teacher_repair_branch(
    original: TrainingRollout,
    detection: AttributionDetection,
    *,
    pair_id: str,
    tokenizer,
    teacher_client: BranchChatClient,
    student_client: BranchChatClient,
    spec: TeacherRepairSpec,
    session_id: str,
) -> RepairBranch | RepairRejection:
    """Build an exact-token branch; its grade must be computed independently."""
    if detection.ambiguous:
        return RepairRejection("ambiguous_span", "claim requires manual audit")
    turn_index = _claim_turn_index(original, detection, tokenizer)
    branch_step = original.steps[turn_index]
    local_boundary = _clause_token_index(tokenizer, branch_step.turn, detection)
    prefix_ids = tuple(branch_step.turn.response_token_ids[:local_boundary])
    prefix_text = _decode_prefix(tokenizer, prefix_ids)
    if detection.claim.asserted_entity.casefold() in prefix_text.casefold():
        return RepairRejection("claim_prefix_collision", "incorrect entity remains in the retained prefix")
    turn_start = branch_step.response_end + 1 - len(branch_step.turn.response_token_ids)
    boundary = turn_start + local_boundary
    served_prefix = tuple(branch_step.turn.prompt_token_ids) + prefix_ids
    messages = list(branch_step.messages[:-1])
    messages.append({"role": "assistant", "content": prefix_text})
    teacher_turn = await teacher_client.continue_turn(
        messages,
        prefix_token_ids=served_prefix,
        continue_final_message=True,
        assistant_message_index=len(messages) - 1,
        sampling_params=spec.teacher_sampling_params,
        max_context_length=spec.max_context_length,
        max_prompt_length=spec.max_prompt_length,
        chat_template_kwargs=spec.chat_template_kwargs,
        session_id=session_id + ":teacher",
    )
    if not teacher_turn.token_ids:
        return RepairRejection("empty_teacher_continuation", "teacher produced no tokens")
    if teacher_turn.prompt_token_ids != served_prefix:
        raise RolloutContractError("Teacher did not condition on the exact repair prefix")
    full_turn_ids = prefix_ids + teacher_turn.token_ids
    full_text = _decode_prefix(tokenizer, full_turn_ids)
    messages[-1] = {"role": "assistant", "content": final_answer_text(full_text)}
    ids = list(original.response_token_ids[:boundary]) + list(teacher_turn.token_ids)
    if original.logprobs is None:
        raise RolloutContractError("Repair requires recorded student behavior logprobs")
    logprobs = list(original.logprobs[:boundary]) + list(teacher_turn.logprobs)
    masks = [0] * boundary + [1] * len(teacher_turn.token_ids)
    sources = [STUDENT_SOURCE] * boundary + [TEACHER_SOURCE] * len(teacher_turn.token_ids)
    steps = list(original.steps[:turn_index])
    steps.append(
        TrainingStep(
            turn=ModelTurn(
                message=messages[-1],
                prompt_token_ids=branch_step.turn.prompt_token_ids,
                response_token_ids=full_turn_ids,
                logprobs=tuple(original.logprobs[turn_start:boundary]) + teacher_turn.logprobs,
                stop_reason=teacher_turn.stop_reason,
                text=full_text,
                metadata={
                    BEHAVIOR_SOURCES_KEY: [STUDENT_SOURCE] * local_boundary
                    + [TEACHER_SOURCE] * len(teacher_turn.token_ids)
                },
            ),
            transition=TrainingTransition(done=False),
            response_end=len(ids) - 1,
            messages=tuple(messages),
        )
    )
    later_steps = original.steps[turn_index + 1 :]
    if len(later_steps) > spec.max_resume_turns:
        return RepairRejection("resume_budget", "branch exceeds the configured turn budget")
    previous_original = branch_step
    for index, later in enumerate(later_steps):
        observations = later.messages[len(previous_original.messages) : -1]
        if any(message.get("role") != "user" for message in observations):
            raise RolloutContractError("Tool observations must be regenerated in a branch-local environment")
        last_assistant = len(messages) - 1
        messages.extend(observations)
        served = original.prompt_token_ids + tuple(ids)
        resumed = await student_client.continue_turn(
            messages,
            prefix_token_ids=served,
            continue_final_message=False,
            assistant_message_index=last_assistant,
            sampling_params=spec.student_sampling_params,
            max_context_length=spec.max_context_length,
            max_prompt_length=spec.max_prompt_length,
            chat_template_kwargs=spec.chat_template_kwargs,
            session_id=f"{session_id}:student:{index}",
        )
        if resumed.prompt_token_ids[: len(served)] != served:
            raise RolloutContractError("Resumed student rewrote the branch token history")
        observation_ids = resumed.prompt_token_ids[len(served) :]
        ids.extend(observation_ids)
        masks.extend([0] * len(observation_ids))
        logprobs.extend([0.0] * len(observation_ids))
        sources.extend([STUDENT_SOURCE] * len(observation_ids))
        ids.extend(resumed.token_ids)
        masks.extend([1] * len(resumed.token_ids))
        logprobs.extend(resumed.logprobs)
        sources.extend([STUDENT_SOURCE] * len(resumed.token_ids))
        messages.append(dict(resumed.assistant_message))
        steps.append(
            TrainingStep(
                turn=ModelTurn(
                    message=dict(resumed.assistant_message),
                    prompt_token_ids=resumed.prompt_token_ids,
                    response_token_ids=resumed.token_ids,
                    logprobs=resumed.logprobs,
                    stop_reason=resumed.stop_reason,
                    text=resumed.text,
                    metadata={BEHAVIOR_SOURCES_KEY: [STUDENT_SOURCE] * len(resumed.token_ids)},
                ),
                transition=TrainingTransition(done=False),
                response_end=len(ids) - 1,
                messages=tuple(messages),
            )
        )
        previous_original = later
    steps[-1] = replace(steps[-1], transition=TrainingTransition(done=True))
    rollout = TrainingRollout(
        task_id=original.task_id,
        messages=tuple(messages),
        prompt_token_ids=original.prompt_token_ids,
        response_token_ids=tuple(ids),
        loss_mask=tuple(masks),
        logprobs=tuple(logprobs),
        grade=GradeResult(Outcome.UNAVAILABLE, None),
        stop_reason=steps[-1].turn.stop_reason,
        steps=tuple(steps),
        metrics={
            "identity_repair": {
                "pair_id": pair_id,
                "boundary_token_index": boundary,
                "teacher_token_count": len(teacher_turn.token_ids),
                "teacher_stop_reason": teacher_turn.stop_reason,
                "teacher_sampling_params": dict(spec.teacher_sampling_params),
                "student_sampling_params": dict(spec.student_sampling_params),
                "claim": {
                    "kind": str(detection.claim.kind),
                    "text": detection.claim.text,
                    "char_start": detection.claim.char_start,
                    "char_end": detection.claim.char_end,
                },
                "judge_version": detection.judge_version,
                "clause_start_char": detection.clause_start_char,
            }
        },
    )
    return RepairBranch(
        rollout, pair_id, boundary, len(teacher_turn.token_ids), tuple(sources), teacher_turn.stop_reason
    )


async def try_build_teacher_repair_branch(
    original: TrainingRollout,
    detection: AttributionDetection,
    *,
    pair_id: str,
    tokenizer,
    teacher_client: BranchChatClient,
    student_client: BranchChatClient,
    spec: TeacherRepairSpec,
    session_id: str,
) -> RepairBranch | RepairRejection:
    """Build the branch, converting generator failures into recorded rejections.

    Repairs are best-effort by policy: a teacher failure keeps the original
    rollout and records the failure instead of discarding the group or
    resampling until success.
    """
    try:
        return await build_teacher_repair_branch(
            original,
            detection,
            pair_id=pair_id,
            tokenizer=tokenizer,
            teacher_client=teacher_client,
            student_client=student_client,
            spec=spec,
            session_id=session_id,
        )
    except (RolloutContractError, RuntimeError, ValueError) as error:
        logger.warning(f"repair pair {pair_id} rejected: {type(error).__name__}: {error}")
        return RepairRejection("generator_error", f"{type(error).__name__}: {error}")


async def expand_identity_repair_group(
    original: TrainingRollout,
    detections: Sequence[AttributionDetection],
    *,
    tokenizer,
    teacher_client: BranchChatClient,
    student_client: BranchChatClient,
    spec: TeacherRepairSpec,
    session_id: str,
) -> list[TrainingRollout]:
    """Return the original rollout plus at most one repair branch.

    Repair at most one verified error per rollout. Ambiguous detections and
    generator failures are recorded in the returned rollouts' metrics as
    rejections and never silently dropped.
    """
    rollouts = [original]
    for detection in detections:
        result = await try_build_teacher_repair_branch(
            original,
            detection,
            pair_id=f"{original.task_id}:{uuid4().hex[:8]}",
            tokenizer=tokenizer,
            teacher_client=teacher_client,
            student_client=student_client,
            spec=spec,
            session_id=session_id,
        )
        if isinstance(result, RepairBranch):
            rollouts[0] = replace(
                original,
                loss_mask=(0,) * result.boundary_token_index + original.loss_mask[result.boundary_token_index :],
                metrics={
                    **original.metrics,
                    "identity_repair_pair": {
                        "pair_id": result.pair_id,
                        "boundary_token_index": result.boundary_token_index,
                    },
                },
            )
            rollouts.append(result.rollout)
            break
        original.metrics.setdefault("identity_repair_rejections", []).append(
            {"reason": result.reason, "detail": result.detail}
        )
        if result.reason == "ambiguous_span":
            break
    return rollouts


def assert_finite_logprobs(values: Sequence[float], context: str) -> None:
    """Reject non-finite behavior logprobs before they reach the trainer."""
    if any(not math.isfinite(value) for value in values):
        raise RolloutContractError(f"Behavior logprobs must be finite ({context})")


class ModelClientBranchChat:
    """Continue an exact served token prefix through the model client chat API."""

    def __init__(self, client) -> None:
        self._client = client

    async def continue_turn(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        prefix_token_ids: tuple[int, ...],
        continue_final_message: bool,
        assistant_message_index: int,
        sampling_params: Mapping[str, Any],
        max_context_length: int,
        max_prompt_length: int | None,
        chat_template_kwargs: Mapping[str, Any],
        session_id: str,
    ) -> TurnContinuation:
        continuation = ChatContinuation(
            served_prefix_token_ids=list(prefix_token_ids),
            assistant_message_index=assistant_message_index,
            continue_final_message=continue_final_message,
        )
        params = dict(sampling_params)
        params.setdefault("logprobs", 0)
        model_input = InferenceEngineInput(
            prompts=[list(messages)],
            chat_completion_params=[{"chat_template_kwargs": dict(chat_template_kwargs)}],
            chat_continuations=[continuation],
            session_ids=[session_id],
            sampling_params=params,
            max_context_length=max_context_length,
            max_prompt_length=max_prompt_length,
        )
        output = await self._client.generate(model_input)
        if output["token_provenance"] != TokenProvenance.ENGINE:
            raise RolloutContractError("The canonical rollout engine requires exact model tokens")
        logprobs = output.get("response_logprobs")
        if logprobs is None or len(logprobs[0]) != len(output["response_ids"][0]):
            raise RolloutContractError("Generator must return one behavior logprob per sampled token")
        assert_finite_logprobs(logprobs[0], session_id)
        return TurnContinuation(
            prompt_token_ids=tuple(output["prompt_ids"][0]),
            token_ids=tuple(output["response_ids"][0]),
            logprobs=tuple(logprobs[0]),
            text=output.get("responses", [""])[0],
            stop_reason=output["stop_reasons"][0],
            assistant_message=output["assistant_messages"][0],
        )
