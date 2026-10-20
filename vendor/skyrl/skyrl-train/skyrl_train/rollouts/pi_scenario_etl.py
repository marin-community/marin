"""Pure ETL for a single, fully hydrated Harbor/Pi trial payload.

The input is one validated object, not a join of Pi text, model requests and
teacher logs. Harbor owns task execution and verification; Pi must collate its
private and executed turns before handing this envelope to SkyRL.
"""

import math
from enum import IntEnum, StrEnum
from typing import Any, Literal, Mapping

import numpy as np
from harbor.models.agent.pi_trajectory import same_pi_assistant
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from skyrl_gym.verification import VerificationResult

from skyrl_train.trajectory_runners.types import TrajectoryBatch


def _token_ids(ids: Any) -> tuple[int, ...]:
    # Hugging Face may return BatchEncoding rather than a bare list even when
    # apply_chat_template(tokenize=True) is called without return_dict=True.
    if isinstance(ids, Mapping):
        ids = ids.get("input_ids")
    if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Pi SFT requires flat student-tokenizer token IDs")
    return tuple(ids)


class PiScenario(StrEnum):
    VERIFIER_GRPO = "verifier_grpo_base"
    DISTRIBUTION = "teacher_distribution_pi_base"
    RLAIF = "rlaif_pi_base"
    TEACHER_TRACE = "teacher_demo_pi_base"
    TEACHER_TRACE_VERIFIED = "teacher_demo_pi_verified_base"
    DAGGER_MIXTURE = "dagger_pi_turn_mixture_base"
    DAGGER_TAKEOVER = "dagger_pi_prefix_takeover_base"
    CRITIQUE_REVISION = "critique_revision_pi_base"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "PiScenario":
        """Require the sole reviewed scenario selector, not arbitrary ETL options."""
        if not isinstance(config, Mapping) or set(config) != {"preset"}:
            raise ValueError("Pi ETL requires exactly scenario.preset")
        return cls(config["preset"])


class TokenSource(IntEnum):
    CONTEXT = 0
    TEACHER = 1
    STUDENT = 2
    REVISION = 3


class _Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelEvidence(_Evidence):
    request_id: str
    model: str
    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    chat_template_kwargs: dict[str, Any]
    assistant: dict[str, Any]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None

    @model_validator(mode="after")
    def complete(self) -> "ModelEvidence":
        if (
            not self.request_id
            or not self.model
            or not self.messages
            or self.assistant.get("role") != "assistant"
            or set(self.chat_template_kwargs) & {"tools", "tokenize", "add_generation_prompt", "messages"}
            or not self.prompt_token_ids
            or not self.response_token_ids
            or (self.logprobs is not None and len(self.response_token_ids) != len(self.logprobs))
            or any(type(value) is not int or value < 0 for value in (*self.prompt_token_ids, *self.response_token_ids))
            or (self.logprobs is not None and any(not math.isfinite(value) for value in self.logprobs))
        ):
            raise ValueError(
                "Pi model evidence needs exact sampled tokens and an assistant response; present logprobs must align"
            )
        return self


class ToolObservation(_Evidence):
    tool_call_id: str
    content: str
    authored_by: Literal["student", "teacher"]


class PiTurn(_Evidence):
    index: int
    actor: Literal["student", "teacher", "revision"]
    executed: ModelEvidence
    executed_assistant: dict[str, Any]
    observations: tuple[ToolObservation, ...] = ()
    teacher_label: ModelEvidence | None = None

    @model_validator(mode="after")
    def check_observations(self) -> "PiTurn":
        if self.actor != "teacher" and self.executed.logprobs is None:
            raise ValueError("Pi student and revision behavior requires exact sampled logprobs")
        if not same_pi_assistant(self.executed.assistant, self.executed_assistant):
            raise ValueError("Pi executed assistant differs from the sampled model action")
        calls = self.executed_assistant.get("tool_calls") or []
        if not isinstance(calls, list) or len(calls) != len(self.observations):
            raise ValueError("Pi turn must include every executed tool observation")
        for call, observation in zip(calls, self.observations, strict=True):
            if (
                not isinstance(call, dict)
                or call.get("id") != observation.tool_call_id
                or observation.authored_by != ("teacher" if self.actor == "teacher" else "student")
            ):
                raise ValueError("Pi tool observation author or call ID differs from its executable turn")
        if self.teacher_label is not None:
            if (
                self.teacher_label.messages != self.executed.messages
                or self.teacher_label.tools != self.executed.tools
                or self.teacher_label.chat_template_kwargs != self.executed.chat_template_kwargs
            ):
                raise ValueError("Pi teacher label must see the exact visited pre-action context")
            if self.actor == "teacher" and self.teacher_label != self.executed:
                raise ValueError("Pi teacher-executed action must be the sampled private label")
            if self.actor != "teacher" and self.teacher_label.request_id == self.executed.request_id:
                raise ValueError("Pi student action must not reuse its private teacher label")
        return self


class JudgeEvidence(_Evidence):
    request_id: str
    trial_id: str
    student_request_ids: tuple[str, ...]
    model: str
    principle: str
    prompt_revision: Literal["rlaif_judge_v1"]
    score: Literal[0, 1]

    @field_validator("score", mode="before")
    @classmethod
    def exact_binary_score(cls, value: Any) -> int:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Pi judge score must be an exact binary integer")
        return value


class CritiqueEvidence(_Evidence):
    trial_id: str
    teacher_model: str
    student_request_id: str
    teacher_request_id: str
    teacher_request: dict[str, Any]
    teacher_response: dict[str, Any]
    feedback: str

    @model_validator(mode="after")
    def attributed(self) -> "CritiqueEvidence":
        if (
            not self.trial_id
            or not self.teacher_model
            or not self.student_request_id
            or not self.teacher_request_id
            or not self.feedback.strip()
            or self.teacher_request.get("model") != self.teacher_model
            or self.teacher_request.get("stream") is not False
            or self.teacher_response.get("role") != "assistant"
            or not isinstance(self.teacher_response.get("content"), str)
            or not self.teacher_response["content"].strip()
            or self.teacher_response["content"].strip() not in self.feedback
        ):
            raise ValueError("Pi private critique requires teacher-call provenance and student-visible feedback")
        return self


class HydratedPiTrial(_Evidence):
    schema_version: Literal[1]
    trial_id: str
    policy_step: int
    turns: tuple[PiTurn, ...]
    planned_executors: tuple[Literal["teacher", "student"], ...] | None = None
    critique_feedback: str | None = None
    critique: CritiqueEvidence | None = None
    judge: JudgeEvidence | None = None
    verifier_status: Literal["graded", "submission_failure", "skipped"]
    verifier_reward: float | None
    verifier_passed: bool | None

    @model_validator(mode="after")
    def check_continuation(self) -> "HydratedPiTrial":
        if not self.trial_id or type(self.policy_step) is not int or self.policy_step < 0 or not self.turns:
            raise ValueError("Pi trial requires a trial ID, policy step and turns")
        if self.verifier_status == "skipped":
            if self.verifier_reward is not None or self.verifier_passed is not None:
                raise ValueError("Skipped Pi verification cannot carry a verifier verdict")
        elif self.verifier_reward is None or not math.isfinite(self.verifier_reward):
            raise ValueError("Graded Pi trial requires a finite verifier reward")
        request_ids: set[str] = set()
        for index, turn in enumerate(self.turns):
            if turn.index != index or turn.executed.request_id in request_ids:
                raise ValueError("Pi turns need ordered, unique executed requests")
            request_ids.add(turn.executed.request_id)
            if turn.teacher_label is not None and turn.actor != "teacher":
                if turn.teacher_label.request_id in request_ids:
                    raise ValueError("Pi private teacher labels need unique request IDs")
                request_ids.add(turn.teacher_label.request_id)
            if index:
                previous = self.turns[index - 1]
                executed_messages = (
                    *previous.executed.messages,
                    previous.executed_assistant,
                    *(
                        {"role": "tool", "tool_call_id": result.tool_call_id, "content": result.content}
                        for result in previous.observations
                    ),
                )
                current_messages = turn.executed.messages
                if current_messages[: len(executed_messages)] != executed_messages:
                    raise ValueError("Pi next visited state must contain only the executed action and observations")
                additions = current_messages[len(executed_messages) :]
                if additions != (
                    ()
                    if self.critique_feedback is None
                    else ({"role": "user", "content": [{"type": "text", "text": self.critique_feedback}]},)
                ):
                    raise ValueError("Pi continuation has missing or unapproved student-visible feedback")
        if self.critique is not None:
            if (
                self.critique.trial_id != self.trial_id
                or self.critique.feedback != self.critique_feedback
                or len(self.turns) != 2
                or self.critique.student_request_id != self.turns[0].executed.request_id
            ):
                raise ValueError("Pi private critique must belong to the executed student attempt and this trial")
            first = self.turns[0]
            visited = (
                *first.executed.messages,
                first.executed.assistant,
                *(
                    {"role": "tool", "tool_call_id": item.tool_call_id, "content": item.content}
                    for item in first.observations
                ),
            )
            teacher_messages = self.critique.teacher_request.get("messages")
            if (
                not isinstance(teacher_messages, list)
                or teacher_messages[: len(visited)] != list(visited)
                or len(teacher_messages) != len(visited) + 1
                or teacher_messages[-1].get("role") != "user"
                or teacher_messages[-1] in self.turns[-1].executed.messages
            ):
                raise ValueError("Pi critique must use executed observations without leaking private instruction")
        elif self.critique_feedback is not None:
            raise ValueError("Pi feedback without its private teacher source is incomplete")
        if self.judge is not None and (
            not self.judge.request_id
            or not self.judge.model
            or not self.judge.principle
            or self.judge.trial_id != self.trial_id
            or self.judge.student_request_ids != tuple(turn.executed.request_id for turn in self.turns)
        ):
            raise ValueError("Pi judgment must identify this complete student trajectory")
        return self


class TrainingRow(_Evidence):
    prompt_token_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    token_sources: tuple[TokenSource, ...]
    optimization_reward: float
    verifier_reward: float | None
    behavior_logprobs: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def aligned(self) -> "TrainingRow":
        if (
            not self.prompt_token_ids
            or not self.response_ids
            or len(self.response_ids) != len(self.loss_mask)
            or len(self.response_ids) != len(self.token_sources)
            or any(value not in (0, 1) for value in self.loss_mask)
            or not any(self.loss_mask)
            or (self.behavior_logprobs is not None and len(self.behavior_logprobs) != len(self.response_ids))
        ):
            raise ValueError("Pi row needs aligned nonempty student-tokenizer IDs, masks and attribution")
        return self


class TrainingRows(_Evidence):
    trial_id: str
    scenario: PiScenario
    rows: tuple[TrainingRow, ...]
    verifier_reward: float | None
    verifier_passed: bool | None
    evidence_messages: tuple[dict[str, Any], ...]
    judge_score: int | None = None


def _teacher_row(label: ModelEvidence, tokenizer: Any, reward: float | None) -> TrainingRow:
    options = {"tokenize": True, "tools": list(label.tools), **label.chat_template_kwargs}
    prefix = _token_ids(tokenizer.apply_chat_template(list(label.messages), add_generation_prompt=True, **options))
    full = _token_ids(
        tokenizer.apply_chat_template([*label.messages, label.assistant], add_generation_prompt=False, **options)
    )
    if not prefix or full[: len(prefix)] != prefix or len(full) == len(prefix):
        raise ValueError("Teacher response does not align with the student tokenizer's visited context")
    response = tuple(full[len(prefix) :])
    return TrainingRow(
        prompt_token_ids=tuple(prefix),
        response_ids=response,
        loss_mask=(1,) * len(response),
        token_sources=(TokenSource.TEACHER,) * len(response),
        optimization_reward=0.0 if reward is None else reward,
        verifier_reward=reward,
    )


def _teacher_trajectory_row(turns: tuple[PiTurn, ...], tokenizer: Any, reward: float | None) -> TrainingRow:
    """Keep one physical SFT row per trial; mask tool/user context between labels."""
    # Pi may canonicalize tool-call JSON before execution. Train SFT on the
    # validated executed rendering so each label matches the next visited
    # student-tokenizer prefix; the sampled teacher evidence remains untouched.
    segments = [
        _teacher_row(turn.executed.model_copy(update={"assistant": turn.executed_assistant}), tokenizer, reward)
        for turn in turns
    ]
    prompt = segments[0].prompt_token_ids
    tokens = prompt
    response: list[int] = []
    mask: list[int] = []
    sources: list[TokenSource] = []
    for segment in segments:
        if segment.prompt_token_ids[: len(tokens)] != tokens:
            raise ValueError("Pi teacher trace cannot assemble a continuous student-tokenizer prefix")
        context = segment.prompt_token_ids[len(tokens) :]
        response.extend((*context, *segment.response_ids))
        mask.extend((0,) * len(context) + (1,) * len(segment.response_ids))
        sources.extend((TokenSource.CONTEXT,) * len(context) + (TokenSource.TEACHER,) * len(segment.response_ids))
        tokens = (*segment.prompt_token_ids, *segment.response_ids)
    return TrainingRow(
        prompt_token_ids=prompt,
        response_ids=tuple(response),
        loss_mask=tuple(mask),
        token_sources=tuple(sources),
        optimization_reward=0.0 if reward is None else reward,
        verifier_reward=reward,
    )


def _student_row(turn: PiTurn, reward: float, verifier_reward: float, source: TokenSource) -> TrainingRow:
    sampled = turn.executed
    if sampled.logprobs is None:
        raise ValueError("Pi student behavior needs exact sampled logprobs")
    return TrainingRow(
        prompt_token_ids=sampled.prompt_token_ids,
        response_ids=sampled.response_token_ids,
        loss_mask=(1,) * len(sampled.response_token_ids),
        token_sources=(source,) * len(sampled.response_token_ids),
        optimization_reward=reward,
        verifier_reward=verifier_reward,
        behavior_logprobs=sampled.logprobs,
    )


def _student_trajectory_row(turns: tuple[PiTurn, ...], reward: float, verifier_reward: float) -> TrainingRow:
    """Assemble exact served prefixes for objectives requiring one row per trial."""
    prompt = turns[0].executed.prompt_token_ids
    tokens = prompt
    response: list[int] = []
    mask: list[int] = []
    sources: list[TokenSource] = []
    logprobs: list[float] = []
    for turn in turns:
        sampled = turn.executed
        if sampled.logprobs is None:
            raise ValueError("Pi student behavior needs exact sampled logprobs")
        if sampled.prompt_token_ids[: len(tokens)] != tokens:
            raise ValueError("Pi GRPO cannot assemble a continuous served token prefix")
        context = sampled.prompt_token_ids[len(tokens) :]
        response.extend((*context, *sampled.response_token_ids))
        mask.extend((0,) * len(context) + (1,) * len(sampled.response_token_ids))
        sources.extend((TokenSource.CONTEXT,) * len(context) + (TokenSource.STUDENT,) * len(sampled.response_token_ids))
        logprobs.extend((0.0,) * len(context) + sampled.logprobs)
        tokens = (*sampled.prompt_token_ids, *sampled.response_token_ids)
    return TrainingRow(
        prompt_token_ids=prompt,
        response_ids=tuple(response),
        loss_mask=tuple(mask),
        token_sources=tuple(sources),
        optimization_reward=reward,
        verifier_reward=verifier_reward,
        behavior_logprobs=tuple(logprobs),
    )


def scenario_training_rows(
    scenario: PiScenario,
    trial: HydratedPiTrial,
    student_tokenizer: Any,
    *,
    expected_trial_id: str,
    expected_policy_step: int,
) -> TrainingRows:
    """Convert a *single* collated payload to objective-specific, attributed rows."""
    if trial.trial_id != expected_trial_id or trial.policy_step != expected_policy_step:
        raise ValueError("Pi trial does not match its leased identity and policy step")
    turns = trial.turns
    rows: list[TrainingRow]
    if trial.verifier_status == "skipped" and scenario is not PiScenario.TEACHER_TRACE:
        raise ValueError("Only unfiltered Pi teacher SFT can skip verification")
    if scenario in (PiScenario.DAGGER_MIXTURE, PiScenario.DAGGER_TAKEOVER):
        assert trial.verifier_reward is not None
        if trial.judge is not None or trial.critique_feedback is not None:
            raise ValueError("Pi DAgger cannot contain judge or critique evidence")
        expected = "turn_mixture" if scenario is PiScenario.DAGGER_MIXTURE else "prefix_takeover"
        if trial.planned_executors is None or len(trial.planned_executors) != len(turns):
            raise ValueError("Pi DAgger needs one planned executor per visited state")
        if scenario is PiScenario.DAGGER_TAKEOVER and any(
            a == "teacher" and b == "student" for a, b in zip(trial.planned_executors, trial.planned_executors[1:])
        ):
            raise ValueError("Pi prefix takeover cannot switch back to student")
        for turn, actor in zip(turns, trial.planned_executors, strict=True):
            if turn.actor != actor or turn.teacher_label is None:
                raise ValueError(f"Pi {expected} needs the planned actor and a private label at every state")
        rows = [
            _teacher_row(turn.teacher_label, student_tokenizer, trial.verifier_reward)
            for turn in turns
            if scenario is PiScenario.DAGGER_MIXTURE or turn.actor == "teacher"
        ]
    elif scenario in (PiScenario.TEACHER_TRACE, PiScenario.TEACHER_TRACE_VERIFIED):
        if scenario is PiScenario.TEACHER_TRACE_VERIFIED and (
            trial.verifier_status != "graded" or trial.verifier_passed is None
        ):
            raise ValueError("Verified Pi teacher SFT needs an explicit graded pass/fail verdict")
        if trial.judge is not None or trial.critique_feedback is not None:
            raise ValueError("Teacher trace cannot contain judge or critique evidence")
        if trial.planned_executors is not None or any(
            turn.actor != "teacher" or turn.teacher_label is not None for turn in turns
        ):
            raise ValueError("Teacher trace requires only executed teacher actions, no DAgger labels")
        rows = [_teacher_trajectory_row(turns, student_tokenizer, trial.verifier_reward)]
    elif scenario is PiScenario.CRITIQUE_REVISION:
        assert trial.verifier_reward is not None
        if (
            len(turns) != 2
            or [turn.actor for turn in turns] != ["student", "revision"]
            or not trial.critique_feedback
            or trial.critique is None
            or trial.judge is not None
            or trial.planned_executors is not None
            or any(turn.teacher_label is not None for turn in turns)
            or turns[-1].observations
            or turns[-1].executed_assistant.get("tool_calls")
        ):
            raise ValueError(
                "Pi critique requires a student attempt, private feedback and one tool-free student revision"
            )
        rows = [_student_row(turns[-1], trial.verifier_reward, trial.verifier_reward, TokenSource.REVISION)]
    else:
        assert trial.verifier_reward is not None
        if (
            any(turn.actor != "student" or turn.teacher_label is not None for turn in turns)
            or trial.planned_executors is not None
            or trial.critique is not None
            or trial.critique_feedback is not None
        ):
            raise ValueError("Pi GRPO requires student-only behavior evidence")
        if scenario is PiScenario.RLAIF:
            if trial.judge is None or not trial.judge.request_id or not trial.judge.model:
                raise ValueError("RLAIF requires a complete, separate post-trial judgment")
            reward = float(trial.judge.score)
        else:
            if trial.judge is not None:
                if scenario is PiScenario.DISTRIBUTION:
                    raise ValueError("Pi distribution must not contain a judge result")
                raise ValueError("Verifier GRPO must not substitute a teacher judgment")
            reward = trial.verifier_reward
        # Tool calls and observations may re-render or truncate the next
        # served prompt. Both the GRPO loss and the distribution teacher
        # must see each turn's exact served IDs, not an invented prefix.
        rows = (
            [_student_row(turn, reward, trial.verifier_reward, TokenSource.STUDENT) for turn in turns]
            if scenario in (PiScenario.VERIFIER_GRPO, PiScenario.DISTRIBUTION)
            else [_student_trajectory_row(turns, reward, trial.verifier_reward)]
        )
    if not rows:
        raise ValueError("Pi trial has no eligible training rows")
    last = trial.turns[-1]
    transcript = (
        *last.executed.messages,
        last.executed_assistant,
        *({"role": "tool", "tool_call_id": item.tool_call_id, "content": item.content} for item in last.observations),
    )
    return TrainingRows(
        trial_id=trial.trial_id,
        scenario=scenario,
        rows=tuple(rows),
        verifier_reward=trial.verifier_reward,
        verifier_passed=trial.verifier_passed,
        evidence_messages=transcript,
        judge_score=trial.judge.score if trial.judge is not None else None,
    )


def training_rows_to_batch(result: TrainingRows) -> TrajectoryBatch:
    """Build optimizer rows while retaining verifier/judge and actor channels separately."""
    rows = result.rows
    with_behavior = [row.behavior_logprobs is not None for row in rows]
    if any(with_behavior) and not all(with_behavior):
        raise ValueError("Pi training rows cannot mix behavior-policy and teacher SFT tokens")
    verifier = (
        VerificationResult.skipped("Pi teacher trace verification skipped")
        if result.verifier_reward is None
        else VerificationResult.verified(result.verifier_reward, passed=result.verifier_passed)
    )
    return TrajectoryBatch(
        prompt_token_ids=[list(row.prompt_token_ids) for row in rows],
        response_ids=[list(row.response_ids) for row in rows],
        rewards=[[0.0] * (len(row.response_ids) - 1) + [row.optimization_reward] for row in rows],
        unshaped_rewards=[0.0 if row.verifier_reward is None else row.verifier_reward for row in rows],
        unshaped_reward_available=[row.verifier_reward is not None for row in rows],
        verification_results=[verifier] * len(rows),
        evidence_messages=[list(result.evidence_messages) for _ in rows],
        loss_masks=[
            [0] * len(row.loss_mask)
            if result.scenario is PiScenario.TEACHER_TRACE_VERIFIED and result.verifier_passed is False
            else list(row.loss_mask)
            for row in rows
        ],
        token_sources=[[int(source) for source in row.token_sources] for row in rows],
        rollout_logprobs=(
            [np.asarray(row.behavior_logprobs, dtype=np.float32) for row in rows] if all(with_behavior) else None
        ),
        data_sources=[None] * len(rows),
        reward_shaping_components=None,
        reward_shaping_loop_spans=None,
        loop_advantages=None,
        reward_shaping_versions=None,
        verifier_tests=None,
        stop_reasons=["complete"] * len(rows),
        exception_types=[None] * len(rows),
        error_treatments=[None] * len(rows),
        server_errors=[None] * len(rows),
        rollout_metrics={"pi/judged_score": float(result.judge_score)} if result.judge_score is not None else {},
        student_topk_indices=None,
        behavior_topk_logprobs=None,
        rollout_routed_experts=None,
        teacher_evidence=None,
        distillation=None,
        token_level_shaping=None,
        response_span_tags=None,
        trajectory_ids=None,
        teacher_route_keys=None,
        is_last_step=[index == len(rows) - 1 for index in range(len(rows))],
        exclude_from_baseline=[False] * len(rows),
    )


def scenario_rows_from_config(
    scenario_config: Mapping[str, Any],
    payload: Mapping[str, Any],
    student_tokenizer: Any,
    *,
    expected_trial_id: str,
    expected_policy_step: int,
) -> TrainingRows:
    """Select the reviewed ETL contract and validate one collated Pi trial."""
    return scenario_training_rows(
        PiScenario.from_config(scenario_config),
        HydratedPiTrial.model_validate(payload),
        student_tokenizer,
        expected_trial_id=expected_trial_id,
        expected_policy_step=expected_policy_step,
    )
