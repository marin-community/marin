"""Convert common rollout records into whole-trajectory or per-step training data."""

from collections.abc import Sequence
import copy
from dataclasses import dataclass, replace

import numpy as np
from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase
from skyrl_gym.verification import (
    VERIFIER_RUNTIME_ERROR,
    RewardResult,
    TrainingDisposition,
    VerificationResult,
)
from taskcompendium.grading_result import GradeResult, Outcome
from skyrl_gym.task_records import grade_skipped, skipped_grade
from rolloutengine.models import RolloutContractError
from skyrl_train.rollouts.training_record import TrainingRollout

from skyrl_train.error_treatment import ErrorTreatment
from skyrl_train.dataset.tasks import TASKCOMPENDIUM_ENVIRONMENT
from skyrl_train.distillation import INVALID_TOPK_INDEX
from skyrl_train.inference_engines.base import UNSTAMPED_POLICY_STEP
from skyrl_train.metric_names import TASK_ROLLOUT_METRIC_PREFIX
from skyrl_train.trajectory_runners.trajectory_processing import (
    apply_overlong_filtering,
    get_batch_failure_metrics,
    get_rollout_metrics,
    scalar_reward_token_credit,
)
from skyrl_train.trajectory_runners.selected_topk import AlignedStudentTopK, align_student_topk
from skyrl_train.trajectory_runners.types import (
    RESPONSE_POLICY_STEPS_KEY,
    RESPONSE_SPAN_TAGS_KEY,
    ROUTED_EXPERTS_KEY,
    TrajectoryBatch,
    TaskRequestBatch,
)
from skyrl_train.utils.harbor_errors import (
    DEFAULT_ERROR_HANDLING_CONFIG,
    ErrorHandlingConfig,
    classify_exception_type,
    passthrough_logprob_error_type,
    treatment_excludes_from_baseline,
)

MASKED_TOKEN_ID = 0


@dataclass(frozen=True)
class _TrainingDecision:
    verification: VerificationResult
    reward: RewardResult
    disposition: TrainingDisposition
    error_treatment: str | None


def verification_result(grade: GradeResult) -> VerificationResult:
    """Preserve the task verdict and its availability in the training contract."""
    if grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}:
        assert grade.reward is not None
        return VerificationResult.verified(
            grade.reward,
            passed=grade.passed,
            diagnostics=grade.diagnostics,
            score_min=grade.score_min,
            score_max=grade.score_max,
        )
    if grade_skipped(grade):
        return VerificationResult.skipped(grade.error or "Verification skipped", diagnostics=grade.diagnostics)
    if grade.status == Outcome.UNAVAILABLE:
        return VerificationResult.unavailable(grade.error or "Verification unavailable", diagnostics=grade.diagnostics)
    return VerificationResult.error(grade.error or grade.status.value, diagnostics=grade.diagnostics)


def rollout_loss_eligible(
    rollout: TrainingRollout,
    error_handling: ErrorHandlingConfig,
    *,
    logprobs_required: bool,
) -> bool:
    """Return execution eligibility before the final grade is available."""
    if rollout.failure is None:
        return True
    treatment, missing_logprobs = _failure_policy(rollout, error_handling, logprobs_required)
    graded = rollout.grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
    return (
        bool(any(rollout.loss_mask))
        and not treatment_excludes_from_baseline(treatment, verifier_available=graded)
        and missing_logprobs is None
        and (not rollout.failure.diagnostics.get("timed_out") or error_handling.preserve_logprobs_on_timeout)
    )


def _failure_policy(
    rollout: TrainingRollout,
    error_handling: ErrorHandlingConfig,
    logprobs_required: bool,
) -> tuple[ErrorTreatment, str | None]:
    assert rollout.failure is not None
    treatment = (
        classify_exception_type(rollout.failure.exception_type, error_handling)
        if error_handling.enable_error_classification
        else ErrorTreatment.MASK
    )
    return (
        treatment,
        passthrough_logprob_error_type(
            treatment,
            has_rollout_logprobs=rollout.logprobs is not None,
            rollout_logprobs_required=logprobs_required,
        ),
    )


def _rollout_rewards(rollout: TrainingRollout) -> RewardResult:
    graded = rollout.grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
    skipped = grade_skipped(rollout.grade)
    token_rewards = None
    token_credit = None
    components = {}
    optimization_reward = rollout.grade.reward if graded else 0.0
    if any(step.transition.reward is not None for step in rollout.steps):
        rewards = [0.0] * len(rollout.response_token_ids)
        optimization_reward = 0.0
        for step in rollout.steps:
            if (graded or skipped) and step.transition.reward is not None:
                optimization_reward += step.transition.reward
                if step.transition.token_rewards is None:
                    rewards[step.response_end] += step.transition.reward
                else:
                    start = step.response_end + 1 - len(step.turn.response_token_ids)
                    rewards[start : step.response_end + 1] = step.transition.token_rewards
                for name, value in step.transition.reward_components.items():
                    components[name] = components.get(name, 0.0) + value
        token_rewards = tuple(rewards)
    if any(step.transition.token_credit is not None for step in rollout.steps):
        credit = [0.0] * len(rollout.response_token_ids)
        for step in rollout.steps:
            if (graded or skipped) and step.transition.token_credit is not None:
                start = step.response_end + 1 - len(step.turn.response_token_ids)
                credit[start : step.response_end + 1] = step.transition.token_credit
        token_credit = tuple(credit)
    return RewardResult(
        unshaped_reward=rollout.grade.reward,
        optimization_reward=optimization_reward,
        token_rewards=token_rewards,
        token_credit=token_credit,
        components=components,
    )


def _student_topk(rollout: TrainingRollout) -> AlignedStudentTopK | None:
    candidates = [step.turn.metadata.get("student_topk_indices") for step in rollout.steps]
    scores = [step.turn.metadata.get("behavior_topk_logprobs") for step in rollout.steps]
    if all(value is None for value in (*candidates, *scores)):
        return None
    if any(value is None for value in (*candidates, *scores)):
        raise RolloutContractError("Student top-K evidence must be present on every model turn")
    generated = []
    admitted_candidates = []
    admitted_scores = []
    for step, step_candidates, step_scores in zip(rollout.steps, candidates, scores, strict=True):
        start = step.response_end + 1 - len(step.turn.response_token_ids)
        for offset, (token, ids, values) in enumerate(
            zip(step.turn.response_token_ids, step_candidates, step_scores, strict=True)
        ):
            if rollout.loss_mask[start + offset]:
                generated.append(token)
                admitted_candidates.append(ids)
                admitted_scores.append(values)
    return align_student_topk(
        rollout.response_token_ids,
        rollout.loss_mask,
        generated,
        admitted_candidates,
        admitted_scores,
    )


def _training_decision(
    rollout: TrainingRollout,
    error_handling: ErrorHandlingConfig,
    *,
    logprobs_required: bool,
) -> _TrainingDecision:
    graded = rollout.grade.status in {Outcome.GRADED, Outcome.SUBMISSION_FAILURE}
    verification = verification_result(rollout.grade)
    if graded:
        disposition = TrainingDisposition.train()
    elif grade_skipped(rollout.grade):
        disposition = (
            TrainingDisposition.train(reason="verification skipped")
            if any(step.transition.reward is not None for step in rollout.steps)
            else TrainingDisposition.mask("No verifier verdict", exception_type="skipped")
        )
    else:
        disposition = TrainingDisposition(
            loss_eligible=False,
            baseline_eligible=False,
            reason=rollout.grade.error or rollout.grade.status.value,
            exception_type="VerifierUnavailable"
            if rollout.grade.status == Outcome.UNAVAILABLE
            else VERIFIER_RUNTIME_ERROR,
        )
    reward = _rollout_rewards(rollout)
    error_treatment = None
    if rollout.failure is not None:
        failure = rollout.failure
        treatment, missing_logprobs = _failure_policy(rollout, error_handling, logprobs_required)
        disposition = TrainingDisposition(
            loss_eligible=rollout_loss_eligible(rollout, error_handling, logprobs_required=logprobs_required),
            baseline_eligible=not treatment_excludes_from_baseline(treatment, verifier_available=graded)
            and missing_logprobs is None,
            reason="Rollout execution failed",
            exception_type=missing_logprobs or failure.exception_type,
        )
        error_treatment = treatment.value
        diagnostics = {**verification.diagnostics, "exception_type": failure.exception_type, **failure.diagnostics}
        verification = (
            replace(verification, diagnostics=diagnostics)
            if graded
            else VerificationResult.error(verification.reason or "Rollout execution failed", diagnostics=diagnostics)
        )
        if treatment is not ErrorTreatment.PASSTHROUGH or not graded:
            reward = replace(
                reward,
                optimization_reward=0.0,
                token_rewards=None if reward.token_rewards is None else (0.0,) * len(reward.token_rewards),
                token_credit=None if reward.token_credit is None else (0.0,) * len(reward.token_credit),
                components={},
            )
    return _TrainingDecision(verification, reward, disposition, error_treatment)


@dataclass(frozen=True)
class WholeTaskProjection:
    """Emit one training sample from each RolloutEngine record."""

    runner_config: DictConfig
    tokenizer: PreTrainedTokenizerBase
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG
    harbor_error_handling: ErrorHandlingConfig | None = None

    def project(self, rollouts: Sequence[TrainingRollout], request: TaskRequestBatch) -> TrajectoryBatch:
        required = logprobs_requested(request, self.runner_config)
        policies = _task_error_policies(request, self.error_handling, self.harbor_error_handling)
        decisions = [
            _training_decision(rollout, policy, logprobs_required=required)
            for rollout, policy in zip(rollouts, policies, strict=True)
        ]
        batch = _training_batch(
            rollouts,
            decisions,
            required,
            self.runner_config,
            self.tokenizer,
            [row.metadata.metric_class for row in request.rows],
        )
        _merge_task_metrics(batch, rollouts, request)
        return batch


@dataclass(frozen=True)
class StepTaskProjection:
    """Emit one training sample per model turn, with its exact served prompt."""

    runner_config: DictConfig
    tokenizer: PreTrainedTokenizerBase
    error_handling: ErrorHandlingConfig = DEFAULT_ERROR_HANDLING_CONFIG
    harbor_error_handling: ErrorHandlingConfig | None = None

    def project(self, rollouts: Sequence[TrainingRollout], request: TaskRequestBatch) -> TrajectoryBatch:
        trajectory_ids = request.trajectory_ids
        if trajectory_ids is None:
            raise ValueError("Per-turn training samples require trajectory_ids")
        required = logprobs_requested(request, self.runner_config)
        policies = _task_error_policies(request, self.error_handling, self.harbor_error_handling)
        samples = []
        decisions = []
        projected_ids = []
        final_steps = []
        for rollout, policy, trajectory_id in zip(rollouts, policies, trajectory_ids, strict=True):
            turns = _step_rollouts(rollout)
            for index, sample in enumerate(turns):
                last = index == len(turns) - 1
                decision = _training_decision(sample, policy, logprobs_required=required)
                if (
                    not last
                    and not sample.steps[0].transition.done
                    and sample.grade.status == Outcome.UNAVAILABLE
                    and not grade_skipped(sample.grade)
                    and (rollout.grade.status == Outcome.GRADED or grade_skipped(rollout.grade))
                    and rollout.failure is None
                ):
                    decision = replace(decision, disposition=TrainingDisposition.train())
                samples.append(sample)
                decisions.append(decision)
                projected_id = copy.copy(trajectory_id)
                projected_id.step = index
                projected_ids.append(projected_id)
                final_steps.append(last)
        batch = _training_batch(samples, decisions, required, self.runner_config, self.tokenizer, None)
        batch["trajectory_ids"] = projected_ids
        batch["is_last_step"] = final_steps
        _merge_task_metrics(batch, rollouts, request)
        return batch


def logprobs_requested(request: TaskRequestBatch, runner_config: DictConfig) -> bool:
    sampling_params = request.sampling_params
    if sampling_params is not None and "logprobs" in sampling_params:
        return sampling_params["logprobs"] is not None
    return runner_config.sampling_params.logprobs is not None


def _training_batch(
    rollouts: Sequence[TrainingRollout],
    decisions: Sequence[_TrainingDecision],
    logprobs_required: bool,
    runner_config: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    env_metric_classes: Sequence[str] | None,
) -> TrajectoryBatch:
    responses = [list(rollout.response_token_ids) or [MASKED_TOKEN_ID] for rollout in rollouts]
    rewards = [decision.reward.to_trainer_reward() for decision in decisions]
    if any(isinstance(reward, list) for reward in rewards):
        rewards = [
            reward if isinstance(reward, list) else scalar_reward_token_credit(reward, response)
            for reward, response in zip(rewards, responses, strict=True)
        ]
    masks = [
        list(rollout.loss_mask)
        if decision.disposition.loss_eligible and rollout.response_token_ids
        else [0] * len(response)
        for rollout, decision, response in zip(rollouts, decisions, responses, strict=True)
    ]
    if runner_config.apply_overlong_filtering:
        masks = apply_overlong_filtering(masks, responses, tokenizer.eos_token_id)
    logprobs = [
        _row_logprobs(rollout, decision, logprobs_required)
        for rollout, decision in zip(rollouts, decisions, strict=True)
    ]
    verification = [decision.verification for decision in decisions]
    batch = TrajectoryBatch(
        prompt_token_ids=[list(rollout.prompt_token_ids) or [MASKED_TOKEN_ID] for rollout in rollouts],
        response_ids=responses,
        rewards=rewards,
        verification_results=verification,
        evidence_messages=[[dict(message) for message in rollout.messages] for rollout in rollouts],
        loss_masks=masks,
        stop_reasons=[rollout.stop_reason for rollout in rollouts],
        rollout_metrics={},
        rollout_logprobs=logprobs if logprobs_required else None,
        exclude_from_baseline=[not decision.disposition.baseline_eligible for decision in decisions],
    )
    _attach_reward_channels(batch, decisions)
    if any(rollout.reasoning_span_penalty for rollout in rollouts):
        batch["reasoning_span_penalties"] = [rollout.reasoning_span_penalty for rollout in rollouts]
    _attach_failure_channels(batch, decisions)
    _attach_student_topk(batch, rollouts)
    _attach_routed_experts(batch, rollouts)
    _attach_policy_steps(batch, rollouts)
    _attach_span_tags(batch, rollouts)
    metrics = None
    if env_metric_classes is not None:
        metrics = [
            {} if environment == TASKCOMPENDIUM_ENVIRONMENT else dict(rollout.metrics)
            for rollout, environment in zip(rollouts, env_metric_classes, strict=True)
        ]
        if any(metrics):
            batch["env_metrics"] = metrics
            batch["env_classes"] = list(env_metric_classes)
    batch["rollout_metrics"] = get_rollout_metrics(responses, rewards, metrics, env_metric_classes, verification)
    return batch


def _row_logprobs(rollout: TrainingRollout, decision: _TrainingDecision, required: bool) -> np.ndarray | None:
    if not rollout.response_token_ids:
        return np.zeros(1, dtype=np.float32)
    if rollout.logprobs is not None:
        values = np.asarray(rollout.logprobs, dtype=np.float32)
        if values.shape != (len(rollout.response_token_ids),) or not np.isfinite(values).all():
            raise RolloutContractError("Behavior logprobs must be finite and align with response tokens")
        return values
    if not decision.disposition.loss_eligible:
        return np.zeros(len(rollout.response_token_ids), dtype=np.float32)
    if required:
        raise RolloutContractError("A trainable rollout is missing requested behavior logprobs")
    return None


def _attach_reward_channels(batch: TrajectoryBatch, decisions: Sequence[_TrainingDecision]) -> None:
    batch["unshaped_rewards"] = [decision.reward.unshaped_reward or 0.0 for decision in decisions]
    availability = [decision.reward.unshaped_reward is not None for decision in decisions]
    if not all(availability):
        batch["unshaped_reward_available"] = availability
    if any(decision.reward.token_credit is not None for decision in decisions):
        batch["token_level_shaping"] = [
            list(decision.reward.token_credit) if decision.reward.token_credit is not None else [0.0] * len(response)
            for decision, response in zip(decisions, batch["response_ids"], strict=True)
        ]


def _attach_failure_channels(batch: TrajectoryBatch, decisions: Sequence[_TrainingDecision]) -> None:
    exception_types = [decision.disposition.exception_type for decision in decisions]
    if any(value is not None for value in exception_types):
        batch["exception_types"] = exception_types
    error_treatments = [decision.error_treatment for decision in decisions]
    if any(value is not None for value in error_treatments):
        batch["error_treatments"] = error_treatments
    server_errors = [
        {
            "category": decision.verification.diagnostics["error_category"],
            "request_id": decision.verification.diagnostics.get("request_id"),
            "status_code": decision.verification.diagnostics.get("status_code"),
        }
        if "error_category" in decision.verification.diagnostics
        else None
        for decision in decisions
    ]
    if any(error is not None for error in server_errors):
        batch["server_errors"] = server_errors


def _attach_student_topk(batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout]) -> None:
    selected = [_student_topk(rollout) for rollout in rollouts]
    template = next((value for value in selected if value is not None), None)
    if template is not None:
        width = template.indices.shape[1]
        indices = []
        scores = []
        for value, response, mask in zip(selected, batch["response_ids"], batch["loss_masks"], strict=True):
            if value is None:
                if any(mask):
                    raise ValueError("Student top-K evidence is missing for a trainable rollout")
                indices.append(np.full((len(response), width), INVALID_TOPK_INDEX, dtype=template.indices.dtype))
                scores.append(np.zeros((len(response), width), dtype=np.float32))
                continue
            if value.indices.shape[1] != width:
                raise ValueError("Student top-K evidence widths must agree across rollouts")
            indices.append(value.indices)
            scores.append(value.topk_logprobs)
        batch["student_topk_indices"] = indices
        batch["behavior_topk_logprobs"] = scores


def _model_channel(
    rollout: TrainingRollout,
    key: str,
    length: int,
    trailing_shape: tuple[int, ...],
    dtype: np.dtype | type[np.generic],
    fill: int = 0,
) -> np.ndarray:
    values = np.full((length, *trailing_shape), fill, dtype=dtype)
    for step in rollout.steps:
        captured = step.turn.metadata.get(key)
        if captured is None:
            continue
        if len(captured) != len(step.turn.response_token_ids):
            raise ValueError(f"{key} must align with generated tokens")
        values[step.response_end + 1 - len(captured) : step.response_end + 1] = captured
    return values


def _attach_routed_experts(batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout]) -> None:
    routes = [step.turn.metadata.get(ROUTED_EXPERTS_KEY) for rollout in rollouts for step in rollout.steps]
    route_template = next((value for value in routes if value is not None and len(value)), None)
    if route_template is not None:
        batch["rollout_routed_experts"] = [
            _model_channel(rollout, ROUTED_EXPERTS_KEY, len(response), route_template.shape[1:], route_template.dtype)
            for rollout, response in zip(rollouts, batch["response_ids"], strict=True)
        ]


def _token_channels(
    batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout], key: str, dtype: type[np.generic], fill: int = 0
) -> list[np.ndarray] | None:
    """One scalar per response token from each turn's ``key`` metadata, or None when no turn carries it."""
    if all(step.turn.metadata.get(key) is None for rollout in rollouts for step in rollout.steps):
        return None
    return [
        _model_channel(rollout, key, len(response), (), dtype, fill)
        for rollout, response in zip(rollouts, batch["response_ids"], strict=True)
    ]


def _attach_policy_steps(batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout]) -> None:
    steps = _token_channels(batch, rollouts, RESPONSE_POLICY_STEPS_KEY, np.int32, fill=UNSTAMPED_POLICY_STEP)
    if steps is not None:
        batch["rollout_policy_steps"] = steps


def _attach_span_tags(batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout]) -> None:
    tags = _token_channels(batch, rollouts, RESPONSE_SPAN_TAGS_KEY, np.int64)
    if tags is not None:
        batch["response_span_tags"] = [row.tolist() for row in tags]


def _task_error_policies(
    request: TaskRequestBatch, default: ErrorHandlingConfig, harbor: ErrorHandlingConfig | None
) -> list[ErrorHandlingConfig]:
    if harbor is None:
        return [default] * len(request.rows)
    return [harbor if "harbor" in row.spec.task.tags else default for row in request.rows]


def _merge_task_metrics(batch: TrajectoryBatch, rollouts: Sequence[TrainingRollout], request: TaskRequestBatch) -> None:
    """Count task evidence and preserve environment-specific metrics."""
    metrics = batch.get("rollout_metrics") or {}
    failed = [
        rollout
        for rollout in rollouts
        if rollout.failure is not None or (rollout.grade.status != Outcome.GRADED and not grade_skipped(rollout.grade))
    ]
    excluded = batch["exclude_from_baseline"]
    final_steps = batch.get("is_last_step") or [True] * len(excluded)
    metrics.update(
        get_batch_failure_metrics(
            num_trials=len(rollouts),
            num_failed_trajectories=len(failed),
            num_failed_instances=len({rollout.task_id for rollout in failed}),
            num_masked_trajectories=sum(masked and last for masked, last in zip(excluded, final_steps, strict=True)),
        )
    )
    counters = {
        "tasks": len(rollouts),
        "turns": sum(len(rollout.steps) for rollout in rollouts),
        "multi_turn_tasks": sum(len(rollout.steps) > 1 for rollout in rollouts),
        "tool_tasks": sum(any(step.turn.message.get("tool_calls") for step in rollout.steps) for rollout in rollouts),
        "generated_tokens": sum(sum(rollout.loss_mask) for rollout in rollouts),
        "missing_logprob_tokens": sum(sum(rollout.loss_mask) for rollout in rollouts if rollout.logprobs is None),
    }
    metrics.update({f"{TASK_ROLLOUT_METRIC_PREFIX}{name}": float(value) for name, value in counters.items()})
    batch["rollout_metrics"] = metrics
    environments = [row.metadata.metric_class for row in request.rows]
    for rollout, environment in zip(rollouts, environments, strict=True):
        if environment != TASKCOMPENDIUM_ENVIRONMENT:
            continue
        for key, value in rollout.metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value


def _step_rollouts(rollout: TrainingRollout) -> list[TrainingRollout]:
    """Use exact served prompts and score each transition at its final action token."""
    if not rollout.steps:
        return [rollout]
    outputs = []
    for index, step in enumerate(rollout.steps):
        last = index == len(rollout.steps) - 1
        grade = rollout.grade
        if grade.status == Outcome.GRADED or grade_skipped(grade):
            grade = step.transition.grade or (grade if last else skipped_grade("No intermediate verifier"))
        reward = step.transition.reward
        if not last and reward is None:
            reward = 0.0
        transition = replace(step.transition, reward=reward, grade=grade)
        turn = step.turn
        output = replace(
            rollout,
            messages=step.messages,
            prompt_token_ids=turn.prompt_token_ids,
            response_token_ids=turn.response_token_ids,
            loss_mask=(1,) * len(turn.response_token_ids),
            logprobs=turn.logprobs,
            grade=grade,
            stop_reason=rollout.stop_reason if last else turn.stop_reason,
            steps=(replace(step, response_end=len(turn.response_token_ids) - 1, transition=transition),),
            metrics=step.transition.metrics,
        )
        outputs.append(output)
    return outputs
