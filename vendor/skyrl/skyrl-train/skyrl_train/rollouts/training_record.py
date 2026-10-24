"""SkyRL training signals projected from Marin's graded attempt records."""

from dataclasses import dataclass, field
from typing import Any

from rolloutengine.models import ConversationRecord, GradedRollout, ModelTurn, RolloutFailure, TaskRolloutSpec
from skyrl_gym.task_records import FOLDED_GRADE_HARNESSES, StepReward, fold_grades
from taskcompendium.grading_result import GradeResult
from taskcompendium.models import NoGrader


@dataclass(frozen=True)
class TrainingTransition:
    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    reward: float | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    grade: GradeResult | None = None
    task_grade: GradeResult | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingStep:
    turn: ModelTurn
    transition: TrainingTransition
    response_end: int
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TrainingRollout:
    """One trainable conversation and its task verdict, retaining every engine conversation."""

    task_id: str
    messages: tuple[dict[str, Any], ...]
    prompt_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    grade: GradeResult
    stop_reason: str
    steps: tuple[TrainingStep, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, bytes] = field(default_factory=dict)
    failure: RolloutFailure | None = None
    conversations: tuple[ConversationRecord, ...] = ()


def training_rollout(result: GradedRollout, spec: TaskRolloutSpec) -> TrainingRollout:
    """Select the last conversation for optimization and keep the full attempt for audit."""
    rollout = result.rollout
    conversation = rollout.conversations[-1] if rollout.conversations else None
    steps: tuple[TrainingStep, ...] = ()
    if conversation is not None:
        steps = tuple(
            TrainingStep(
                step.turn,
                TrainingTransition(
                    done=step.transition.done,
                    observations=step.transition.observations,
                    reward=data.reward,
                    token_rewards=data.token_rewards,
                    token_credit=data.token_credit,
                    reward_components=data.reward_components,
                    grade=data.grade,
                    task_grade=data.task_grade,
                    metrics=step.transition.metrics,
                ),
                step.response_end,
                step.messages,
            )
            for step in conversation.steps
            for data in [step.transition.data if isinstance(step.transition.data, StepReward) else StepReward()]
        )
    grade = result.grade
    if isinstance(spec.task.grader, NoGrader) and steps:
        grades = [step.transition.grade for step in steps if step.transition.grade is not None]
        if steps[-1].transition.task_grade is not None:
            grade = steps[-1].transition.task_grade
        elif grades:
            grade = fold_grades(grades) if spec.harness in FOLDED_GRADE_HARNESSES else grades[-1]
    return TrainingRollout(
        task_id=spec.task.id,
        messages=() if conversation is None else conversation.messages,
        prompt_token_ids=() if conversation is None else conversation.prompt_token_ids,
        response_token_ids=() if conversation is None else conversation.response_token_ids,
        loss_mask=() if conversation is None else conversation.loss_mask,
        logprobs=None if conversation is None else conversation.logprobs,
        grade=grade,
        stop_reason=rollout.stop_reason,
        steps=steps,
        metrics=rollout.metrics,
        outputs=rollout.outputs,
        failure=rollout.failure,
        conversations=rollout.conversations,
    )
