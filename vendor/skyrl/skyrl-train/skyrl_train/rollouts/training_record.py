"""SkyRL training signals projected from Marin's graded attempt records."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from rolloutengine.models import ConversationRecord, GradedRollout, ModelTurn, RolloutFailure, TaskRolloutSpec
from skyrl_gym.task_records import FOLDED_GRADE_HARNESSES, StepReward, fold_grades
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import NoGrader

REASONING_VIOLATION_METADATA_KEY = "reasoning_violation"
REASONING_VIOLATIONS_DIAGNOSTIC = "reasoning_violations"


class ReasoningSpanPolicy(StrEnum):
    """How training treats a rollout with a malformed reasoning span.

    PENALIZE subtracts a fixed penalty from the reward. REJECT grades the rollout 0.
    """

    IGNORE = "ignore"
    PENALIZE = "penalize"
    REJECT = "reject"


@dataclass(frozen=True)
class ReasoningSpanSettings:
    """A default reasoning-span policy with per-agent overrides for Nemotron Ultra tasks."""

    policy: ReasoningSpanPolicy
    penalty: float = 0.0
    agent_policies: Mapping[str, ReasoningSpanPolicy] = field(default_factory=dict)

    def __post_init__(self):
        penalized = ReasoningSpanPolicy.PENALIZE in {self.policy, *self.agent_policies.values()}
        if penalized != (self.penalty > 0):
            raise ValueError("reasoning_spans.penalty must be positive exactly when some policy is penalize")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ReasoningSpanSettings":
        return cls(
            ReasoningSpanPolicy(config["policy"]),
            float(config["penalty"]),
            {agent: ReasoningSpanPolicy(policy) for agent, policy in config["agent_policies"].items()},
        )

    def policy_for(self, agent: str | None) -> ReasoningSpanPolicy:
        return self.policy if agent is None else self.agent_policies.get(agent, self.policy)


@dataclass(frozen=True)
class TurnReasoningViolation:
    conversation: int
    turn: int
    violation: str


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
    reasoning_span_penalty: float = 0.0


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


def _reasoning_violations(rollout: TrainingRollout) -> list[TurnReasoningViolation]:
    return [
        TurnReasoningViolation(index, turn, violation)
        for index, conversation in enumerate(rollout.conversations)
        for turn, step in enumerate(conversation.steps)
        if (violation := step.turn.metadata.get(REASONING_VIOLATION_METADATA_KEY)) is not None
    ]


def reject_malformed_reasoning(rollout: TrainingRollout) -> TrainingRollout:
    """Grade a rollout with a malformed reasoning span as a reward-0 submission failure.

    Step rewards are cleared so the zero grade is the whole reward.
    """
    violations = _reasoning_violations(rollout)
    if not violations:
        return rollout
    grade = GradeResult(
        Outcome.SUBMISSION_FAILURE,
        0.0,
        "Malformed reasoning span",
        passed=False,
        diagnostics={REASONING_VIOLATIONS_DIAGNOSTIC: [asdict(violation) for violation in violations]},
    )
    steps = tuple(
        replace(step, transition=replace(step.transition, reward=None, token_rewards=None, reward_components={}))
        for step in rollout.steps
    )
    return replace(rollout, grade=grade, steps=steps)


def penalize_malformed_reasoning(rollout: TrainingRollout, penalty: float) -> TrainingRollout:
    """Charge ``penalty`` to a rollout with a malformed reasoning span.

    The grade keeps the task reward. Finalization subtracts the penalty from the optimization reward, as it
    does the length penalties, so outcome metrics stay unpenalized.
    """
    violations = _reasoning_violations(rollout)
    if not violations or rollout.grade.reward is None:
        return rollout
    grade = replace(
        rollout.grade,
        diagnostics={
            **rollout.grade.diagnostics,
            REASONING_VIOLATIONS_DIAGNOSTIC: [asdict(violation) for violation in violations],
            "reasoning_span_penalty": penalty,
        },
    )
    return replace(rollout, grade=grade, reasoning_span_penalty=penalty)
