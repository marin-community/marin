"""SkyRL training signals projected from Marin's graded attempt records."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from rolloutengine.models import ConversationRecord, GradedRollout, ModelTurn, RolloutFailure, TaskRolloutSpec
from skyrl_gym.task_records import FOLDED_GRADE_HARNESSES, StepReward, fold_grades
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import NoGrader
from taskcompendium.reasoning_spans import ReasoningViolation

REASONING_VIOLATIONS_DIAGNOSTIC = "reasoning_violations"
REASONING_VIOLATION_TOKEN_METADATA_KEY = "reasoning_violation_token"


class ReasoningSpanPolicy(StrEnum):
    """How training treats a rollout with a malformed reasoning span.

    GRADE keeps the task grade. REJECT grades the rollout as a reward-0 submission failure. PENALIZE keeps
    the task grade and charges a fixed penalty in train rollouts.
    """

    GRADE = "grade"
    REJECT = "reject"
    PENALIZE = "penalize"


class ReasoningPenaltyTarget(StrEnum):
    """Where PENALIZE charges its penalty: the rollout's reward, or the token where the span goes wrong.

    A token penalty is added to the advantages after normalization, so no estimator rescales it.
    """

    ROLLOUT = "rollout"
    MARKER = "marker"


@dataclass(frozen=True)
class ReasoningSpanSettings:
    policy: ReasoningSpanPolicy
    penalty: float
    penalty_target: ReasoningPenaltyTarget

    def __post_init__(self):
        # A recipe that leaves PENALIZE keeps the inherited penalty, which then goes unused.
        if self.policy == ReasoningSpanPolicy.PENALIZE and self.penalty <= 0:
            raise ValueError("reasoning_spans.penalty must be positive when the policy is penalize")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ReasoningSpanSettings":
        return cls(
            ReasoningSpanPolicy(config["policy"]),
            float(config["penalty"]),
            ReasoningPenaltyTarget(config["penalty_target"]),
        )


@dataclass(frozen=True)
class TurnReasoningViolation:
    conversation: int
    turn: int
    violation: ReasoningViolation
    token: int | None


@dataclass(frozen=True)
class TrainingTransition:
    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    reward: float | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reasoning_span_penalties: tuple[float, ...] | None = None
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
        TurnReasoningViolation(
            index, turn, step.turn.reasoning_violation, step.turn.metadata.get(REASONING_VIOLATION_TOKEN_METADATA_KEY)
        )
        for index, conversation in enumerate(rollout.conversations)
        for turn, step in enumerate(conversation.steps)
        if step.turn.reasoning_violation is not None
    ]


def reject_malformed_reasoning(rollout: TrainingRollout) -> TrainingRollout:
    """Grade a rollout with a malformed reasoning span in any turn as a reward-0 submission failure.

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


def _penalized_violations(rollout: TrainingRollout) -> list[TurnReasoningViolation]:
    """A rollout without a reward has nothing to penalize."""
    return [] if rollout.grade.reward is None else _reasoning_violations(rollout)


def _penalty_diagnostics(
    grade: GradeResult, violations: list[TurnReasoningViolation], name: str, penalty: float
) -> GradeResult:
    diagnostics = {REASONING_VIOLATIONS_DIAGNOSTIC: [asdict(violation) for violation in violations], name: penalty}
    return replace(grade, diagnostics={**grade.diagnostics, **diagnostics})


def penalize_malformed_reasoning(rollout: TrainingRollout, penalty: float) -> TrainingRollout:
    """Record ``penalty`` against a rollout with a malformed reasoning span, keeping its task grade."""
    violations = _penalized_violations(rollout)
    if not violations:
        return rollout
    grade = _penalty_diagnostics(rollout.grade, violations, "reasoning_span_penalty", penalty)
    return replace(rollout, grade=grade, reasoning_span_penalty=penalty)


def penalize_malformed_reasoning_tokens(rollout: TrainingRollout, penalty: float) -> TrainingRollout:
    """Record ``penalty`` at the token where each malformed reasoning span goes wrong, keeping the task grade.

    A violation with no token in the optimized conversation, such as one in literal marker text or an earlier
    conversation, charges the whole rollout instead.
    """
    violations = _penalized_violations(rollout)
    if not violations:
        return rollout
    optimized = len(rollout.conversations) - 1
    if any(violation.conversation != optimized or violation.token is None for violation in violations):
        return penalize_malformed_reasoning(rollout, penalty)
    steps = list(rollout.steps)
    for violation in violations:
        assert violation.token is not None
        step = steps[violation.turn]
        penalties = [0.0] * len(step.turn.response_token_ids)
        penalties[violation.token] = penalty
        steps[violation.turn] = replace(
            step, transition=replace(step.transition, reasoning_span_penalties=tuple(penalties))
        )
    grade = _penalty_diagnostics(rollout.grade, violations, "reasoning_span_token_penalty", penalty)
    return replace(rollout, grade=grade, steps=tuple(steps))


def penalize_reasoning_spans(rollout: TrainingRollout, settings: ReasoningSpanSettings) -> TrainingRollout:
    """Apply the PENALIZE policy at its target; other policies leave the rollout unchanged."""
    if settings.policy != ReasoningSpanPolicy.PENALIZE:
        return rollout
    if settings.penalty_target == ReasoningPenaltyTarget.MARKER:
        return penalize_malformed_reasoning_tokens(rollout, settings.penalty)
    return penalize_malformed_reasoning(rollout, settings.penalty)
