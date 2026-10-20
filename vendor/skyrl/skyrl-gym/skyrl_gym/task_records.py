"""Convert model evidence and verifier results to the task execution records."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from rolloutengine.models import ModelTurn, Prompt, Transition
from taskcompendium.models import TaskSpec
from taskcompendium.submission import conversation_messages
from taskcompendium.grading_result import GradeResult, Outcome

from skyrl_gym.verification import RewardResult, RolloutEvidence, VerificationResult, VerificationStatus

VERIFICATION_SKIPPED = "verification_skipped"
FOLDED_GRADE_HARNESSES = frozenset({"nemotron_ultra", "openenv"})


def episode_prompt(task: TaskSpec) -> Prompt:
    """Render a direct task's source messages and advertised tools without a verifier instruction."""
    tools = [
        {"type": "function", "function": definition.model_dump(exclude_none=True)}
        for definition in (*task.final_tools, *task.interaction_tools)
    ]
    return Prompt(
        tuple(conversation_messages(task.context.events)),
        {"tools": tools} if tools else {},
    )


@dataclass(frozen=True)
class StepReward:
    """SkyRL training signals attached to one harness step."""

    reward: float | None = None
    grade: GradeResult | None = None
    task_grade: GradeResult | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskTransition:
    """A task episode's observations and training signals for one model turn."""

    done: bool
    observations: tuple[dict[str, Any], ...] = ()
    reset_conversation: tuple[dict[str, Any], ...] | None = None
    token_ids: tuple[int, ...] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    grade: GradeResult | None = None
    task_grade: GradeResult | None = None
    token_rewards: tuple[float, ...] | None = None
    token_credit: tuple[float, ...] | None = None
    reward_components: dict[str, float] = field(default_factory=dict)

    def for_harness(self) -> Transition:
        return Transition(
            done=self.done,
            observations=self.observations,
            token_ids=self.token_ids,
            metrics=self.metrics,
            data=StepReward(
                reward=self.reward,
                grade=self.grade,
                task_grade=self.task_grade,
                token_rewards=self.token_rewards,
                token_credit=self.token_credit,
                reward_components=self.reward_components,
            ),
        )


def skipped_grade(reason: str, *, diagnostics: dict | None = None) -> GradeResult:
    return GradeResult(
        Outcome.UNAVAILABLE,
        None,
        reason,
        diagnostics={**(diagnostics or {}), VERIFICATION_SKIPPED: True},
    )


def grade_skipped(grade: GradeResult) -> bool:
    return grade.status == Outcome.UNAVAILABLE and grade.diagnostics.get(VERIFICATION_SKIPPED) is True


def rollout_evidence(turn: ModelTurn) -> RolloutEvidence:
    return RolloutEvidence(
        messages=(turn.message,),
        response=turn.text,
        stop_reason=turn.stop_reason,
        generated_token_count=len(turn.response_token_ids),
        response_token_ids=turn.response_token_ids,
        metadata=turn.metadata,
    )


def grade_result(verification: VerificationResult) -> GradeResult:
    if verification.status == VerificationStatus.VERIFIED:
        return GradeResult(
            Outcome.GRADED,
            verification.score,
            passed=verification.passed,
            diagnostics=dict(verification.diagnostics),
            score_min=verification.score_min,
            score_max=verification.score_max,
        )
    if verification.status == VerificationStatus.SKIPPED:
        return skipped_grade(verification.reason, diagnostics=dict(verification.diagnostics))
    statuses = {
        VerificationStatus.UNAVAILABLE: Outcome.UNAVAILABLE,
        VerificationStatus.ERROR: Outcome.INFRA_ERROR,
    }
    return GradeResult(
        statuses[verification.status], None, verification.reason, diagnostics=dict(verification.diagnostics)
    )


def graded_transition(
    turn: ModelTurn, verification: VerificationResult, reward: RewardResult, metrics: dict
) -> TaskTransition:
    reward.validate_for(rollout_evidence(turn))
    return TaskTransition(
        done=True,
        reward=reward.optimization_reward,
        token_rewards=reward.token_rewards,
        token_credit=reward.token_credit,
        reward_components=dict(reward.components),
        grade=grade_result(verification),
        metrics=metrics,
    )


def terminal_grade(results: Sequence[GradeResult]) -> GradeResult:
    """Return the last task verdict without averaging tool or correction turns."""
    if not results:
        return GradeResult(Outcome.UNAVAILABLE, None, "The task produced no grade")
    return results[-1]


def fold_grades(results: Sequence[GradeResult]) -> GradeResult:
    """Average independent turn scores and preserve an unscored terminal result."""
    if not results:
        return GradeResult(Outcome.UNAVAILABLE, None, "The task produced no grade")
    if len(results) == 1 or results[-1].reward is None:
        return results[-1]
    scored = [result for result in results if result.reward is not None]
    normalized = [
        min(1.0, max(0.0, (result.reward - result.score_min) / (result.score_max - result.score_min)))
        for result in scored
    ]
    return GradeResult(
        Outcome.GRADED,
        sum(normalized) / len(scored),
        passed=all(
            result.passed if result.passed is not None else score >= 1.0 for result, score in zip(scored, normalized)
        ),
        diagnostics={"steps": tuple(result.diagnostics for result in results), "num_scored_steps": len(scored)},
    )
