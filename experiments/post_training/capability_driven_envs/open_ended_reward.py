# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Group-relative reward records for open-ended Taskforge tasks.

Taskforge authors one semantic task and freezes its reference and rubric before
rollouts. A downstream campaign samples a cohort of answers, obtains blinded
pairwise ratings under that rubric, and calls :func:`build_scorepad` to turn the
ratings into auditable rewards. Model transport is deliberately outside this
module: the scorepad records judge provenance but does not hard-code an endpoint
or model.
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator
from verifyit.modes.grade_judge import grade_paired_ordinal


class HardCheckResult(BaseModel):
    """One deterministic check and the reward cap imposed when it fails."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    check_id: str = Field(min_length=1)
    passed: bool
    cap_if_failed: float = Field(ge=0, le=1)
    detail: str = ""


class CandidateOutcome(BaseModel):
    """One completed rollout and the deterministic checks applied before judging."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    content_digest: str = Field(min_length=1)
    output_tokens: int = Field(ge=0)
    hard_checks: tuple[HardCheckResult, ...] = ()

    @model_validator(mode="after")
    def unique_checks(self) -> "CandidateOutcome":
        ids = [check.check_id for check in self.hard_checks]
        if len(set(ids)) != len(ids):
            raise ValueError("hard-check IDs must be unique within a candidate")
        return self


class PairwiseRating(BaseModel):
    """The judge's score and preference for one blinded, ordered candidate pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: int = Field(ge=0)
    right: int = Field(ge=0)
    score_left: float = Field(ge=1, le=5)
    score_right: float = Field(ge=1, le=5)
    ranking: float = Field(ge=1, le=6)
    reasoning: str = ""


class CandidateReward(BaseModel):
    """One candidate's comparative score, applied caps, and centered reward."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    ordinal_reward: float = Field(ge=0, le=1)
    capped_reward: float = Field(ge=0, le=1)
    centered_reward: float
    over_output_budget: bool
    applied_cap: float = Field(ge=0, le=1)


class OpenEndedScorepad(BaseModel):
    """A replayable record of one rubric-grounded cohort judgment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_digest: str = Field(min_length=1)
    rubric_digest: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)
    judge_prompt_digest: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    candidates: tuple[CandidateOutcome, ...]
    edges: tuple[tuple[int, int], ...]
    ratings: tuple[PairwiseRating, ...]
    rewards: tuple[CandidateReward, ...]


def swapped_ring_edges(cohort_size: int) -> tuple[tuple[int, int], ...]:
    """Return a bounded comparison schedule covering every candidate in both positions.

    The schedule has two directed comparisons per candidate rather than the
    quadratic cost of a complete round robin. Two-candidate cohorts reduce to
    the two possible orders.
    """
    if cohort_size < 2:
        raise ValueError("a comparative cohort needs at least two candidates")
    if cohort_size == 2:
        return ((0, 1), (1, 0))
    forward = tuple((index, (index + 1) % cohort_size) for index in range(cohort_size))
    return (*forward, *((right, left) for left, right in forward))


def build_scorepad(
    candidates: tuple[CandidateOutcome, ...],
    ratings: tuple[PairwiseRating, ...],
    *,
    task_digest: str,
    rubric_digest: str,
    judge_model: str,
    judge_prompt_digest: str,
    max_output_tokens: int,
) -> OpenEndedScorepad:
    """Apply budget and hard-check caps, then center comparative cohort rewards.

    An over-budget candidate automatically loses a comparison against an
    in-budget candidate. If both are over budget, the comparison is a low-score
    tie. Deterministic check caps apply after comparative judging. The resulting
    rewards are centered but not variance-normalized; the RL algorithm remains
    responsible for any further advantage shaping.
    """
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    edges = swapped_ring_edges(len(candidates))
    if tuple((rating.left, rating.right) for rating in ratings) != edges:
        raise ValueError("ratings must follow the scorepad's complete swapped-ring edge schedule")

    adjusted = []
    for rating in ratings:
        left_over = candidates[rating.left].output_tokens > max_output_tokens
        right_over = candidates[rating.right].output_tokens > max_output_tokens
        record = rating.model_dump(exclude={"reasoning"})
        if left_over != right_over:
            record.update(
                score_left=1.0 if left_over else 5.0,
                score_right=1.0 if right_over else 5.0,
                ranking=6.0 if left_over else 1.0,
            )
        elif left_over:
            record.update(score_left=1.0, score_right=1.0, ranking=3.5)
        adjusted.append(record)

    ordinal = grade_paired_ordinal(len(candidates), list(edges), adjusted)
    uncapped: list[tuple[float, float, bool]] = []
    for candidate, verdict in zip(candidates, ordinal, strict=True):
        over_budget = candidate.output_tokens > max_output_tokens
        failed_caps = tuple(check.cap_if_failed for check in candidate.hard_checks if not check.passed)
        cap = min((*failed_caps, 0.0 if over_budget else 1.0))
        uncapped.append((verdict.reward, min(verdict.reward, cap), over_budget))
    mean_reward = sum(reward for _, reward, _ in uncapped) / len(uncapped)
    rewards = tuple(
        CandidateReward(
            candidate_id=candidate.candidate_id,
            ordinal_reward=raw,
            capped_reward=capped,
            centered_reward=capped - mean_reward,
            over_output_budget=over,
            applied_cap=min(
                (*(check.cap_if_failed for check in candidate.hard_checks if not check.passed), 0.0 if over else 1.0)
            ),
        )
        for candidate, (raw, capped, over) in zip(candidates, uncapped, strict=True)
    )
    return OpenEndedScorepad(
        task_digest=task_digest,
        rubric_digest=rubric_digest,
        judge_model=judge_model,
        judge_prompt_digest=judge_prompt_digest,
        max_output_tokens=max_output_tokens,
        candidates=candidates,
        edges=edges,
        ratings=ratings,
        rewards=rewards,
    )
