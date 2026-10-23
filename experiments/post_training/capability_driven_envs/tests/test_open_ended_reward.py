# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.post_training.capability_driven_envs.open_ended_reward import (
    CandidateOutcome,
    HardCheckResult,
    OpenEndedScorepad,
    PairwiseRating,
    build_scorepad,
    swapped_ring_edges,
)


def ratings_for(edges: tuple[tuple[int, int], ...]) -> tuple[PairwiseRating, ...]:
    return tuple(
        PairwiseRating(
            left=left,
            right=right,
            score_left=4,
            score_right=2,
            ranking=2,
            reasoning=f"candidate {left} is better supported",
        )
        for left, right in edges
    )


def scorepad(candidates: tuple[CandidateOutcome, ...], ratings: tuple[PairwiseRating, ...]) -> OpenEndedScorepad:
    return build_scorepad(
        candidates,
        ratings,
        task_digest="task-digest",
        rubric_digest="rubric-digest",
        judge_model="judge/model-v1",
        judge_prompt_digest="prompt-digest",
        max_output_tokens=100,
    )


def test_swapped_ring_covers_every_candidate_in_both_positions_without_quadratic_round_robin():
    edges = swapped_ring_edges(5)

    assert len(edges) == 10
    assert {left for left, _ in edges} == set(range(5))
    assert {right for _, right in edges} == set(range(5))
    assert all((right, left) in edges for left, right in edges)
    assert swapped_ring_edges(2) == ((0, 1), (1, 0))


def test_scorepad_centers_comparative_rewards_and_preserves_judge_provenance():
    candidates = tuple(
        CandidateOutcome(candidate_id=f"candidate-{index}", content_digest=f"digest-{index}", output_tokens=50)
        for index in range(3)
    )
    edges = swapped_ring_edges(len(candidates))

    pad = scorepad(candidates, ratings_for(edges))
    restored = OpenEndedScorepad.model_validate_json(pad.model_dump_json())

    assert restored == pad
    assert pad.edges == edges
    assert pad.rubric_digest == "rubric-digest"
    assert pad.judge_model == "judge/model-v1"
    assert sum(reward.centered_reward for reward in pad.rewards) == pytest.approx(0)


def test_output_budget_forces_a_loss_even_when_the_judge_preferred_the_verbose_candidate():
    candidates = (
        CandidateOutcome(candidate_id="concise", content_digest="concise-digest", output_tokens=50),
        CandidateOutcome(candidate_id="verbose", content_digest="verbose-digest", output_tokens=101),
    )
    # Every raw judgment prefers whichever candidate appears on the left. The
    # budget rule must override the comparison in both orders.
    pad = scorepad(candidates, ratings_for(swapped_ring_edges(2)))
    concise, verbose = pad.rewards

    assert concise.capped_reward > verbose.capped_reward
    assert not concise.over_output_budget
    assert verbose.over_output_budget
    assert verbose.capped_reward == 0
    assert verbose.applied_cap == 0


def test_hard_check_caps_comparative_reward_before_centering():
    candidates = (
        CandidateOutcome(
            candidate_id="failed-check",
            content_digest="failed-digest",
            output_tokens=50,
            hard_checks=(HardCheckResult(check_id="numbers_reproduce", passed=False, cap_if_failed=0.2),),
        ),
        CandidateOutcome(candidate_id="passed", content_digest="passed-digest", output_tokens=50),
    )
    pad = scorepad(candidates, ratings_for(swapped_ring_edges(2)))
    failed, passed = pad.rewards

    assert failed.applied_cap == 0.2
    assert failed.capped_reward <= 0.2
    assert passed.applied_cap == 1


def test_scorepad_rejects_missing_or_reordered_pairwise_results():
    candidates = (
        CandidateOutcome(candidate_id="a", content_digest="a-digest", output_tokens=1),
        CandidateOutcome(candidate_id="b", content_digest="b-digest", output_tokens=1),
        CandidateOutcome(candidate_id="c", content_digest="c-digest", output_tokens=1),
    )
    ratings = ratings_for(swapped_ring_edges(3))

    with pytest.raises(ValueError, match="complete swapped-ring"):
        scorepad(candidates, ratings[:-1])
    with pytest.raises(ValueError, match="complete swapped-ring"):
        scorepad(candidates, tuple(reversed(ratings)))


def test_scorepad_rejects_duplicate_candidate_identity():
    candidates = (
        CandidateOutcome(candidate_id="same", content_digest="one", output_tokens=1),
        CandidateOutcome(candidate_id="same", content_digest="two", output_tokens=1),
    )

    with pytest.raises(ValueError, match="unique"):
        scorepad(candidates, ratings_for(swapped_ring_edges(2)))
