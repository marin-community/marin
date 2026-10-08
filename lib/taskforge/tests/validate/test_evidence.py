# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

from rolloutengine.contracts import TOTAL_TURN_TIMEOUT_STOP_REASON, RolloutData
from taskcompendium.grading_result import GradeResult, Outcome

from taskforge.validate.evidence import Complete, Evidence, Incomplete, RewardStats
from taskforge.validate.outcome import Cause, Graded, TrialKind, Ungraded


def rollout(grade: GradeResult, stop_reason: str = "stop") -> RolloutData:
    return RolloutData("task", (), (), (), (), None, grade, stop_reason)


def graded(
    reward: float | None, status: Outcome = Outcome.GRADED, passed: bool | None = None, stop_reason: str = "stop"
) -> Graded:
    return Graded(rollout(GradeResult(status, reward, passed=passed), stop_reason))


def ungraded(cause: Cause) -> Ungraded:
    return Ungraded(cause, "detail", None)


def test_statistics_use_graded_outcomes_only_and_ungraded_ones_make_evidence_incomplete():
    evidence = Evidence(
        {
            TrialKind.SOLVER: (
                graded(1.0),
                graded(0.5, stop_reason=TOTAL_TURN_TIMEOUT_STOP_REASON),
                graded(None, Outcome.SUBMISSION_FAILURE),
                ungraded(Cause.MACHINE_START),
                ungraded(Cause.UNCLASSIFIED),
            ),
            TrialKind.CONTROL: (graded(1.0), ungraded(Cause.MACHINE_START)),
        }
    )

    assert evidence.reward_stats(TrialKind.SOLVER) == RewardStats(graded=3, mean_reward=0.5, solved=1, timed_out=1)
    assert evidence.status == Incomplete(Counter({Cause.MACHINE_START: 2, Cause.UNCLASSIFIED: 1}))


def test_a_grader_pass_flag_overrides_the_full_reward_rule():
    evidence = Evidence({TrialKind.SOLVER: (graded(0.7, passed=True), graded(1.0, passed=False))})

    assert evidence.reward_stats(TrialKind.SOLVER).solved == 1
    assert evidence.status == Complete()


def test_a_kind_with_no_graded_outcome_has_no_mean():
    evidence = Evidence({TrialKind.SOLVER: (ungraded(Cause.MODEL_UNAVAILABLE),)})

    assert evidence.reward_stats(TrialKind.SOLVER) == RewardStats(graded=0, mean_reward=None, solved=0, timed_out=0)
    assert evidence.reward_stats(TrialKind.ADVERSARY).graded == 0
