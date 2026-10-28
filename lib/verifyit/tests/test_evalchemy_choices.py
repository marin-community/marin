# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen responses graded by the original Evalchemy letter-choice graders."""

import pytest
from verifyit.candidate import grade_candidate
from verifyit.modes.grade_mcq import grade
from verifyit.spec import EmptyOutputPolicy, McqProfile, McqSpec


def source_rewards(profile, expected, response, tmp_path):
    spec = McqSpec(
        expected=expected,
        options=10 if profile is McqProfile.EVALCHEMY_MMLU_PRO else 4,
        profile=profile,
        empty_output=EmptyOutputPolicy.GRADE,
    )
    (tmp_path / "answer.txt").write_text(response)
    return grade_candidate(spec, response, {}), grade(spec, tmp_path, tmp_path)


@pytest.mark.parametrize(
    "response,extracted,expected,reward",
    [
        (r"\boxed{A}", "A", "A", 1),
        ("Answer: c. Later B.", "B", "C", 0),
        ("<think>A</think>Final: D", "D", "D", 1),
        ("", "", "A", 0),
        ("No answer.", "No answer", "A", 0),
        ("D/", "D", "D", 1),
    ],
)
def test_gpqa_last_letter_matches_original_grader(response, extracted, expected, reward, tmp_path):
    for result in source_rewards(McqProfile.EVALCHEMY_GPQA, expected, response, tmp_path):
        assert result.reward == reward
        assert result.detail["extracted"] == extracted


@pytest.mark.parametrize(
    "response,extracted,expected,reward",
    [
        ("B. This is valid.", "B", "B", 1),
        ("b. This is valid.", None, "B", 0),
        ("The ANSWER IS (c). Later B.", "C", "C", 1),
        ("aNsWeR: d. Later mention B.", "D", "B", 0),
        ("Answer: A. Answer: B.", "A", "B", 0),
        ("<think>answer is A</think>Answer: B", "B", "B", 1),
        ("", None, "B", 0),
        ("J", "J", "J", 1),
    ],
)
def test_mmlu_marker_priority_matches_original_grader(response, extracted, expected, reward, tmp_path):
    for result in source_rewards(McqProfile.EVALCHEMY_MMLU_PRO, expected, response, tmp_path):
        assert result.reward == reward
        assert result.detail["extracted"] == extracted
