# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from verifyit.grade import GradingInfraError
from verifyit.modes.grade_judge import BASE_URL_ENV
from verifyit.spec import JudgeSpec

from taskcompendium.importers.evalchemy.scorers.main import score


@pytest.mark.parametrize(
    ("reply", "points", "passed"),
    [
        ("Partial credit\nVERDICT: 6.75", 6.75, 0),
        ("VERDICT: 7", 7, 1),
        ("VERDICT: 10", 10, 1),
        ("VERDICT: 0", 0, 0),
        ("VERDICT: 10.001", None, None),
        ("VERDICT: -1", None, None),
        ("**VERDICT: 7**", None, None),
        ("VERDICT: 7/10", None, None),
        ("VERDICT: 6\nVERDICT: 7", None, None),
        ("VERDICT: 7\nexplanation", None, None),
    ],
)
def test_research_verdict(judge_server, monkeypatch, reply, points, passed):
    monkeypatch.setenv(BASE_URL_ENV, f"http://127.0.0.1:{judge_server.server_port}/v1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    judge_server.reply = reply
    judge = JudgeSpec(
        question="Find x.",
        references=("x = {2}",),
        prompt_template="{question}\n{reference}\n{candidate}",
        model="fixture",
    )
    if points is None:
        with pytest.raises(GradingInfraError):
            score("frontierscience_research", {}, "Two.", judge)
    else:
        result = score("frontierscience_research", {}, "Two.", judge)
        assert result.reward == passed
        assert result.detail["metrics"] == {"pass_rate": passed, "rubric_points": points}
    [request] = judge_server.requests
    assert request["messages"] == [{"role": "user", "content": "Find x.\nx = {2}\nTwo."}]


def test_blank_research_final_scores_zero_without_a_judge():
    result = score("frontierscience_research", {}, "  ", JudgeSpec())
    assert result.reward == 0
    assert result.detail["metrics"] == {"pass_rate": 0, "rubric_points": 0}
