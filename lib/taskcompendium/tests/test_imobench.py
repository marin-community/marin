# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IMO-AnswerBench autograder verdicts through the judge transport."""

import pytest
from verifyit.grade import Status
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate

from taskcompendium.importers.imobench import answer_judge_spec


@pytest.fixture
def connection(judge_server, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    return JudgeConnection(
        f"http://127.0.0.1:{judge_server.server_port}/v1", "fixture-key", default_model="fixture-judge"
    )


@pytest.mark.parametrize(
    "reply,reward",
    [
        ("<thinking>\n1. **Golden Answer:** 3\n4. **Conclusion:** Correct\n</thinking>\n\\boxed{Correct}", 1.0),
        ("<thinking>Conclusion: Incorrect</thinking> \\boxed{incorrect}", 0.0),
        ("\\boxed{Correct}", 1.0),
    ],
)
def test_answer_grader_reads_the_box_after_thinking(judge_server, connection, reply, reward):
    judge_server.reply = reply
    verdict = grade_judge_candidate(answer_judge_spec("Find all $N$.", "$N=3$"), "So $N = 3$.", connection=connection)
    assert verdict.status is Status.SCORED
    assert verdict.reward == reward
    prompt = judge_server.requests[-1]["messages"][-1]["content"]
    assert prompt.endswith("Problem: Find all $N$.\nModel Solution: So $N = 3$.\nGolden Answer: $N=3$\n")
