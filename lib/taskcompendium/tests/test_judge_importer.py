# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for TaskTrove reference-answer judge conversion."""

from pathlib import Path

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_judge import import_task
from taskcompendium.judging import JudgeReply
from taskcompendium.models import (
    AssistantFinal,
    Chat,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    NoEnvironment,
    Outcome,
    Protocol,
    Rejected,
)

FIXTURES = Path(__file__).parent / "fixtures/tasktrove/judge"


class FakeJudge:
    def __init__(self, score: str):
        self.score = score

    def complete(self, prompt: str, policy: JudgeModelPolicy, timeout: float) -> JudgeReply:
        return JudgeReply(f"The answer is evaluated by the fixture.\nSCORE: {self.score}", policy.model, "fixture")


def _spec(row: str):
    archive = read_archive((FIXTURES / f"judge-row-{row}.tar.gz").read_bytes(), row, "qa-short-answer")
    config = JudgeConfig(JudgeModelPolicy("fixture", "small", "fixture", "https://fixture.invalid/v1"), JudgeView())
    return archive, import_task(archive, config)


def test_reference_judge_import_preserves_contract_and_strips_delivery_wrapper():
    archive, result = _spec("11676")

    assert not isinstance(result, Rejected)
    assert isinstance(result.environment, NoEnvironment)
    assert result.verifier.parameters["references"] == (
        "No, because the buyer lacks legal title to encumber the property.",
    )
    assert result.verifier.parameters["exact_gate"] is True
    assert result.verifier.parameters["rubric"] == "reference"
    assert result.verifier.judge is not None
    assert "/app/response.txt" not in result.instructions
    assert not result.instructions.endswith("Remember to put your answer inside \\boxed{}.")
    assert "contract for deed transaction" in result.instructions
    assert archive.source.row == "11676"


def test_reference_judge_grades_exact_gate_before_fixture_model(tmp_path):
    _, result = _spec("11678")
    assert not isinstance(result, Rejected)
    protocol = Protocol("judge", Chat(), AssistantFinal())

    exact = grade_attempt(result, protocol, r"$ \frac{4\pi a \sin(\theta_0)}{\lambda} $", tmp_path)
    judged = grade_attempt(
        result,
        protocol,
        "The phase difference is four pi a sin(theta0) over lambda.",
        tmp_path,
        judge_client=FakeJudge("1"),
    )
    bad = grade_attempt(
        result,
        protocol,
        "The aperture has no phase difference.",
        tmp_path,
        judge_client=FakeJudge("0"),
    )

    assert exact.status is Outcome.GRADED and exact.reward == 1.0
    assert judged.status is Outcome.GRADED and judged.reward == 1.0
    assert bad.status is Outcome.GRADED and bad.reward == 0.0
