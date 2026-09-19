# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for TaskTrove reference-answer judge conversion."""

from pathlib import Path

import msgspec
import pytest

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.tasktrove import CLEAN_09_RELEASE, read_archive
from taskcompendium.importers.tasktrove_judge import import_task
from taskcompendium.judging import JudgeReply
from taskcompendium.models import (
    VERIFIER_REVISION,
    AssistantFinal,
    BoxedLatex,
    Capability,
    JsonPath,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    Outcome,
    Rejected,
    Rendering,
    TaskRequirements,
    WorkspaceState,
)
from taskcompendium.rendering import render_instruction

FIXTURES = Path(__file__).parent / "fixtures/tasktrove/judge"
CLEAN_09_FIXTURES = Path(__file__).parent / "fixtures/tasktrove-clean-09/judge"


class FakeJudge:
    def __init__(self, score: str):
        self.score = score
        self.prompts: list[str] = []

    def complete(self, prompt: str, policy: JudgeModelPolicy, timeout: float) -> JudgeReply:
        self.prompts.append(prompt)
        return JudgeReply(f"The answer is evaluated by the fixture.\nSCORE: {self.score}", policy.model, "fixture")


@pytest.mark.parametrize("path", ["/repo/report.md", "/assets/report.md"])
def test_judge_receives_evidence_from_declared_workspace_roots(tmp_path, path):
    _, spec = _spec("11676")
    assert not isinstance(spec, Rejected)
    step = spec.steps[0]
    config = msgspec.structs.replace(step.verifier.judge, view=JudgeView(files=(path,)))
    spec = msgspec.structs.replace(
        spec,
        requirements=TaskRequirements(
            (Capability.FILESYSTEM,), WorkspaceState(workdir="/repo", additional_directories=("/assets",))
        ),
        steps=(msgspec.structs.replace(step, verifier=msgspec.structs.replace(step.verifier, judge=config)),),
    )
    relative = "report.md" if path.startswith("/repo/") else "__external__/assets/report.md"
    evidence = tmp_path / relative
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("The report contains the requested evidence.")
    judge = FakeJudge("1")
    result = grade_attempt(spec, Rendering("plain", AssistantFinal()), "See the report.", tmp_path, judge_client=judge)
    assert (result.status, result.reward) == (Outcome.GRADED, 1.0)
    assert "The report contains the requested evidence." in judge.prompts[0]


def _spec(row: str):
    archive = read_archive((FIXTURES / f"judge-row-{row}.tar.gz").read_bytes(), row, "qa-short-answer")
    config = JudgeConfig(JudgeModelPolicy("fixture", "small", "fixture", "https://fixture.invalid/v1"), JudgeView())
    return archive, import_task(archive, config)


def test_reference_judge_import_preserves_contract_and_strips_delivery_wrapper():
    archive, result = _spec("11676")

    assert not isinstance(result, Rejected)
    assert result.requirements.capabilities == ()
    assert result.steps[0].verifier.parameters["references"] == (
        "No, because the buyer lacks legal title to encumber the property.",
    )
    assert result.steps[0].verifier.parameters["exact_gate"] is True
    assert result.steps[0].verifier.parameters["rubric"] == "reference"
    assert result.steps[0].verifier.judge is not None
    assert "/app/response.txt" not in result.steps[0].instructions
    assert not result.steps[0].instructions.endswith("Remember to put your answer inside \\boxed{}.")
    assert "contract for deed transaction" in result.steps[0].instructions
    assert archive.source.row == "11676"


def test_reference_judge_grades_exact_gate_before_fixture_model(tmp_path):
    _, result = _spec("11678")
    assert not isinstance(result, Rejected)
    protocol = Rendering("judge", AssistantFinal())

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


@pytest.mark.parametrize(
    ("filename", "reference"),
    [
        (
            "openqa-80f6c461ebcf.tar.gz",
            "Because the bounds for x and y are independent, allowing for simpler limits of integration when "
            "integrating with respect to x and y before z.",
        ),
        ("openqa-c7e9374b56ea.tar.gz", "Continuous exposure (control)"),
    ],
)
def test_clean09_openqa_is_release_pinned_and_uses_a_private_reference_gate(tmp_path, filename, reference):
    archive = read_archive((CLEAN_09_FIXTURES / filename).read_bytes(), filename, "qa-short-answer", CLEAN_09_RELEASE)
    config = JudgeConfig(JudgeModelPolicy("fixture", "small", "fixture", "https://fixture.invalid/v1"), JudgeView())
    specification = import_task(archive, config)

    assert not isinstance(specification, Rejected)
    assert specification.metadata.source.dataset == CLEAN_09_RELEASE.root
    assert specification.steps[0].verifier.implementation_revision == VERIFIER_REVISION
    assert "judge" not in specification.steps[0].instructions.lower()
    assert grade_attempt(specification, Rendering("plain", AssistantFinal()), reference, tmp_path).reward == 1.0


@pytest.mark.parametrize("row", ["11676", "11677", "11678"])
def test_judge_task_answer_wrapper_belongs_only_to_rendering(row, tmp_path):
    archive, spec = _spec(row)
    assert not isinstance(spec, Rejected)
    assert r"\boxed" in archive.instructions
    assert r"\boxed" not in spec.steps[0].instructions
    assert r"\boxed" not in spec.steps[0].verifier.parameters["question"]
    plain = Rendering("plain", AssistantFinal())
    structured = Rendering("json", AssistantFinal(JsonPath()))
    boxed = Rendering("boxed", AssistantFinal(BoxedLatex()))
    assert r"\boxed" not in render_instruction(spec, plain)
    assert r"\boxed" not in render_instruction(spec, structured)
    assert r"\boxed" in render_instruction(spec, boxed)
    reference = spec.steps[0].verifier.parameters["references"][0]
    assert grade_attempt(spec, plain, reference, tmp_path).reward == 1.0
    assert grade_attempt(spec, boxed, r"\boxed{" + reference + "}", tmp_path).reward == 1.0
