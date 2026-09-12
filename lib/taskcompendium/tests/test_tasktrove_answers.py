# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove answer-family conversion contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from tasktrove_verify.grade import grade as source_grade

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.tasktrove import read_archive
from taskcompendium.importers.tasktrove_answers import import_task
from taskcompendium.models import (
    AssistantFinal,
    Chat,
    JsonPath,
    NoEnvironment,
    Outcome,
    Protocol,
    Rejected,
    RejectionReason,
    XmlPath,
)

FIXTURES = Path(__file__).parent / "fixtures/tasktrove/answers"


def _archive(name: str, family: str):
    return read_archive((FIXTURES / name).read_bytes(), name.split("-row-")[1].split(".")[0], family)


def test_mcq_source_has_answer_only_contract_and_strips_wrappers():
    result = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))

    assert not isinstance(result, Rejected)
    assert isinstance(result.environment, NoEnvironment)
    assert "/app/answer.txt" not in result.instructions
    assert "last line of your response" not in result.instructions
    assert "luminance contrast ratio" in result.instructions


def test_mcq_grading_matches_pinned_source_oracle():
    specification = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))
    assert not isinstance(specification, Rejected)
    protocol = Protocol("mcq", Chat(), AssistantFinal())

    correct = grade_attempt(specification, protocol, "C", Path("/tmp"))
    wrong = grade_attempt(specification, protocol, "D", Path("/tmp"))

    assert correct.status is Outcome.GRADED and correct.reward == 1.0
    assert wrong.status is Outcome.GRADED and wrong.reward == 0.0


def test_mcq_grading_preserves_answer_across_json_and_xml_renderings():
    specification = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))
    assert not isinstance(specification, Rejected)

    json_result = grade_attempt(
        specification,
        Protocol("mcq-json", Chat(), AssistantFinal(JsonPath("$.answer"))),
        '{"answer":"C"}',
        Path("/tmp"),
    )
    xml_result = grade_attempt(
        specification,
        Protocol("mcq-xml", Chat(), AssistantFinal(XmlPath("/answer"))),
        "<answer>C</answer>",
        Path("/tmp"),
    )

    assert json_result.status is Outcome.GRADED and json_result.reward == 1.0
    assert xml_result.status is Outcome.GRADED and xml_result.reward == 1.0


def test_defective_exact_source_is_rejected_without_fixing_gold():
    result = import_task(_archive("exact-row-113.tar.gz", "math-answer"))

    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.BROKEN_GRADER


@pytest.mark.parametrize(
    "name,family,good,bad",
    [
        ("mcq-row-1972.tar.gz", "qa-short-answer", "C", "D"),
        ("exact-row-119.tar.gz", "math-answer", "off", "on"),
    ],
)
def test_real_answer_family_matches_source_grading_and_rejects_empty(tmp_path, name, family, good, bad):
    archive = _archive(name, family)
    specification = import_task(archive)
    assert not isinstance(specification, Rejected)
    source_output = tmp_path / "source-answer"
    contract = replace(archive.verifier, output=str(source_output))
    protocol = Protocol("plain", Chat(), AssistantFinal())
    for answer, reward in ((good, 1.0), (bad, 0.0)):
        source_output.write_text(f"Answer: {answer}" if family == "qa-short-answer" else answer)
        original = source_grade(contract, tmp_path, tmp_path)
        lowered = grade_attempt(specification, protocol, answer, tmp_path)
        assert original.reward == lowered.reward == reward
    source_output.write_text("")
    assert source_grade(contract, tmp_path, tmp_path).reward == 0.0
    empty = grade_attempt(specification, protocol, "", tmp_path)
    assert empty.status == Outcome.EXTRACTION_ERROR and empty.reward is None


@pytest.mark.parametrize("gold", ["Z", "0", "CC"])
def test_mcq_rejects_invalid_source_option_before_lowering(gold):
    archive = _archive("mcq-row-1972.tar.gz", "qa-short-answer")
    archive.files["tests/verifier.toml"] = archive.files["tests/verifier.toml"].replace(
        b'expected = "C"', f'expected = "{gold}"'.encode()
    )
    result = import_task(archive)
    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.BROKEN_GRADER
