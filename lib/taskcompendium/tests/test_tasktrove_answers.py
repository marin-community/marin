# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove answer-family conversion contracts."""

from dataclasses import replace
from pathlib import Path

import pytest
from tasktrove_verify.grade import grade as source_grade

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.tasktrove import CLEAN_09_RELEASE, read_archive
from taskcompendium.importers.tasktrove_answers import import_task
from taskcompendium.models import (
    VERIFIER_REVISION,
    AssistantFinal,
    JsonPath,
    Outcome,
    Rejected,
    RejectionReason,
    Rendering,
    XmlPath,
)

FIXTURES = Path(__file__).parent / "fixtures/tasktrove/answers"
CLEAN_09_FIXTURES = Path(__file__).parent / "fixtures/tasktrove-clean-09/answers"


def _archive(name: str, family: str):
    return read_archive((FIXTURES / name).read_bytes(), name.split("-row-")[1].split(".")[0], family)


def test_mcq_source_has_answer_only_contract_and_strips_wrappers():
    result = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))

    assert not isinstance(result, Rejected)
    assert result.requirements.capabilities == ()
    assert "/app/answer.txt" not in result.steps[0].instructions
    assert "last line of your response" not in result.steps[0].instructions
    assert "luminance contrast ratio" in result.steps[0].instructions


def test_mcq_grading_matches_pinned_source_oracle():
    specification = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))
    assert not isinstance(specification, Rejected)
    protocol = Rendering("mcq", AssistantFinal())

    correct = grade_attempt(specification, protocol, "C", Path("/tmp"))
    wrong = grade_attempt(specification, protocol, "D", Path("/tmp"))

    assert correct.status is Outcome.GRADED and correct.reward == 1.0
    assert wrong.status is Outcome.GRADED and wrong.reward == 0.0


def test_mcq_grading_preserves_answer_across_json_and_xml_renderings():
    specification = import_task(_archive("mcq-row-1972.tar.gz", "qa-short-answer"))
    assert not isinstance(specification, Rejected)

    json_result = grade_attempt(
        specification,
        Rendering("mcq-json", AssistantFinal(JsonPath("$.answer"))),
        '{"answer":"C"}',
        Path("/tmp"),
    )
    xml_result = grade_attempt(
        specification,
        Rendering("mcq-xml", AssistantFinal(XmlPath("/answer"))),
        "<answer>C</answer>",
        Path("/tmp"),
    )

    assert json_result.status is Outcome.GRADED and json_result.reward == 1.0
    assert xml_result.status is Outcome.GRADED and xml_result.reward == 1.0


@pytest.mark.parametrize(
    ("filename", "row", "correct", "wrong"),
    [
        ("mcq-1961bdb52b5a.tar.gz", "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz", "C", "A"),
        ("mcq-cce3426cf566.tar.gz", "Nemotron-RL-knowledge-mcqa-cce3426cf566.tar.gz", "G", "A"),
    ],
)
def test_clean09_mcq_is_release_pinned_and_grades_without_prompted_evaluation(tmp_path, filename, row, correct, wrong):
    archive = read_archive((CLEAN_09_FIXTURES / filename).read_bytes(), row, "qa-short-answer", CLEAN_09_RELEASE)
    specification = import_task(archive)

    assert not isinstance(specification, Rejected)
    assert specification.metadata.source.dataset == CLEAN_09_RELEASE.root
    assert specification.metadata.source.revision == CLEAN_09_RELEASE.version
    assert specification.steps[0].verifier.implementation_revision == VERIFIER_REVISION
    assert "verifier" not in specification.steps[0].instructions.lower()
    assert "last line of your response" not in specification.steps[0].instructions.lower()
    assert r"\boxed" not in specification.steps[0].instructions
    rendering = Rendering("plain", AssistantFinal())
    assert grade_attempt(specification, rendering, correct, tmp_path).reward == 1.0
    assert grade_attempt(specification, rendering, wrong, tmp_path).reward == 0.0


def test_defective_exact_source_is_rejected_without_fixing_gold():
    result = import_task(_archive("exact-row-113.tar.gz", "math-answer"))

    assert isinstance(result, Rejected)
    assert result.reason == RejectionReason.BROKEN_GRADER


def test_valid_ascending_exact_source_is_not_rejected_for_generic_wording():
    archive = _archive("exact-row-113.tar.gz", "math-answer")
    archive.files["instruction.md"] = archive.files["instruction.md"].replace(
        b"sort these words in descending order", b"sort these words in ascending order"
    )

    result = import_task(archive)

    assert not isinstance(result, Rejected), result


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
    protocol = Rendering("plain", AssistantFinal())
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
