# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove free-form judge archive conversion and provenance."""

import json

import pytest
from tasktrove_verify.spec import RUBRIC_CHECKLIST, RUBRIC_REFERENCE, JudgeSpec, render_spec

from taskcompendium.importers.tasktrove.judge import import_task
from taskcompendium.importers.tasktrove.models import TaskArchive
from taskcompendium.lowering import render_instruction
from taskcompendium.models import AnswerType, VerifierKind
from taskcompendium.submission import PlainText
from taskcompendium.verifier_registry import resolve_verifier
from taskcompendium.verifiers.judge import JudgeVerifier


def _archive(family: str, converter: str, spec: JudgeSpec, *, tags: tuple[str, ...] = ("judge", "source")):
    manifest = (
        "[metadata]\n"
        f'family = "{family}"\n'
        f'converter = "{converter}"\n'
        'mode = "judge"\n'
        f"tags = {json.dumps(tags)}\n"
    ).encode()
    files = {
        "task.toml": manifest,
        "instruction.md": spec.question.encode(),
        "tests/verifier.toml": render_spec(spec).encode(),
    }
    if spec.context:
        files[f"tests/{spec.context}"] = b"Private rubric context."
    return TaskArchive("source", "row-1.tar.gz", "release", "revision", files)


@pytest.mark.parametrize(
    "family,converter,rubric",
    [
        ("qa-short-answer", "nemotron_openqa", RUBRIC_REFERENCE),
        ("llm-judge-freeform", "judge_rubric", RUBRIC_CHECKLIST),
    ],
)
def test_import_supported_judge_templates_keeps_rubric_private(family, converter, rubric):
    spec = JudgeSpec(
        references=("private reference",) if rubric == RUBRIC_REFERENCE else (),
        criteria=("private criterion",) if rubric == RUBRIC_CHECKLIST else (),
        question="Answer the source question.",
        context="context.txt" if rubric == RUBRIC_CHECKLIST else "",
        rubric=rubric,
    )
    source_archive = _archive(family, converter, spec)
    converted = import_task(source_archive)

    public_prompt = render_instruction(converted, PlainText(id="plain"))
    assert converted.answer_type is AnswerType.TEXT
    assert converted.source.importer_revision == "taskcompendium-tasktrove-judge-v0.1"
    assert converted.tags == ("judge", "source")
    assert converted.context.events[0].content == spec.question
    assert "private reference" not in public_prompt
    assert "private criterion" not in public_prompt
    private_verifier = resolve_verifier(converted.verifier)
    assert isinstance(private_verifier, JudgeVerifier)
    assert converted.verifier.kind is VerifierKind.JUDGE


def test_import_rejects_instruction_that_does_not_contain_judge_question():
    source_archive = _archive(
        "qa-short-answer", "nemotron_openqa", JudgeSpec(references=("answer",), question="Expected question")
    )
    source_archive.files["instruction.md"] = b"Different instruction"

    with pytest.raises(ValueError, match="question is not present"):
        import_task(source_archive)


def test_import_rejects_unrecognized_judge_converter():
    source_archive = _archive("qa-short-answer", "unknown", JudgeSpec(question="Question"))

    with pytest.raises(ValueError, match="Unsupported TaskTrove judge template"):
        import_task(source_archive)
