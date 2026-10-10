# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean MCQA import and pure grading coverage."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from verifyit.grade import grade as source_grade
from verifyit.spec import McqSpec, Mode

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.tasktrove.convert import MAX_ARCHIVE_MEMBERS, read_archive
from taskcompendium.importers.tasktrove.mcqa import import_task
from taskcompendium.models import (
    AnswerType,
    ConversationTrace,
    GradingAttempt,
    JsonAnswer,
    TextMessage,
    VerifyitGrader,
)
from taskcompendium.submission import render_instruction

FIXTURE = Path(__file__).parent / "fixtures/tasktrove/mcq-1961bdb52b5a.tar.gz"
TASKTROVE_SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"


TASKTROVE_PATH = "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz"
RELEASE_URI = "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9"

RELEASE_REVISION = "2026.09.10.9"


def _archive(release_revision: str = RELEASE_REVISION):
    return read_archive(FIXTURE.read_bytes(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, release_revision)


def test_import_preserves_release_identity():
    specification = import_task(_archive())
    assert specification.source.dataset == RELEASE_URI
    assert specification.source.revision == RELEASE_REVISION
    assert specification.source.row == f"{TASKTROVE_SOURCE}:{TASKTROVE_PATH}"
    later_release = import_task(_archive("2026.09.10.10"))
    assert later_release.id != specification.id


def test_import_removes_source_submission_instructions():
    specification = import_task(_archive())
    prompt = specification.context.events[0].content
    assert "verifier" not in prompt.lower()
    assert "/app/answer.txt" not in prompt
    assert "theranostics clinical trials" in prompt
    public = render_instruction(specification)
    assert "verifier" not in public.lower()
    assert specification.environment_requirements.capabilities == ()
    assert specification.answer_type is AnswerType.TEXT


def test_imported_mcqa_matches_source_grading(tmp_path):
    specification = import_task(_archive())
    assert isinstance(specification.grader, VerifyitGrader)
    assert specification.grader.mode == Mode.MCQ
    source_contract = McqSpec(expected="C", options=10, output=str(tmp_path / "source-answer.txt"))
    for source_response, response, reward in (
        ("Answer: C", "C", 1.0),
        ("Answer: D", "D", 0.0),
        ("Answer: Z", "Z", 0.0),
    ):
        (tmp_path / "source-answer.txt").write_text(source_response)
        assert source_grade(source_contract, tmp_path, tmp_path).reward == reward
        result = grade_answer(
            specification,
            GradingAttempt(
                ConversationTrace(
                    events=(*specification.context.events, TextMessage(role="assistant", content=response))
                ),
            ),
        )
        assert (result.status, result.reward) == (Outcome.GRADED, reward)


def test_imported_mcqa_extracts_json_and_scores_unformatted_answers_zero():
    specification = import_task(_archive())
    json_result = grade_answer(
        specification.model_copy(update={"answer_format": JsonAnswer()}),
        GradingAttempt(
            ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content='{"answer":"C"}'))
            ),
        ),
    )
    malformed = grade_answer(
        specification,
        GradingAttempt(
            ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content="Answer: C"))
            ),
        ),
    )
    assert (json_result.status, json_result.reward) == (Outcome.GRADED, 1.0)
    assert (malformed.status, malformed.reward) == (Outcome.GRADED, 0.0)


def test_import_rejects_non_mcqa_source_before_lowering():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "exact"\nexpected = ["C"]\n'

    with pytest.raises(ValueError, match="MCQ verifier"):
        import_task(archive)


def test_import_rejects_unknown_source_submission_format():
    archive = _archive()
    archive.files["instruction.md"] = archive.files["instruction.md"].replace(
        b"The last line of your response should be in the following format:",
        b"The last line of your response should be JSON in the following format:",
    )

    with pytest.raises(ValueError, match="Unsupported MCQA instruction format"):
        import_task(archive)


def test_import_accepts_plain_source_answer_line_template():
    archive = _archive()
    archive.files["instruction.md"] = (
        archive.files["instruction.md"]
        .replace(b"Answer: \\boxed{A/B/C/D/E/F/G/H/I/J}", b"Answer: A/B/C/D/E/F/G/H/I/J")
        .replace(b"Answer: \\boxed{B}", b"Answer: B")
    )

    specification = import_task(archive)

    assert specification.answer_type is AnswerType.TEXT
    assert "Answer:" not in specification.context.events[0].content


def test_archive_rejects_caller_identity_that_disagrees_with_metadata():
    with pytest.raises(ValueError, match="source identity"):
        read_archive(FIXTURE.read_bytes(), "other_source", TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


def test_archive_rejects_excessive_empty_members():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            archive.addfile(tarfile.TarInfo(f"empty-{index}"), io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        read_archive(data.getvalue(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


def test_imported_mcqa_resolves_verifier_in_fresh_process(tmp_path):
    path = tmp_path / "specification.json"
    path.write_text(import_task(_archive()).model_dump_json())
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import grade_answer; "
        "from taskcompendium.models import GradingAttempt, TaskSpec, ConversationTrace, TextMessage; "
        "specification = TaskSpec.model_validate_json(Path(sys.argv[1]).read_text()); "
        "result = grade_answer(specification, "
        "GradingAttempt(ConversationTrace(events=(*specification.context.events, "
        "TextMessage(role='assistant', content='C'))))); "
        "print(json.dumps({'status':result.status, 'reward':result.reward}))"
    )
    completed = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}
