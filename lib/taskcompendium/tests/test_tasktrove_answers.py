# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean MCQA import and direct-chat Harbor coverage."""

import io
import json
import subprocess
import sys
import tarfile

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.convert import MAX_ARCHIVE_MEMBERS, read_archive
from taskcompendium.importers.tasktrove.mcqa import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor, read_specification
from taskcompendium.models import AnswerType, ConversationTrace, TaskSpec, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, JsonAnswer, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial

TASKTROVE_SOURCE = "synthetic__mcqa-demo"
TASKTROVE_PATH = "synthetic-mcqa.tar.gz"
RELEASE_URI = "https://example.invalid/tasktrove-clean"
RELEASE_REVISION = "synthetic-release-1"


def _archive_bytes():
    manifest = f"""[metadata]
tasktrove_source = "{TASKTROVE_SOURCE}"
tasktrove_path = "{TASKTROVE_PATH}"
family = "qa-short-answer"
converter = "nemotron_mcqa"
template_id = "c814af4f124d"
mode = "mcq"
tags = ["qa", "mcq", "synthetic"]
"""
    instruction = (
        "You are answering a multiple-choice question. Read the question below and write your final "
        "answer to `/app/answer.txt`.\n\n"
        "The verifier extracts a single letter (A/B/C/...) from your answer file using a regex pattern; "
        "the simplest valid output is a file containing exactly\n`Answer: X` (where X is your chosen letter).\n\n"
        "---\n\nAnswer the following multiple choice question. The last line of your response "
        "should be in the following format: 'Answer: A/B/C/D/E' (e.g. 'Answer: D').\n\n"
        "Synthetic question: Which label is assigned to this example?\n"
        "A: Alpha\nB: Bravo\nC: Charlie\nD: Delta\nE: Echo"
    )
    files = {
        "task.toml": manifest.encode(),
        "instruction.md": instruction.encode(),
        "tests/verifier.toml": b'mode = "mcq"\nexpected = "D"\noptions = 5\n',
    }
    archive_data = io.BytesIO()
    with tarfile.open(fileobj=archive_data, mode="w:gz") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return archive_data.getvalue()


def _archive(release_revision: str = RELEASE_REVISION):
    return read_archive(
        _archive_bytes(),
        TASKTROVE_SOURCE,
        TASKTROVE_PATH,
        RELEASE_URI,
        release_revision,
    )


def test_import_preserves_release_identity():
    specification = import_task(_archive())
    assert specification.source.dataset == RELEASE_URI
    assert specification.source.revision == RELEASE_REVISION
    assert specification.source.row == f"{TASKTROVE_SOURCE}:{TASKTROVE_PATH}"
    later_release = import_task(_archive("synthetic-release-2"))
    assert later_release.id != specification.id


def test_import_preserves_ordered_source_tags_through_harbor_export(tmp_path):
    specification = import_task(_archive())
    expected_tags = ("qa", "mcq", "synthetic")

    assert specification.tags == expected_tags
    assert TaskSpec.model_validate_json(specification.model_dump_json()).tags == expected_tags

    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )

    assert read_specification(task / "specification.json").tags == expected_tags


def test_supported_mcqa_template_preserves_synthetic_choices_and_tags():
    specification = import_task(_archive())
    prompt = specification.context.events[0].content

    assert specification.tags == ("qa", "mcq", "synthetic")
    assert "Synthetic question" in prompt
    assert [prompt.index(option) for option in ("A: Alpha", "B: Bravo", "C: Charlie", "D: Delta", "E: Echo")] == sorted(
        prompt.index(option) for option in ("A: Alpha", "B: Bravo", "C: Charlie", "D: Delta", "E: Echo")
    )
    assert "/app/answer.txt" not in prompt
    assert "verifier" not in prompt.lower()


def test_import_removes_source_submission_instructions():
    specification = import_task(_archive())
    prompt = specification.context.events[0].content
    assert "verifier" not in prompt.lower()
    assert "/app/answer.txt" not in prompt
    assert "Synthetic question" in prompt
    public = render_instruction(specification, PlainText(id="plain"))
    assert "verifier" not in public.lower()
    assert specification.environment_requirements.capabilities == ()
    assert specification.answer_type is AnswerType.TEXT


async def test_imported_mcqa_matches_source_grading(tmp_path):
    specification = import_task(_archive())
    assert specification.verifier.kind is VerifierKind.MCQ_ANSWER
    assert json.loads(specification.verifier.parameters_json) == {"expected": "D", "options": 5}
    source_contract = McqSpec(expected="D", options=5, output=str(tmp_path / "source-answer.txt"))
    convention = PlainText(id="plain")
    for source_response, response, reward in (
        ("Answer: D", "D", 1.0),
        ("Answer: C", "C", 0.0),
        ("Answer: Z", "Z", 0.0),
    ):
        (tmp_path / "source-answer.txt").write_text(source_response)
        assert source_grade(source_contract, tmp_path, tmp_path).reward == reward
        result = await grade_answer(
            specification,
            convention,
            GradingAttempt(
                ConversationTrace(
                    events=(*specification.context.events, TextMessage(role="assistant", content=response))
                ),
                object(),
            ),
        )
        assert (result.status, result.reward) == (Outcome.GRADED, reward)


async def test_imported_mcqa_extracts_json_and_rejects_malformed_answers():
    specification = import_task(_archive())
    convention = PlainText(id="plain")
    json_result = await grade_answer(
        specification,
        JsonAnswer(id="json"),
        GradingAttempt(
            ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content='{"answer":"D"}'))
            ),
            object(),
        ),
    )
    malformed = await grade_answer(
        specification,
        convention,
        GradingAttempt(
            ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content="Answer: C"))
            ),
            object(),
        ),
    )
    assert (json_result.status, json_result.reward) == (Outcome.GRADED, 1.0)
    assert (malformed.status, malformed.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)


def test_import_rejects_non_mcqa_source_before_lowering():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "exact"\nexpected = ["C"]\n'

    with pytest.raises(ValueError, match="MCQ verifier"):
        import_task(archive)


def test_import_rejects_unknown_mcqa_template_id():
    archive = _archive()
    archive.files["task.toml"] = archive.files["task.toml"].replace(
        b'template_id = "c814af4f124d"',
        b'template_id = "unreviewed-template"',
    )

    with pytest.raises(ValueError, match="Unsupported TaskTrove MCQA source"):
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
        read_archive(_archive_bytes(), "other_source", TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


def test_archive_rejects_excessive_empty_members():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            archive.addfile(tarfile.TarInfo(f"empty-{index}"), io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        read_archive(data.getvalue(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


async def test_imported_mcqa_runs_through_direct_chat_harbor(tmp_path):
    specification = import_task(_archive())
    environment_config = HarborEnvironmentConfig()
    task = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        environment_config,
        tmp_path / "task",
    )

    result = await run_replay_trial(task, {"role": "assistant", "content": "D"}, tmp_path / "trials", "mcqa")

    outcome = json.loads((tmp_path / "trials/mcqa/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome == {"status": "graded", "reward": 1.0, "error": None}


async def test_synthetic_mcqa_runs_positive_and_negative_harbor_trials(tmp_path):
    task = lower_to_harbor(
        import_task(_archive()),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )

    correct = await run_replay_trial(task, {"role": "assistant", "content": "D"}, tmp_path / "trials", "correct")
    incorrect = await run_replay_trial(task, {"role": "assistant", "content": "B"}, tmp_path / "trials", "incorrect")

    correct_outcome = json.loads((tmp_path / "trials/correct/verifier/taskcompendium-result.json").read_text())
    incorrect_outcome = json.loads((tmp_path / "trials/incorrect/verifier/taskcompendium-result.json").read_text())
    assert correct.exception_info is None, correct.exception_info
    assert incorrect.exception_info is None, incorrect.exception_info
    assert correct_outcome == {"status": "graded", "reward": 1.0, "error": None}
    assert incorrect_outcome == {"status": "graded", "reward": 0.0, "error": None}


def test_imported_mcqa_resolves_verifier_in_fresh_process(tmp_path):
    task = lower_to_harbor(
        import_task(_archive()),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    script = (
        "import asyncio, json, sys; from pathlib import Path; "
        "from taskcompendium.verifier_registry import grade_answer; "
        "from taskcompendium.submission import GradingAttempt; "
        "from taskcompendium.models import ConversationTrace, TextMessage; "
        "from taskcompendium.lowering import read_submission_convention, read_specification; "
        "root = Path(sys.argv[1]); "
        "specification = read_specification(root / 'specification.json'); "
        "result = asyncio.run(grade_answer(specification, "
        "read_submission_convention(root / 'submission_convention.json'), "
        "GradingAttempt(ConversationTrace(events=(*specification.context.events, "
        "TextMessage(role='assistant', content='D'))), object()))); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )

    completed = subprocess.run([sys.executable, "-c", script, str(task)], capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}
