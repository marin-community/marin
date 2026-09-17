# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean MCQA import and direct-chat Harbor coverage."""

import io
import json
import tarfile
from pathlib import Path

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import Outcome, grade_answer
from taskcompendium.harbor.runner import HarborLaunch, run_trial
from taskcompendium.importers.tasktrove import MAX_ARCHIVE_MEMBERS, RELEASE, RELEASE_ROOT, read_archive
from taskcompendium.importers.tasktrove_mcqa import import_task
from taskcompendium.lowering import HarborTaskBinding, lower_to_harbor
from taskcompendium.models import AnswerKind, MultipleChoiceAnswer
from taskcompendium.rendering import AnswerFormat, Rendering, render_instruction

FIXTURE = Path(__file__).parent / "fixtures/tasktrove/mcq-1961bdb52b5a.tar.gz"
TASKTROVE_SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
TASKTROVE_PATH = "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz"


def _archive():
    return read_archive(FIXTURE.read_bytes(), TASKTROVE_SOURCE, TASKTROVE_PATH)


def test_import_preserves_release_provenance_source_grading_and_prompt_hygiene(tmp_path):
    archive = _archive()
    specification = import_task(archive)

    assert specification.source.dataset == RELEASE_ROOT
    assert specification.source.revision == RELEASE
    assert specification.source.row == f"{TASKTROVE_SOURCE}:{TASKTROVE_PATH}"
    assert "/" not in specification.id
    assert specification.requirements.capabilities == ()
    assert specification.answer_kind is AnswerKind.OPTION_LETTER
    assert isinstance(specification.verifier, MultipleChoiceAnswer)
    assert "verifier" not in specification.instructions.lower()
    assert "/app/answer.txt" not in specification.instructions
    assert "theranostics clinical trials" in specification.instructions
    public = render_instruction(specification, Rendering("plain", AnswerFormat.PLAIN))
    assert "Return the selected option letter as plain text." in public
    assert "verifier" not in public.lower()

    source_contract = McqSpec(expected="C", options=10, output=str(tmp_path / "source-answer.txt"))
    rendering = Rendering("plain", AnswerFormat.PLAIN)
    for source_response, response, reward in (
        ("Answer: C", "C", 1.0),
        ("Answer: D", "D", 0.0),
        ("Answer: Z", "Z", 0.0),
    ):
        (tmp_path / "source-answer.txt").write_text(source_response)
        assert source_grade(source_contract, tmp_path, tmp_path).reward == reward
        result = grade_answer(specification, rendering, response)
        assert (result.status, result.reward) == (Outcome.GRADED, reward)
    json_result = grade_answer(specification, Rendering("json", AnswerFormat.JSON), '{"answer":"C"}')
    malformed = grade_answer(specification, rendering, "Answer: C")
    assert (json_result.status, json_result.reward) == (Outcome.GRADED, 1.0)
    assert (malformed.status, malformed.reward) == (Outcome.EXTRACTION_ERROR, None)


def test_import_rejects_non_mcqa_source_before_lowering():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "exact"\nexpected = ["C"]\n'

    with pytest.raises(ValueError, match="MCQ verifier"):
        import_task(archive)


def test_archive_rejects_caller_identity_that_disagrees_with_metadata():
    with pytest.raises(ValueError, match="source identity"):
        read_archive(FIXTURE.read_bytes(), "other_source", TASKTROVE_PATH)


def test_archive_rejects_excessive_empty_members():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            archive.addfile(tarfile.TarInfo(f"empty-{index}"), io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        read_archive(data.getvalue(), TASKTROVE_SOURCE, TASKTROVE_PATH)


async def test_imported_mcqa_runs_through_direct_chat_harbor(tmp_path):
    specification = import_task(_archive())
    binding = HarborTaskBinding()
    task = lower_to_harbor(specification, Rendering("plain", AnswerFormat.PLAIN), binding, tmp_path / "task")

    result = await run_trial(
        task,
        binding,
        HarborLaunch("replay", agent_kwargs={"response": "C"}),
        tmp_path / "trials",
        "mcqa",
    )

    outcome = json.loads((tmp_path / "trials/mcqa/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome == {"status": "graded", "reward": 1.0, "error": None}
