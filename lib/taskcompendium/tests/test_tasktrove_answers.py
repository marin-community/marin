# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean MCQA import and direct-chat Harbor coverage."""

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import McqSpec

from taskcompendium.grading import Outcome, grade_answer
from taskcompendium.harbor.runner import HarborLaunch, run_trial
from taskcompendium.importers.tasktrove import MAX_ARCHIVE_MEMBERS, read_archive
from taskcompendium.importers.tasktrove_mcqa import import_task
from taskcompendium.lowering import HarborTaskBinding, lower_to_harbor
from taskcompendium.models import AnswerKind
from taskcompendium.rendering import AnswerFormat, Rendering, render_instruction

FIXTURE = Path(__file__).parent / "fixtures/tasktrove/mcq-1961bdb52b5a.tar.gz"
TASKTROVE_SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
TASKTROVE_PATH = "Nemotron-RL-knowledge-mcqa-1961bdb52b5a.tar.gz"
RELEASE_URI = "s3://marin-us-east-02a/marin/tasktrove/clean/2026.09.10.9"
RELEASE_REVISION = "2026.09.10.9"


def _archive(release_revision: str = RELEASE_REVISION):
    return read_archive(FIXTURE.read_bytes(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, release_revision)


def test_import_preserves_release_provenance_source_grading_and_prompt_hygiene(tmp_path):
    archive = _archive()
    specification = import_task(archive)

    assert specification.source.dataset == RELEASE_URI
    assert specification.source.revision == RELEASE_REVISION
    assert specification.source.row == f"{TASKTROVE_SOURCE}:{TASKTROVE_PATH}"
    assert "/" not in specification.id
    assert specification.requirements.capabilities == ()
    assert specification.answer_kind is AnswerKind.OPTION_LETTER
    assert "verifier" not in specification.instructions.lower()
    assert "/app/answer.txt" not in specification.instructions
    assert "theranostics clinical trials" in specification.instructions
    public = render_instruction(specification, Rendering("plain", AnswerFormat.PLAIN))
    assert "verifier" not in public.lower()

    later_release = import_task(_archive("2026.09.10.10"))
    assert later_release.id != specification.id

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
        read_archive(FIXTURE.read_bytes(), "other_source", TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


def test_archive_rejects_excessive_empty_members():
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        for index in range(MAX_ARCHIVE_MEMBERS + 1):
            archive.addfile(tarfile.TarInfo(f"empty-{index}"), io.BytesIO())

    with pytest.raises(ValueError, match="member limit"):
        read_archive(data.getvalue(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


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


def test_imported_mcqa_resolves_verifier_in_fresh_process(tmp_path):
    task = lower_to_harbor(
        import_task(_archive()),
        Rendering("plain", AnswerFormat.PLAIN),
        HarborTaskBinding(),
        tmp_path / "task",
    )
    script = (
        "import json, sys; from pathlib import Path; "
        "from taskcompendium.grading import grade_answer; "
        "from taskcompendium.lowering import read_rendering, read_specification; "
        "root = Path(sys.argv[1]); "
        "result = grade_answer(read_specification(root / 'specification.json'), "
        "read_rendering(root / 'rendering.json'), 'C'); "
        "print(json.dumps({'status': result.status, 'reward': result.reward}))"
    )

    completed = subprocess.run([sys.executable, "-c", script, str(task)], capture_output=True, text=True, check=True)

    assert json.loads(completed.stdout) == {"status": "graded", "reward": 1.0}
