# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Clean IFEval source import and Harbor coverage."""

import io
import json
import tarfile

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.modes import ifeval as source_ifeval
from tasktrove_verify.spec import IfevalSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove import import_task as import_tasktrove
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationTrace, TaskSpec, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial

TASKTROVE_SOURCE = "synthetic-ifeval"
TASKTROVE_PATH = "ifeval-task.tar.gz"
RELEASE_URI = "fixture://tasktrove"
RELEASE_REVISION = "test-revision"
SUBMISSION_SCAFFOLD = """You are running in a shell-based sandbox.
The verifier reads ONLY `/app/answer.txt`.

Your answer.txt must be the raw answer prose itself.
cat > /app/answer.txt
Verify the file exists and contains your answer.

---

"""
GOOD_ANSWER = "yard watched the rain.\n\nyard walked home."
BAD_ANSWER = "yard watched the rain.\n\nShe walked home."


def _archive():
    metadata = f"""[metadata]
tasktrove_source = "{TASKTROVE_SOURCE}"
tasktrove_path = "{TASKTROVE_PATH}"
family = "instruction-following"
converter = "nemotron_ifeval"
mode = "ifeval"
tags = ["instruction-following", "ifeval", "synthetic"]
"""
    members = {
        "task.toml": metadata.encode(),
        "instruction.md": (
            (
                SUBMISSION_SCAFFOLD + "Make sure to follow these requirements: write two paragraphs. "
                "Start the first word of each sentence with yard."
            ).encode()
        ),
        "tests/verifier.toml": (
            b'mode = "ifeval"\n\n[[constraints]]\nname = "length_constraints:number_paragraphs"\n'
            b"[constraints.params]\nnum_paragraphs = 2\n\n[[constraints]]\n"
            b'name = "first_word:first_word_sent"\n[constraints.params]\nfirst_word = "yard"\n'
        ),
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for name, data in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
    return read_archive(buffer.getvalue(), TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)


def test_import_preserves_source_pin_and_ordered_tags():
    specification = import_tasktrove(_archive())
    assert specification.source.dataset == RELEASE_URI
    assert specification.source.revision == RELEASE_REVISION
    assert specification.source.row == f"{TASKTROVE_SOURCE}:{TASKTROVE_PATH}"
    assert specification.tags == ("instruction-following", "ifeval", "synthetic")
    assert TaskSpec.model_validate_json(specification.model_dump_json()).tags == specification.tags


def test_import_removes_only_known_submission_scaffold():
    specification = import_tasktrove(_archive())
    prompt = specification.context.events[0].content
    public = render_instruction(specification, PlainText(id="plain"))

    assert "Make sure to follow these requirements" in prompt
    assert "two paragraphs" in prompt
    assert "first word of each sentence" in prompt
    assert "/app/answer.txt" not in public
    assert "verifier" not in public.lower()
    assert specification.environment_requirements.capabilities == ()
    assert specification.answer_type is AnswerType.TEXT
    assert specification.verifier.kind is VerifierKind.IFEVAL


async def test_imported_ifeval_matches_source_checker_for_good_and_bad_answers(tmp_path):
    specification = import_tasktrove(_archive())
    source_spec = parse_spec(_archive().files["tests/verifier.toml"].decode())
    assert isinstance(source_spec, IfevalSpec)
    convention = PlainText(id="plain")

    for answer, expected_reward in ((GOOD_ANSWER, 1.0), (BAD_ANSWER, 0.0)):
        (tmp_path / "answer.txt").write_text(answer)
        assert source_grade(source_spec, tmp_path, tmp_path).reward == expected_reward
        result = await grade_answer(
            specification,
            convention,
            GradingAttempt(
                ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=answer))),
                {},
                object(),
            ),
        )
        assert (result.status, result.reward) == (Outcome.GRADED, expected_reward)


async def test_ifeval_checker_crash_is_an_infrastructure_error(monkeypatch):
    specification = import_tasktrove(_archive())

    def crash(text, params):
        raise RuntimeError("checker failure")

    monkeypatch.setitem(source_ifeval.CONSTRAINTS, "first_word:first_word_sent", crash)
    result = await grade_answer(
        specification,
        PlainText(id="plain"),
        GradingAttempt(
            ConversationTrace(
                events=(*specification.context.events, TextMessage(role="assistant", content=GOOD_ANSWER))
            ),
            {},
            object(),
        ),
    )

    assert result.status is Outcome.INFRA_ERROR
    assert result.reward is None


@pytest.mark.parametrize(
    ("answer", "reward"),
    ((GOOD_ANSWER, 1.0), (BAD_ANSWER, 0.0)),
    ids=("all-source-constraints-pass", "first-word-constraint-fails"),
)
async def test_imported_ifeval_runs_through_harbor(tmp_path, answer: str, reward: float):
    task = lower_to_harbor(
        import_tasktrove(_archive()),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "task",
    )
    result = await run_replay_trial(task, {"role": "assistant", "content": answer}, tmp_path / "trials", "ifeval")

    outcome = json.loads((tmp_path / "trials/ifeval/verifier/taskcompendium-result.json").read_text())
    assert result.exception_info is None, result.exception_info
    assert outcome == {"status": "graded", "reward": reward, "error": None}


def test_import_rejects_archive_with_unrecognized_source_prompt():
    archive = _archive()
    archive.files["instruction.md"] = archive.files["instruction.md"].replace(
        b"The verifier reads ONLY `/app/answer.txt`", b"The answer is written somewhere else"
    )

    with pytest.raises(ValueError, match="Unsupported IFEval submission scaffold"):
        import_tasktrove(archive)


def test_import_rejects_non_ifeval_contract():
    archive = _archive()
    archive.files["tests/verifier.toml"] = b'mode = "exact"\nexpected = ["yard"]\n'

    with pytest.raises(ValueError, match="IFEval verifier"):
        import_tasktrove(archive)
