# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove Reasoning Gym imports keep generated answers private and score upstream."""

import hashlib
import json
import tarfile
from io import BytesIO
from types import SimpleNamespace

import pytest
import reasoning_gym
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import ReasoningGymSpec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.reasoning_gym import IMPORTER_REVISION, import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial

TASKTROVE_SOURCE = "laion__nemotron-gym-reasoning-gym-v2"
TASKTROVE_PATH = "reasoning-gym-synthetic.tar.gz"
RELEASE_URI = "synthetic-tasktrove-clean"
RELEASE_REVISION = "synthetic-revision"
PREAMBLE = (
    "You are solving a procedurally-generated reasoning task from Reasoning Gym. Read the problem below and write "
    "your final answer to `/app/answer.txt`. The verifier will try the upstream Reasoning Gym scorer first, then fall "
    "back to normalized exact-match.\n\n---\n\n"
)


def _archive_bytes() -> tuple[bytes, dict]:
    entry = reasoning_gym.create_dataset("course_schedule", seed=231, size=1)[0]
    entry["metadata"]["private_marker"] = "synthetic-hidden-marker"
    files = {
        "task.toml": (
            (
                f'[metadata]\nfamily = "other"\nconverter = "nemotron_reasoning"\nmode = "reasoning-gym"\n'
                f'tasktrove_source = "{TASKTROVE_SOURCE}"\ntasktrove_path = "{TASKTROVE_PATH}"\n'
                'tags = ["reasoning", "reasoning-gym", "course-schedule", "nemotron"]\n'
            ).encode()
        ),
        "instruction.md": (PREAMBLE + entry["question"]).encode(),
        "tests/verifier.toml": b'mode = "reasoning-gym"\ndataset = "course_schedule"\n',
        "tests/entry.json": json.dumps(entry, separators=(",", ":")).encode(),
    }
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        for name, contents in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            archive.addfile(info, BytesIO(contents))
    return payload.getvalue(), entry


def _imported():
    data, entry = _archive_bytes()
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    return archive, import_task(archive), entry


def test_import_rewrites_prompt_preserves_tags_and_archive_digest():
    data, _ = _archive_bytes()
    archive = read_archive(data, TASKTROVE_SOURCE, TASKTROVE_PATH, RELEASE_URI, RELEASE_REVISION)
    specification = import_task(archive)

    prompt = specification.context.events[0].content
    assert specification.verifier.kind is VerifierKind.REASONING_GYM_ANSWER
    assert specification.answer_type is AnswerType.TEXT
    assert specification.source.importer_revision == IMPORTER_REVISION
    assert specification.tags == ("reasoning", "reasoning-gym", "course-schedule", "nemotron")
    assert "/app/answer.txt" not in prompt
    assert "fallback" not in prompt.lower()
    assert "Give your answer in plain text." in prompt
    assert archive.archive_sha256 == hashlib.sha256(data).hexdigest()
    assert specification.environment_requirements.capabilities == ()


async def test_imported_entry_matches_source_grader_and_harbor(tmp_path):
    archive, specification, entry = _imported()
    contract = ReasoningGymSpec(dataset="course_schedule")
    score_answer = reasoning_gym.get_score_answer_fn("course_schedule")
    correct = entry["answer"]
    wrong = "False" if correct == "True" else "True"
    convention = PlainText(id="plain")
    source_tests = tmp_path / "source-tests"
    source_tests.mkdir()
    (source_tests / "entry.json").write_text(json.dumps(entry))
    for label, candidate, expected in (("correct", correct, 1.0), ("wrong", wrong, 0.0)):
        (tmp_path / "answer.txt").write_text(candidate)
        source_result = source_grade(contract, source_tests, tmp_path)
        assert source_result.reward == expected
        assert score_answer(candidate, entry) == expected

        attempt = GradingAttempt(
            ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=candidate))),
            {},
            object(),
        )
        result = await grade_answer(specification, convention, attempt)
        assert (result.status, result.reward) == (Outcome.GRADED, expected)

        task = lower_to_harbor(specification, convention, HarborEnvironmentConfig(), tmp_path / f"task-{label}")
        trial = await run_replay_trial(task, {"role": "assistant", "content": candidate}, tmp_path / "trials", label)
        outcome = json.loads((tmp_path / f"trials/{label}/verifier/taskcompendium-result.json").read_text())
        assert trial.exception_info is None, trial.exception_info
        assert outcome == {"status": "graded", "reward": expected, "error": None}
        assert score_answer(candidate, entry) == outcome["reward"]

    assert archive.archive_sha256
    prompt = specification.context.events[0].content
    verifier_parameters = json.loads(specification.verifier.parameters_json)
    assert entry["metadata"]["private_marker"] not in prompt
    assert entry["metadata"]["private_marker"] in json.dumps(verifier_parameters)


async def test_reasoning_gym_scorer_failure_propagates(monkeypatch):
    _, specification, _ = _imported()
    convention = PlainText(id="plain")
    attempt = GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content="answer"))),
        {},
        object(),
    )

    def get_score_answer_fn(dataset):
        def score_answer(candidate, entry):
            raise RuntimeError("synthetic scorer failure")

        return score_answer

    monkeypatch.setattr(
        "taskcompendium.verifiers.reasoning_gym.import_module",
        lambda name: SimpleNamespace(get_score_answer_fn=get_score_answer_fn),
    )

    with pytest.raises(RuntimeError, match="synthetic scorer failure"):
        await grade_answer(specification, convention, attempt)


@pytest.mark.parametrize(
    "score,error",
    [(True, TypeError), (float("nan"), ValueError), (-0.1, ValueError), (1.1, ValueError)],
)
async def test_reasoning_gym_rejects_invalid_scorer_results(monkeypatch, score, error):
    _, specification, _ = _imported()
    convention = PlainText(id="plain")
    attempt = GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content="answer"))),
        {},
        object(),
    )
    monkeypatch.setattr(
        "taskcompendium.verifiers.reasoning_gym.import_module",
        lambda name: SimpleNamespace(get_score_answer_fn=lambda dataset: lambda candidate, entry: score),
    )

    with pytest.raises(error, match="Reasoning Gym scorer"):
        await grade_answer(specification, convention, attempt)
