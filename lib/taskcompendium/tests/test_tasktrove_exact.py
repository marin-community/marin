# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove exact imports with direct-chat Harbor coverage."""

import json
from dataclasses import replace

import pytest
from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import ExactSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.exact import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial
from .tasktrove_fixtures import FIXTURE_DATASET_URI, FIXTURE_REVISION, exact_archive


def _archive():
    return exact_archive()


def _attempt(specification, response: str):
    return GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=response))),
        {},
    )


def _source_task():
    archive = _archive()
    specification = import_task(archive)
    source_contract = parse_spec(archive.files["tests/verifier.toml"].decode())
    assert isinstance(source_contract, ExactSpec)
    return specification, source_contract


def test_exact_import_preserves_tags_and_removes_file_harness():
    specification, source_contract = _source_task()
    assert specification.answer_type is AnswerType.TEXT
    assert specification.verifier.kind is VerifierKind.EXACT_ANSWER
    assert specification.tags == ("test", "puzzle", "ordered-list")
    assert specification.source.dataset == FIXTURE_DATASET_URI
    assert specification.source.revision == FIXTURE_REVISION
    assert specification.source.row == "laion__all-puzzles-v2:synthetic-exact-task"
    instruction = render_instruction(specification, PlainText(id="plain"))
    assert "/app/answer.txt" not in instruction
    assert "verifier reads" not in instruction.lower()
    assert ", ".join(source_contract.expected) not in instruction


@pytest.mark.parametrize(
    "contract,answers",
    [
        pytest.param(
            ExactSpec(expected=("green", "blue")),
            (("GREEN,  blue", 1.0), ("green\nblue", 1.0), ("blue, green", 0.0), ("green", 0.0)),
            id="ordered-list",
        ),
        pytest.param(
            ExactSpec(expected=("green", "blue", "green"), ordered=False),
            (("blue,green,green", 1.0), ("green,blue", 0.0), ("blue,blue,green", 0.0)),
            id="unordered-duplicates",
        ),
        pytest.param(
            ExactSpec(expected=("green", "blue", "green")),
            (("green,blue,green", 1.0), ("blue,green,green", 0.0)),
            id="ordered-duplicates",
        ),
        pytest.param(
            ExactSpec(expected=("New York", "blue"), ignore_case=False, ignore_whitespace=False),
            ((" New York, blue ", 1.0), ("New  York,blue", 0.0), ("new York,blue", 0.0)),
            id="strict-normalization",
        ),
        pytest.param(
            ExactSpec(expected=("New York", "blue")),
            ((" NEW  York, BLUE ", 1.0), (",New York,,blue,", 1.0)),
            id="normalized-list",
        ),
        pytest.param(
            ExactSpec(expected=("green", "blue")),
            (
                (r"Answer: \boxed{green, blue}", 1.0),
                (r"\boxed{green, blue} then \boxed{blue, green}", 0.0),
                (r"\boxed{green, blue", 0.0),
            ),
            id="boxed-list",
        ),
        pytest.param(
            ExactSpec(expected=("Straße Park",)),
            ((" STRASSE  Park ", 1.0), (r"Answer: \boxed{Straße Park}", 1.0), ("Straße", 0.0)),
            id="scalar",
        ),
        pytest.param(
            ExactSpec(expected=("green, blue",)),
            (("green, blue", 1.0), ("green\nblue", 0.0), ("blue, green", 0.0)),
            id="scalar-with-comma",
        ),
        pytest.param(
            ExactSpec(expected=(r"\boxed{wrong}",)),
            ((r"\boxed{wrong}", 1.0), ("wrong", 0.0)),
            id="whole-answer-fallback",
        ),
    ],
)
async def test_exact_import_matches_source_verifier(tmp_path, contract, answers):
    archive = _archive()
    verifier_toml = (
        'mode = "exact"\n'
        f"expected = {json.dumps(contract.expected)}\n"
        f"ignore_case = {str(contract.ignore_case).lower()}\n"
        f"ignore_whitespace = {str(contract.ignore_whitespace).lower()}\n"
        f"ordered = {str(contract.ordered).lower()}\n"
    )
    archive = replace(archive, files={**archive.files, "tests/verifier.toml": verifier_toml.encode()})
    specification = import_task(archive)
    answer_path = tmp_path / "answer.txt"
    source_contract = replace(contract, output=str(answer_path))
    for answer, reward in answers:
        answer_path.write_text(answer)
        source_result = source_grade(source_contract, tmp_path, tmp_path)
        imported_result = await grade_answer(specification, PlainText(id="plain"), _attempt(specification, answer))
        assert source_result.reward == reward
        assert (imported_result.status, imported_result.reward) == (Outcome.GRADED, reward)


async def test_exact_import_harbor_replay_returns_correct_rewards(tmp_path):
    specification, source_contract = _source_task()
    correct = ", ".join(source_contract.expected)
    incorrect_order = ", ".join(reversed(source_contract.expected))
    task_dir = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "harbor-task",
    )
    for response, name, reward in ((correct, "valid-exact-list", 1.0), (incorrect_order, "wrong-exact-list", 0.0)):
        harbor_result = await run_replay_trial(
            task_dir,
            {"role": "assistant", "content": response},
            tmp_path / "trials",
            name,
        )
        outcome = json.loads((tmp_path / f"trials/{name}/verifier/taskcompendium-result.json").read_text())
        assert harbor_result.exception_info is None, harbor_result.exception_info
        assert outcome == {"status": "graded", "reward": reward, "error": None}
