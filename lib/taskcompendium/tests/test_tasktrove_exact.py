# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove exact imports with direct-chat Harbor coverage."""

import json

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
        object(),
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
    assert specification.verifier.kind is VerifierKind.EXACT_LIST_ANSWER
    assert specification.tags == ("test", "puzzle", "ordered-list")
    assert specification.source.dataset == FIXTURE_DATASET_URI
    assert specification.source.revision == FIXTURE_REVISION
    assert specification.source.row == "laion__all-puzzles-v2:synthetic-exact-task"
    instruction = render_instruction(specification, PlainText(id="plain"))
    assert "/app/answer.txt" not in instruction
    assert "verifier reads" not in instruction.lower()
    assert ", ".join(source_contract.expected) not in instruction


async def test_exact_import_matches_source_verifier_on_ordered_lists(tmp_path):
    specification, source_contract = _source_task()
    correct = ", ".join(source_contract.expected)
    incorrect_order = ", ".join(reversed(source_contract.expected))
    answer_path = tmp_path / "answer.txt"
    contract = ExactSpec(
        expected=source_contract.expected,
        ignore_case=source_contract.ignore_case,
        ignore_whitespace=source_contract.ignore_whitespace,
        ordered=source_contract.ordered,
        output=str(answer_path),
    )
    for answer, reward in ((correct, 1.0), (incorrect_order, 0.0)):
        answer_path.write_text(answer)
        source_result = source_grade(contract, tmp_path, tmp_path)
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
