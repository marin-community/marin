# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove puzzle choice and ordered-list imports with Harbor coverage."""

import json

import pytest

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.puzzle_choices_and_lists import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import (
    AnswerType,
    VerifierKind,
)
from taskcompendium.submission import AnswerCall, JsonAnswer, PlainText
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial
from .submission_helpers import answer_attempt
from .tasktrove_fixtures import FIXTURE_DATASET_URI, FIXTURE_REVISION, exact_archive


def _task():
    return import_task(exact_archive())


def test_exact_import_preserves_source_contract_and_intrinsic_puzzle():
    specification = _task()
    assert specification.answer_type is AnswerType.TEXT
    assert specification.verifier.kind is VerifierKind.EXACT_ANSWER
    assert json.loads(specification.verifier.parameters_json) == {
        "expected": ["green", "blue"],
        "ignore_case": True,
        "collapse_whitespace": True,
        "ordering": "ordered",
    }
    assert specification.tags == ("test", "puzzle", "ordered-list")
    assert specification.source.dataset == FIXTURE_DATASET_URI
    assert specification.source.revision == FIXTURE_REVISION
    assert specification.source.row == "laion__all-puzzles-v2:synthetic-exact-task"
    prompt = specification.context.events[0].content
    assert "reverse alphabetical order" in prompt
    assert "/app/answer.txt" not in prompt
    assert "Return only the requested answer." not in prompt


@pytest.mark.parametrize("convention", [PlainText(id="plain"), JsonAnswer(id="json"), AnswerCall(id="call")])
@pytest.mark.parametrize("response, reward", [("GREEN,  blue", 1.0), ("blue, green", 0.0)])
async def test_exact_import_uses_selected_submission_convention(convention, response, reward):
    specification = _task()
    result = await grade_answer(specification, convention, answer_attempt(specification, convention, response))
    assert (result.status, result.reward) == (Outcome.GRADED, reward)


@pytest.mark.parametrize("response, reward", [("green, blue", 1.0), ("blue, green", 0.0)])
async def test_exact_import_harbor_replay_returns_correct_rewards(tmp_path, response, reward):
    task_dir = lower_to_harbor(
        _task(),
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "harbor-task",
    )
    trial_name = "valid-exact-list" if reward else "wrong-exact-list"
    harbor_result = await run_replay_trial(
        task_dir,
        {"role": "assistant", "content": response},
        tmp_path / "trials",
        trial_name,
    )
    outcome = json.loads((tmp_path / f"trials/{trial_name}/verifier/taskcompendium-result.json").read_text())
    assert harbor_result.exception_info is None, harbor_result.exception_info
    assert outcome == {"status": "graded", "reward": reward, "error": None}
