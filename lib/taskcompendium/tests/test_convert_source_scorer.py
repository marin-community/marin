# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A source scorer package calls the scorer installed in its grader image and returns the scorer's reward."""

import hashlib
from pathlib import Path

import pytest
from shellbox.machine import Backend, DockerImage, MachineSpec

from taskcompendium.convert.conversation import conversation_task
from taskcompendium.convert.environment import grading_environment
from taskcompendium.convert.source_scorer import source_scorer_package
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, Source, TaskSpec, TextMessage
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.task_grading import grade_task

from .test_runtime import LocalGradingMachines

IMAGE = "scorer@sha256:" + "c" * 64
ROW = RawRow("task", Source(dataset="fixture", revision="1", row="0", importer_revision="1"), {})
SCORER = b"""
def compute_score(answer, words):
    found = [word in answer.split() for word in words]
    return {"score": sum(found) / len(found), "detail": {"found": found}}
"""


@pytest.fixture
def scorer(tmp_path) -> Path:
    # The scorer stands in for a file installed in the grader image at a pinned digest.
    path = tmp_path / "image/scorer.py"
    path.parent.mkdir()
    path.write_bytes(SCORER)
    return path


def scorer_task(scorer: Path) -> TaskSpec:
    package = source_scorer_package(
        invocation={
            "function": "fixture_scorer:compute_score",
            "source_path": str(scorer),
            "source_sha256": hashlib.sha256(SCORER).hexdigest(),
            "args": ["answer", "contract.words"],
            "reward_key": "score",
        },
        config={"contract": {"words": ["red", "blue"]}},
        environment=grading_environment(IMAGE, (Backend.DOCKER,)),
        timeout=30,
    )
    return conversation_task(
        ROW, events=(TextMessage(role="user", content="Name two colors of the flag."),), package=package
    )


def grade(task: TaskSpec, answer: str, root: Path) -> GradeResult:
    trace = ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    return grade_task(
        task,
        GradingAttempt(trace),
        machine_factory=LocalGradingMachines(root),
        machine_spec=MachineSpec(DockerImage(IMAGE)),
    )


@pytest.mark.parametrize("answer,expected", [("red blue", 1.0), ("red green", 0.5), ("green", 0.0)])
def test_scorer_receives_the_reply_and_contract_and_its_reward_is_the_grade(tmp_path, scorer, answer, expected):
    result = grade(scorer_task(scorer), answer, tmp_path / "machine")
    assert (result.status, result.reward) == (Outcome.GRADED, expected), result.error


def test_scorer_that_differs_from_the_pinned_digest_never_produces_a_reward(tmp_path, scorer):
    task = scorer_task(scorer)
    scorer.write_bytes(SCORER.replace(b"sum(found)", b"len(found)"))
    result = grade(task, "green", tmp_path / "machine")
    # An image whose scorer changed is an environment fault, not a wrong answer.
    assert (result.status, result.reward) == (Outcome.INFRA_ERROR, None)
