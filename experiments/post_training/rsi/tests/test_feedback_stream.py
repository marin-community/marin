# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The tasks that a synthesis session accepts reach training before the session ends."""

import pytest

pytest.importorskip("taskforge")

import asyncio
from types import SimpleNamespace

from rigging.filesystem.storage_path import StoragePath
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskforge.capability import CapabilityIdea

from experiments.post_training.rsi import feedback
from experiments.post_training.rsi.streams import from_iterable

IDEA = CapabilityIdea(
    capability_id="gap.strings.sort",
    subject_id="gap",
    subject_name="Gaps",
    catalog_version="test",
    capability={"id": "gap.strings.sort"},
    prerequisite_edges=(),
    capability_hash="0",
)


def task(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        context=ConversationInput(events=(TextMessage(role="user", content=task_id),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=NoGrader(reason="test"),
        source=Source(dataset="test", revision="1", row=task_id, importer_revision="1"),
    )


def config(tmp_path, held_out_share: float) -> feedback.SynthesisConfig:
    return feedback.SynthesisConfig(
        agentic=None,
        tasks_per_idea=2,
        instances_per_task=1,
        probes_per_task=1,
        concurrency=1,
        output=StoragePath(str(tmp_path)),
        held_out_idea_share=held_out_share,
    )


@pytest.fixture
def session(monkeypatch):
    """A session that accepts one task, then waits until ``finish`` is set before it ends."""
    finish = asyncio.Event()
    accepted = SimpleNamespace(task=task("accepted"))
    written = []

    async def generate_tasks(idea, n, agentic_config, on_accept):
        await on_accept(accepted)
        await finish.wait()
        return SimpleNamespace(idea=idea, tasks=(accepted,), failure=None)

    async def instances(item, synthesis_config):
        return [task(f"{item.task.id}-instance")], [task(f"{item.task.id}-probe")]

    monkeypatch.setattr(feedback.agentic, "generate_tasks", generate_tasks)
    monkeypatch.setattr(feedback, "_instances", instances)
    monkeypatch.setattr(feedback, "write_session", lambda output, result, probes, *, held_out: written.append(probes))
    return finish, written


def test_accepted_tasks_reach_training_before_the_session_ends(session, tmp_path):
    finish, written = session

    async def run() -> tuple[list[str], list[str]]:
        stream = feedback.generate_tasks(from_iterable([IDEA]), config(tmp_path, held_out_share=0.0))
        early = [(await anext(stream)).id, (await anext(stream)).id]
        finish.set()
        rest = [item.id async for item in stream]
        return early, rest

    early, rest = asyncio.run(run())
    assert early == [f"{IDEA.capability_id}.accepted", f"{IDEA.capability_id}.accepted-instance"]
    assert rest == []
    # The session's record still gets the probes of every accepted task.
    assert [[probe.id for probe in probes] for probes in written[0]] == [["accepted-probe"]]


def test_a_held_out_idea_gives_probes_but_no_training_tasks(session, tmp_path):
    finish, written = session
    finish.set()

    async def run() -> list[str]:
        return [item.id async for item in feedback.generate_tasks(from_iterable([IDEA]), config(tmp_path, 1.0))]

    assert asyncio.run(run()) == []
    assert [[probe.id for probe in probes] for probes in written[0]] == [["accepted-probe"]]


def test_ideas_are_ranked_by_the_failed_tasks_that_they_cite():
    def idea(name: str, evidence: list[str]) -> CapabilityIdea:
        return IDEA.model_copy(update={"capability_id": name, "capability": {"id": name, "evidence_tasks": evidence}})

    ideas = [idea("format", ["a"]), idea("sorting", ["b", "c", "d"]), idea("none", [])]
    assert [feedback.cited_failures(item) for item in ideas] == [1, 3, 0]
    assert [item.capability_id for item in sorted(ideas, key=feedback.cited_failures, reverse=True)] == [
        "sorting",
        "format",
        "none",
    ]
