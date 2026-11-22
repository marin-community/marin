# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field

import pytest
from finelog.client import FlushResult, LogClient
from finelog.embedded import require_embedded_server

import rolloutengine.finelog_observer as finelog_observer
from rolloutengine.engine import RolloutEngine
from rolloutengine.finelog_observer import (
    EVENTS_NAMESPACE,
    MESSAGES_NAMESPACE,
    ContentKind,
    FinelogObserver,
    FinelogRolloutLog,
    RolloutScope,
)
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import Prompt, RolloutInterrupted, ToolCallCompleted
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_spec

SCOPE = RolloutScope(run_id="experiment", run_attempt_id="a1", phase="eval", step=3)
ROLLOUT_ID = "experiment/eval/3/arithmetic_0"
SYSTEM = {"role": "system", "content": "Answer with a number."}
QUESTION = {"role": "user", "content": "What is six plus six? Antwort: zw\u00f6lf."}
OPTIONS = {"temperature": 0.0}
CHECK = {"role": "user", "content": "Check it."}


@dataclass
class FakeTable:
    rows: list = field(default_factory=list)

    def write(self, rows) -> None:
        self.rows.extend(rows)

    def flush(self, timeout=None) -> FlushResult:
        return FlushResult.SUCCEEDED


@dataclass
class FakeClient:
    tables: dict[str, FakeTable] = field(default_factory=dict)

    def get_table(self, namespace, schema, **bounds) -> FakeTable:
        return self.tables.setdefault(namespace, FakeTable())


async def run_two_turns(log: FinelogRolloutLog, *, content_fraction: float, run_id: str = ROLLOUT_ID, options=None):
    """Run a task attempt of two model turns, with one tool call after the first turn."""
    first = Prompt((SYSTEM, QUESTION), OPTIONS if options is None else options)

    async def agent(context, complete) -> None:
        turn = await complete(first)
        context.events.emit(ToolCallCompleted("call-1", "calculator", 2, False, 0.0))
        await complete(Prompt((*first.messages, turn.message, CHECK), first.options))

    observer = FinelogObserver(log, SCOPE, content_fraction=content_fraction)
    model = ReplayModel([{"role": "assistant", "content": "Let me think."}, {"role": "assistant", "content": "12"}])
    return await RolloutEngine({}, {"agent": AgentHarness(agent)}, observer=observer).run_task(
        rollout_spec(arithmetic_task(), harness="agent"), model.complete, run_id=run_id
    )


@pytest.fixture
def finelog_client(tmp_path):
    server = require_embedded_server()(log_dir=str(tmp_path / "finelog"))
    client = LogClient.connect(server.address)
    try:
        yield client
    finally:
        client.close()
        server.stop()


async def test_rows_round_trip_through_a_finelog_server(finelog_client):
    log = FinelogRolloutLog(finelog_client)
    started_ms = int(time.time() * 1000)
    record = await run_two_turns(log, content_fraction=1.0)
    finished_ms = int(time.time() * 1000) + 1
    assert await asyncio.to_thread(log.flush, 30) == (FlushResult.SUCCEEDED, FlushResult.SUCCEEDED)

    events = finelog_client.query(f'SELECT * FROM "{EVENTS_NAMESPACE}" ORDER BY event_index').to_pylist()
    messages = finelog_client.query(
        f'SELECT * FROM "{MESSAGES_NAMESPACE}" ORDER BY event_index, message_index'
    ).to_pylist()

    assert [(row["event_index"], row["event_type"]) for row in events] == list(
        enumerate(
            [
                "attempt_started",
                "conversation_started",
                "model_turn_completed",
                "tool_call_completed",
                "transition_completed",
                "model_turn_completed",
                "transition_completed",
                "graded",
                "attempt_ended",
            ]
        )
    )
    scope = ("experiment", "a1", "eval", 3, "arithmetic", ROLLOUT_ID)
    columns = ("run_id", "run_attempt_id", "phase", "step", "task_id", "rollout_id")
    assert {tuple(row[column] for column in columns) for row in [*events, *messages]} == {scope}
    assert all(started_ms <= row["timestamp_ms"] <= finished_ms for row in [*events, *messages])

    started, tool, graded, ended = events[0], events[3], events[7], events[8]
    assert (started["harness"], started["trace_mode"], started["dataset"]) == ("agent", "exact_tokens", "fixture")
    assert (tool["tool_call_id"], tool["tool_name"], tool["observation_bytes"], tool["done"]) == (
        "call-1",
        "calculator",
        2,
        False,
    )
    assert (graded["grade_status"], graded["reward"]) == ("graded", 1.0)
    assert (ended["stop_reason"], ended["failed_operation"], ended["turns"], ended["total_tokens"]) == (
        "stop",
        None,
        2,
        9,
    )

    # The first request has two messages and its options; each later event adds only its new message.
    assert [(row["event_index"], row["message_index"], row["kind"]) for row in messages] == [
        (1, 0, ContentKind.MESSAGE),
        (1, 1, ContentKind.MESSAGE),
        (1, 2, ContentKind.OPTIONS),
        (2, 0, ContentKind.MESSAGE),
        (4, 0, ContentKind.MESSAGE),
        (5, 0, ContentKind.MESSAGE),
    ]
    assert {row["event_index"]: row["content_rows"] for row in events if row["content_captured"]} == {
        1: 3,
        2: 1,
        4: 1,
        5: 1,
    }
    transcript = tuple(json.loads(row["content"]) for row in messages if row["kind"] == ContentKind.MESSAGE)
    assert transcript == record.rollout.conversations[-1].messages
    assert json.loads(messages[2]["content"]) == OPTIONS
    assert [row["role"] for row in messages] == ["system", "user", None, "assistant", "user", "assistant"]


async def test_an_interrupted_task_attempt_records_the_failed_operation():
    async def model(request):
        raise ConnectionError("inference server unavailable")

    async def agent(_context, complete) -> None:
        await complete(Prompt((QUESTION,), OPTIONS))

    client = FakeClient()
    observer = FinelogObserver(FinelogRolloutLog(client), SCOPE, content_fraction=1.0)
    with pytest.raises(RolloutInterrupted):
        await RolloutEngine({}, {"agent": AgentHarness(agent)}, observer=observer).run_task(
            rollout_spec(arithmetic_task(), harness="agent"), model, run_id=ROLLOUT_ID
        )

    ended = client.tables[EVENTS_NAMESPACE].rows[-1]
    assert (ended.event_type, ended.stop_reason, ended.failed_operation, ended.turns) == (
        "attempt_ended",
        None,
        "model",
        0,
    )
    assert ended.error == repr(ConnectionError("inference server unavailable"))


async def test_content_fraction_captures_whole_task_attempts():
    client = FakeClient()
    log = FinelogRolloutLog(client)
    for number in range(40):
        await run_two_turns(log, content_fraction=0.5, run_id=f"experiment/eval/3/arithmetic_{number}")
    events, messages = client.tables[EVENTS_NAMESPACE].rows, client.tables[MESSAGES_NAMESPACE].rows

    message_rows = Counter(row.rollout_id for row in messages)
    assert set(message_rows) == {row.rollout_id for row in events if row.content_captured}
    assert set(message_rows.values()) == {6}
    assert 10 <= len(message_rows) <= 30
    # A task attempt without captured content still reports the size of its content.
    unwritten = [row for row in events if row.content_captured is False]
    assert len({row.rollout_id for row in unwritten}) == 40 - len(message_rows)
    assert {row.event_type for row in unwritten} == {
        "conversation_started",
        "model_turn_completed",
        "transition_completed",
    }
    assert all(row.content_rows and row.content_chars for row in unwritten)


async def test_long_message_is_cut_and_marked(monkeypatch):
    monkeypatch.setattr(finelog_observer, "MAX_MESSAGE_CHARS", 20)
    client = FakeClient()
    await run_two_turns(FinelogRolloutLog(client), content_fraction=1.0)

    question = next(row for row in client.tables[MESSAGES_NAMESPACE].rows if row.role == "user")
    assert (len(question.content), question.truncated) == (20, True)
    assert question.content_chars == len(json.dumps(QUESTION))


async def test_a_value_that_json_cannot_hold_does_not_stop_the_events():
    client = FakeClient()
    await run_two_turns(FinelogRolloutLog(client), content_fraction=1.0, options={"stop": {"END"}})

    # An observer error ends the delivery for the task attempt, so the events after the first request would be lost.
    assert [row.event_index for row in client.tables[EVENTS_NAMESPACE].rows] == list(range(9))
    [options] = [row for row in client.tables[MESSAGES_NAMESPACE].rows if row.kind == ContentKind.OPTIONS]
    assert json.loads(options.content) == {"stop": "{'END'}"}
