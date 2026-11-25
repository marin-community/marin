# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import time
from collections import Counter
from dataclasses import dataclass, field, replace

import pytest
from finelog.client import FlushResult, LogClient
from finelog.embedded import require_embedded_server
from taskcompendium.grading_result import Outcome

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
from rolloutengine.models import (
    ConversationStarted,
    Graded,
    Prompt,
    RolloutEvent,
    RolloutInterrupted,
    TokenUsage,
    ToolCallCompleted,
)
from rolloutengine.test_support import ReplayModel, arithmetic_task, rollout_spec

SCOPE = RolloutScope(run_id="experiment", run_attempt_id="a1", phase="eval", step=3)
ROLLOUT_ID = "experiment/eval/3/arithmetic_0"
SYSTEM = {"role": "system", "content": "Answer with a number."}
QUESTION = {"role": "user", "content": "What is six plus six? Antwort: zw\u00f6lf."}
OPTIONS = {"temperature": 0.0}
CALL = {"id": "call-1", "type": "function", "function": {"name": "calculator", "arguments": '{"a": 6, "b": 6}'}}
OBSERVATION = {"role": "tool", "tool_call_id": "call-1", "content": "12"}


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


async def run_two_turns(
    log: FinelogRolloutLog,
    *,
    content_fraction: float,
    run_id: str = ROLLOUT_ID,
    options=None,
    observation=OBSERVATION,
    tool_call=ToolCallCompleted("call-1", "calculator", 2, False, 0.0),
):
    """Run a task attempt of two model turns, with one tool call after the first turn.

    The second turn reports the endpoint's own token counts in ``usage``.
    """
    first = Prompt((SYSTEM, QUESTION), OPTIONS if options is None else options)

    async def agent(context, complete) -> None:
        turn = await complete(first)
        context.events.emit(tool_call)
        await complete(Prompt((*first.messages, turn.message, observation), first.options))

    replay = ReplayModel(
        [{"role": "assistant", "content": "", "tool_calls": [CALL]}, {"role": "assistant", "content": "12"}]
    )

    async def model(request):
        turn = await replay.complete(request)
        return turn if len(replay.requests) == 1 else replace(turn, usage=TokenUsage(30, 4))

    observer = FinelogObserver(log, SCOPE, content_fraction=content_fraction)
    return await RolloutEngine({}, {"agent": AgentHarness(agent)}, observer=observer).run_task(
        rollout_spec(arithmetic_task(), harness="agent"), model, run_id=run_id
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
    turns = [row for row in events if row["event_type"] == "model_turn_completed"]
    # The first turn counts its token IDs; the second has the endpoint's usage, which the row prefers.
    assert [(row["turn_index"], row["stop_reason"], row["prompt_tokens"], row["response_tokens"]) for row in turns] == [
        (0, "stop", 2, 1),
        (1, "stop", 30, 4),
    ]
    transitions = [row for row in events if row["event_type"] == "transition_completed"]
    assert [(row["turn_index"], row["done"]) for row in transitions] == [(0, False), (1, True)]
    assert (graded["grade_status"], graded["reward"]) == ("graded", 1.0)
    assert (ended["stop_reason"], ended["failed_operation"], ended["turns"], ended["total_tokens"]) == (
        "stop",
        None,
        2,
        37,
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
    assert [(row["role"], row["tool_call_id"]) for row in messages] == [
        ("system", None),
        ("user", None),
        (None, None),
        ("assistant", None),
        ("tool", "call-1"),
        ("assistant", None),
    ]


async def test_values_that_the_encoder_refuses_do_not_lose_the_other_rows(finelog_client):
    log = FinelogRolloutLog(finelog_client)
    # Nothing checks what a harness reports before its rows are queued: here numbers as tool call IDs, and a lone
    # surrogate in a role and in a tool name. The grader rejects the odd message only after that.
    odd = {"role": "tool\ud800", "tool_call_id": 7, "content": "12"}
    result = await run_two_turns(
        log,
        content_fraction=1.0,
        run_id="odd",
        observation=odd,
        tool_call=ToolCallCompleted(7, "calculator\ud800", 2, False, 0.0),
    )
    assert (result.grade.status, result.grade.reward) == (Outcome.SUBMISSION_FAILURE, 0.0)
    await run_two_turns(log, content_fraction=1.0)
    # A verifier's error detail reaches the grade unchecked.
    observer = FinelogObserver(log, SCOPE, content_fraction=1.0)
    for rollout_id, error in (("error-number", 7), ("error-surrogate", "bad\ud800")):
        await observer.on_event(RolloutEvent(rollout_id, "task", 0, 0.0, Graded(Outcome.GRADED, 0.0, error, None, 0.0)))
    # The rows of all task attempts go out in one request, which one refused value would lose.
    assert await asyncio.to_thread(log.flush, 30) == (FlushResult.SUCCEEDED, FlushResult.SUCCEEDED)

    calls = finelog_client.query(
        f'SELECT rollout_id, tool_call_id, tool_name FROM "{EVENTS_NAMESPACE}" WHERE event_index = 3 ORDER BY rollout_id'
    ).to_pylist()
    assert [tuple(row.values()) for row in calls] == [
        (ROLLOUT_ID, "call-1", "calculator"),
        ("odd", "7", "calculator\\ud800"),
    ]
    observations = finelog_client.query(
        f'SELECT rollout_id, role, tool_call_id FROM "{MESSAGES_NAMESPACE}" WHERE event_index = 4 ORDER BY rollout_id'
    ).to_pylist()
    assert [tuple(row.values()) for row in observations] == [(ROLLOUT_ID, "tool", "call-1"), ("odd", "tool\\ud800", "7")]
    errors = finelog_client.query(
        f"SELECT rollout_id, error FROM \"{EVENTS_NAMESPACE}\" WHERE rollout_id LIKE 'error-%' ORDER BY rollout_id"
    ).to_pylist()
    assert [tuple(row.values()) for row in errors] == [("error-number", "7"), ("error-surrogate", "bad\\ud800")]


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


async def test_samples_of_one_prompt_group_are_captured_independently():
    client = FakeClient()
    observer = FinelogObserver(FinelogRolloutLog(client), SCOPE, content_fraction=0.25)
    request = ConversationStarted((QUESTION,), OPTIONS, 0.0)
    # The rollout IDs of the samples of one prompt group differ only in the sample number.
    groups = [
        [f"run/a1/train/{group // 64}/task-{group:05d}_{sample}/0" for sample in range(8)] for group in range(1000)
    ]
    for rollout_id in (rollout_id for group in groups for rollout_id in group):
        await observer.on_event(RolloutEvent(rollout_id, "task", 0, 0.0, request))

    written = {row.rollout_id for row in client.tables[MESSAGES_NAMESPACE].rows}
    captured = [sum(rollout_id in written for rollout_id in group) for group in groups]
    # Independent draws leave about 10% of the groups with no captured sample, and almost none with all eight.
    assert 0.05 < captured.count(0) / len(groups) < 0.15
    assert captured.count(8) < 10


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
