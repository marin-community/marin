# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay approvals: the admin page and a GitHub 👍 both queue an idea, and an idea approved on both runs once."""

import datetime
import hashlib
import json

import pytest
from belay.app import (
    DEFAULT_SETTINGS,
    EventKind,
    IdeaReport,
    IntakeApproval,
    IntakeComment,
    RunExpectation,
    SessionFinish,
    SessionState,
    agent_caller_ok,
    approve_idea,
    approve_intake_comment,
    claim_events,
    create_api,
    decline_idea,
    expect_run,
    finish_session,
    migrate,
    note_overdue_runs,
    offer_pending_work,
    requeue_idea,
    sync_intake_comments,
)
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, event, text

POSTED = datetime.datetime(2026, 10, 26, 12, 0, tzinfo=datetime.UTC)
ROCKET = datetime.datetime(2026, 11, 2, 9, 0, tzinfo=datetime.UTC)


@pytest.fixture
def conn(database_url: str):
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS belay_test CASCADE; CREATE SCHEMA belay_test"))
        connection.execute(text("SET search_path TO belay_test"))
        migrate(connection)
        yield connection
        connection.execute(text("DROP SCHEMA belay_test CASCADE"))


def comment(*, approved_upstream: bool, approved_at: datetime.datetime | None = None) -> IntakeComment:
    return IntakeComment(
        comment_id="1",
        author="someone",
        body="Track: speedrun\nTry X.",
        url="https://github.com/marin-community/marin/issues/9514#issuecomment-1",
        posted_at=POSTED,
        approved_upstream=approved_upstream,
        approved_at=approved_at,
    )


def approved_events(conn) -> int:
    return conn.execute(text("SELECT count(*) FROM events WHERE kind = 'approved'")).scalar_one()


def idea_id(conn) -> int:
    return conn.execute(text("SELECT id FROM ideas WHERE comment_id = '1'")).scalar_one()


def test_page_then_github_approval_runs_once(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    assert approve_idea(conn, idea_id(conn), "maintainer")
    sync_intake_comments(conn, [comment(approved_upstream=True, approved_at=ROCKET)])
    assert approved_events(conn) == 1


def test_github_then_page_approval_runs_once(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=True, approved_at=ROCKET)])
    assert approved_events(conn) == 1
    assert not approve_idea(conn, idea_id(conn), "maintainer")
    sync_intake_comments(conn, [comment(approved_upstream=True, approved_at=ROCKET)])
    assert approved_events(conn) == 1


def test_approval_before_cutover_waits_for_the_page(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=True)])
    assert approved_events(conn) == 0
    assert approve_idea(conn, idea_id(conn), "maintainer")
    assert approved_events(conn) == 1


def test_escalated_idea_is_not_offered_again(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    approve_idea(conn, idea_id(conn), "maintainer")
    conn.execute(text("UPDATE events SET session_id = 's1'"))  # a session took the approval
    conn.execute(text("UPDATE events SET at = now() - interval '2 hours'"))
    conn.execute(text("UPDATE ideas SET note = 'Escalated: needs a maintainer'"))
    assert offer_pending_work(conn) == 0
    conn.execute(text("UPDATE ideas SET note = 'Maintainer ClassicLarry: retest on current recipe'"))
    assert offer_pending_work(conn) == 1


def test_comment_already_in_the_registry_is_not_queued(conn) -> None:
    ideas = [{"xid": "X0220", "intake": {"url": "https://github.com/marin-community/marin/issues/9514#issuecomment-1"}}]
    conn.execute(
        text("INSERT INTO registry (id, revision, data) VALUES (1, 'r', CAST(:d AS JSONB))"),
        {"d": json.dumps({"ideas": ideas})},
    )
    sync_intake_comments(conn, [comment(approved_upstream=True, approved_at=ROCKET)])
    assert approved_events(conn) == 0
    assert not approve_idea(conn, idea_id(conn), "maintainer")
    row = conn.execute(text("SELECT state, registry_id FROM ideas WHERE comment_id = '1'")).one()
    assert tuple(row) == ("recorded", "X0220")


def test_agent_endpoints_take_the_session_identity_or_an_approver() -> None:
    settings = {**DEFAULT_SETTINGS, "service_token_sha256": None}
    assert agent_caller_ok(settings, "loom-vm@hai-gcp-models.iam.gserviceaccount.com", None)
    assert agent_caller_ok(settings, "larry.dial@openathena.ai", None)
    assert not agent_caller_ok(settings, "someone@example.com", None)
    assert not agent_caller_ok(settings, None, "any token")


def activity_rows(conn) -> int:
    return conn.execute(text("SELECT count(*) FROM activity")).scalar_one()


def test_decline_refuses_an_unknown_or_running_idea_and_writes_nothing(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    before = activity_rows(conn)
    with pytest.raises(HTTPException) as unknown:
        decline_idea(conn, 999, "ClassicLarry", "no")
    conn.execute(text("UPDATE ideas SET state = 'running'"))
    with pytest.raises(HTTPException) as running:
        decline_idea(conn, idea_id(conn), "ClassicLarry", "no")
    assert (unknown.value.status_code, running.value.status_code) == (404, 409)
    assert activity_rows(conn) == before
    conn.execute(text("UPDATE ideas SET state = 'pending'"))
    assert decline_idea(conn, idea_id(conn), "ClassicLarry", "diagnostics only") == {"state": "declined"}


def test_a_retried_failed_session_report_requeues_once(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    approve_idea(conn, idea_id(conn), "maintainer")
    conn.execute(text("INSERT INTO sessions (id, state) VALUES ('s1', 'running')"))
    conn.execute(text("UPDATE events SET session_id = 's1'"))
    failed = SessionFinish(state=SessionState.FAILED, summary="crashed")
    assert finish_session(conn, "s1", failed) == {"state": "failed"}
    assert finish_session(conn, "s1", failed) == {"state": "failed"}
    assert conn.execute(text("SELECT count(*) FROM events WHERE session_id IS NULL")).scalar_one() == 1


def test_a_retried_claim_gets_its_events_back(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    approve_idea(conn, idea_id(conn), "maintainer")
    conn.execute(text("UPDATE settings SET value = 'true' WHERE key = 'dispatch_enabled'"))
    first = claim_events(conn, "s1")
    assert first["launch"] and len(first["events"]) == 1
    retried = claim_events(conn, "s1")
    assert retried["launch"] and [e["id"] for e in retried["events"]] == [e["id"] for e in first["events"]]


def test_a_finish_report_needs_a_terminal_state() -> None:
    with pytest.raises(ValidationError):
        SessionFinish(state=SessionState.RUNNING, summary="still going")


def test_a_retried_requeue_queues_once(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    approve_idea(conn, idea_id(conn), "maintainer")
    requeue_idea(conn, idea_id(conn), "ClassicLarry", "retest on current recipe")
    requeue_idea(conn, idea_id(conn), "ClassicLarry", "retest on current recipe")
    assert conn.execute(text("SELECT count(*) FROM events WHERE kind = 'work_pending'")).scalar_one() == 1


def enable_dispatch(conn) -> None:
    conn.execute(text("UPDATE settings SET value = 'true' WHERE key = 'dispatch_enabled'"))


def test_an_approval_records_its_node_grant(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    assert approve_idea(conn, idea_id(conn), "ClassicLarry", node_grant=12)
    assert conn.execute(text("SELECT node_grant FROM ideas")).scalar_one() == 12


def test_a_session_report_cannot_grant_nodes() -> None:
    with pytest.raises(ValidationError):
        IdeaReport.model_validate({"note": "x", "node_grant": 12})


def test_a_draining_session_takes_only_the_kinds_it_asks_for(conn) -> None:
    sync_intake_comments(conn, [comment(approved_upstream=False)])
    approve_idea(conn, idea_id(conn), "maintainer")
    enable_dispatch(conn)
    claim_events(conn, "s1")
    conn.execute(
        text(
            "INSERT INTO events (kind, ref, summary) VALUES "
            "('run_finished', 'belay-X0001-seed0', 'finished'), ('approved', '99', 'another idea')"
        )
    )
    held = claim_events(conn, "s1", [EventKind.RUN_FINISHED])
    assert [e["kind"] for e in held["events"]] == ["approved", "run_finished"]
    assert conn.execute(text("SELECT kind FROM events WHERE session_id IS NULL")).scalars().all() == ["approved"]


def test_an_overdue_run_is_noted_once(conn) -> None:
    late = datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=2)
    expect_run(conn, RunExpectation(name="belay-X0001-seed0", expected_at=late))
    assert note_overdue_runs(conn) == 1
    assert note_overdue_runs(conn) == 0
    assert conn.execute(text("SELECT kind FROM events")).scalars().all() == ["run_overdue"]


def test_an_intake_comment_the_registry_holds_cannot_be_approved_from_the_admin_page(conn) -> None:
    url = "https://github.com/marin-community/marin/issues/9514#issuecomment-1"
    ideas = [{"xid": "X0220", "intake": {"url": url}}]
    conn.execute(
        text("INSERT INTO registry (id, revision, data) VALUES (1, 'r', CAST(:d AS JSONB))"),
        {"d": json.dumps({"ideas": ideas})},
    )
    held = IntakeApproval(comment_id="1", url=url, author="a", body="b", posted_at=POSTED)
    with pytest.raises(HTTPException) as refused:
        approve_intake_comment(conn, held, "ClassicLarry")
    assert refused.value.status_code == 409
    fresh = IntakeApproval(comment_id="2", url=url[:-1] + "2", author="a", body="Try Y.", posted_at=POSTED, nodes=4)
    assert approve_intake_comment(conn, fresh, "ClassicLarry")["state"] == "approved"
    assert approved_events(conn) == 1


class _Services:
    """The kernel's services for a test app on the ``belay_api_test`` schema."""

    name = "belay"
    data_url = ""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url)

        @event.listens_for(self._engine, "connect")
        def _schema(connection, _record):
            # Outside a transaction, so a rollback when the pool takes the connection back does not undo it.
            connection.autocommit = True
            cursor = connection.cursor()
            cursor.execute("SET search_path TO belay_api_test")
            cursor.close()
            connection.autocommit = False

    def engine(self):
        return self._engine


def test_only_an_approver_can_change_the_node_cap_or_grant_nodes(database_url: str) -> None:
    token = "session-token"
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS belay_api_test CASCADE; CREATE SCHEMA belay_api_test"))
        connection.execute(text("SET search_path TO belay_api_test"))
        migrate(connection)
        connection.execute(
            text("UPDATE settings SET value = CAST(:v AS JSONB) WHERE key = 'service_token_sha256'"),
            {"v": json.dumps(hashlib.sha256(token.encode()).hexdigest())},
        )
    client = TestClient(create_api(_Services(database_url)).app)
    headers = {"X-Belay-Token": token}
    assert client.get("/agent/state", headers=headers).status_code == 200
    assert client.post("/admin/settings", json={"node_cap": 20}, headers=headers).status_code == 401
    assert client.post("/admin/ideas/1/approve", json={"nodes": 20}, headers=headers).status_code == 401
    with engine.begin() as connection:
        connection.execute(text("DROP SCHEMA belay_api_test CASCADE"))


def test_a_finished_session_records_its_token_usage(conn) -> None:
    conn.execute(text("INSERT INTO sessions (id, state) VALUES ('s1', 'running')"))
    report = SessionFinish(
        state=SessionState.SUCCEEDED,
        summary="ok",
        turns=12,
        input_tokens=4_000,
        output_tokens=900,
        cache_read_tokens=80_000,
    )
    finish_session(conn, "s1", report)
    row = conn.execute(text("SELECT turns, input_tokens, output_tokens, cache_read_tokens FROM sessions")).one()
    assert tuple(row) == (12, 4_000, 900, 80_000)
