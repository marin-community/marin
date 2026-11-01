# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay approvals: the admin page and a GitHub 👍 both queue an idea, and an idea approved on both runs once."""

import datetime
import json

import pytest
from belay.app import (
    DEFAULT_SETTINGS,
    IntakeComment,
    SessionFinish,
    SessionState,
    agent_caller_ok,
    approve_idea,
    claim_events,
    decline_idea,
    finish_session,
    migrate,
    offer_pending_work,
    requeue_idea,
    sync_intake_comments,
)
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, text

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
