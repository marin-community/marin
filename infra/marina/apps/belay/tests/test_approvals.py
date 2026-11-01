# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Belay approvals: the admin page and a GitHub 👍 both queue an idea, and an idea approved on both runs once."""

import datetime
import json

import pytest
from belay.app import IntakeComment, approve_idea, migrate, offer_pending_work, sync_intake_comments
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
