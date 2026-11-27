# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from datetime import UTC, datetime

import httpx

from infra.marina.apps.review_queue import app


def test_author_reply_does_not_assign_the_author_to_their_own_pull_request():
    # GitHub can store a reply to review feedback as a COMMENTED review.
    timeline = [
        {
            "event": "review_requested",
            "requested_reviewer": {"login": "Reviewer"},
            "created_at": "2026-11-24T15:00:00Z",
        },
        {
            "event": "reviewed",
            "state": "changes_requested",
            "user": {"type": "User", "login": "Reviewer"},
            "submitted_at": "2026-11-24T17:00:00Z",
        },
        {
            "event": "reviewed",
            "state": "commented",
            "user": {"type": "User", "login": "Author"},
            "submitted_at": "2026-11-24T17:05:00Z",
        },
    ]

    assert [(fact["login"], fact["response_hours"]) for fact in app.review_facts(timeline, "author")] == [
        ("reviewer", 2.0)
    ]


def test_history_discovers_later_pages_and_keeps_unmodified_prs(monkeypatch):
    cutoff = datetime(2026, 11, 19, tzinfo=UTC)

    def pull(number, day):
        return {
            "number": number,
            "title": f"PR {number}",
            "html_url": f"https://github.com/marin-community/marin/pull/{number}",
            "created_at": "2026-11-18T09:00:00Z",
            "updated_at": f"2026-11-{day}T01:00:00Z",
            "draft": False,
            "requested_reviewers": [],
            "requested_teams": [],
        }

    pages = [[pull(1, 26)], [pull(2, 24), pull(3, 18)]]
    visited = []

    async def github_get(_client, path, **params):
        assert path == "pulls" and params["state"] == "all" and params["sort"] == "updated"
        page = params["page"]
        visited.append(page)
        links = {"Link": '<https://api.github.com/pulls?page=2>; rel="next"'} if page == 1 else {}
        return httpx.Response(200, headers=links, json=pages[page - 1])

    monkeypatch.setattr(app, "github_get", github_get)
    initial = asyncio.run(app.recent_prs(None, [], cutoff, cutoff))
    assert [pr["number"] for pr in initial] == [1, 2]
    assert visited == [1, 2]

    visited.clear()
    pages[0] = [pull(4, 26), pull(5, 25)]
    incremental = asyncio.run(app.recent_prs(None, initial, cutoff, datetime(2026, 11, 26, tzinfo=UTC)))
    assert [pr["number"] for pr in incremental] == [1, 4, 2]
    assert visited == [1]


def test_activity_distinguishes_other_humans_feedback_and_approval():
    def event(login, kind, state=None, user_type="User"):
        return {
            "event": kind,
            "state": state,
            "user": {"login": login, "type": user_type},
            "submitted_at": "2026-11-24T17:00:00Z",
        }

    timeline = [
        event("Author", "commented"),
        event("lint-bot", "reviewed", "approved", "Bot"),
        event("Commenter", "commented"),
        event("Approver", "reviewed", "approved"),
        event("Approver", "commented"),
        event("RaisedIssue", "reviewed", "approved"),
        event("RaisedIssue", "reviewed", "changes_requested"),
        event("PendingReviewer", "reviewed", "pending"),
    ]

    assert app.review_activity(timeline, "author") == {
        "commenter": "feedback",
        "approver": "approved",
        "raisedissue": "feedback",
    }
