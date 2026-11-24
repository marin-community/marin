# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime

from scripts.ops import pr_review_reminders as reminders


def test_monday_digest_keeps_only_due_reviewers_and_active_internal_prs(monkeypatch):
    friday = "2026-11-20T18:00:00Z"
    thursday = "2026-11-19T23:00:00Z"
    monday = datetime.fromisoformat("2026-11-23T18:00:00Z")
    pull = dict(
        number=10,
        title="Fix <queue> & wakeups",
        draft=False,
        created_at=thursday,
        updated_at=friday,
        author="member",
        author_type="User",
        reviewers=["early", "requested-again", "commenter", "reviewer", "inline", "committer"],
    )
    pulls = [
        pull,
        dict(pull, number=11, draft=True),
        dict(pull, number=12, author="external"),
        dict(pull, number=13, created_at=friday),
        dict(pull, number=14, reviewers=[]),
        dict(pull, number=15, updated_at="2026-11-16T17:59:59Z"),
    ]

    def event(kind, timestamp, **values):
        fields = dict(event=kind, created_at=timestamp, submitted_at=None, updated_at=None, requested=None, user=None)
        return dict(fields, actor=None, **values)

    timeline = [
        event("review_requested", thursday, requested="early"),
        event("commented", "2026-11-20T16:00:00Z", user="early"),
        event("review_requested", friday, requested="requested-again"),
        event("commented", friday, user="commenter"),
        event("reviewed", None, submitted_at=friday, user="reviewer"),
    ]

    def read(endpoint, fields):
        if endpoint.startswith("pulls?"):
            return pulls
        return {
            "issues/10/timeline": timeline,
            "pulls/10/comments": [{"user": "inline", "updated_at": friday}],
            "pulls/10/commits": [{"user": "committer", "date": friday}],
        }[endpoint]

    monkeypatch.setattr(reminders, "records", read)
    monkeypatch.setattr(reminders, "has_write_access", lambda login: login == "member")
    (line,) = reminders.reminder_lines(monday)
    assert "<https://github.com/marin-community/marin/pull/10|#10>" in line
    assert "Fix &lt;queue&gt; &amp; wakeups" in line
    assert "Updated 3d ago" in line
    assert "early (3d 2h)" in line
    assert all(name not in line for name in pull["reviewers"] if name != "early")


def test_digest_splits_at_pr_boundaries_only_when_needed():
    lines = [f"<https://github.com/marin-community/marin/pull/{number}|#{number}>  reviewer" for number in range(100)]
    parts = reminders.messages(lines)
    assert len(parts) == 2
    assert all(len(part.encode()) <= 4000 for part in parts)
    assert [line for part in parts for line in part.splitlines()[1:]] == lines
    assert reminders.messages([]) == []
