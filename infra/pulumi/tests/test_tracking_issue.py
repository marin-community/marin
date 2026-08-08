# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from iac.github.tracking_issue import SyncAction, sync_tracking_issue

REPO = "marin-community/marin"
LABEL = "iam-drift"
MARKER = "<!-- marin-iam-drift -->"
FP = "deadbeef"
BODY = f"{MARKER}\n<!-- fingerprint:{FP} -->\nfindings"


class FakeGh:
    """Records `gh` argument lists and answers `issue list` from a scripted open-issue set."""

    def __init__(self, open_issues: list[dict]) -> None:
        self._open_issues = open_issues
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[:2] == ["issue", "list"]:
            return json.dumps(self._open_issues)
        return ""

    def commands(self) -> list[str]:
        return [args[1] for args in self.calls if args[0] == "issue"]


def test_first_drift_creates_a_labelled_issue():
    gh = FakeGh(open_issues=[])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=BODY,
        fingerprint=FP,
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.CREATED
    create = next(args for args in gh.calls if args[1] == "create")
    assert "--label" in create and LABEL in create and "agent-generated" in create
    assert BODY in create


def test_same_fingerprint_is_a_noop():
    gh = FakeGh(open_issues=[{"number": 5, "body": BODY}])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=BODY,
        fingerprint=FP,
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.UNCHANGED
    assert gh.commands() == ["list"]  # no edit/create/comment


def test_changed_fingerprint_edits_the_existing_issue():
    stale = f"{MARKER}\n<!-- fingerprint:0000 -->\nold"
    gh = FakeGh(open_issues=[{"number": 5, "body": stale}])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=BODY,
        fingerprint=FP,
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.UPDATED
    edit = next(args for args in gh.calls if args[1] == "edit")
    assert "5" in edit and BODY in edit
    # A changed set also comments so watchers are notified (a body edit alone is silent).
    assert gh.commands() == ["list", "edit", "comment"]


def test_clean_scan_comments_and_closes_open_issue():
    gh = FakeGh(open_issues=[{"number": 5, "body": BODY}])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=None,
        fingerprint="",
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.RESOLVED
    assert gh.commands() == ["list", "comment", "close"]


def test_clean_scan_with_no_open_issue_does_nothing():
    gh = FakeGh(open_issues=[])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=None,
        fingerprint="",
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.CLEAN
    assert gh.commands() == ["list"]


def test_marker_mismatch_is_treated_as_no_existing_issue():
    # A same-label issue that isn't the scan's sticky one (no marker) must not be adopted.
    gh = FakeGh(open_issues=[{"number": 9, "body": "unrelated issue"}])

    action = sync_tracking_issue(
        repo=REPO,
        label=LABEL,
        marker=MARKER,
        title="t",
        body=BODY,
        fingerprint=FP,
        changed_comment="🤖 changed",
        resolved_comment="🤖 clear",
        run=gh,
    )

    assert action is SyncAction.CREATED
