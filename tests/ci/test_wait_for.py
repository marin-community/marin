# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest

from scripts.ci import wait_for

PR_URL = "https://github.com/marin-community/marin/pull/123"


def _pr_snapshot(
    *,
    state: str = "OPEN",
    mergeable: str = "MERGEABLE",
    review_decision: str | None = "REVIEW_REQUIRED",
    is_draft: bool = False,
) -> wait_for.PrSnapshot:
    return wait_for.PrSnapshot(
        state=state,
        mergeable=mergeable,
        review_decision=review_decision,
        is_draft=is_draft,
        url=PR_URL,
    )


def _pr_source(monkeypatch: pytest.MonkeyPatch, snapshots: Iterator[wait_for.PrSnapshot]) -> wait_for.PrSource:
    monkeypatch.setattr(wait_for, "gh_pr_snapshot", lambda _pr, _repo: next(snapshots))
    return wait_for.PrSource(wait_for.parse_spec("github.pr 123"), "marin-community/marin")


@pytest.mark.parametrize(
    ("before", "after", "reasons"),
    [
        (_pr_snapshot(), _pr_snapshot(state="MERGED"), ["merged"]),
        (_pr_snapshot(), _pr_snapshot(state="CLOSED"), ["closed"]),
        (_pr_snapshot(mergeable="UNKNOWN"), _pr_snapshot(mergeable="CONFLICTING"), ["conflicted"]),
        (_pr_snapshot(is_draft=True), _pr_snapshot(is_draft=False), ["ready_for_review"]),
        (
            _pr_snapshot(review_decision="REVIEW_REQUIRED"),
            _pr_snapshot(review_decision="APPROVED"),
            ["review_decision"],
        ),
        (
            _pr_snapshot(review_decision="APPROVED"),
            _pr_snapshot(review_decision="CHANGES_REQUESTED"),
            ["review_decision"],
        ),
        (
            _pr_snapshot(mergeable="UNKNOWN", review_decision="REVIEW_REQUIRED", is_draft=True),
            _pr_snapshot(mergeable="CONFLICTING", review_decision="APPROVED", is_draft=False),
            ["conflicted", "ready_for_review", "review_decision"],
        ),
    ],
    ids=["merged", "closed", "conflicted", "ready-for-review", "approved", "changes-requested", "simultaneous"],
)
def test_pr_source_actionable_transition_returns_before_and_after(
    monkeypatch: pytest.MonkeyPatch,
    before: wait_for.PrSnapshot,
    after: wait_for.PrSnapshot,
    reasons: list[str],
) -> None:
    source = _pr_source(monkeypatch, iter((before, after)))

    assert source.check() is None
    assert source.check() == {
        "reasons": reasons,
        "before": before.payload(),
        "after": after.payload(),
    }


def test_pr_source_non_actionable_mergeability_change_does_not_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _pr_source(
        monkeypatch,
        iter(
            (
                _pr_snapshot(mergeable="UNKNOWN"),
                _pr_snapshot(mergeable="MERGEABLE"),
            )
        ),
    )

    assert source.check() is None
    assert source.check() is None


@pytest.mark.parametrize(
    ("snapshot", "reason"),
    [(_pr_snapshot(state="MERGED"), "merged"), (_pr_snapshot(mergeable="CONFLICTING"), "conflicted")],
)
def test_pr_source_actionable_initial_state_fires(
    monkeypatch: pytest.MonkeyPatch, snapshot: wait_for.PrSnapshot, reason: str
) -> None:
    source = _pr_source(monkeypatch, iter((snapshot,)))

    assert source.check() == {
        "reasons": [reason],
        "before": None,
        "after": snapshot.payload(),
    }


def test_significant_comment_filter_ignores_exact_loom_placeholder_and_fires_after_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    placeholder = wait_for.GhRecord(
        id=1,
        author=wait_for.LOOM_BOT,
        body="Working on this in loom: https://loom.oa.dev/s/qhb71pit",
        url="https://github.com/marin-community/marin/pull/123#issuecomment-1",
        state=None,
        kind="issue_comment",
    )
    finding = wait_for.GhRecord(
        id=placeholder.id,
        author=placeholder.author,
        body="The timeout path exits before writing the final event payload.",
        url=placeholder.url,
        state=None,
        kind=placeholder.kind,
    )
    responses = iter(([], [], [placeholder], [], [finding], []))
    monkeypatch.setattr(wait_for, "gh_api_list", lambda _repo, _path, *, kind: next(responses))
    source = wait_for.CommentSource(
        wait_for.parse_spec("github.pr_comment 123"),
        "marin-community/marin",
        ignore_authors=set(),
        comment_filter=wait_for.CommentFilter.SIGNIFICANT,
    )

    assert source.check() is None
    assert source.check() is None
    assert source.check() == {
        "comments": [
            {
                "author": wait_for.LOOM_BOT,
                "body": finding.body,
                "url": finding.url,
                "kind": "issue_comment",
                "significance": "concern",
            }
        ]
    }


def test_loom_placeholder_text_from_human_remains_significant() -> None:
    body = "Working on this in loom: https://loom.oa.dev/s/qhb71pit"

    assert wait_for.classify_significance(body, "human-reviewer") is wait_for.Significance.CONCERN
