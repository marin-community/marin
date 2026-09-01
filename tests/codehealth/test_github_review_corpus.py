# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import json
import subprocess
from copy import deepcopy

import pytest

from infra.codehealth import github_review_corpus as github

REPOSITORY = "owner/repo"
START = dt.datetime(2026, 8, 1, tzinfo=dt.UTC)
END = dt.datetime(2026, 9, 1, tzinfo=dt.UTC)


def test_github_client_serializes_graphql_id_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '{"data":{"rateLimit":{"cost":1}}}', "")

    monkeypatch.setattr(github.subprocess, "run", fake_run)

    github.GitHubClient().graphql("query($ids: [ID!]!) { rateLimit { cost } }", {"ids": ["node-a", "node-b"]})

    fields = [commands[0][index + 1] for index, value in enumerate(commands[0][:-1]) if value == "-F"]
    assert "ids[]=node-a" in fields
    assert "ids[]=node-b" in fields


def test_github_client_returns_no_diff_only_for_exact_too_large_response(monkeypatch: pytest.MonkeyPatch) -> None:
    too_large = json.dumps(
        {
            "message": "diff exceeded the maximum number of files",
            "errors": [{"resource": "PullRequest", "field": "diff", "code": "too_large"}],
            "status": "406",
        }
    )

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, too_large, "gh: request failed (HTTP 406)\n")

    monkeypatch.setattr(github.subprocess, "run", fake_run)

    assert github.GitHubClient().rest_text("repos/owner/repo/pulls/7", "application/vnd.github.diff") is None


def test_github_client_preserves_non_too_large_diff_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "gh: authentication failed (HTTP 401)\n")

    monkeypatch.setattr(github.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        github.GitHubClient().rest_text("repos/owner/repo/pulls/7", "application/vnd.github.diff")


def _page(nodes: list[dict], *, total: int | None = None, previous: bool = False, next_: bool = False) -> dict:
    return {
        "totalCount": len(nodes) if total is None else total,
        "pageInfo": {
            "hasPreviousPage": previous,
            "startCursor": "before" if previous else None,
            "hasNextPage": next_,
            "endCursor": "after" if next_ else None,
        },
        "nodes": nodes,
    }


def _actor(login: str, typename: str = "User") -> dict:
    return {"__typename": typename, "login": login}


def _review(review_id: int, author: dict, submitted: str, *, body: str = "") -> dict:
    return {
        "id": f"review-node-{review_id}",
        "databaseId": review_id,
        "body": body,
        "state": "COMMENTED",
        "createdAt": submitted,
        "updatedAt": submitted,
        "submittedAt": submitted,
        "url": f"https://example.test/review/{review_id}",
        "author": author,
        "authorAssociation": "MEMBER",
        "commit": {"oid": "head"},
        "comments": {"totalCount": 0},
    }


def _pull(*, changed_files: int = 1) -> dict:
    return {
        "id": "pull-node-7",
        "number": 7,
        "url": "https://example.test/pull/7",
        "title": "Frozen review corpus",
        "body": "PR body",
        "state": "OPEN",
        "isDraft": False,
        "author": _actor("author"),
        "authorAssociation": "MEMBER",
        "createdAt": "2026-07-01T00:00:00Z",
        "updatedAt": "2026-08-28T00:00:00Z",
        "closedAt": None,
        "mergedAt": None,
        "baseRefName": "main",
        "baseRefOid": "base",
        "headRefName": "feature",
        "headRefOid": "head",
        "additions": 1,
        "deletions": 1,
        "changedFiles": changed_files,
    }


def _commit() -> dict:
    return {
        "commit": {
            "oid": "head",
            "message": "Change code",
            "authoredDate": "2026-08-01T00:00:00Z",
            "committedDate": "2026-08-01T00:00:00Z",
            "author": {"user": {"login": "author"}},
            "parents": {"totalCount": 1, "nodes": [{"oid": "base"}]},
        }
    }


def _file(path: str = "example.py") -> dict:
    return {
        "path": path,
        "changeType": "MODIFIED",
        "additions": 1,
        "deletions": 1,
    }


def _thread_comment(comment_id: int, review_id: int, *, updated: str, author: dict | None = None) -> dict:
    return {
        "id": f"comment-node-{comment_id}",
        "databaseId": comment_id,
        "body": "edited old comment",
        "state": "SUBMITTED",
        "createdAt": "2026-07-01T00:00:00Z",
        "updatedAt": updated,
        "url": f"https://example.test/comment/{comment_id}",
        "author": author or _actor("reviewer"),
        "authorAssociation": "MEMBER",
        "diffHunk": "@@ -1 +1 @@\n-old\n+new",
        "path": "example.py",
        "line": 1,
        "originalLine": 1,
        "startLine": None,
        "originalStartLine": None,
        "commit": {"oid": "head"},
        "originalCommit": {"oid": "head"},
        "replyTo": None,
        "pullRequestReview": {"databaseId": review_id},
    }


def _thread(comment: dict) -> dict:
    return {
        "id": "thread-1",
        "isResolved": True,
        "isOutdated": False,
        "path": "example.py",
        "line": 1,
        "originalLine": 1,
        "startLine": None,
        "originalStartLine": None,
        "diffSide": "RIGHT",
        "startDiffSide": None,
        "subjectType": "LINE",
        "resolvedBy": _actor("author"),
        "comments": _page([comment]),
    }


class FakeGitHub(github.GitHubClient):
    def __init__(self, scan_pull: dict, hydrated_pull: dict, *, seed: dict | None = None) -> None:
        super().__init__()
        self.scan_pull = deepcopy(scan_pull)
        self.hydrated_pull = deepcopy(hydrated_pull)
        self.seed = deepcopy(seed)
        self.text_calls = 0
        self.older_review_page: dict | None = None
        self.review_page: dict | None = None
        self.thread_comment_page: dict | None = None
        self.file_page: dict | None = None
        self.seed_candidate: dict | None = self.scan_pull | {"__typename": "PullRequest"}
        self.diff: str | None = "diff --git a/example.py b/example.py\n"
        self.mutate_once = False

    def graphql(self, query: str, variables: dict[str, object]) -> dict:
        self._record_graphql_request(1)
        if query == github.SCAN_QUERY:
            return {"repository": {"pullRequests": _page([deepcopy(self.scan_pull)])}, "rateLimit": {"cost": 1}}
        if query == github.SCAN_REVIEWS_QUERY:
            assert self.older_review_page is not None
            return {
                "repository": {"pullRequest": {"reviews": deepcopy(self.older_review_page)}},
                "rateLimit": {"cost": 1},
            }
        if query == github.SEED_PULL_REQUEST_QUERY:
            return {
                "repository": {"candidate": deepcopy(self.seed_candidate)},
                "rateLimit": {"cost": 1},
            }
        if query.lstrip().startswith("query BatchHydrate"):
            numbers = sorted(name for name in variables if name.startswith("number"))
            repository = {f"pr{index}": deepcopy(self.hydrated_pull) for index, _ in enumerate(numbers)}
            return {"repository": repository, "rateLimit": {"cost": 1}}
        if query.lstrip().startswith("query BatchConnections"):
            if "reviews(first:" in query:
                assert self.review_page is not None
                page = {"reviews": deepcopy(self.review_page)}
            elif "files(first:" in query:
                assert self.file_page is not None
                page = {"files": deepcopy(self.file_page)}
            else:
                raise AssertionError("unexpected batched connection")
            return {"repository": {"pr0": page}, "rateLimit": {"cost": 1}}
        if query == github.THREAD_COMMENTS_PAGE_QUERY:
            assert self.thread_comment_page is not None
            return {
                "node": {"comments": deepcopy(self.thread_comment_page)},
                "rateLimit": {"cost": 1},
            }
        if query == github.FINGERPRINT_QUERY:
            ids = variables["ids"]
            assert isinstance(ids, list | tuple)
            nodes = [deepcopy(self.hydrated_pull) for _ in ids]
            if self.mutate_once:
                nodes[0]["updatedAt"] = "2026-08-29T00:00:00Z"
                self.mutate_once = False
            return {"nodes": nodes, "rateLimit": {"cost": 1}}
        raise AssertionError("unexpected GraphQL query")

    def rest_records(self, endpoint: str) -> list[dict]:
        self._record_rest_requests(1)
        if "/pulls/comments?" in endpoint:
            return [deepcopy(self.seed)] if self.seed and "pull_request_url" in self.seed else []
        if "/issues/comments?" in endpoint:
            return [deepcopy(self.seed)] if self.seed and "issue_url" in self.seed else []
        raise AssertionError(f"unexpected REST endpoint: {endpoint}")

    def rest_text(self, endpoint: str, accept: str) -> str | None:
        del endpoint, accept
        self._record_rest_requests(1)
        self.text_calls += 1
        return self.diff


def _hydrated_pull(*, review: dict, thread: dict | None, changed_files: int = 1) -> dict:
    pull = _pull(changed_files=changed_files)
    pull.update(
        {
            "comments": _page([]),
            "reviews": _page([review]),
            "reviewThreads": _page([thread] if thread else []),
            "commits": _page([_commit()]),
            "files": _page([_file()] if changed_files else []),
        }
    )
    return pull


def _scan_pull(
    reviews: list[dict],
    *,
    changed_files: int = 1,
    review_thread_count: int = 0,
    review_total: int | None = None,
    previous_reviews: bool = False,
) -> dict:
    pull = _pull(changed_files=changed_files)
    pull.update(
        {
            "comments": _page([]),
            "reviews": _page(reviews, total=review_total, previous=previous_reviews),
            "reviewThreads": {"totalCount": review_thread_count},
            "commits": {"totalCount": 1},
        }
    )
    return pull


def test_collect_corpus_discovers_edited_old_inline_comment_and_retains_full_context() -> None:
    review = _review(55, _actor("reviewer"), "2026-07-01T00:00:00Z")
    review["comments"]["totalCount"] = 1
    comment = _thread_comment(101, 55, updated="2026-08-20T00:00:00Z")
    hydrated = _hydrated_pull(review=review, thread=_thread(comment))
    scan = deepcopy(hydrated)
    scan["updatedAt"] = "2026-07-20T00:00:00Z"
    scan["reviews"] = _page([review])
    scan["reviewThreads"] = {"totalCount": 1}
    seed = {
        "id": 101,
        "body": "edited old comment",
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-08-20T00:00:00Z",
        "pull_request_url": "https://api.github.test/repos/owner/repo/pulls/7",
        "user": {"login": "reviewer", "type": "User"},
    }
    client = FakeGitHub(scan, hydrated, seed=seed)

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    bundle = result.bundles[0]
    assert [(event.kind, event.database_id, event.in_window) for event in bundle.events] == [
        ("inline_comment", 101, True),
        ("review", 55, False),
    ]
    assert bundle.events[1].body == ""
    assert bundle.threads[0].is_resolved
    assert bundle.files[0].filename == "example.py"
    assert bundle.diff.startswith("diff --git")
    assert result.candidate_pull_requests == 1
    assert result.usage.rest_requests == 3
    assert result.usage.projected_rest_requests == 153


def test_collect_corpus_scans_past_ten_newer_bots_for_human_review() -> None:
    bots = [_review(index, _actor("bot", "Bot"), f"2026-08-{index:02d}T00:00:00Z") for index in range(10, 20)]
    human = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Please simplify this.")
    scan = _scan_pull(bots, review_total=11, previous_reviews=True)
    hydrated = _hydrated_pull(review=human, thread=None)
    hydrated["reviews"] = _page([human, bots[0]])
    client = FakeGitHub(scan, hydrated)
    client.older_review_page = _page([human], total=11)

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins={"bot"}, client=client)

    assert result.candidate_pull_requests == 1
    assert [(event.database_id, event.is_bot) for event in result.bundles[0].events] == [(1, False), (10, True)]
    assert result.usage.graphql_requests == 4


def test_collect_corpus_reuses_unchanged_stored_pull_request() -> None:
    review = _review(55, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    hydrated = _hydrated_pull(review=review, thread=None)
    client = FakeGitHub(scan, hydrated)
    cached = github.PullRequestFingerprint(
        updated_at="2026-08-28T00:00:00Z",
        head_sha="head",
        base_sha="base",
        changed_files=1,
        commits=1,
        reviews=1,
        review_threads=0,
        issue_comments=0,
    )
    reused: list[int] = []

    result = github.collect_corpus(
        REPOSITORY,
        START,
        END,
        bot_logins=set(),
        client=client,
        cached_fingerprints={7: cached},
        reused_pull_request_sink=reused.append,
    )

    assert result.bundles == ()
    assert result.reused_pull_requests == 1
    assert reused == [7]
    assert client.text_calls == 0
    assert result.usage.rest_requests == 2


def test_collect_corpus_trusts_same_window_checkpoint_over_edited_event_seed() -> None:
    review = _review(55, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    seed = {
        "id": 101,
        "body": "edited comment",
        "created_at": "2026-08-01T00:00:00Z",
        "updated_at": "2026-08-20T00:00:00Z",
        "pull_request_url": "https://api.github.test/repos/owner/repo/pulls/7",
        "user": {"login": "reviewer", "type": "User"},
    }
    client = FakeGitHub(scan, _hydrated_pull(review=review, thread=None), seed=seed)

    result = github.collect_corpus(
        REPOSITORY,
        START,
        END,
        bot_logins=set(),
        client=client,
        checkpointed_pr_numbers={7},
    )

    assert result.bundles == ()
    assert result.reused_pull_requests == 1
    assert client.text_calls == 0


def test_collect_corpus_paginates_complete_review_threads() -> None:
    review = _review(55, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    review["comments"]["totalCount"] = 2
    recent = _thread_comment(102, 55, updated="2026-08-20T00:00:00Z")
    old = _thread_comment(101, 55, updated="2026-07-01T00:00:00Z")
    thread = _thread(recent)
    thread["comments"] = _page([recent], total=2, next_=True)
    scan = _scan_pull([review], review_thread_count=1)
    hydrated = _hydrated_pull(review=review, thread=thread)
    client = FakeGitHub(scan, hydrated)
    client.thread_comment_page = _page([old], total=2)

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    bundle = result.bundles[0]
    assert bundle.threads[0].comment_ids == (102, 101)
    assert [event.database_id for event in bundle.events if event.kind == "inline_comment"] == [101, 102]


def test_collect_corpus_rejects_thread_comment_without_review() -> None:
    review = _review(55, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    review["comments"]["totalCount"] = 1
    comment = _thread_comment(101, 55, updated="2026-08-20T00:00:00Z")
    comment["pullRequestReview"] = None
    scan = _scan_pull([review], review_thread_count=1)
    hydrated = _hydrated_pull(review=review, thread=_thread(comment))

    with pytest.raises(RuntimeError, match="thread comment 101 has no review"):
        github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=FakeGitHub(scan, hydrated))


def test_collect_corpus_paginates_exact_changed_file_metadata() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review], changed_files=2)
    hydrated = _hydrated_pull(review=review, thread=None, changed_files=2)
    hydrated["files"] = _page([_file("first.py")], total=2, next_=True)
    client = FakeGitHub(scan, hydrated)
    client.file_page = _page([_file("second.py")], total=2)

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    assert [item.filename for item in result.bundles[0].files] == ["first.py", "second.py"]
    assert result.usage.rest_requests == 3


def test_collect_corpus_rejects_changed_file_cap_before_rest_context() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review], changed_files=3_001)
    hydrated = _hydrated_pull(review=review, thread=None, changed_files=3_001)
    client = FakeGitHub(scan, hydrated)

    with pytest.raises(RuntimeError, match="3,000-file API cap"):
        github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    assert client.text_calls == 0


def test_collect_corpus_rejects_duplicate_records_across_hydration_pages() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    hydrated = _hydrated_pull(review=review, thread=None)
    hydrated["reviews"] = _page([review], total=2, next_=True)
    client = FakeGitHub(scan, hydrated)
    client.review_page = _page([review], total=2)

    with pytest.raises(RuntimeError, match="reviews contains duplicate records"):
        github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)


def test_collect_corpus_rejects_nonadvancing_hydration_cursor() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    hydrated = _hydrated_pull(review=review, thread=None)
    hydrated["reviews"] = _page([review], total=3, next_=True)
    client = FakeGitHub(scan, hydrated)
    client.review_page = _page([_review(2, _actor("reviewer"), "2026-08-02T00:00:00Z")], total=3, next_=True)

    with pytest.raises(RuntimeError, match="reviews pagination cursor did not advance"):
        github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)


def test_limited_collection_does_not_expand_with_seed_only_pull_requests() -> None:
    bot_review = _review(1, _actor("bot", "Bot"), "2026-08-01T00:00:00Z")
    scan = _scan_pull([bot_review])
    hydrated = _hydrated_pull(review=bot_review, thread=None)
    seed = {
        "id": 201,
        "body": "edited comment on another old PR",
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-08-20T00:00:00Z",
        "pull_request_url": "https://api.github.test/repos/owner/repo/pulls/8",
        "user": {"login": "reviewer", "type": "User"},
    }
    client = FakeGitHub(scan, hydrated, seed=seed)

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins={"bot"}, limit=1, client=client)

    assert result.candidate_pull_requests == 1
    assert result.bundles == ()


def test_collect_corpus_ignores_issue_comment_seed_without_pull_request() -> None:
    bot_review = _review(1, _actor("bot", "Bot"), "2026-07-01T00:00:00Z")
    scan = _scan_pull([bot_review])
    scan["updatedAt"] = "2026-07-01T00:00:00Z"
    hydrated = _hydrated_pull(review=bot_review, thread=None)
    seed = {
        "id": 201,
        "body": "edited issue comment",
        "created_at": "2026-07-01T00:00:00Z",
        "updated_at": "2026-08-20T00:00:00Z",
        "issue_url": "https://api.github.test/repos/owner/repo/issues/8",
        "user": {"login": "reviewer", "type": "User"},
    }
    client = FakeGitHub(scan, hydrated, seed=None)
    client.seed = seed
    client.seed_candidate = {"__typename": "Issue"}

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins={"bot"}, client=client)

    assert result.candidate_pull_requests == 0
    assert result.bundles == ()


def test_collect_corpus_retries_one_mutated_pull_request_snapshot() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    hydrated = _hydrated_pull(review=review, thread=None)
    client = FakeGitHub(scan, hydrated)
    client.mutate_once = True

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    assert len(result.bundles) == 1
    assert client.text_calls == 2
    assert result.usage.rest_requests == 4


def test_collect_corpus_marks_too_large_diff_as_unavailable() -> None:
    review = _review(1, _actor("reviewer"), "2026-08-01T00:00:00Z", body="Review this.")
    scan = _scan_pull([review])
    hydrated = _hydrated_pull(review=review, thread=None)
    client = FakeGitHub(scan, hydrated)
    client.diff = None

    result = github.collect_corpus(REPOSITORY, START, END, bot_logins=set(), client=client)

    bundle = result.bundles[0]
    assert bundle.diff is None
    assert [item.filename for item in bundle.files] == ["example.py"]
