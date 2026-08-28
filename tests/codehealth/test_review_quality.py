# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace

import pytest

from infra.codehealth import review_quality as review


def _pull_request() -> dict:
    return {
        "number": 42,
        "title": "Tighten review context",
        "mergedAt": "2026-08-27T12:00:00Z",
        "author": {"login": "author"},
        "headRefOid": "a" * 40,
        "baseRefOid": "b" * 40,
    }


def _comment(**overrides) -> review.Comment:
    values = {
        "pr_number": 42,
        "pr_title": "Tighten review context",
        "merged_at": "2026-08-27T12:00:00Z",
        "pr_author": "author",
        "head_sha": "a" * 40,
        "base_sha": "b" * 40,
        "comment_id": 7,
        "comment_type": "inline",
        "author": "reviewer",
        "is_bot": False,
        "file": "infra/example.py",
        "line": 12,
        "body": "This branch is inverted.",
        "source_url": "https://example.test/inline/7",
        "context": "@@ -10,3 +10,3 @@\n-if ready:\n+if not ready:",
    }
    values.update(overrides)
    return review.Comment(**values)


def test_fetch_pr_comments_preserves_inline_and_pull_request_context(monkeypatch) -> None:
    def paginated(args: list[str]) -> list[dict]:
        endpoint = args[1]
        if endpoint.endswith("/files"):
            return [
                {
                    "filename": "infra/example.py",
                    "status": "modified",
                    "additions": 2,
                    "deletions": 1,
                    "patch": "@@ -10,3 +10,4 @@\n-old\n+new",
                }
            ]
        if endpoint.endswith("/comments") and "/pulls/" in endpoint:
            return [
                {
                    "id": 7,
                    "user": {"login": "reviewer", "type": "User"},
                    "path": "infra/example.py",
                    "line": 12,
                    "body": "This branch is inverted.",
                    "diff_hunk": "@@ -10,3 +10,3 @@\n-if ready:\n+if not ready:",
                    "html_url": "https://example.test/inline/7",
                }
            ]
        if endpoint.endswith("/reviews"):
            return [
                {
                    "id": 8,
                    "user": {"login": "reviewer", "type": "User"},
                    "body": "Please simplify the control flow.",
                    "html_url": "https://example.test/review/8",
                }
            ]
        return [
            {
                "id": 9,
                "user": {"login": "dependabot[bot]", "type": "Bot"},
                "body": "Automated update",
                "html_url": "https://example.test/issue/9",
            }
        ]

    monkeypatch.setattr(review, "_gh_paginated", paginated)

    comments = review.fetch_pr_comments("marin-community/marin", _pull_request(), set())

    inline, review_body, issue = comments
    assert inline.context == "@@ -10,3 +10,3 @@\n-if ready:\n+if not ready:"
    assert inline.source_url == "https://example.test/inline/7"
    assert review_body.context is not None
    assert "File: infra/example.py (modified, +2/-1)" in review_body.context
    assert review_body.context == issue.context
    assert issue.is_bot


def test_fetch_pr_comment_sets_is_parallel_and_preserves_pr_order(monkeypatch) -> None:
    barrier = threading.Barrier(2)

    def paginated(args: list[str]) -> list[dict]:
        endpoint = args[1]
        if endpoint.endswith("/files"):
            barrier.wait(timeout=2)
            return []
        if "/issues/" in endpoint:
            pr_number = int(endpoint.split("/issues/")[1].split("/")[0])
            return [
                {
                    "id": pr_number,
                    "user": {"login": "reviewer", "type": "User"},
                    "body": "Please simplify this branch.",
                    "html_url": f"https://example.test/issue/{pr_number}",
                }
            ]
        return []

    monkeypatch.setattr(review, "_gh_paginated", paginated)
    first = _pull_request()
    first["number"] = 41
    second = _pull_request()
    second["number"] = 42

    fetched = review.fetch_pr_comment_sets("marin-community/marin", [first, second], set(), concurrency=2)

    assert list(fetched) == [41, 42]
    assert [comments[0].pr_number for comments in fetched.values()] == [41, 42]


def test_pull_request_context_is_bounded() -> None:
    context = review._pull_request_context(
        [
            {
                "filename": f"file-{index}.py",
                "status": "modified",
                "additions": 1,
                "deletions": 1,
                "patch": "x" * (review.MAX_FILE_PATCH * 2),
            }
            for index in range(10)
        ]
    )

    assert context is not None
    assert len(context) <= review.MAX_COMMENT_CONTEXT
    assert "file-0.py" in context
    assert "x" * (review.MAX_FILE_PATCH + 1) not in context


def test_agent_marked_replies_are_not_reviewer_feedback() -> None:
    assert not review._is_reviewer_comment(_comment(body="  🤖 Fixed in abc123."))
    assert review._is_reviewer_comment(_comment(body="Please simplify this branch."))


def test_resolution_sends_context_only_for_uncached_comments() -> None:
    comments = [
        _comment(comment_id=1, body="cached", context="old context"),
        _comment(comment_id=2, body="new", context="new diff hunk"),
    ]
    cached = review.CommentClassification(
        **{"class": "ack"},
        catchable_strict=False,
        catchable_generous=False,
        confidence=1.0,
        reason="acknowledgment",
    )
    seen: list[review.CommentToClassify] = []

    def classifier(items: list[review.CommentToClassify]) -> dict[int, review.CommentClassification]:
        seen.extend(items)
        return {
            item.id: review.CommentClassification(
                **{"class": "bug"},
                catchable_strict=False,
                catchable_generous=True,
                confidence=0.9,
                reason="diff-local logic error",
            )
            for item in items
        }

    resolved = review.resolve_classifications(
        comments,
        {("inline", 1): ("cached", cached)},
        classifier,
        batch_size=20,
        concurrency=1,
    )

    assert resolved[0] is cached
    assert len(seen) == 1
    assert seen[0].context == "new diff hunk"


def test_resolution_excludes_low_confidence_catchable_verdicts_from_outcomes() -> None:
    comments = [
        _comment(comment_id=1, body="cached"),
        _comment(comment_id=2, body="fresh"),
    ]
    low_confidence = review.CommentClassification(
        **{"class": "structure"},
        catchable_strict=True,
        catchable_generous=True,
        confidence=review.MIN_CLASSIFICATION_CONFIDENCE - 0.01,
        reason="possible catalog match",
    )

    resolved = review.resolve_classifications(
        comments,
        {("inline", 1): ("cached", low_confidence)},
        lambda items: {item.id: low_confidence for item in items},
        batch_size=20,
        concurrency=1,
    )

    assert [(item.klass, item.confidence, item.reason) for item in resolved] == [
        ("structure", review.MIN_CLASSIFICATION_CONFIDENCE - 0.01, "possible catalog match"),
        ("structure", review.MIN_CLASSIFICATION_CONFIDENCE - 0.01, "possible catalog match"),
    ]
    assert all(not item.catchable_strict and not item.catchable_generous for item in resolved)


def test_model_classifier_uses_tool_free_structured_request(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeResponses:
        def parse(self, **kwargs):
            calls.append(kwargs)
            parsed = kwargs["text_format"].model_validate(
                {
                    "results": [
                        {
                            "id": 0,
                            "class": "bug",
                            "catchable_strict": True,
                            "catchable_generous": False,
                            "confidence": 0.9,
                            "reason": "deterministic branch check",
                        }
                    ]
                }
            )
            return SimpleNamespace(output_parsed=parsed)

    class FakeOpenAI:
        def __init__(self, *, timeout: float):
            assert timeout == review.CLASSIFIER_TIMEOUT
            self.responses = FakeResponses()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(review, "OpenAI", FakeOpenAI)
    classifier = review.make_model_classifier("gpt-5.6-terra", "medium")
    result = classifier(
        [
            review.CommentToClassify(
                id=0,
                file="infra/example.py",
                line=12,
                body="This branch is inverted.",
                context="@@ -10 +10 @@\n-if ready:\n+if not ready:",
            )
        ]
    )

    call = calls[0]
    assert set(call) == {"model", "input", "reasoning", "text_format"}
    assert call["model"] == "gpt-5.6-terra"
    assert call["reasoning"] == {"effort": "medium"}
    assert "Diff context:" in call["input"]
    assert result[0].catchable_strict
    assert result[0].catchable_generous


def test_model_classifier_preserves_structured_request_failure(monkeypatch) -> None:
    failure = ValueError("invalid API key")

    class FailingOpenAI:
        def __init__(self, *, timeout: float):
            del timeout
            raise failure

    monkeypatch.setattr(review, "OpenAI", FailingOpenAI)
    classifier = review.make_model_classifier("gpt-5.6-terra", "medium")
    items = [review.CommentToClassify(id=0, file=None, line=None, body="Review this.", context=None)]

    with pytest.raises(RuntimeError, match="batch of 1 comments: invalid API key") as raised:
        classifier(items)

    assert raised.value.__cause__ is failure


def test_classification_fails_when_a_batch_omits_a_comment() -> None:
    items = [
        review.CommentToClassify(id=0, file=None, line=None, body="first", context=None),
        review.CommentToClassify(id=1, file=None, line=None, body="second", context=None),
    ]

    with pytest.raises(RuntimeError):
        review.classify_comments(lambda _: {}, items, batch_size=20, concurrency=1)
