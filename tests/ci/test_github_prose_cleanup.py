# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from scripts.ci.github_prose_cleanup import cleanup_result_from_event


@pytest.mark.parametrize("kind", ["issue", "pull_request"])
@pytest.mark.parametrize(
    "body",
    [
        "Bound retries to five attempts so a failed upload cannot stall the pipeline.",
        "## Summary\n\nThis PR bounds **upload retries**.\n\n- [x] Check retries\n\nFixes #123",
    ],
)
def test_unchanged_description_does_not_trigger_cleanup(kind: str, body: str) -> None:
    result = cleanup_result_from_event({kind: {"number": 42, "body": body}}, body)

    assert not result.changed
    assert result.cleaned_body == body
    assert not result.archive_comment_body


@pytest.mark.parametrize("kind", ["issue", "pull_request"])
def test_attribution_edit_preserves_remaining_description_and_archives_original(kind: str) -> None:
    rewritten = (
        "## Summary\n\nThis PR bounds **upload retries**.\n\n"
        "- [x] Check retries\n\n"
        "```python\nretries = 5\n```\n\n"
        "[Evidence](https://example.com/results)\n\nFixes #123"
    )
    original = rewritten + "\n\nWritten by Claude"

    result = cleanup_result_from_event({kind: {"number": 42, "body": original}}, rewritten)

    assert result.changed
    assert result.cleaned_body == rewritten
    assert original in result.archive_comment_body
    assert result.original_body_hash == hashlib.sha256(original.encode()).hexdigest()
    assert result.archive_marker in result.archive_comment_body
