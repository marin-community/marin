# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from scripts.pm.scrub_experiment_issue_tldrs import (
    IssueSummaryBlock,
    extract_existing_doc_issue_number,
    issue_needs_summary_refresh,
    render_tldr_block,
    upsert_tldr_block,
)


def _issue(*, body: str | None, labels: list[str]):
    return SimpleNamespace(
        body=body,
        labels=[SimpleNamespace(name=name) for name in labels],
    )


def test_upsert_tldr_block_appends_when_missing():
    body = "Original issue body."
    updated = upsert_tldr_block(body, "<!-- experiment-tldr:start -->\nnew\n<!-- experiment-tldr:end -->")

    assert updated.startswith("Original issue body.\n\n")
    assert updated.endswith("<!-- experiment-tldr:end -->")


def test_upsert_tldr_block_replaces_existing_managed_block():
    body = "\n".join(
        [
            "Original issue body.",
            "",
            "<!-- experiment-tldr:start -->",
            "old",
            "<!-- experiment-tldr:end -->",
        ]
    )

    updated = upsert_tldr_block(body, "<!-- experiment-tldr:start -->\nnew\n<!-- experiment-tldr:end -->")

    assert "old" not in updated
    assert updated.count("<!-- experiment-tldr:start -->") == 1
    assert "new" in updated


def test_render_tldr_block_tracks_doc_issue_marker():
    block = IssueSummaryBlock(
        summary="A short summary.",
        relevant_links=["https://example.com/doc"],
    )

    rendered = render_tldr_block(block, doc_issue_number=123)

    assert "## Summary" in rendered
    assert "### Helpful links" in rendered
    assert extract_existing_doc_issue_number(rendered) == 123


def test_issue_without_managed_block_needs_refresh():
    assert issue_needs_summary_refresh(_issue(body="Original body", labels=["experiment"]), refresh_existing=False)


def test_issue_with_managed_block_but_no_tldr_label_needs_refresh():
    body = "<!-- experiment-tldr:start -->\nSummary\n<!-- experiment-tldr:end -->"
    assert issue_needs_summary_refresh(_issue(body=body, labels=["experiment"]), refresh_existing=False)


def test_issue_with_managed_block_and_tldr_label_is_skipped_without_refresh_existing():
    body = "<!-- experiment-tldr:start -->\nSummary\n<!-- experiment-tldr:end -->"
    assert not issue_needs_summary_refresh(_issue(body=body, labels=["experiment", "tldr"]), refresh_existing=False)


def test_refresh_existing_revisits_adequately_labeled_issue():
    body = "<!-- experiment-tldr:start -->\nSummary\n<!-- experiment-tldr:end -->"
    assert issue_needs_summary_refresh(_issue(body=body, labels=["experiment", "tldr"]), refresh_existing=True)
