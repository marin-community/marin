# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from scripts.pm.scrub_experiment_issue_tldrs import (
    dedupe_preserving_order,
    extract_existing_doc_issue_number,
    issue_needs_summary_refresh,
)


def _issue(*, body: str | None, labels: list[str]):
    return SimpleNamespace(
        body=body,
        labels=[SimpleNamespace(name=name) for name in labels],
    )


def test_dedupe_preserving_order_keeps_first_occurrence():
    assert dedupe_preserving_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_extract_existing_doc_issue_number_reads_marker():
    body = "\n".join(
        [
            "<!-- experiment-tldr:start -->",
            "## Summary",
            "",
            "Example summary.",
            "",
            "<!-- experiment-tldr:doc-issue=123 -->",
            "<!-- experiment-tldr:end -->",
        ]
    )

    assert extract_existing_doc_issue_number(body) == 123


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
