# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.pm.scrub_experiment_issue_tldrs import (
    IssueAnalysis,
    extract_existing_doc_issue_number,
    render_tldr_block,
    sanitize_analysis,
    upsert_tldr_block,
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
    analysis = IssueAnalysis(
        summary="A short summary.",
        documentation_sufficient=False,
        relevant_links=["https://example.com/doc"],
        needs_doc_issue=True,
        doc_issue_title="Docs gap",
        doc_issue_body="Need more docs",
    )

    block = render_tldr_block(analysis, doc_issue_number=123)

    assert "Follow-up: #123." in block
    assert extract_existing_doc_issue_number(block) == 123


def test_sanitize_analysis_filters_unknown_links():
    analysis = IssueAnalysis(
        summary="A short summary.",
        documentation_sufficient=True,
        relevant_links=["https://allowed", "https://disallowed", "https://allowed"],
        needs_doc_issue=False,
        doc_issue_title=None,
        doc_issue_body=None,
    )

    sanitized = sanitize_analysis(analysis, ["https://allowed"])

    assert sanitized.relevant_links == ["https://allowed"]


def test_sanitize_analysis_requires_doc_issue_payload_when_needed():
    analysis = IssueAnalysis(
        summary="A short summary.",
        documentation_sufficient=False,
        relevant_links=[],
        needs_doc_issue=True,
        doc_issue_title="",
        doc_issue_body=None,
    )

    try:
        sanitize_analysis(analysis, [])
    except ValueError as exc:
        assert "Documentation gap issues require" in str(exc)
    else:
        raise AssertionError("Expected sanitize_analysis() to reject incomplete doc issue payloads")
