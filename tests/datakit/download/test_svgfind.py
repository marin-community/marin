# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the svgfind row → SFT document formatter."""

from marin.datakit.download.svgfind import svgfind_row_to_doc

SAMPLE_ROW = {
    "id": "10000206",
    "title": "messaging",
    "data_pack": "ui-outlines",
    "tags": [
        "messaging",
        "messaging app",
        "messaging service",
        "chat app",
        "chat service",
        "chat",
    ],
    "license": "CREATIVECOMMONS",
    "license_owner": "Adrian Adam",
    "download_url": "https://www.svgfind.com/download/10000206/messaging.svg",
    "svg_content": (
        '<svg fill="#000" width="800" height="800" viewBox="144 144 512 512" '
        'xmlns="http://www.w3.org/2000/svg"><path d="m1 2 3 4z"/></svg>'
    ),
}


def test_svgfind_row_to_doc_matches_expected_layout():
    expected_text = (
        "Create an SVG which matches the following description.\n"
        "Title: messaging\n"
        "Data Pack: ui-outlines\n"
        "Tags: messaging, messaging app, messaging service, chat app, chat service, chat\n"
        "\n"
        '<svg fill="#000" width="800" height="800" viewBox="144 144 512 512" '
        'xmlns="http://www.w3.org/2000/svg"><path d="m1 2 3 4z"/></svg>'
    )
    [doc] = svgfind_row_to_doc(SAMPLE_ROW)
    assert doc == {
        "id": "10000206",
        "text": expected_text,
        "source": "nyuuzyou/svgfind/creativecommons",
    }


def test_svgfind_row_to_doc_preserves_svg_verbatim():
    [doc] = svgfind_row_to_doc(SAMPLE_ROW)
    assert doc["text"].endswith(SAMPLE_ROW["svg_content"])


def test_svgfind_row_to_doc_preserves_tag_order_and_duplicates():
    row = {**SAMPLE_ROW, "tags": ["a", "a b", "a", "c"]}
    [doc] = svgfind_row_to_doc(row)
    assert "Tags: a, a b, a, c\n" in doc["text"]


def test_svgfind_row_to_doc_handles_empty_tags():
    row = {**SAMPLE_ROW, "tags": []}
    [doc] = svgfind_row_to_doc(row)
    assert "Tags: \n" in doc["text"]


def test_svgfind_row_to_doc_drops_rows_missing_title_or_svg():
    assert svgfind_row_to_doc({**SAMPLE_ROW, "title": ""}) == []
    assert svgfind_row_to_doc({**SAMPLE_ROW, "svg_content": ""}) == []
