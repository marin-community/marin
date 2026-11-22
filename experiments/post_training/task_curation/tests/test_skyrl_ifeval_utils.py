# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regressions for IFEval checks shared by the SkyRL and Nemotron sources."""

import pytest

from experiments.post_training.task_curation.datasets.skyrl.scorers import ifeval_utils


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("First paragraph.\n\n***\n\nSecond paragraph.", True),
        ("First paragraph.\n\n* * *\n\nSecond paragraph.", True),
        ("First paragraph.\n\nSecond paragraph.", False),
        ("First paragraph.\n\n***\n\n", False),
    ],
)
def test_paragraphs_require_two_nonempty_parts_with_a_markdown_divider(reply: str, expected: bool):
    assert ifeval_utils.verify_paragraph_count(reply, 2) is expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("First paragraph.\n* One\n* Two\n\n***\n\nSecond paragraph.", True),
        ("First paragraph.\n* One\n* Two\n\n* * *\n\nSecond paragraph.", True),
        ("**Key points:**\n* One\n* Two", True),
        ("First paragraph.\n* One\n\n***\n\nSecond paragraph.", False),
    ],
)
def test_bullets_exclude_dividers_and_bold_headings(reply: str, expected: bool):
    assert ifeval_utils.verify_bullet_points(reply, 2) is expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("<<A Title>>\n\nSECTION 1\nBody.\n\nSECTION 2\nBody.", True),
        ("<<Guide to SECTION 1>>\n\nSECTION 1\nBody.\n\nSECTION 2\nBody.", True),
        ("<<A Title>>\n\nSECTION 1\nBody.", False),
        ("<<A Title>>\n\nBody without a section marker.", False),
    ],
)
def test_sections_exclude_the_title_before_the_first_marker(reply: str, expected: bool):
    assert ifeval_utils.validate_sections(reply, 2, "SECTION") is expected


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("**First** and **second** are highlighted.", True),
        ("**Only one** is highlighted.", False),
        ("There is no highlighted text.", False),
    ],
)
def test_highlights_count_each_bold_span_once(reply: str, expected: bool):
    assert ifeval_utils.validate_highlighted_sections(reply, 2) is expected
