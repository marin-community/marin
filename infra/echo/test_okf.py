# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Round-trip and validation tests for the OKF wiki document format."""

import okf
import pytest

DOC = """---
type: wiki-note
title: Grafana access
use_when: when you need to inspect training dashboards
tags:
  - ops
  - grafana
author: someone@openathena.ai
---

Use the IAP route via grafana.oa.dev.
"""


def test_parse_wiki_extracts_fields():
    fields = okf.parse_wiki(DOC)
    assert fields.title == "Grafana access"
    assert fields.use_when == "when you need to inspect training dashboards"
    assert fields.tags == ("ops", "grafana")
    assert fields.body == "Use the IAP route via grafana.oa.dev."


def test_parse_wiki_ignores_body_leading_heading():
    fields = okf.parse_wiki("---\ntype: wiki-note\ntitle: T\nuse_when: when X\n---\n\n# Notes\n\nBody line.")
    assert fields.body == "# Notes\n\nBody line."


def test_parse_requires_frontmatter():
    with pytest.raises(ValueError, match="frontmatter"):
        okf.parse_wiki("no frontmatter, just prose")


def test_parse_requires_type():
    with pytest.raises(ValueError, match="type"):
        okf.parse("---\ntitle: X\n---\n\nbody")


def test_parse_wiki_reports_missing_fields():
    with pytest.raises(ValueError, match="use_when"):
        okf.parse_wiki("---\ntype: wiki-note\ntitle: X\n---\n\nbody")


def test_parse_wiki_rejects_scalar_tags():
    with pytest.raises(ValueError, match="list of strings"):
        okf.parse_wiki("---\ntype: wiki-note\ntitle: X\nuse_when: when X\ntags: ops\n---\n\nbody")


def test_round_trip_through_okf():
    entry = {
        "id": 7,
        "title": "Grafana access",
        "use_when": "when you need dashboards",
        "tags": ["ops", "grafana"],
        "body": "Use the IAP route.",
        "author": "a@openathena.ai",
        "updated_at": "2026-07-27T00:00:00Z",
    }
    text = okf.wiki_to_okf(entry, resource="https://echo.oa.dev/wiki/7")
    assert "type: wiki-note" in text
    assert "resource: https://echo.oa.dev/wiki/7" in text
    assert "tags:\n- ops\n- grafana" in text
    fields = okf.parse_wiki(text)
    assert (fields.title, fields.use_when, fields.tags, fields.body) == (
        "Grafana access",
        "when you need dashboards",
        ("ops", "grafana"),
        "Use the IAP route.",
    )
