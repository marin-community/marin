# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure logic of context_search.py — no Cloud SQL connection."""

import argparse
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("context_search", Path(__file__).parent / "context_search.py")
cs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cs)


def test_escape_like_neutralizes_wildcards():
    assert cs.escape_like("ragged_all_to_all") == "ragged\\_all\\_to\\_all"
    assert cs.escape_like("50%") == "50\\%"
    assert cs.escape_like("a\\b") == "a\\\\b"


def test_chunk_filters_builds_predicates_and_params():
    args = argparse.Namespace(source="discord", kind="message", since="2026-07-01")
    predicates, params = cs.chunk_filters(args)
    assert predicates == ["source = %s", "kind = %s", "date >= %s"]
    assert params == ["discord", "message", "2026-07-01"]


def test_chunk_filters_empty_when_unfiltered():
    predicates, params = cs.chunk_filters(argparse.Namespace(source=None, kind=None, since=None))
    assert predicates == []
    assert params == []
    assert cs.where_clause(predicates) == ""
    assert cs.where_clause(["source = %s"]) == "WHERE source = %s"


def test_bounded_limit_rejects_out_of_range():
    assert cs.bounded_limit("50") == 50
    for bad in ("0", "-1", "101"):
        with pytest.raises(argparse.ArgumentTypeError):
            cs.bounded_limit(bad)


def test_iso_date_and_nonblank_validate():
    assert cs.iso_date("2026-07-01") == "2026-07-01"
    with pytest.raises(ValueError):
        cs.iso_date("07/01/2026")
    with pytest.raises(argparse.ArgumentTypeError):
        cs.nonblank("   ")


def test_parser_rejects_blank_query_and_bad_limit():
    parser = cs.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "  "])
    with pytest.raises(SystemExit):
        parser.parse_args(["search", "moe", "--limit", "999"])
    args = parser.parse_args(["grep", "ragged_all_to_all", "--source", "discord", "--limit", "5"])
    assert args.pattern == "ragged_all_to_all" and args.source == "discord" and args.limit == 5
