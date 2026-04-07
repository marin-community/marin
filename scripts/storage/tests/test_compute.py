# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the compute.py collapsing logic and deletion set computation."""

from __future__ import annotations


from scripts.storage.cleanup import (
    DirEntry,
    _collapse_deletions,
    _most_common_rule,
    _parent_prefix,
)


def _make_entry(
    prefix: str,
    status: str = "delete",
    matched_rule: str = "test%",
    bucket: str = "marin-us-central2",
    coldline_count: int = 100,
    coldline_bytes: int = 1_000_000,
) -> DirEntry:
    return DirEntry(
        bucket=bucket,
        prefix=prefix,
        status=status,
        matched_rule=matched_rule,
        standard_count=0,
        standard_bytes=0,
        nearline_count=0,
        nearline_bytes=0,
        coldline_count=coldline_count,
        coldline_bytes=coldline_bytes,
        archive_count=0,
        archive_bytes=0,
    )


# ---------------------------------------------------------------------------
# _parent_prefix
# ---------------------------------------------------------------------------


def test_parent_prefix_depth_2():
    assert _parent_prefix("a/b/") == "a/"


def test_parent_prefix_depth_1():
    assert _parent_prefix("a/") is None


def test_parent_prefix_depth_3():
    assert _parent_prefix("a/b/c/") == "a/b/"


def test_parent_prefix_root():
    assert _parent_prefix("") is None


# ---------------------------------------------------------------------------
# _most_common_rule
# ---------------------------------------------------------------------------


def test_most_common_rule_single():
    entries = [_make_entry("a/", matched_rule="r1")]
    assert _most_common_rule(entries) == "r1"


def test_most_common_rule_majority():
    entries = [
        _make_entry("a/", matched_rule="r1"),
        _make_entry("b/", matched_rule="r2"),
        _make_entry("c/", matched_rule="r1"),
    ]
    assert _most_common_rule(entries) == "r1"


# ---------------------------------------------------------------------------
# _collapse_deletions
# ---------------------------------------------------------------------------


def test_collapse_all_siblings_delete():
    """If all children of a parent are delete, collapse to parent."""
    entries = [
        _make_entry("data/a/"),
        _make_entry("data/b/"),
        _make_entry("data/c/"),
    ]
    result = _collapse_deletions(entries)
    assert len(result) == 1
    assert result[0].prefix == "data/"
    assert result[0].status == "delete"
    assert result[0].coldline_count == 300
    assert result[0].coldline_bytes == 3_000_000


def test_collapse_mixed_siblings_no_collapse():
    """If one sibling is keep, don't collapse."""
    entries = [
        _make_entry("data/a/"),
        _make_entry("data/b/", status="keep"),
        _make_entry("data/c/"),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 2
    prefixes = {e.prefix for e in delete_entries}
    assert prefixes == {"data/a/", "data/c/"}


def test_collapse_single_child_no_collapse():
    """A single child doesn't collapse (needs >= 2 siblings)."""
    entries = [_make_entry("data/a/")]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 1
    assert delete_entries[0].prefix == "data/a/"


def test_collapse_recursive():
    """Collapsing should happen recursively bottom-up."""
    entries = [
        _make_entry("top/a/x/"),
        _make_entry("top/a/y/"),
        _make_entry("top/b/x/"),
        _make_entry("top/b/y/"),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 1
    assert delete_entries[0].prefix == "top/"
    assert delete_entries[0].coldline_count == 400


def test_collapse_parent_is_keep_blocks():
    """If the parent exists and is keep, don't collapse children into it."""
    entries = [
        _make_entry("data/", status="keep"),
        _make_entry("data/a/"),
        _make_entry("data/b/"),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 2
    prefixes = {e.prefix for e in delete_entries}
    assert prefixes == {"data/a/", "data/b/"}


def test_collapse_merges_existing_parent_delete():
    """If the parent is also delete, merge its stats with collapsed children."""
    entries = [
        _make_entry("data/", coldline_count=10, coldline_bytes=100),
        _make_entry("data/a/", coldline_count=50, coldline_bytes=500),
        _make_entry("data/b/", coldline_count=40, coldline_bytes=400),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 1
    assert delete_entries[0].prefix == "data/"
    assert delete_entries[0].coldline_count == 100  # 10 + 50 + 40
    assert delete_entries[0].coldline_bytes == 1000  # 100 + 500 + 400


def test_collapse_different_buckets_independent():
    """Entries from different buckets should not collapse together."""
    entries = [
        _make_entry("data/a/", bucket="marin-us-central1"),
        _make_entry("data/b/", bucket="marin-us-central2"),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 2


def test_collapse_preserves_matched_rule():
    """Collapsed parent should get the most common matched_rule."""
    entries = [
        _make_entry("data/a/", matched_rule="rule_a%"),
        _make_entry("data/b/", matched_rule="rule_b%"),
        _make_entry("data/c/", matched_rule="rule_a%"),
    ]
    result = _collapse_deletions(entries)
    delete_entries = [e for e in result if e.status == "delete"]
    assert len(delete_entries) == 1
    assert delete_entries[0].matched_rule == "rule_a%"


def test_storage_class_breakdown():
    entry = DirEntry(
        bucket="test",
        prefix="x/",
        status="delete",
        matched_rule="r%",
        standard_count=10,
        standard_bytes=100,
        nearline_count=0,
        nearline_bytes=0,
        coldline_count=5,
        coldline_bytes=50,
        archive_count=3,
        archive_bytes=30,
    )
    assert entry.storage_class_breakdown == "STANDARD:10;COLDLINE:5;ARCHIVE:3"
    assert entry.object_count == 18
    assert entry.total_bytes == 180


def test_depth():
    assert _make_entry("a/").depth == 1
    assert _make_entry("a/b/").depth == 2
    assert _make_entry("a/b/c/").depth == 3
