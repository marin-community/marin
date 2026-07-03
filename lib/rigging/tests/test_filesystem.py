# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for filesystem helpers."""

from pathlib import Path

import pytest
from rigging.filesystem import StoragePath, atomic_rename, prefix_join, unique_temp_path


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("s3://marin-na/marin", "s3://marin-na/marin/a/b"),
        ("s3://marin-na/marin/", "s3://marin-na/marin/a/b"),
        ("s3://marin-na/marin//", "s3://marin-na/marin/a/b"),
        ("/tmp/marin/", "/tmp/marin/a/b"),
        ("file:///tmp/marin", "file:///tmp/marin/a/b"),
        ("mirror://", "mirror://a/b"),
    ],
)
def test_prefix_join_uses_exactly_one_separator(prefix, expected):
    """Object-store keys are not normalized: a doubled ``/`` addresses a different key,
    silently splitting writers from readers (marin-community/marin#6904)."""
    assert prefix_join(prefix, "a/b") == expected


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("gs://bucket/a//b/", "gs://bucket/a/b"),
        ("s3://bucket", "s3://bucket"),
        ("mirror://", "mirror://"),
        ("mirror://a/b", "mirror://a/b"),
        ("file:///tmp//x/", "file:///tmp/x"),
        ("/tmp//marin/", "/tmp/marin"),
        ("rel/path", "rel/path"),
    ],
)
def test_storage_path_normalize_canonicalizes(raw, canonical):
    """parse -> str emits the canonical single-separator form: interior ``//`` collapsed,
    trailing ``/`` stripped, scheme and authority preserved."""
    assert StoragePath.normalize(raw) == canonical


def test_storage_path_join_uses_single_separators():
    base = StoragePath.parse("s3://marin-na/marin/")
    assert str(base / "slimpajama-6b/2026.06.28") == "s3://marin-na/marin/slimpajama-6b/2026.06.28"
    # A bare empty-authority scheme keeps its join convention (no third slash).
    assert str(StoragePath.parse("mirror://") / "a/b") == "mirror://a/b"


def test_storage_path_join_rejects_non_relative():
    base = StoragePath.parse("gs://bucket/prefix")
    with pytest.raises(ValueError, match="non-relative"):
        base / "/abs/path"
    with pytest.raises(ValueError, match="non-relative"):
        base / "gs://other/loc"


def test_storage_path_relative_to_is_structural():
    """Containment compares parsed segments, not string prefixes, so a doubled
    separator on either side cannot fork the answer (marin-community/marin#6838)."""
    base = StoragePath.parse("gs://bucket/cache")
    shard = StoragePath.parse("gs://bucket/cache//train/shard-00001.bin")
    assert shard.relative_to(base) == "train/shard-00001.bin"

    with pytest.raises(ValueError, match="not under"):
        StoragePath.parse("gs://other/cache/x").relative_to(base)


def test_unique_temp_path_produces_distinct_paths():
    """Each call to unique_temp_path returns a different path."""
    paths = {unique_temp_path("/some/output.txt") for _ in range(10)}
    assert len(paths) == 10
    for p in paths:
        assert p.startswith("/some/output.txt.tmp.")


def test_atomic_rename_uses_unique_temp_paths(tmp_path):
    """Concurrent atomic_rename calls use distinct temp paths (UUID collision avoidance)."""
    output = str(tmp_path / "out.txt")
    observed_temps = []

    for _ in range(5):
        with atomic_rename(output) as temp_path:
            observed_temps.append(temp_path)
            Path(temp_path).write_text("data")

    assert len(set(observed_temps)) == 5, "Each call should produce a unique temp path"
    for tp in observed_temps:
        assert ".tmp." in tp


def test_atomic_rename_cleans_up_on_error(tmp_path):
    """Temp file is removed when the context raises an exception."""
    output = str(tmp_path / "out.txt")

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_rename(output) as temp_path:
            Path(temp_path).write_text("bad")
            raise RuntimeError("boom")

    assert not Path(temp_path).exists()
    assert not Path(output).exists()
