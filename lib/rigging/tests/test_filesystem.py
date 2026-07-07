# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for filesystem helpers."""

from pathlib import Path

import pytest
from rigging.filesystem import (
    StoragePath,
    _bucket_from_gcs_url,
    atomic_rename,
    prefix_join,
    rebase_file_path,
    split_gcs_path,
    unique_temp_path,
)


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


@pytest.mark.parametrize(
    ("raw", "bucket", "key", "name", "parent"),
    [
        ("gs://bucket/a/b/c", "bucket", "a/b/c", "c", "gs://bucket/a/b"),
        ("gs://bucket/a//b/", "bucket", "a/b", "b", "gs://bucket/a"),
        ("gs://bucket/only", "bucket", "only", "only", "gs://bucket"),
        ("gs://bucket", "bucket", "", "", "gs://bucket"),
        ("/tmp/x/y", "", "tmp/x/y", "y", "/tmp/x"),
        ("/", "", "", "", "/"),
        ("rel/path", "", "rel/path", "path", "rel"),
    ],
)
def test_storage_path_accessors(raw, bucket, key, name, parent):
    sp = StoragePath.parse(raw)
    assert sp.bucket == bucket
    assert sp.key == key
    assert sp.name == name
    assert str(sp.parent) == parent


@pytest.mark.parametrize(
    ("uri", "bucket", "path"),
    [
        ("gs://bucket/path/to/resource", "bucket", "path/to/resource"),
        ("gs://bucket/a//b/", "bucket", "a/b"),
        ("gs://bucket", "bucket", "."),
        ("gs://bucket/", "bucket", "."),
    ],
)
def test_split_gcs_path(uri, bucket, path):
    got_bucket, got_path = split_gcs_path(uri)
    assert got_bucket == bucket
    assert got_path == Path(path)


def test_split_gcs_path_rejects_non_gcs():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        split_gcs_path("s3://bucket/x")


@pytest.mark.parametrize(
    ("url", "bucket"),
    [
        ("gs://bucket/path", "bucket"),
        ("gcs://bucket/path", "bucket"),
        ("gs://bucket", "bucket"),
        ("s3://bucket/path", None),
        ("/local/path", None),
    ],
)
def test_bucket_from_gcs_url(url, bucket):
    assert _bucket_from_gcs_url(url) == bucket


@pytest.mark.parametrize(
    ("base_in", "file_path", "base_out", "new_ext", "old_ext", "expected"),
    [
        # Extension swap under a trailing-slash output base (a common caller pattern).
        (
            "gs://b/in",
            "gs://b/in/sub/doc.jsonl.gz",
            "gs://b/out/data/",
            ".parquet",
            ".jsonl.gz",
            "gs://b/out/data/sub/doc.parquet",
        ),
        # Trailing slash on the input base collapses structurally.
        ("gs://b/in/", "gs://b/in/a/b/doc.json", "gs://b/out", ".parquet", ".json", "gs://b/out/a/b/doc.parquet"),
        # Without old_extension, everything after the last dot is replaced.
        ("gs://b/in", "gs://b/in/doc.jsonl.zst", "gs://b/out", ".parquet", None, "gs://b/out/doc.jsonl.parquet"),
        # Without old_extension and no dot in the name, new_extension is appended.
        ("gs://b/in", "gs://b/in/noext", "gs://b/out", ".txt", None, "gs://b/out/noext.txt"),
        # No extension change — pure rebase.
        ("gs://b/in", "gs://b/in/x/doc", "gs://b/out", None, None, "gs://b/out/x/doc"),
        # Local paths.
        ("/tmp/in", "/tmp/in/x/doc.jsonl", "/tmp/out", ".parquet", ".jsonl", "/tmp/out/x/doc.parquet"),
    ],
)
def test_rebase_file_path(base_in, file_path, base_out, new_ext, old_ext, expected):
    assert rebase_file_path(base_in, file_path, base_out, new_ext, old_ext) == expected


def test_rebase_file_path_rejects_file_outside_base():
    with pytest.raises(ValueError, match="not under"):
        rebase_file_path("gs://b/in", "gs://b/other/doc.jsonl", "gs://b/out")


def test_rebase_file_path_wrong_old_extension():
    with pytest.raises(ValueError, match="does not end with"):
        rebase_file_path("gs://b/in", "gs://b/in/doc.jsonl", "gs://b/out", ".parquet", ".csv")


def test_rebase_file_path_old_extension_requires_new():
    with pytest.raises(ValueError, match="requires new_extension"):
        rebase_file_path("gs://b/in", "gs://b/in/doc.jsonl", "gs://b/out", old_extension=".jsonl")


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
