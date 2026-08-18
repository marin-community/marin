# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for paged listings over standard S3 and GCS fsspec filesystems."""

from datetime import UTC, datetime

import s3fs
from gcsfs.core import GCSFileSystem
from rigging.filesystem.cross_region import CrossRegionGuardedFS
from rigging.filesystem.paged_listing import with_listing

MODIFIED = datetime(2025, 1, 1, tzinfo=UTC)


def _s3_filesystem(monkeypatch, responses, expected_delimiter):
    fs = s3fs.S3FileSystem(key="test", secret="test", skip_instance_cache=True)
    calls = []

    def call_s3(method, **kwargs):
        assert method == "list_objects_v2"
        assert kwargs["Delimiter"] == expected_delimiter
        calls.append((kwargs["Prefix"], kwargs.get("ContinuationToken")))
        return responses[kwargs.get("ContinuationToken")]

    monkeypatch.setattr(fs, "call_s3", call_s3)
    return with_listing(fs), calls


def test_s3_level_pages_yield_one_page_per_continuation_token(monkeypatch):
    responses = {
        None: {
            "CommonPrefixes": [{"Prefix": "iris/"}],
            "Contents": [{"Key": "a.bin", "Size": 10, "LastModified": MODIFIED}],
            "NextContinuationToken": "page-2",
        },
        "page-2": {
            "Contents": [{"Key": "b.bin", "Size": 20, "LastModified": MODIFIED}],
        },
    }
    fs, calls = _s3_filesystem(monkeypatch, responses, expected_delimiter="/")

    pages = list(fs.listing.level_pages("bucket"))

    assert len(pages) == 2
    first_files, first_subdirs = pages[0]
    assert first_subdirs == ["bucket/iris/"]
    assert [(f["name"], f["size"]) for f in first_files] == [("bucket/a.bin", 10)]
    second_files, second_subdirs = pages[1]
    assert second_subdirs == []
    assert [f["name"] for f in second_files] == ["bucket/b.bin"]
    # The root key is empty, so every request carries the empty prefix.
    assert calls == [("", None), ("", "page-2")]


def test_s3_flat_pages_resume_by_token_and_drop_self_marker(monkeypatch):
    responses = {
        None: {
            "Contents": [
                {"Key": "root/", "Size": 0, "LastModified": MODIFIED},  # prefix marker for the listed path
                {"Key": "root/a.bin", "Size": 10, "LastModified": MODIFIED},
            ],
            "NextContinuationToken": "page-2",
        },
        "page-2": {
            "Contents": [{"Key": "root/deep/b.bin", "Size": 20, "LastModified": MODIFIED}],
        },
    }
    fs, calls = _s3_filesystem(monkeypatch, responses, expected_delimiter="")

    pages = list(fs.listing.flat_pages("bucket/root"))

    # The flat listing is recursive: objects below sub-prefixes appear directly.
    assert [[e["name"] for e in page] for page in pages] == [["bucket/root/a.bin"], ["bucket/root/deep/b.bin"]]
    assert calls == [("root/", None), ("root/", "page-2")]


def test_s3_level_pages_preserve_an_empty_path_segment(monkeypatch):
    responses = {None: {"CommonPrefixes": [{"Prefix": "root/a//"}]}}
    fs, _ = _s3_filesystem(monkeypatch, responses, expected_delimiter="/")

    pages = list(fs.listing.level_pages("bucket/root/a/"))

    assert pages == [([], ["bucket/root/a//"])]


def test_gcs_level_pages_split_files_from_subdirs_and_drop_self_marker(monkeypatch):
    fs = GCSFileSystem(token="anon", skip_instance_cache=True)

    def call(method, template, bucket, *, prefix, delimiter, maxResults, pageToken, json_out):
        assert (bucket, prefix, delimiter, pageToken) == ("bucket", "root/", "/", None)
        return {
            "prefixes": ["root/sub/"],
            "items": [
                {"name": "root/", "size": "0"},  # prefix marker for the listed path
                {"name": "root/a.txt", "size": "5"},
            ],
        }

    monkeypatch.setattr(fs, "call", call)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: False)
    pages = list(with_listing(guarded).listing.level_pages("bucket/root"))

    assert len(pages) == 1
    files, subdirs = pages[0]
    assert [f["name"] for f in files] == ["bucket/root/a.txt"]
    assert subdirs == ["bucket/root/sub/"]


def test_gcs_flat_pages_resume_by_page_token_and_normalize_like_ls(monkeypatch):
    responses = {
        None: {
            "items": [
                {"name": "root/", "size": "0", "timeCreated": "2025-01-01T00:00:00Z"},  # self marker
                {"name": "root/a.txt", "size": "5", "storageClass": "STANDARD", "updated": "2025-01-01T00:00:00Z"},
            ],
            "nextPageToken": "page-2",
        },
        "page-2": {
            "items": [{"name": "root/deep/b.txt", "size": "7"}],
        },
    }
    fs = GCSFileSystem(token="anon", skip_instance_cache=True)
    calls = []

    def call(method, template, bucket, *, prefix, delimiter, maxResults, pageToken, json_out):
        assert (method, template, delimiter, json_out) == ("GET", "b/{}/o", None, True)
        calls.append((bucket, prefix, pageToken))
        return responses[pageToken]

    monkeypatch.setattr(fs, "call", call)

    pages = list(with_listing(fs).listing.flat_pages("bucket/root"))

    assert [[e["name"] for e in page] for page in pages] == [["bucket/root/a.txt"], ["bucket/root/deep/b.txt"]]
    first = pages[0][0]
    # Items are normalized the same way ls output is: int sizes, bucket-prefixed names.
    assert first["size"] == 5
    assert first["type"] == "file"
    assert [c[:2] for c in calls] == [("bucket", "root/"), ("bucket", "root/")]
    assert [c[2] for c in calls] == [None, "page-2"]
