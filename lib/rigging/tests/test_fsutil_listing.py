# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the backend-agnostic delimiter-level listing primitive."""

from datetime import UTC, datetime

from rigging.fsutil.listing import iter_object_pages


class _GcsFileSystem:
    """A GCS-like filesystem whose ``ls`` returns one whole level per call."""

    protocol = ("gs", "gcs")

    def __init__(self, level):
        self._level = level

    def ls(self, path, *, detail):
        assert detail is True
        return self._level


class _PagedS3FileSystem:
    """An S3-like filesystem whose delimiter listing spans continuation tokens."""

    protocol = "s3"

    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def split_path(self, path):
        bucket, _, key = path.partition("/")
        return bucket, key, None

    def call_s3(self, method, **kwargs):
        assert method == "list_objects_v2"
        assert kwargs["Delimiter"] == "/"
        self.calls.append((kwargs["Prefix"], kwargs.get("ContinuationToken")))
        return self._responses[kwargs.get("ContinuationToken")]


def test_iter_object_pages_gcs_splits_files_from_subdirs_and_drops_self_marker():
    fs = _GcsFileSystem(
        [
            {"name": "bucket/root", "size": 0, "type": "directory"},  # prefix marker for the listed path
            {"name": "bucket/root/sub", "size": 0, "type": "directory"},
            {"name": "bucket/root/a.txt", "size": 5, "type": "file"},
        ]
    )

    pages = list(iter_object_pages(fs, "bucket/root"))

    assert len(pages) == 1
    files, subdirs = pages[0]
    assert [f["name"] for f in files] == ["bucket/root/a.txt"]
    assert subdirs == ["bucket/root/sub"]


def test_iter_object_pages_s3_yields_one_page_per_continuation_token():
    modified = datetime(2025, 1, 1, tzinfo=UTC)
    responses = {
        None: {
            "CommonPrefixes": [{"Prefix": "iris/"}],
            "Contents": [{"Key": "a.bin", "Size": 10, "LastModified": modified}],
            "NextContinuationToken": "page-2",
        },
        "page-2": {
            "Contents": [{"Key": "b.bin", "Size": 20, "LastModified": modified}],
        },
    }
    fs = _PagedS3FileSystem(responses)

    pages = list(iter_object_pages(fs, "bucket"))

    assert len(pages) == 2
    first_files, first_subdirs = pages[0]
    assert first_subdirs == ["bucket/iris/"]
    assert [(f["name"], f["size"]) for f in first_files] == [("bucket/a.bin", 10)]
    second_files, second_subdirs = pages[1]
    assert second_subdirs == []
    assert [f["name"] for f in second_files] == ["bucket/b.bin"]
    # The root key is empty, so every request carries the empty prefix and the delimiter.
    assert fs.calls == [("", None), ("", "page-2")]
