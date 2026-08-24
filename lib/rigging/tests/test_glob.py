# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

from rigging.filesystem.glob import glob_with_metadata


def test_glob_with_metadata_overlaps_pattern_requests(monkeypatch):
    barrier = threading.Barrier(2)

    class ConcurrentFilesystem:
        def glob(self, pattern, *, detail):
            assert detail is True
            barrier.wait(timeout=2)
            output = pattern.replace("*.parquet", "part.parquet")
            return {output: {"size": len(output)}}

    filesystem = ConcurrentFilesystem()

    def filesystem_for(url):
        return filesystem, url.removeprefix("s3://")

    monkeypatch.setattr("rigging.filesystem.glob.filesystem_for", filesystem_for)

    entries = glob_with_metadata(
        (
            "s3://bucket/first/*.parquet",
            "s3://bucket/second/*.parquet",
        ),
        workers=2,
    )

    assert [entry.path for entry in entries] == [
        "s3://bucket/first/part.parquet",
        "s3://bucket/second/part.parquet",
    ]
