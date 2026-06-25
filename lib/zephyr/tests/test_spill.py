# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the opaque spill row format."""

import fsspec
from zephyr.spill import SpillReader, SpillWriter


def test_spill_round_trip_through_fsspec_filesystem():
    """SpillWriter must write through fsspec, not PyArrow's native filesystem.

    Regression for #5616: ``pq.ParquetWriter(raw_path)`` resolved the path
    through PyArrow's built-in S3 client and bypassed the configured fsspec
    filesystem, so R2's ``fixed_upload_size`` (set cluster-wide via
    ``FSSPEC_S3``) was never applied and multipart uploads failed with
    ``InvalidPart``. The ``memory://`` filesystem is reachable only through
    fsspec — PyArrow's native resolver rejects the URI — so a successful round
    trip proves the write stream is fsspec-managed.
    """
    path = "memory://spill-regression/run.spill"
    items = [{"id": i, "payload": "x" * i} for i in range(50)]

    fs = fsspec.filesystem("memory")
    try:
        # A tiny row-group budget forces several chunks, exercising the
        # background writer and on-disk chunk boundaries.
        with SpillWriter(path, row_group_bytes=1024) as writer:
            for item in items:
                writer.write([item])

        assert list(SpillReader(path)) == items
    finally:
        if fs.exists(path):
            fs.rm(path)
