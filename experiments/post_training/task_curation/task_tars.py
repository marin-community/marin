# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read task archives and cohort parquet payloads."""

import gzip
import io
import tarfile


def open_task_tar(payload: bytes) -> tarfile.TarFile:
    """Open a task's tar archive, gzip-decompressed when the payload is gzipped."""
    raw = gzip.decompress(payload) if payload[:2] == b"\x1f\x8b" else payload
    return tarfile.open(fileobj=io.BytesIO(raw))
