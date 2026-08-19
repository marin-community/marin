# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import aiohttp
import pytest
from rigging.filesystem.s3_errors import is_transient_s3_error


class _S3ResponseError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientConnectionError(),
        aiohttp.ClientPayloadError(),
        TimeoutError(),
        _S3ResponseError("SlowDown"),
        OSError("specified parts could not be found"),
        # Polars scans s3:// through Rust object_store, which raises a plain
        # OSError whose only signal is the message.
        OSError(
            "object-store error: Generic S3 error: Error performing HEAD "
            "http://bucket.example/c0000.parquet in 274ms, after 2 retries, max_retries: 2, "
            "retry_timeout: 10s  - HTTP error: error sending request (path: s3://bucket/c0000.parquet)"
        ),
        OSError("object-store error: connection closed before message completed"),
    ],
)
def test_is_transient_s3_error_recognizes_incomplete_responses(error):
    assert is_transient_s3_error(error)


@pytest.mark.parametrize(
    "error",
    [
        _S3ResponseError("AccessDenied"),
        ValueError("invalid key"),
        OSError("object-store error: Generic S3 error: AccessDenied"),
    ],
)
def test_is_transient_s3_error_rejects_permanent_failures(error):
    assert not is_transient_s3_error(error)
