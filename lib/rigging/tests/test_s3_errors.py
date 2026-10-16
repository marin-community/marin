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
    ],
)
def test_is_transient_s3_error_recognizes_incomplete_responses(error):
    assert is_transient_s3_error(error)


@pytest.mark.parametrize("error", [_S3ResponseError("AccessDenied"), ValueError("invalid key")])
def test_is_transient_s3_error_rejects_permanent_failures(error):
    assert not is_transient_s3_error(error)
