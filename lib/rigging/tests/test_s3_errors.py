# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno

import aiohttp
import pytest
from botocore.exceptions import ClientError, ConnectionClosedError, HTTPClientError
from rigging.filesystem.s3_errors import is_transient_s3_error


class _S3ResponseError(Exception):
    def __init__(self, code: str):
        self.response = {"Error": {"Code": code}}


def _client_error(code: str, status: int) -> ClientError:
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "PutObject")


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientConnectionError(),
        aiohttp.ClientPayloadError(),
        TimeoutError(),
        _S3ResponseError("SlowDown"),
        OSError("specified parts could not be found"),
        ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"),
        ConnectionClosedError(endpoint_url="https://cwobject.com"),
        HTTPClientError(error="connection reset by peer"),
        _client_error("BadGateway", 502),
        _client_error("502", 502),
        _client_error("GatewayTimeout", 504),
        _client_error("InternalServerError", 500),
    ],
)
def test_is_transient_s3_error_recognizes_incomplete_responses(error):
    assert is_transient_s3_error(error)


@pytest.mark.parametrize(
    "error",
    [
        _S3ResponseError("AccessDenied"),
        ValueError("invalid key"),
        _client_error("AccessDenied", 403),
        _client_error("InvalidAccessKeyId", 403),
        _client_error("SignatureDoesNotMatch", 403),
        _client_error("NoSuchBucket", 404),
        _client_error("PreconditionFailed", 412),
        _client_error("NotImplemented", 501),
    ],
)
def test_is_transient_s3_error_rejects_permanent_failures(error):
    assert not is_transient_s3_error(error)
