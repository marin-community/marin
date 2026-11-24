# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import errno

import aiohttp
import pytest
from botocore.exceptions import ClientError, IncompleteReadError
from google.api_core.exceptions import NotFound
from rigging.filesystem.s3_errors import is_transient_s3_error, storage_error_status
from s3fs.errors import translate_boto_error


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


def _client_error(code: str, status: int) -> ClientError:
    return ClientError({"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, "GetObject")


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (_client_error("AccessDenied", 403), 403),
        (translate_boto_error(_client_error("AccessDenied", 403)), 403),
        (translate_boto_error(_client_error("BadGateway", 502)), 502),
        (NotFound("gone"), 404),
        (ConnectionResetError(errno.ECONNRESET, "Connection reset by peer"), None),
        (IncompleteReadError(actual_bytes=3, expected_bytes=9), None),
        (PermissionError(errno.EACCES, "denied"), None),
    ],
    ids=["client-error", "s3fs-refusal", "s3fs-unmapped", "gcs", "reset", "short-body", "local"],
)
def test_storage_error_status_reports_the_service_answer_through_translation(error, status):
    assert storage_error_status(error) == status
