# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify retryable S3 and object-store transport failures."""

import aiohttp
import botocore.exceptions
import google.api_core.exceptions

OBJECT_STORE_ERRORS = (
    OSError,
    aiohttp.ClientError,
    botocore.exceptions.BotoCoreError,
    botocore.exceptions.ClientError,
    google.api_core.exceptions.GoogleAPIError,
)
"""Exception families the local, S3 and GCS filesystem adapters raise when a storage operation fails."""

_TRANSIENT_S3_TRANSPORT_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    botocore.exceptions.ConnectionClosedError,
    botocore.exceptions.ConnectTimeoutError,
    botocore.exceptions.EndpointConnectionError,
    botocore.exceptions.ProxyConnectionError,
    botocore.exceptions.ReadTimeoutError,
    botocore.exceptions.ResponseStreamingError,
    TimeoutError,
)

_TRANSIENT_S3_ERROR_CODES = frozenset(
    {
        "InvalidPart",
        "InternalError",
        "ServiceUnavailable",
        "SlowDown",
        "RequestTimeout",
        "RequestTimeTooSkewed",
    }
)

# s3fs may translate a botocore ClientError into an OSError, discarding the
# structured response while preserving the service error in the message.
_TRANSIENT_S3_MESSAGE_FRAGMENTS = (
    "specified parts could not be found",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
)


def is_transient_s3_error(error: BaseException) -> bool:
    """Return whether an S3 operation can be retried after an incomplete response."""
    if isinstance(error, _TRANSIENT_S3_TRANSPORT_ERRORS):
        return True
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_S3_ERROR_CODES:
            return True
    message = str(error)
    return any(fragment in message for fragment in _TRANSIENT_S3_MESSAGE_FRAGMENTS)


def is_transient_s3_error_code(code: str | None) -> bool:
    """Return whether an S3 response error code is transient."""
    return code in _TRANSIENT_S3_ERROR_CODES


def storage_error_status(error: BaseException) -> int | None:
    """Return the HTTP status a storage service answered with, or None when it sent no response.

    Follows ``__cause__`` so an error that s3fs translated into an ``OSError`` keeps its service's status.
    """
    if isinstance(error, botocore.exceptions.ClientError):
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    elif isinstance(error, google.api_core.exceptions.GoogleAPICallError):
        status = error.code
    elif isinstance(error, aiohttp.ClientResponseError):
        status = error.status
    else:
        status = None
    if isinstance(status, int):
        return status
    return None if error.__cause__ is None else storage_error_status(error.__cause__)
