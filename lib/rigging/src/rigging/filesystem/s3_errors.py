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
    # Every failure to send or read a request: closed connections, read timeouts and broken response streams.
    botocore.exceptions.HTTPClientError,
    botocore.exceptions.ConnectTimeoutError,
    botocore.exceptions.EndpointConnectionError,
    botocore.exceptions.ProxyConnectionError,
    ConnectionResetError,
    TimeoutError,
)

# Server-side and gateway failures; a refusal (403/404) or failed precondition (412) carries a 4xx status.
_TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})

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
        if code in _TRANSIENT_S3_ERROR_CODES or response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (
            _TRANSIENT_HTTP_STATUSES
        ):
            return True
    message = str(error)
    return any(fragment in message for fragment in _TRANSIENT_S3_MESSAGE_FRAGMENTS)


def is_transient_s3_error_code(code: str | None) -> bool:
    """Return whether an S3 response error code is transient."""
    return code in _TRANSIENT_S3_ERROR_CODES
