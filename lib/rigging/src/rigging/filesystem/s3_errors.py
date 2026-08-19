# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify retryable S3 and object-store transport failures."""

import aiohttp
import botocore.exceptions

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
#
# The last three come from Rust object_store, which Polars uses instead of
# botocore when it scans an s3:// path directly. It raises a plain OSError with
# no structured response, so its transport failures are only visible in the
# message. All three are connection-level, never a bad request.
_TRANSIENT_S3_MESSAGE_FRAGMENTS = (
    "specified parts could not be found",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
    "error sending request",
    "connection closed before message completed",
    "operation timed out",
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
