# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Classify retryable S3 and object-store transport failures."""

import aiohttp

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
    if isinstance(error, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError)):
        return True
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if code in _TRANSIENT_S3_ERROR_CODES:
            return True
    message = str(error)
    return any(fragment in message for fragment in _TRANSIENT_S3_MESSAGE_FRAGMENTS)
