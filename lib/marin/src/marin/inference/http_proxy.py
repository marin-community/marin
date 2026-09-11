# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared HTTP header policy for inference proxies."""

from collections.abc import Mapping

REQUEST_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)
RESPONSE_DROP_HEADERS = frozenset({"content-length", "connection", "keep-alive", "transfer-encoding"})


def forwardable_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop hop-by-hop headers and empty bearer credentials."""

    forwardable: dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in REQUEST_DROP_HEADERS:
            continue
        if key.lower() == "authorization" and not value.removeprefix("Bearer").strip():
            continue
        forwardable[key] = value
    return forwardable


def forwardable_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in RESPONSE_DROP_HEADERS}
