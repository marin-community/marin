# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared retry-aware ``requests.Session`` factory for HTTP-based download modules."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

DEFAULT_STATUS_FORCELIST = (429, 500, 502, 503, 504)


def build_retrying_session(
    *,
    total: int = 5,
    backoff_factor: float = 1.0,
    status_forcelist: tuple[int, ...] = DEFAULT_STATUS_FORCELIST,
) -> requests.Session:
    """Return a ``requests.Session`` that retries idempotent GETs on transient HTTP errors.

    The retry-aware adapter is mounted on both ``http://`` and ``https://``.
    """
    retries = Retry(
        total=total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
