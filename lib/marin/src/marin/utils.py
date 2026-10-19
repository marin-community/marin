# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import datasets
import fsspec
import httpx
import requests
from huggingface_hub.utils import HfHubHTTPError
from rigging.timing import ExponentialBackoff, retry_with_backoff

_HF_RETRY_KEYWORDS = (
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "temporarily unavailable",
)


def _is_retryable_status(status: int) -> bool:
    return status == 429 or status >= 500


def _hf_should_retry(exc: Exception) -> bool:
    # huggingface_hub >= 1.0 raises httpx-based errors; datasets still uses requests for some URLs.
    if isinstance(exc, HfHubHTTPError | httpx.HTTPStatusError):
        return _is_retryable_status(exc.response.status_code)
    if isinstance(exc, requests.exceptions.HTTPError):
        return exc.response is not None and _is_retryable_status(exc.response.status_code)
    if isinstance(
        exc,
        httpx.NetworkError | httpx.TimeoutException | requests.exceptions.ConnectionError | requests.exceptions.Timeout,
    ):
        return True
    message = str(exc).lower()
    return any(keyword in message for keyword in _HF_RETRY_KEYWORDS)


def load_dataset_with_backoff(
    *,
    context: str,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    **dataset_kwargs: Any,
):
    """Call ``datasets.load_dataset`` with exponential backoff tuned for HF rate limits."""
    return retry_with_backoff(
        lambda: datasets.load_dataset(**dataset_kwargs),
        retryable=_hf_should_retry,
        max_attempts=max_attempts,
        backoff=ExponentialBackoff(initial=initial_delay, maximum=max_delay, factor=2.0, jitter=0.25),
        operation=context,
    )


def is_path_like(path: str) -> bool:
    """Return True if path is a URL (gs://, s3://, etc.) or an existing local path.

    Use this to distinguish file paths from HuggingFace dataset/model identifiers.
    """
    protocol, _ = fsspec.core.split_protocol(path)
    if protocol is not None:
        return True
    return os.path.exists(path)


def get_directory_friendly_name(name: str) -> str:
    """Convert a huggingface repo name to a directory friendly name."""
    return name.replace("/", "--").replace(".", "-").replace("#", "-")
