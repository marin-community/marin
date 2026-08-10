# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import datasets
import fsspec
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


def _hf_should_retry(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.HTTPError):
        # HfHubHTTPError subclasses HTTPError; retry it on unknown status because the
        # hub SDK can raise without an attached response on transient failures.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is None:
            return isinstance(exc, HfHubHTTPError)
        return status == 429 or status >= 500
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
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
