# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from typing import Any

import braceexpand
import datasets
import fsspec
import requests
from huggingface_hub.utils import HfHubHTTPError
from rigging.filesystem import url_to_fs
from rigging.timing import ExponentialBackoff, retry_with_backoff


def fsspec_exists(file_path: str) -> bool:
    """
    Check if a file exists in a fsspec filesystem.

    Args:
        file_path (str): The path of the file

    Returns:
        bool: True if the file exists, False otherwise.
    """

    # Use fsspec to check if the file exists
    fs = url_to_fs(file_path)[0]
    return fs.exists(file_path)


def fsspec_glob(file_path: str) -> list[str]:
    """
    Get a list of files in a fsspec filesystem that match a pattern.

    We extend fsspec glob to also work with braces, using braceexpand.

    Args:
        file_path (str): a file path or pattern, possibly with *, **, ?, or {}'s

    Returns:
        list[str]: A list of files that match the pattern. returned files have the protocol prepended to them.
    """

    # Use fsspec to get a list of files
    fs = url_to_fs(file_path)[0]
    protocol = fsspec.core.split_protocol(file_path)[0]

    def join_protocol(file: str) -> str:
        if protocol:
            return f"{protocol}://{file}"
        return file

    out: list[str] = []

    # glob has to come after braceexpand
    for file in braceexpand.braceexpand(file_path):
        out.extend(join_protocol(file) for file in fs.glob(file))

    return out


def fsspec_mkdirs(dir_path: str, exist_ok: bool = True) -> None:
    """
    Create a directory in a fsspec filesystem.

    Args:
        dir_path (str): The path of the directory
    """

    # Use fsspec to create the directory
    fs = url_to_fs(dir_path)[0]
    fs.makedirs(dir_path, exist_ok=exist_ok)


def fsspec_isdir(dir_path: str) -> bool:
    """
    Check if a path is a directory in fsspec filesystem.
    """
    fs, _ = url_to_fs(dir_path)
    return fs.isdir(dir_path)


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


def fsspec_size(file_path: str) -> int:
    """Get file size (in bytes) of a file on an `fsspec` filesystem."""
    fs = url_to_fs(file_path)[0]

    return fs.size(file_path)


def fsspec_mtime(file_path: str) -> datetime:
    """Get file modification time (in seconds since epoch) of a file on an `fsspec` filesystem."""
    fs = url_to_fs(file_path)[0]

    return fs.modified(file_path)


def fsspec_url(fs: fsspec.AbstractFileSystem, path: str) -> str:
    """Re-attach ``fs``'s protocol to a bare ``path`` so it round-trips through ``url_to_fs``.

    ``fsspec`` glob/find results drop the protocol prefix (e.g. ``gs://``), which makes them
    ambiguous to reopen on a non-local filesystem. Local paths are returned unchanged.
    """
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    if protocol in (None, "file"):
        return path
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


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
