# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar

import braceexpand
import datasets
import fsspec
import requests
from huggingface_hub.utils import HfHubHTTPError
from rigging.filesystem import url_to_fs
from rigging.timing import ExponentialBackoff, retry_with_backoff

logger = logging.getLogger(__name__)
T = TypeVar("T")


def fsspec_exists(file_path):
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


def fsspec_glob(file_path):
    """
    Get a list of files in a fsspec filesystem that match a pattern.

    We extend fsspec glob to also work with braces, using braceexpand.

    Args:
        file_path (str): a file path or pattern, possibly with *, **, ?, or {}'s

    Returns:
        list: A list of files that match the pattern. returned files have the protocol prepended to them.
    """

    # Use fsspec to get a list of files
    fs = url_to_fs(file_path)[0]
    protocol = fsspec.core.split_protocol(file_path)[0]

    def join_protocol(file):
        if protocol:
            return f"{protocol}://{file}"
        return file

    out = []

    # glob has to come after braceexpand
    for file in braceexpand.braceexpand(file_path):
        out.extend(join_protocol(file) for file in fs.glob(file))

    return out


def fsspec_mkdirs(dir_path, exist_ok=True):
    """
    Create a directory in a fsspec filesystem.

    Args:
        dir_path (str): The path of the directory
    """

    # Use fsspec to create the directory
    fs = url_to_fs(dir_path)[0]
    fs.makedirs(dir_path, exist_ok=exist_ok)


def fsspec_isdir(dir_path):
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


def call_with_hf_backoff(
    fn: Callable[[], T],
    *,
    context: str,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
) -> T:
    """Call ``fn`` with exponential backoff tuned for HF rate limits."""
    return retry_with_backoff(
        fn,
        retryable=_hf_should_retry,
        max_attempts=max_attempts,
        backoff=ExponentialBackoff(initial=initial_delay, maximum=max_delay, factor=2.0, jitter=0.25),
        operation=context,
    )


def load_dataset_with_backoff(
    *,
    context: str,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    **dataset_kwargs: Any,
):
    return call_with_hf_backoff(
        lambda: datasets.load_dataset(**dataset_kwargs),
        context=context,
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
    )


def fsspec_size(file_path: str) -> int:
    """Get file size (in bytes) of a file on an `fsspec` filesystem."""
    fs = url_to_fs(file_path)[0]

    return fs.size(file_path)


def fsspec_mtime(file_path: str) -> datetime:
    """Get file modification time (in seconds since epoch) of a file on an `fsspec` filesystem."""
    fs = url_to_fs(file_path)[0]

    return fs.modified(file_path)


def is_path_like(path: str) -> bool:
    """Return True if path is a URL (gs://, s3://, etc.) or an existing local path.

    Use this to distinguish file paths from HuggingFace dataset/model identifiers.
    """
    protocol, _ = fsspec.core.split_protocol(path)
    if protocol is not None:
        return True
    return os.path.exists(path)


def rebase_file_path(base_in_path, file_path, base_out_path, new_extension=None, old_extension=None):
    """
    Rebase a file path from one directory to another, with an option to change the file extension.

    Args:
        base_in_path (str): The base directory of the input file
        file_path (str): The path of the file
        base_out_path (str): The base directory of the output file
        new_extension (str, optional): If provided, the new file extension to use (including the dot, e.g., '.txt')
        old_extension (str, optional): If provided along with new_extension, specifies the old extension to replace.
                                       If not provided (but `new_extension` is), the function will replace everything
                                       after the last dot.

    Returns:
        str: The rebased file path
    """

    rel_path = os.path.relpath(file_path, base_in_path)

    # Construct the output file path
    if old_extension and not new_extension:
        raise ValueError("old_extension requires new_extension to be set")

    if new_extension:
        if old_extension:
            rel_path = rel_path[: rel_path.rfind(old_extension)] + new_extension
        else:
            rel_path = rel_path[: rel_path.rfind(".")] + new_extension
    result = os.path.join(base_out_path, rel_path)
    return result


@contextmanager
def remove_tpu_lockfile_on_exit() -> Iterator[None]:
    """
    Context manager to remove the TPU lockfile on exit.

    Example:
    ```
    with remove_tpu_lockfile_on_exit():
        # do something with TPU
    ```

    """
    try:
        yield
    finally:
        _hacky_remove_tpu_lockfile()


def _hacky_remove_tpu_lockfile():
    """
    This is a hack to remove the lockfile that TPU pods create on the host filesystem.

    libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    Ordinarily a lockfile would be removed when the process exits, but a long-running worker process may not exit until
    the node is shut down. This means that the lockfile can persist across tasks. This doesn't apply to tasks that fork a
    new process to do the TPU work, but does apply to tasks that run the TPU code in the same long-running worker
    process.
    """
    try:
        os.unlink("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        pass
    except PermissionError:
        result = subprocess.run(["sudo", "rm", "-f", "/tmp/libtpu_lockfile"], capture_output=True)
        if result.returncode != 0:
            logger.error("Failed to remove lockfile: %s", result.stderr.decode(errors="replace"))


def get_directory_friendly_name(name: str) -> str:
    """Convert a huggingface repo name to a directory friendly name."""
    return name.replace("/", "--").replace(".", "-").replace("#", "-")


def asdict_excluding(obj, exclude: set[str]) -> dict:
    """
    Convert a dataclass to a dictionary, excluding specified fields.
    Useful when you have not easily serializable fields, such as `RuntimeEnv` in ResourceConfig.
    This does not check recursively for nested dataclasses- it checks only the top-level dataclass for
    the specified fields to exclude.

    Args:
        obj: The dataclass object to convert.
        exclude: A set of field names to exclude from the dictionary.

    Returns:
        A dictionary representation of the dataclass, excluding the specified fields.
    """
    if not is_dataclass(obj):
        raise ValueError("Only dataclasses are supported")

    result = {}
    for f in fields(obj):
        if f.name not in exclude:
            value = getattr(obj, f.name)
            if is_dataclass(value):
                result[f.name] = asdict_excluding(value, exclude=set())  # nested objects
            else:
                result[f.name] = value
    return result
