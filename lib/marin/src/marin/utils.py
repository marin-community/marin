# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar

import braceexpand
import datasets
import datasets.features
import datasets.features.features
import fsspec
import requests
import transformers
from iris.marin_fs import url_to_fs
from huggingface_hub.utils import HfHubHTTPError

logger = logging.getLogger(__name__)


def _patch_datasets_list_feature():
    """Monkey-patch older datasets versions to support the `List` feature type.

    The `List` feature type was introduced in datasets 3.0.0. Older versions only have `Sequence`.
    This allows loading datasets created with newer versions on environments with older datasets.
    """
    # Always try to register List if it's not in _FEATURE_TYPES
    # This is more robust than version checking
    feature_types = getattr(datasets.features.features, "_FEATURE_TYPES", None)
    if feature_types is not None and "List" not in feature_types:
        # Register `List` as an alias for `Sequence`
        feature_types["List"] = datasets.features.Sequence
        logger.warning(
            f"datasets version {datasets.__version__} doesn't have 'List' feature type. "
            f"Registered 'List' as alias for 'Sequence' for compatibility."
        )
        return True
    return False


# Apply the patch immediately on import
_patch_datasets_list_feature()
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


def fsspec_rm(path: str):
    """
    Check if a file/directory exists in a fsspec filesystem. If it exists, remove it (recursively).

    Args:
        path (str): The path of the file

    Returns:
        bool: True if the file existed (and was removed or already gone), False if it never existed.
    """
    fs = url_to_fs(path)[0]
    if fs.exists(path):
        try:
            fs.rm(path, recursive=True)
        except FileNotFoundError as e:
            logger.info("File already removed (race condition): %s", e)
        return True

    return False


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


def fsspec_cpdir(dir_path: str, target_path: str) -> None:
    """
    Recursively copies all contents of dir_path to target_path.

    Args:
        dir_path (str): The path of the directory to copy.
        target_path (str): The target path.
    """

    fs = fsspec.core.get_fs_token_paths(target_path, mode="wb")[0]
    fs.put(os.path.join(dir_path, "*"), target_path, recursive=True)


_HF_RETRY_KEYWORDS = (
    "too many requests",
    "rate limit",
    "timed out",
    "timeout",
    "connection reset",
    "temporarily unavailable",
)


def _hf_should_retry(exc: Exception) -> bool:
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "status_code"):
            status = response.status_code
        if status is None:
            return True
        return status == 429 or status >= 500
    if isinstance(exc, requests.exceptions.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status == 429 or (status is not None and status >= 500)
    if isinstance(exc, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    message = str(exc).lower()
    return any(keyword in message for keyword in _HF_RETRY_KEYWORDS)


def _hf_sleep_with_jitter(delay: float, max_delay: float) -> tuple[float, float]:
    jitter = random.uniform(0.5, 1.5)
    sleep_seconds = min(delay * jitter, max_delay)
    time.sleep(sleep_seconds)
    next_delay = min(delay * 2, max_delay)
    return sleep_seconds, next_delay


def call_with_hf_backoff(
    fn: Callable[[], T],
    *,
    context: str,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    logger: logging.Logger | None = None,
) -> T:
    """Call ``fn`` with exponential backoff tuned for HF rate limits."""

    log_obj = logger or logging.getLogger(__name__)
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network failure
            retryable = _hf_should_retry(exc)
            if not retryable or attempt == max_attempts:
                raise

            sleep_seconds, delay = _hf_sleep_with_jitter(delay, max_delay)
            log_obj.warning(
                "HF request failed for %s (attempt %s/%s): %s. Retrying in %.1fs",
                context,
                attempt,
                max_attempts,
                exc,
                sleep_seconds,
            )

    raise RuntimeError(f"Exceeded max attempts ({max_attempts}) for HF request: {context}")


def load_dataset_with_backoff(
    *,
    context: str,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 120.0,
    logger: logging.Logger | None = None,
    **dataset_kwargs: Any,
):
    return call_with_hf_backoff(
        lambda: datasets.load_dataset(**dataset_kwargs),
        context=context,
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        logger=logger,
    )


def load_tokenizer_with_backoff(
    tokenizer_name: str,
    *,
    tokenizer_kwargs: dict[str, Any] | None = None,
    context: str | None = None,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    logger: logging.Logger | None = None,
):
    kwargs = tokenizer_kwargs or {}
    load_context = context or f"tokenizer={tokenizer_name}"
    return call_with_hf_backoff(
        lambda: transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs),
        context=load_context,
        max_attempts=max_attempts,
        initial_delay=initial_delay,
        max_delay=max_delay,
        logger=logger,
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


def remove_tpu_lockfile_on_exit(fn=None):
    """
    Context manager to remove the TPU lockfile on exit. Can be used as a context manager or decorator.

    Example:
    ```
    with remove_tpu_lockfile_on_exit():
        # do something with TPU
    ```

    """
    if fn is None:
        return _remove_tpu_lockfile_on_exit_cm()
    else:

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with _remove_tpu_lockfile_on_exit_cm():
                return fn(*args, **kwargs)

        return wrapper


@contextmanager
def _remove_tpu_lockfile_on_exit_cm():
    try:
        yield
    finally:
        _hacky_remove_tpu_lockfile()


def _hacky_remove_tpu_lockfile():
    """
    This is a hack to remove the lockfile that TPU pods create on the host filesystem.

    libtpu only allows one process to access the TPU at a time, and it uses a lockfile to enforce this.
    Ordinarily a lockfile would be removed when the process exits, but in the case of Ray, the process is
    a long-running daemon that doesn't typically exit until the node is shut down. This means that the lockfile
    persists across Ray tasks. This doesn't apply to tasks that fork a new process to do the TPU work, but
    does apply to tasks that run the TPU code in the same process as the Ray worker.
    """
    try:
        os.unlink("/tmp/libtpu_lockfile")
    except FileNotFoundError:
        pass
    except PermissionError:
        try:
            os.system("sudo rm -f /tmp/libtpu_lockfile")
        except Exception:
            logger.error("Failed to remove lockfile")
            pass


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
