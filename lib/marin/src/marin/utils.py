# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import hashlib
import logging
import os
import random
import re
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar

import braceexpand
import datasets
import fsspec
import requests
import transformers
from huggingface_hub.utils import HfHubHTTPError

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
    fs = fsspec.core.url_to_fs(file_path)[0]
    return fs.exists(file_path)


def fsspec_rm(path: str):
    """
    Check if a file/directory exists in a fsspec filesystem. If it exists, remove it (recursively).

    Args:
        path (str): The path of the file

    Returns:
        bool: True if the file exists, False otherwise.
    """

    # Use fsspec to check if the file exists
    fs = fsspec.core.url_to_fs(path)[0]
    if fs.exists(path):
        try:
            fs.rm(path, recursive=True)
        except FileNotFoundError as e:
            print(f"Error removing the file: {e}. Likely caused by the race condition and file is already removed.")

        # TODO (@siddk) - I think you don't need the finally?
        finally:
            return True  # noqa: B012

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
    fs = fsspec.core.url_to_fs(file_path)[0]
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
    fs = fsspec.core.url_to_fs(dir_path)[0]
    fs.makedirs(dir_path, exist_ok=exist_ok)


def fsspec_get_curr_subdirectories(dir_path):
    """
    Get all subdirectories under this current directory only. Does not return the parent directory.

    Args:
        dir_path (str): The path of the directory

    Returns:
        list: A list of subdirectories.
    """
    fs, _ = fsspec.core.url_to_fs(dir_path)
    protocol = fsspec.core.split_protocol(dir_path)[0]

    # List only immediate subdirectories
    subdirectories = fs.ls(dir_path, detail=True)

    def join_protocol(path):
        return f"{protocol}://{path}" if protocol else path

    subdirectories = [join_protocol(subdir["name"]) for subdir in subdirectories if subdir["type"] == "directory"]
    return subdirectories


def fsspec_dir_only_contains_files(dir_path):
    """
    Check if a directory only contains files in a fsspec filesystem.
    """
    fs, _ = fsspec.core.url_to_fs(dir_path)
    ls_res = fs.ls(dir_path, detail=True)
    if len(ls_res) == 0:
        return False
    return all(item["type"] == "file" for item in ls_res)


def fsspec_get_atomic_directories(dir_path):
    """
    Get all directories under this directory that only contains files within them
    """
    subdirectories = []

    if fsspec_isdir(dir_path):
        for subdir in fsspec_get_curr_subdirectories(dir_path):
            if fsspec_dir_only_contains_files(subdir):
                subdirectories.append(subdir)
            else:
                subdirectories.extend(fsspec_get_atomic_directories(subdir))

    return subdirectories


def fsspec_isdir(dir_path):
    """
    Check if a path is a directory in fsspec filesystem.
    """
    fs, _ = fsspec.core.url_to_fs(dir_path)
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
    max_attempts: int = 10,
    initial_delay: float = 5.0,
    max_delay: float = 120.0,
    logger: logging.Logger | None = None,
):
    """Load a tokenizer with backoff and local file locking for distributed settings.

    When multiple workers try to load the same tokenizer, this uses:
    1. Local file lock to ensure only one process per machine downloads
    2. Random jitter to spread out requests across machines
    3. Aggressive exponential backoff for HuggingFace rate limits
    4. local_files_only=True when loading from cache to avoid API calls

    Args:
        tokenizer_name: HuggingFace tokenizer name or local path.
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained.
        context: Context string for logging.
        max_attempts: Maximum retry attempts for HuggingFace requests.
        initial_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.
        logger: Logger instance.

    Returns:
        The loaded tokenizer.
    """
    from filelock import FileLock

    kwargs = tokenizer_kwargs or {}
    load_context = context or f"tokenizer={tokenizer_name}"
    log_obj = logger or logging.getLogger(__name__)

    # Skip locking for local paths
    if os.path.exists(tokenizer_name):
        return transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)

    # Create a unique lock file for this tokenizer
    tokenizer_hash = hashlib.md5(tokenizer_name.encode()).hexdigest()[:12]
    lock_file = f"/tmp/tokenizer_{tokenizer_hash}.lock"
    success_file = f"/tmp/tokenizer_{tokenizer_hash}.success"

    # Check if another process on this machine already downloaded successfully
    # Use local_files_only=True to avoid any HuggingFace API calls
    if os.path.exists(success_file):
        log_obj.debug(f"Tokenizer {tokenizer_name} already cached locally, loading offline")
        try:
            return transformers.AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True, **kwargs)
        except Exception as e:
            # If local loading fails, the cache might be corrupted - remove marker and retry
            log_obj.warning(f"Failed to load tokenizer from local cache: {e}, will re-download")
            os.remove(success_file)

    # Add random jitter (0-30 seconds) to spread out requests across machines
    # This is the main mechanism to prevent rate limiting across distributed workers
    jitter = random.uniform(0, 30.0)
    log_obj.info(f"Adding {jitter:.1f}s jitter before loading tokenizer {tokenizer_name}")
    time.sleep(jitter)

    # Use file lock to ensure only one process per machine downloads
    with FileLock(lock_file, timeout=600):  # 10 minute timeout
        # Re-check after acquiring lock - use local_files_only=True
        if os.path.exists(success_file):
            log_obj.debug(f"Tokenizer {tokenizer_name} downloaded by another process, loading offline")
            try:
                return transformers.AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True, **kwargs)
            except Exception as e:
                log_obj.warning(f"Failed to load from local cache after lock: {e}")
                os.remove(success_file)

        # Download from HuggingFace with aggressive backoff
        log_obj.info(f"Downloading tokenizer {tokenizer_name} from HuggingFace")
        tokenizer = call_with_hf_backoff(
            lambda: transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs),
            context=load_context,
            max_attempts=max_attempts,
            initial_delay=initial_delay,
            max_delay=max_delay,
            logger=log_obj,
        )

        # Mark as successfully downloaded for other processes on this machine
        with open(success_file, "w") as f:
            f.write(tokenizer_name)
        log_obj.info(f"Tokenizer {tokenizer_name} cached successfully")

        return tokenizer


def save_tokenizer_to_gcs(
    tokenizer_name: str,
    gcs_path: str,
    *,
    tokenizer_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Save a HuggingFace tokenizer to GCS for distributed access.

    Downloads the tokenizer from HuggingFace Hub (if needed) and saves it to GCS.
    This allows distributed workers to load from GCS instead of hitting the HF API.

    Args:
        tokenizer_name: HuggingFace tokenizer name (e.g., "meta-llama/Meta-Llama-3.1-8B").
        gcs_path: GCS path to save the tokenizer (e.g., "gs://bucket/tokenizers/llama3").
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained.
        logger: Logger instance.

    Returns:
        The GCS path where the tokenizer was saved.
    """
    import tempfile

    log_obj = logger or logging.getLogger(__name__)
    kwargs = tokenizer_kwargs or {}

    # Check if already cached in GCS
    marker_path = os.path.join(gcs_path, ".tokenizer_cached")
    if fsspec_exists(marker_path):
        log_obj.info(f"Tokenizer already cached at {gcs_path}")
        return gcs_path

    log_obj.info(f"Downloading tokenizer {tokenizer_name} from HuggingFace")
    tokenizer = call_with_hf_backoff(
        lambda: transformers.AutoTokenizer.from_pretrained(tokenizer_name, **kwargs),
        context=f"save tokenizer {tokenizer_name} to GCS",
        logger=log_obj,
    )

    # Save to a temporary local directory first, then upload to GCS
    with tempfile.TemporaryDirectory() as temp_dir:
        log_obj.info(f"Saving tokenizer to temporary directory {temp_dir}")
        tokenizer.save_pretrained(temp_dir)

        # Upload all tokenizer files to GCS
        fs = fsspec.core.url_to_fs(gcs_path)[0]
        for filename in os.listdir(temp_dir):
            local_path = os.path.join(temp_dir, filename)
            remote_path = os.path.join(gcs_path, filename)
            log_obj.debug(f"Uploading {filename} to {remote_path}")
            fs.put(local_path, remote_path)

        # Write marker file to indicate successful caching
        with fsspec.open(marker_path, "w") as f:
            f.write(tokenizer_name)

    log_obj.info(f"Tokenizer {tokenizer_name} cached to {gcs_path}")
    return gcs_path


def load_tokenizer_from_gcs(
    gcs_path: str,
    *,
    tokenizer_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> transformers.PreTrainedTokenizer:
    """Load a tokenizer from a GCS path.

    Args:
        gcs_path: GCS path where the tokenizer is stored.
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained.
        logger: Logger instance.

    Returns:
        The loaded tokenizer.
    """
    import tempfile

    log_obj = logger or logging.getLogger(__name__)
    kwargs = tokenizer_kwargs or {}

    log_obj.debug(f"Loading tokenizer from {gcs_path}")

    # Download to a temporary local directory and load from there
    # This avoids issues with fsspec and transformers compatibility
    with tempfile.TemporaryDirectory() as temp_dir:
        fs = fsspec.core.url_to_fs(gcs_path)[0]

        # List and download all files
        for remote_file in fs.ls(gcs_path, detail=False):
            filename = os.path.basename(remote_file)
            if filename.startswith("."):
                continue  # Skip marker files
            local_path = os.path.join(temp_dir, filename)
            fs.get(remote_file, local_path)

        tokenizer = transformers.AutoTokenizer.from_pretrained(temp_dir, local_files_only=True, **kwargs)

    return tokenizer


def ensure_tokenizer_cached(
    tokenizer_name: str,
    cache_base_path: str,
    *,
    tokenizer_kwargs: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> str:
    """Ensure a tokenizer is cached in GCS, downloading if necessary.

    This is the main entry point for distributed tokenization. It:
    1. Computes a deterministic GCS path based on the tokenizer name
    2. Checks if the tokenizer is already cached
    3. Downloads and caches if not

    Args:
        tokenizer_name: HuggingFace tokenizer name (e.g., "meta-llama/Meta-Llama-3.1-8B").
        cache_base_path: Base GCS path for caching (e.g., "gs://bucket/tokenizers").
        tokenizer_kwargs: Additional kwargs to pass to AutoTokenizer.from_pretrained.
        logger: Logger instance.

    Returns:
        The GCS path where the tokenizer is cached.
    """
    log_obj = logger or logging.getLogger(__name__)

    # If tokenizer_name is already a GCS path, return it directly
    if tokenizer_name.startswith("gs://"):
        log_obj.debug(f"Tokenizer {tokenizer_name} is already a GCS path")
        return tokenizer_name

    # If tokenizer_name is a local path, return it directly
    if os.path.exists(tokenizer_name):
        log_obj.debug(f"Tokenizer {tokenizer_name} is a local path")
        return tokenizer_name

    # Compute a safe directory name from the tokenizer name
    # e.g., "meta-llama/Meta-Llama-3.1-8B" -> "meta-llama--Meta-Llama-3.1-8B"
    safe_name = tokenizer_name.replace("/", "--")
    gcs_path = os.path.join(cache_base_path, safe_name)

    return save_tokenizer_to_gcs(
        tokenizer_name,
        gcs_path,
        tokenizer_kwargs=tokenizer_kwargs,
        logger=log_obj,
    )


def fsspec_size(file_path: str) -> int:
    """Get file size (in bytes) of a file on an `fsspec` filesystem."""
    fs = fsspec.core.url_to_fs(file_path)[0]

    return fs.size(file_path)


def fsspec_mtime(file_path: str) -> datetime:
    """Get file modification time (in seconds since epoch) of a file on an `fsspec` filesystem."""
    fs = fsspec.core.url_to_fs(file_path)[0]

    return fs.modified(file_path)


def validate_marin_gcp_path(path: str) -> str:
    """
    Validate the given path according to the marin GCP convention.

    This function ensures that the provided path follows the required format for
    GCS paths in a specific bucket structure. The expected format is either:
    gs://marin-$REGION/scratch//* (any structure after scratch)
    or
    gs://marin-$REGION/(documents|attributes|filtered)/$EXPERIMENT/$DATASET/$VERSION/

    Parameters:
    path (str): The GCS path to validate.

    Returns:
    str: The original path if it's valid.

    Raises:
    ValueError: If the path doesn't match the expected format.
                The error message provides details on the correct structure.

    Example:
    >>> validate_marin_gcp_path("gs://marin-us-central1/documents/exp1/dataset1/v1/")
    'gs://marin-us-central1/documents/exp1/dataset1/v1/'
    >>> validate_marin_gcp_path("gs://marin-us-central1/attributes/exp1/dataset1/v1/")
    'gs://marin-us-central1/attributes/exp1/dataset1/v1/'
    >>> validate_marin_gcp_path("gs://marin-us-central1/filtered/exp1/dataset1/v1/")
    'gs://marin-us-central1/filtered/exp1/dataset1/v1/'
    >>> validate_marin_gcp_path("gs://marin-us-central1/scratch/documents/exp1/dataset1/v1/")
    'gs://marin-us-central1/scratch/documents/exp1/dataset1/v1/'
    >>> validate_marin_gcp_path("gs://marin-us-central1/scratch/decontamination/decontamination_demo.jsonl.gz")
    'gs://marin-us-central1/scratch/decontamination/decontamination_demo.jsonl.gz'
    """
    pattern = r"^gs://marin-[^/]+/(scratch/.+|(documents|attributes|filtered)/[^/]+/[^/]+/[^/]+(/.*)?$)"
    if not re.match(pattern, path):
        raise ValueError(
            "Invalid path format. It should follow either:\n"
            "1. gs://marin-$REGION/scratch/* (any structure after scratch)\n"
            "2. gs://marin-$REGION/{documents|attributes|filtered}/$EXPERIMENT/$DATASET/$VERSION/"
        )
    return path


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
    # TODO: if old_extension is not None, but new_extension is None, raise an error or warning?
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
