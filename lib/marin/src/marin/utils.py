# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools
import hashlib
import logging
import os
import random
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar

import braceexpand
import datasets
import fsspec
import requests
import transformers
from rigging.filesystem import url_to_fs
from rigging.timing import ExponentialBackoff, retry_with_backoff
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


def load_tokenizer_with_backoff(
    tokenizer_name: str,
    *,
    context: str | None = None,
    tokenizer_kwargs: dict[str, Any] | None = None,
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


# ============================================================================
# EXECUTOR STEP FOR TOKENIZER CACHING
# ============================================================================


@dataclass(frozen=True)
class CacheTokenizerConfig:
    """Configuration for the tokenizer caching executor step."""

    tokenizer_name: str
    gcs_path: str


def _cache_tokenizer(config: CacheTokenizerConfig) -> str:
    """ExecutorStep function to cache a tokenizer to GCS.

    This is idempotent - if the tokenizer is already cached, it returns immediately.
    """
    return save_tokenizer_to_gcs(
        tokenizer_name=config.tokenizer_name,
        gcs_path=config.gcs_path,
        logger=logger,
    )


def create_cache_tokenizer_step(
    tokenizer_name: str,
    gcs_path: str,
    name_prefix: str,
):
    """Create an ExecutorStep to pre-cache a tokenizer to GCS.

    This step should run before training steps to ensure the tokenizer is
    available in GCS. Workers will then load from GCS instead of hitting
    the HuggingFace API, avoiding rate limiting.

    Args:
        tokenizer_name: HuggingFace tokenizer name (e.g., "meta-llama/Meta-Llama-3.1-8B").
        gcs_path: GCS path to cache the tokenizer (e.g., "gs://bucket/tokenizers/llama3").
        name_prefix: Experiment name prefix for the step name.

    Returns:
        ExecutorStep that caches the tokenizer to GCS.

    Example:
        cache_step = create_cache_tokenizer_step(
            tokenizer_name="meta-llama/Meta-Llama-3.1-8B",
            gcs_path="gs://marin-us-central1/raw/tokenizers/meta-llama--Meta-Llama-3.1-8B",
            name_prefix="my_experiment",
        )
        training_steps = [...]
        all_steps = [cache_step, *training_steps]
    """
    # Import here to avoid circular dependency
    from marin.execution.executor import ExecutorStep

    return ExecutorStep(
        name=f"{name_prefix}/cache_tokenizer",
        description=f"Pre-cache tokenizer {tokenizer_name} to GCS to avoid HuggingFace rate limiting",
        fn=_cache_tokenizer,
        config=CacheTokenizerConfig(
            tokenizer_name=tokenizer_name,
            gcs_path=gcs_path,
        ),
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
