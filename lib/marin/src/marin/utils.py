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
import gzip
import json
import logging
import os
import posixpath
import random
import re
import shutil
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, TypeVar

import braceexpand
import datasets
import fsspec
import pandas as pd
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


def fsspec_cp(source_path: str, target_path: str) -> None:
    """
    Copies source file to target path.

    Args:
        source_path (str): The path of the file to copy.
        target_path (str): The target path.
    """

    fs = fsspec.core.get_fs_token_paths(target_path, mode="wb")[0]
    fs.put(source_path, target_path)


def _fsspec_is_local(fs: fsspec.AbstractFileSystem) -> bool:
    protocol = getattr(fs, "protocol", "file")
    if isinstance(protocol, str):
        return protocol == "file"
    return "file" in protocol


def _fsspec_join(fs: fsspec.AbstractFileSystem, *parts: str) -> str:
    if _fsspec_is_local(fs):
        return os.path.join(*parts)
    return posixpath.join(*parts)


def _fsspec_dirname(fs: fsspec.AbstractFileSystem, path: str) -> str:
    if _fsspec_is_local(fs):
        return os.path.dirname(path)
    return posixpath.dirname(path)


def _fsspec_relpath(fs: fsspec.AbstractFileSystem, path: str, start: str) -> str:
    if _fsspec_is_local(fs):
        return os.path.relpath(path, start)
    return posixpath.relpath(path, start)


def fsspec_copyfile_between_fs(
    *,
    fs_in: fsspec.AbstractFileSystem,
    src: str,
    fs_out: fsspec.AbstractFileSystem,
    dst: str,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Copy a single file between fsspec filesystems.

    Uses server-side copy when possible; otherwise streams bytes through the current process.
    """
    if _fsspec_is_local(fs_in) and _fsspec_is_local(fs_out):
        shutil.copy2(src, dst)
        return

    if type(fs_in) is type(fs_out):
        fs_out.copy(src, dst)
        return

    with fs_in.open(src, "rb") as r, fs_out.open(dst, "wb") as w:
        while True:
            chunk = r.read(chunk_size_bytes)
            if not chunk:
                break
            w.write(chunk)


def fsspec_copy_path_into_dir(
    *,
    src_path: str,
    dst_path: str,
    fs_in: fsspec.AbstractFileSystem | None = None,
    fs_out: fsspec.AbstractFileSystem | None = None,
    chunk_size_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Copy a file or directory into `dst_path` on the destination filesystem.

    If `src_path` is a directory, copies the full tree into `dst_path/<relpath>`.
    If `src_path` is a file, copies it into `dst_path/<basename(src_path)>`.
    """
    if fs_in is None:
        fs_in, src_root = fsspec.core.url_to_fs(src_path)
    else:
        if "://" in src_path:
            _, src_root = fsspec.core.url_to_fs(src_path)
        else:
            src_root = src_path

    if fs_out is None:
        fs_out, dst_root = fsspec.core.url_to_fs(dst_path)
    else:
        if "://" in dst_path:
            _, dst_root = fsspec.core.url_to_fs(dst_path)
        else:
            dst_root = dst_path

    if not fs_in.exists(src_root):
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    fs_out.makedirs(dst_root, exist_ok=True)

    if _fsspec_is_local(fs_in) and _fsspec_is_local(fs_out):
        if os.path.isdir(src_root):
            shutil.copytree(src_root, dst_root, dirs_exist_ok=True)
            return
        shutil.copy2(src_root, _fsspec_join(fs_out, dst_root, os.path.basename(src_root)))
        return

    if fs_in.isdir(src_root):
        for src in fs_in.find(src_root):
            if fs_in.isdir(src):
                continue
            rel = _fsspec_relpath(fs_in, src, src_root)
            dst = _fsspec_join(fs_out, dst_root, rel)
            parent = _fsspec_dirname(fs_out, dst)
            fs_out.makedirs(parent, exist_ok=True)
            fsspec_copyfile_between_fs(
                fs_in=fs_in,
                src=src,
                fs_out=fs_out,
                dst=dst,
                chunk_size_bytes=chunk_size_bytes,
            )
        return

    dst = _fsspec_join(fs_out, dst_root, os.path.basename(src_root))
    parent = _fsspec_dirname(fs_out, dst)
    fs_out.makedirs(parent, exist_ok=True)
    fsspec_copyfile_between_fs(
        fs_in=fs_in,
        src=src_root,
        fs_out=fs_out,
        dst=dst,
        chunk_size_bytes=chunk_size_bytes,
    )


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


def get_gcs_path(file_path):
    """
    Get the GCS path. If a path starts from gs:// then it returns the path as it is. else appends gs:// to the path.
    """
    if file_path.startswith("gs://"):
        return file_path
    return f"gs://{file_path}"


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


def is_in_ci() -> bool:
    """
    Check if the code is running in a CI environment.

    Returns:
        bool: True if running in CI, False otherwise.
    """
    return "CI" in os.environ


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


def parquet_to_jsonl_gz(input_path: str, docs_dir: str, text_field: str = "text") -> None:
    """
    Convert Parquet files to gzip-compressed JSONL format for Dolma deduplication.

    This function reads Parquet files and converts them to the JSONL.gz format required
    by the Dolma deduplication pipeline. It handles both single Parquet files and
    directories containing multiple Parquet files.

    Args:
        input_path (str): Path to a single Parquet file or directory containing Parquet files.
                         If a directory, recursively searches for all .parquet files.
        docs_dir (str): Output directory where converted JSONL.gz files will be written.
        text_field (str, optional): Name of the column in Parquet files containing the text
                                   content. Defaults to "text".

    Returns:
        None

    Raises:
        None: Errors are logged but do not raise exceptions to allow processing to continue.

    Notes:
        - Creates the output directory if it doesn't exist
        - Skips empty Parquet files with a warning
        - Skips files missing the specified text_field with an error
        - Generates synthetic IDs for records missing an 'id' field
        - Converts the text_field column to a 'text' field in the output JSONL
        - Writes gzip-compressed JSONL files with .jsonl.gz extension
        - Uses the original Parquet filename (without .parquet extension) for output files
    """
    os.makedirs(docs_dir, exist_ok=True)
    # find all Parquet files
    if input_path.endswith(".parquet"):
        parquet_files = [input_path]
    else:
        path_to_glob = os.path.join(input_path, "**/*.parquet")
        parquet_files = fsspec_glob(path_to_glob)
    for pq in parquet_files:
        try:
            df = pd.read_parquet(pq)
        except Exception as e:
            logger.error(f"Failed to read parquet file {pq}: {e}")
            continue

        # Skip empty Parquet files gracefully
        if df.empty:
            print(f"Parquet file {pq} contains 0 rows, skipping conversion", flush=True)
            continue
        out_name = os.path.splitext(os.path.basename(pq))[0] + ".jsonl.gz"
        out_path = os.path.join(docs_dir, out_name)

        print(f"Converting {pq} with columns: {list(df.columns)}", flush=True)

        if text_field not in df.columns:
            logger.error(f"Parquet file {pq} missing '{text_field}' field, skipping this file")
            continue

        with gzip.open(out_path, "wt") as f:
            for rec in df.to_dict(orient="records"):
                rec["text"] = rec[text_field]

                if "id" not in rec:
                    # Generate a synthetic ID if missing
                    logger.warning(f"Adding synthetic id to {pq}")
                    rec["id"] = f"synthetic_{hash(str(rec))}"

                f.write(json.dumps(rec) + "\n")
