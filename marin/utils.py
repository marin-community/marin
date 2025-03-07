import functools
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime

import braceexpand
import fsspec

logger = logging.getLogger(__name__)


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
