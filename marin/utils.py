import os

import fsspec
import functools


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


def fsspec_glob(file_path):
    """
    Get a list of files in a fsspec filesystem that match a pattern.

    Args:
        file_path (str): The path of the file

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

    return [join_protocol(file) for file in fs.glob(file_path)]

def fsspec_mkdirs(dir_path, exist_ok=True):
    """
    Create a directory in a fsspec filesystem.

    Args:
        dir_path (str): The path of the directory
    """

    # Use fsspec to create the directory
    fs = fsspec.core.url_to_fs(dir_path)[0]
    fs.makedirs(dir_path, exist_ok=exist_ok)


def rebase_file_path(base_in_dir, file_path, base_out_dir, new_extension=None, old_extension=None):
    """
    Rebase a file path from one directory to another.

    Args:
        base_in_dir (str): The base directory of the input file
        file_path (str): The path of the file
        base_out_dir (str): The base directory of the output file

    Returns:
        str: The rebased file path
    """

    rel_path = os.path.relpath(file_path, base_in_dir)

    # Construct the output file path
    if new_extension:
        if old_extension:
            rel_path = rel_path[:rel_path.rfind(old_extension)] + new_extension
        else:
            rel_path = rel_path[:rel_path.rfind(".")] + new_extension
    result = os.path.join(base_out_dir, rel_path)
    return result



def get_gcs_path(file_path):
    """
    Get the GCS path. If a path starts from gs:// then it returns the path as it is. else appends gs:// to the path.
    """
    if file_path.startswith("gs://"):
        return file_path
    return f"gs://{file_path}"


