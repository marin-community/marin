import os
import re
import fsspec
from typing import Literal


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


def fsspec_rm(file_path):
    """
    Check if a file exists in a fsspec filesystem. If it exists, remove it.

    Args:
        file_path (str): The path of the file

    Returns:
        bool: True if the file exists, False otherwise.
    """

    # Use fsspec to check if the file exists
    fs = fsspec.core.url_to_fs(file_path)[0]
    if fs.exists(file_path):
        try:
            fs.rm(file_path)
        except FileNotFoundError as e:
            print(f"Error removing the file: {e}. Likely caused by the race condition and file is already removed.")
        finally:
            return True
    return False


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
    
    subdirectories = [join_protocol(subdir['name']) for subdir in subdirectories if subdir['type'] == 'directory']
    return subdirectories

def fsspec_dir_only_contains_files(dir_path):
    """
    Check if a directory only contains files in a fsspec filesystem.
    """
    fs, _ = fsspec.core.url_to_fs(dir_path)
    ls_res = fs.ls(dir_path, detail=True)
    if len(ls_res) == 0:
        return False
    return all(item['type'] == 'file' for item in ls_res)

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



import re

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
        raise ValueError(f"Invalid path format. It should follow either:\n"
                         f"1. gs://marin-$REGION/scratch/* (any structure after scratch)\n"
                         f"2. gs://marin-$REGION/{{documents|attributes|filtered}}/$EXPERIMENT/$DATASET/$VERSION/")
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
                                       If not provided (but `new_extension` is), the function will replace everything after the last dot.

    Returns:
        str: The rebased file path
    """

    rel_path = os.path.relpath(file_path, base_in_path)

    # Construct the output file path
    if new_extension:
        if old_extension:
            rel_path = rel_path[:rel_path.rfind(old_extension)] + new_extension
        else:
            rel_path = rel_path[:rel_path.rfind(".")] + new_extension
    result = os.path.join(base_out_path, rel_path)
    return result



def get_gcs_path(file_path):
    """
    Get the GCS path. If a path starts from gs:// then it returns the path as it is. else appends gs:// to the path.
    """
    if file_path.startswith("gs://"):
        return file_path
    return f"gs://{file_path}"


