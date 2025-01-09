"""A script to list out the files on GCS to prepare as training data.
We generate braceexpand paths for the files with the same prefix and suffix.
"""

import os
import re
from typing import Generator

import fire
from braceexpand import braceexpand
from google.cloud import storage


def get_subdirectories(bucket_name, directory_path, suffix) -> Generator[str, None, None]:
    """Given a GCS bucket name, directory path and suffix, list all the subdirectories that contain files with the given suffix."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    subdirectories = set()
    blobs = bucket.list_blobs(prefix=directory_path)
    for blob in blobs:
        if blob.name.endswith(suffix):
            subdir = os.path.dirname(blob.name)
            if subdir not in subdirectories:
                subdirectories.add(subdir)
                yield subdir


def list_files_in_subdir(bucket_name, subdir, suffix):
    """Given a GCS bucket name, subdir and suffix, list all the files in the subdirectory with the given suffix.
    And generate braceexpand paths for the files with the same prefix and suffix.
    Note that we intentionally limit this to subdirectories in order to generate braceexpand paths for the files in the same subdir.
    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=subdir)

    paths = []
    for blob in blobs:
        if blob.name.endswith(suffix):
            paths.append(f"gs://{bucket_name}/{blob.name}")

    ranges = find_number_ranges(paths)
    braceexpand_paths = []
    for start, end in ranges:
        path = os.path.join(f"gs://{bucket_name}", subdir, f"{{{start}..{end}}}{suffix}")
        braceexpand_paths.append(path)
    validate_with_braceexpand(braceexpand_paths, paths)
    return braceexpand_paths


def find_number_ranges(file_paths):
    numbers = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        match = re.search(r"^(\d+)_", file_name)
        if match:
            numbers.append(int(match.group(1)))

    numbers.sort()
    ranges = []
    start = end = numbers[0]

    for i in range(1, len(numbers)):
        if numbers[i] == end + 1:
            end = numbers[i]
        else:
            ranges.append((start, end))
            start = end = numbers[i]

    ranges.append((start, end))
    return ranges


def validate_with_braceexpand(braceexpand_paths, paths):
    be_paths = []
    for be_path in braceexpand_paths:
        be_paths.extend(list(braceexpand(be_path)))

    if set(be_paths) != set(paths):
        print("Braceexpand paths are not equal to the original paths")
        print("Braceexpand paths:")
        print(be_paths)
        print("Original paths:")
        print(paths)


def main(
    bucket_name="marin-data", directory="processed/fineweb/fw-v1.0/", suffix="_processed_html.jsonl.gz", limit=None
):
    counter = 0
    subdirectories = get_subdirectories(bucket_name, directory, suffix)
    for subdir in subdirectories:
        for braceexpand_path in list_files_in_subdir(bucket_name, subdir, suffix):
            print(f"- {braceexpand_path}")
            counter += 1
            if limit and counter >= limit:
                return


if __name__ == "__main__":
    fire.Fire(main)
