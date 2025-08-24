"""
This file checks that two directories have matching ids for each row.
This is useful especially since we store attributes separately from the original file.
We want to make sure that the attributes match up one to one with the original text, so we check
their respective IDs.

Usage:
python -m marin.validation.check_equality \
    --dir1 "gs://marin-data/processed/fineweb/fw-v1.0/md/CC-MAIN-2022-40/000_00000" \
    --dir2 "gs://marin-data/scratch/chrisc/test-fineweb/fw-v1.0/md/CC-MAIN-2022-40/000_00000"
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import fsspec

from marin.utils import fsspec_glob


def get_matching_files(dir1: str, dir2: str) -> list[tuple[str, str]]:
    """Get matching files from two directories."""
    files1 = fsspec_glob(os.path.join(dir1, "**/*.jsonl.gz"))
    files2 = fsspec_glob(os.path.join(dir2, "**/*.jsonl.gz"))
    return [(f1, f2) for f1 in files1 for f2 in files2 if f1.split("/")[-1] == f2.split("/")[-1]]


def check_file_equality(
    file_pair: tuple[str, str], require_one_to_one_mapping: bool = False
) -> tuple[str, bool, list[int]]:
    """Check if two files have matching ids for each row."""
    file1, file2 = file_pair

    mismatch_lines = []

    file1_ids = set()
    file2_ids = set()
    with fsspec.open(file1, "rt", compression="gzip") as f1, fsspec.open(file2, "rt", compression="gzip") as f2:
        for line1, line2 in zip(f1, f2, strict=False):
            if require_one_to_one_mapping and json.loads(line1)["id"] != json.loads(line2)["id"]:
                mismatch_lines.append((line1, line2))
            file1_ids.add(json.loads(line1)["id"])
            file2_ids.add(json.loads(line2)["id"])

    if require_one_to_one_mapping:
        is_equal = len(mismatch_lines) == 0
        return (file1.split("/")[-1], is_equal)
    else:
        is_equal = file1_ids == file2_ids
        return (file1.split("/")[-1], is_equal)


def main(dir1: str, dir2: str, require_one_to_one_mapping: bool = False):
    matching_files = get_matching_files(dir1, dir2)
    print(f"Found {len(matching_files)} matching files.")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_file = {
            executor.submit(check_file_equality, file_pair, require_one_to_one_mapping): file_pair
            for file_pair in matching_files
        }
        for future in as_completed(future_to_file):
            results.append(future.result())

    all_equal = all(result[1] for result in results)
    print(f"All files have matching ids: {all_equal}")

    number_of_files_with_mismatches = sum(not result[1] for result in results)
    print(f"Number of files with mismatches: {number_of_files_with_mismatches}")
    if not all_equal:
        print("Files with mismatching ids:")
        for filename, is_equal in results:
            if not is_equal:
                print(f"Mismatch exists in {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check equality of ids in matching files from two directories.")
    parser.add_argument("--dir1", help="First directory path")
    parser.add_argument("--dir2", help="Second directory path")
    parser.add_argument("--require-one-to-one-mapping", help="Require one to one mapping of ids", action="store_true")
    args = parser.parse_args()

    main(args.dir1, args.dir2, args.require_one_to_one_mapping)
