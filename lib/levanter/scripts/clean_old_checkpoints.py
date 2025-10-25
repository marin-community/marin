# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

## Script to aid in cleaning up old checkpoints from a directory of runs. just prints directories that can be deleted.
## Doesn't delete the latest complete checkpoint or the latest one with step ending in '000'.
# You can pass this to gsutil to delete the directories, e.g.:
# python clean_old_checkpoints.py gs://my-bucket/my-dir | xargs -I {} gsutil -m rm -r {}
import os
import sys
from datetime import datetime, timezone

import fsspec


AGE = 30  # days


def is_dir_of_checkpoints(path):
    fs = fsspec.filesystem("gcs")
    # if the children are named like step-XXXXX, then it's a checkpoint directory
    children = fs.ls(path)
    return any("step-" in child for child in children)


def list_deletable_directories(base_dir, age):
    fs = fsspec.filesystem("gcs")
    run_ids = fs.ls(base_dir)

    for run_id in run_ids:
        if not is_dir_of_checkpoints(run_id):
            continue

        checkpoint_paths = fs.ls(run_id)
        max_complete_checkpoint = None
        max_000_checkpoint = None

        # remove any paths that are not checkpoint directories (i.e. they don't have step-XXXXX in their name)
        checkpoint_paths = [path for path in checkpoint_paths if "/step-" in path]

        for path in checkpoint_paths:
            checkpoint_dir = fs.ls(path)
            if os.path.join(path, "metadata.json") in checkpoint_dir:
                # Check if this checkpoint is the largest complete checkpoint
                step_number = int(path.split("-")[-1])
                if max_complete_checkpoint is None or step_number > int(max_complete_checkpoint.split("-")[-1]):
                    max_complete_checkpoint = path

                # Check if this checkpoint ends in '000' and is the largest of such
                if str(step_number).endswith("000"):
                    if max_000_checkpoint is None or step_number > int(max_000_checkpoint.split("-")[-1]):
                        max_000_checkpoint = path

        # Add all checkpoint directories except the ones we need to keep
        for path in checkpoint_paths:
            if path == max_complete_checkpoint or path == max_000_checkpoint:
                continue

            try:
                new = False
                for file in ["metadata.json", "worker-0.cert"]:
                    details = fs.ls(f"{path}/{file}", detail=True)
                    if details:
                        mtime = details[0]["mtime"]
                        this_age = (datetime.now(timezone.utc) - mtime).days
                        if this_age < age:
                            new = True
                            break

                if new:
                    continue

            except FileNotFoundError:
                pass

            yield path


# Usage example:
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List directories that can be deleted.")
    parser.add_argument("base_dir", help="The base directory to clean up.", type=str, nargs="+")
    parser.add_argument("--age", help="The age in days of the checkpoints to delete.", type=int, default=30)
    args = parser.parse_args()
    if len(sys.argv) < 2:
        print("Usage: python clean_old_checkpoints.py <base_dir>")
        sys.exit(1)
    for base_dir in args.base_dir:
        for path in list_deletable_directories(base_dir, args.age):
            print(f"gs://{path}")
