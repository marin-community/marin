#!/usr/bin/env python3
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

"""Validate JSONL files by attempting to load them and reporting any corrupted files.

Usage:
    # Dry run (just report corrupted files)
    uv run python lib/marin/src/marin/run/ray_run.py \
        --cluster infra/marin-us-central1.yaml \
        -- python scripts/data/validate_jsonl_files.py \
        --path gs://marin-us-central1/raw/dolma3_pool-d37843/data/

    # With a specific glob pattern
    uv run python lib/marin/src/marin/run/ray_run.py \
        --cluster infra/marin-us-central1.yaml \
        -- python scripts/data/validate_jsonl_files.py \
        --path gs://marin-us-central1/raw/dolma3_pool-d37843/data/ \
        --glob "**/*.jsonl.zst"
"""

import argparse
import logging
import os

from zephyr import Backend, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_jsonl_file(path: str) -> dict:
    """Attempt to load a JSONL file and return validation result.

    Args:
        path: Path to the JSONL file

    Returns:
        Dict with path, status ('valid' or 'corrupted'), and error message if corrupted
    """
    from zephyr.readers import load_jsonl

    try:
        # Iterate through all records to fully validate the file
        count = 0
        for _ in load_jsonl(path):
            count += 1
        return {"path": path, "status": "valid", "record_count": count}
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Corrupted file: {path} - {error_msg}")
        return {"path": path, "status": "corrupted", "error": error_msg}


def main():
    parser = argparse.ArgumentParser(description="Validate JSONL files for corruption")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Base path to search for JSONL files (e.g., gs://marin-us-central1/raw/dolma3_pool-d37843/data/)",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="**/*.jsonl*",
        help="Glob pattern to append to path (default: **/*.jsonl*)",
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=1024,
        help="Maximum parallelism for validation (default: 1024)",
    )
    args = parser.parse_args()

    # Ensure path doesn't have trailing slash for consistent joining
    base_path = args.path.rstrip("/")

    # Use MARIN_PREFIX if path is relative
    if not base_path.startswith("gs://") and not base_path.startswith("/"):
        marin_prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
        base_path = f"{marin_prefix}/{base_path}"

    # Combine base path and glob pattern
    full_pattern = f"{base_path}/{args.glob}"

    logger.info(f"Searching for JSONL files with pattern: {full_pattern}")

    # Build and execute the validation pipeline
    pipeline = Dataset.from_files(full_pattern).map(validate_jsonl_file)

    results = list(Backend.execute(pipeline, max_parallelism=args.max_parallelism))

    # Separate valid and corrupted files
    valid_files = [r for r in results if r["status"] == "valid"]
    corrupted_files = [r for r in results if r["status"] == "corrupted"]

    # Log summary
    logger.info("=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total files checked: {len(results)}")
    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Corrupted files: {len(corrupted_files)}")

    if corrupted_files:
        logger.info("")
        logger.info("CORRUPTED FILES:")
        logger.info("-" * 80)
        for f in corrupted_files:
            logger.info(f"  {f['path']}")
            logger.info(f"    Error: {f['error']}")
        logger.info("-" * 80)

        # Also print just the paths for easy copy-paste
        logger.info("")
        logger.info("Corrupted file paths (for copy-paste):")
        for f in corrupted_files:
            print(f["path"])
    else:
        logger.info("No corrupted files found!")


if __name__ == "__main__":
    main()
