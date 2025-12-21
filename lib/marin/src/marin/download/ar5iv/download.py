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

"""
Download and process Ar5iv dataset from a zip file.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=10GB \
    lib/marin/src/marin/download/ar5iv/download.py \
    --input_path gs://bucket/ar5iv.zip \
    --output_path gs://bucket/output
"""

import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass

import draccus
import fsspec
from zephyr import Dataset, execute
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)


@dataclass
class DownloadConfig:
    input_path: str
    output_path: str
    max_files: int | None = None  # Maximum number of shards to process


def process_shard(shard_task: dict) -> dict:
    """
    Process a single shard by extracting its files from the zip in GCS and uploading the merged JSONL.

    Args:
        shard_task: Dict with keys 'input_path', 'output_path', 'shard_id', 'file_list'
    """
    input_path = shard_task["input_path"]
    output_path = shard_task["output_path"]
    shard_id = shard_task["shard_id"]
    file_list = shard_task["file_list"]
    gcs_path = f"{output_path}/{shard_id}.jsonl.gz"

    with fsspec.open(str(input_path), "rb") as f:
        with zipfile.ZipFile(f) as zf:
            with atomic_rename(gcs_path) as temp_path, fsspec.open(temp_path, "wt", compression="gzip") as out_f:
                for filename in file_list:
                    with zf.open(filename, "r") as file_handle:
                        content = file_handle.read()
                        record = {
                            "filename": filename,
                            "format": "html",
                            "content": content.decode("utf-8", errors="replace"),
                        }
                        print(json.dumps(record), file=out_f)

            logger.info(f"Shard {shard_id} with {len(file_list)} files uploaded to {gcs_path}")
            return {"shard_id": shard_id, "num_files": len(file_list), "output_path": gcs_path}


def download(cfg: DownloadConfig) -> None:
    """
    Download and process Ar5iv dataset from a zip file in GCS.

    This function can be called by the executor framework or used standalone.
    """
    logger.info("Starting transfer of Ar5iv dataset...")
    logger.info(f"Source: {cfg.input_path}")

    # Use fsspec+zipfile to list all files
    with fsspec.open(str(cfg.input_path), "rb") as f:
        with zipfile.ZipFile(f) as zf:
            all_files = zf.infolist()

            # Group by shard directory
            # We assume structure: something like: shard_id/.../file
            # shard_id is derived from the second last component if files are nested.
            # Adjust as needed if directory structure differs.
            shard_dict = defaultdict(list)
            for info in all_files:
                if info.is_dir():
                    continue
                # E.g. path might look like: "003/something.html"
                # Extract shard_id from the directory:
                # Split by "/" and take the first part if we assume structure {shard_id}/file
                parts = info.filename.strip("/").split("/")
                if len(parts) < 2:
                    # File at root level - decide how to handle this case.
                    # If no directory structure is given, skip or treat differently.
                    continue
                shard_id = parts[-2]  # get the second-last directory as shard_id
                shard_dict[shard_id].append(info.filename)

            # Apply max_files limit if provided
            shard_ids = list(shard_dict.keys())
            if cfg.max_files is not None:
                shard_ids = shard_ids[: cfg.max_files]

            logger.info(f"Found {len(shard_ids)} shards to process.")

            # Build task list for each shard
            shard_tasks = []
            for shard_id in shard_ids:
                shard_tasks.append(
                    {
                        "input_path": cfg.input_path,
                        "output_path": cfg.output_path,
                        "shard_id": shard_id,
                        "file_list": shard_dict[shard_id],
                    }
                )

    # Execute pipeline with zephyr
    pipeline = (
        Dataset.from_list(shard_tasks)
        .map(process_shard)
        .write_jsonl(f"{cfg.output_path}/.metrics/part-{{shard:05d}}.jsonl", skip_existing=True)
    )
    execute(pipeline)

    logger.info("Transfer completed successfully!")


@draccus.wrap()
def main(cfg: DownloadConfig) -> None:
    """CLI entrypoint for downloading and processing Ar5iv dataset."""
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    download(cfg)
