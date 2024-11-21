"""
ar5iv/download.py

Download script for the AR5IV dataset, provided by FAU.

Home Page: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/
"""

import json
import logging
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import draccus
import fsspec
import ray
import requests
from tqdm import tqdm

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_path: Path
    output_path: str
    chunk_size: int = 20 * 1024 * 1024  # 1MB
    max_files: int = None  # Maximum number of files to download


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def process_shard(shard_list, shard_idx, output_path):
    gcs_path = f"{output_path}/{shard_idx}.jsonl.gz"
    with fsspec.open(f"gs://{gcs_path}", "wt", compression="gzip") as f:
        for data in shard_list:
            try:
                print(json.dumps(data), file=f)
            except Exception as e:
                logger.exception(f"Error processing file {data['filename']}: {e}")

    logger.info(f"Shard {shard_idx} with {len(shard_list)} files uploaded to {gcs_path}")


def get_file_size(url):
    """Get content length from headers without downloading"""
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    try:
        print("Starting transfer of Ar5iv dataset...")
        print(f"Source: {cfg.input_path}")

        total_size = get_file_size(cfg.input_path)

        print(f"Downloading and extracting to {cfg.output_path}...")

        # Download zip to memory
        zip_content = BytesIO()
        with (
            fsspec.open(cfg.input_path, "rb") as source,
            tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading ZIP") as pbar,
        ):
            while True:
                chunk = source.read(cfg.chunk_size)
                if not chunk:
                    break
                zip_content.write(chunk)
                pbar.update(len(chunk))

        zip_content.seek(0)

        # Process and upload files
        with zipfile.ZipFile(zip_content) as zip_ref:
            file_list = [f for f in zip_ref.filelist if not f.filename.endswith("/")]
            print("Files in ZIP:", file_list[:10])
            # Sort file list based on shard_id
            file_list.sort(key=lambda x: x.filename.split("/")[-2])

            if cfg.max_files:
                file_list = file_list[: cfg.max_files]
                # Start of Selection
                from collections import defaultdict

                shard_dict = defaultdict(list)
                for file_info in file_list:
                    # Extract shard id from file path
                    shard_id = file_info.filename.split("/")[-2]
                    shard_dict[shard_id].append(file_info)

            print(f"\nExtracting and uploading {len(file_list)} files...")

            shard_idx = 0
            shard_futures = []

            MAX_CONCURRENT = 100
            for shard_id, shard in shard_dict.items():
                # Extract content for each file in the shard
                extracted_shard = []
                for file_info in shard:
                    with zip_ref.open(file_info.filename) as file:
                        content = file.read()

                    extracted_shard.append(
                        {
                            "filename": file_info.filename,
                            "format": "html",
                            "content": str(content),
                        }
                    )

                if len(shard_futures) >= MAX_CONCURRENT:
                    ready_refs, shard_futures = ray.wait(shard_futures, num_returns=1)
                    try:
                        ray.get(ready_refs)
                    except Exception as e:
                        logger.exception(f"Error processing shard {shard_idx}: {e}")
                        continue

                shard_futures.append(process_shard.remote(extracted_shard, shard_id, cfg.output_path))
                shard_idx += 1

            # Wait for all shards to finish
            try:
                ray.get(shard_futures)
            except Exception as e:
                logger.exception(f"Error processing shard {shard_idx}: {e}")

        zip_content.close()
        print("\nTransfer completed successfully!")

    except Exception as e:
        print(f"Error during transfer: {e}")
        raise
