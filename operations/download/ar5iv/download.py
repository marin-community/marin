"""
ar5iv/download.py

Download script for the AR5IV dataset, provided by FAU.

Home Page: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/
"""

import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import fsspec
import ray
import requests
from tqdm import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_path: Path
    output_path: str
    chunk_size: int = 20 * 1024 * 1024  # 1MB
    max_files: int = None  # Maximum number of files to download


def get_file_size(url):
    """Get content length from headers without downloading"""
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


@ray.remote(memory=300 * 1024 * 1024 * 1024)
def download(cfg: DownloadConfig) -> None:
    if fsspec_exists(cfg.output_path):
        logger.info(f"Output path already exists: {cfg.output_path}. Skipping download.")
        return

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
        shard_dict = defaultdict(list)

        # Process and upload files
        with zipfile.ZipFile(zip_content) as zip_ref:
            file_list = [f for f in zip_ref.filelist if not f.filename.endswith("/")]

            # Sort file list based on shard_id
            file_list.sort(key=lambda x: x.filename.split("/")[-2])

            downloaded_files = fsspec_glob(f"{cfg.output_path}/*.jsonl.gz")
            downloaded_shards = set([f.split("/")[-1].split(".")[0] for f in downloaded_files])

            if cfg.max_files:
                file_list = file_list[: cfg.max_files]

            skip_shards = set([f for f in file_list if f.filename.split("/")[-2] in downloaded_shards])
            file_list = [f for f in file_list if f.filename.split("/")[-2] not in downloaded_shards]

            print(f"Skipping {len(skip_shards)} already downloaded shards, downloading {len(file_list)} shards...")
            print(f"Files to download: {file_list}")

            for file_info in file_list:
                # Extract shard id from file path
                shard_id = file_info.filename.split("/")[-2]

                if shard_id not in skip_shards:
                    shard_dict[shard_id].append(file_info)

            print(f"\nExtracting and uploading {len(file_list)} files...")

            for shard_id, shard in tqdm(shard_dict.items(), desc="Processing shards"):
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

                gcs_path = f"{cfg.output_path}/{shard_id}.jsonl.gz"
                with fsspec.open(f"gs://{gcs_path}", "wt", compression="gzip") as f:
                    for data in extracted_shard:
                        try:
                            print(json.dumps(data), file=f)
                        except Exception as e:
                            logger.exception(f"Error processing file {data['filename']}: {e}")

                logger.info(f"Shard {shard_id} with {len(extracted_shard)} files uploaded to {gcs_path}")

        zip_content.close()
        print("\nTransfer completed successfully!")

    except Exception as e:
        print(f"Error during transfer: {e}")
        raise
