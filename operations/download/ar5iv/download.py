import json
import logging
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import fsspec
import ray
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    input_path: Path
    output_path: str
    chunk_size: int = 20 * 1024 * 1024  # 20MB - not heavily used now, but left for compatibility
    max_files: int = None  # Maximum number of shards to process


@ray.remote(memory=10 * 1024 * 1024 * 1024)
def process_shard(input_path, output_path, shard_id: str, file_list: list) -> None:
    """
    Process a single shard by extracting its files from the zip in GCS and uploading the merged JSONL.
    """
    try:
        # Open the GCS zip file again for random access
        with fsspec.open(str(input_path), "rb") as f:
            with zipfile.ZipFile(f) as zf:
                gcs_path = f"{output_path}/{shard_id}.jsonl.gz"
                success_path = f"{output_path}/{shard_id}.SUCCESS"

                # Avoid overwriting if shard already exists
                if fsspec_exists(success_path):
                    logger.info(f"Shard {shard_id} already exists at {gcs_path}, skipping...")
                    return

                with (
                    fsspec.open(gcs_path, "wt", compression="gzip") as out_f,
                    fsspec.open(success_path, "wt") as success_f,
                ):
                    for filename in file_list:
                        with zf.open(filename, "r") as file_handle:
                            content = file_handle.read()
                            record = {
                                "filename": filename,
                                "format": "html",
                                "content": str(content.decode("utf-8", errors="replace")),
                            }
                            success_record = {
                                "filename": filename,
                                "format": "html",
                            }
                            print(json.dumps(record), file=out_f)
                            print(json.dumps(success_record), file=success_f)

                logger.info(f"Shard {shard_id} with {len(file_list)} files uploaded to {gcs_path}")

    except Exception as e:
        logger.exception(f"Error processing shard {shard_id}: {e}")
        raise


@ray.remote(memory=10 * 1024 * 1024 * 1024)
def download(cfg: DownloadConfig) -> None:
    try:
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

                # Filter out shards we already processed
                downloaded_files = fsspec_glob(f"{cfg.output_path}/*.SUCCESS")
                downloaded_shards = set(f.split("/")[-1].split(".")[0] for f in downloaded_files)

                # Apply max_files limit if provided
                shard_ids = [s for s in shard_dict.keys() if s not in downloaded_shards]
                if cfg.max_files is not None:
                    shard_ids = shard_ids[: cfg.max_files]

                logger.info(f"Found {len(shard_ids)} shards to process.")

                # Launch Ray tasks for each shard
                tasks = []
                for shard_id in shard_ids:
                    files_for_shard = shard_dict[shard_id]
                    tasks.append(process_shard.remote(cfg.input_path, cfg.output_path, shard_id, files_for_shard))

                # Wait for all shards to be processed
                with tqdm(total=len(tasks), desc="Processing shards") as pbar:
                    done, pending = ray.wait(tasks, num_returns=len(tasks), timeout=None)
                    while pending:
                        done_iter, pending = ray.wait(pending, num_returns=len(pending), timeout=None)
                        done += done_iter
                        pbar.update(len(done_iter))

        logger.info("Transfer completed successfully!")

    except Exception as e:
        logger.exception(f"Error during transfer: {e}")
        raise
