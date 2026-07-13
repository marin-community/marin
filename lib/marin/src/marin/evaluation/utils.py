# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import time

from fsspec.callbacks import TqdmCallback
from rigging.filesystem import StoragePath
from rigging.filesystem import filesystem as marin_filesystem

logger = logging.getLogger(__name__)


def download_from_gcs(gcs_path: str, destination_path: str) -> None:
    """
    Downloads the folder at `gcs_path` to `destination_path`,
    unless `destination_path` already exists.
    """
    if os.path.exists(destination_path):
        print(f"Skipping download: {destination_path} already exists.")
        return

    print(f"Downloading {gcs_path} from GCS to {destination_path}.")
    start_time: float = time.time()
    fs = marin_filesystem("gcs")

    if not fs.exists(gcs_path):
        raise FileNotFoundError(f"{gcs_path} does not exist in GCS.")

    # The slash is needed to download the contents of the folder to `destination_path`
    os.makedirs(destination_path, exist_ok=True)
    fs.get(gcs_path + "/", destination_path, recursive=True, callback=TqdmCallback())

    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {gcs_path} to {destination_path} ({elapsed_time_seconds:.2f}s).")


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Uploads a folder `local_path` to Google Cloud Storage (GCS).
    """
    print(f"Uploading {local_path}.")
    fs = marin_filesystem("gcs")
    # The slash is needed to upload the contents of the folder to `gcs_path`
    fs.put(local_path + "/", gcs_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {gcs_path}.")


def discover_hf_checkpoints(base_path: str) -> list[str]:
    """Discover Hugging Face checkpoints under ``base_path``, sorted by mtime ascending (most recent last).

    A directory counts as a checkpoint when it contains both ``config.json``
    (matched by the glob) and ``tokenizer_config.json``.
    """
    config_paths = [str(m) for m in StoragePath(os.path.join(base_path, "**/config.json")).glob()]
    config_paths.sort(key=lambda path: StoragePath(path).mtime())
    return [
        os.path.dirname(path)
        for path in config_paths
        if StoragePath(os.path.join(os.path.dirname(path), "tokenizer_config.json")).exists()
    ]
