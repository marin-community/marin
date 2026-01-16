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

import logging
import os
import subprocess
import time

import fsspec
import psutil
from fsspec.callbacks import TqdmCallback
from fsspec.implementations.local import LocalFileSystem

from marin.utils import fsspec_exists, fsspec_glob, fsspec_mtime

logger = logging.getLogger(__name__)


def is_remote_path(path: str) -> bool:
    """
    Checks if the given path is a remote path, e.g., Google Cloud Storage (GCS) path.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return not isinstance(fs, LocalFileSystem)


def download_from_gcs(gcs_path: str, destination_path: str) -> None:
    """
    Downloads the folder at `gcs_path` to `destination_path`,
    unless `destination_path` already exists with a complete download.

    A download is considered complete if config.json exists in the destination.
    If the directory exists but config.json is missing, the incomplete download
    is removed and re-downloaded.
    """
    import shutil

    if os.path.exists(destination_path):
        config_path = os.path.join(destination_path, "config.json")
        if os.path.exists(config_path):
            print(f"Skipping download: {destination_path} already exists with config.json.")
            return
        else:
            # Incomplete download detected - remove and re-download
            print(f"Incomplete download detected at {destination_path} (missing config.json). Removing and re-downloading.")
            shutil.rmtree(destination_path)

    print(f"Downloading {gcs_path} from GCS to {destination_path}.")
    start_time: float = time.time()
    fs = fsspec.filesystem("gcs")

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
    fs = fsspec.filesystem("gcs")
    # The slash is needed to upload the contents of the folder to `gcs_path`
    fs.put(local_path + "/", gcs_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {gcs_path}.")


def kill_process_on_port(port: int) -> None:
    for conn in psutil.net_connections(kind="inet"):
        laddr = getattr(conn, "laddr", None)
        if not laddr or getattr(laddr, "port", None) != port:
            continue
        pid = getattr(conn, "pid", None)
        if pid is None:
            continue
        try:
            proc = psutil.Process(pid)
            print(f"Killing process {proc.name()} (PID {pid}) on port {port}")
            proc.kill()
        except psutil.NoSuchProcess:
            print(f"Process {pid} no longer exists.")
        except psutil.AccessDenied:
            print(f"Access denied killing PID {pid} on port {port}.")
        except Exception as e:
            print(f"Error killing PID {pid} on port {port}: {e}")


def set_cuda_visible_devices():
    """Sets the CUDA_VISIBLE_DEVICES environment variable based on available GPUs."""
    # Run `nvidia-smi` to get the number of available GPUs
    result = subprocess.run(["nvidia-smi", "--list-gpus"], stdout=subprocess.PIPE, text=True)
    gpu_list = result.stdout.strip().split("\n")

    # Get the indices of all detected GPUs
    available_gpus = [str(i) for i in range(len(gpu_list))]

    if available_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(available_gpus)
        print(f"Auto-selected GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        print("No available GPUs found.")


def discover_checkpoints(base_path: str, initial_glob_pattern: str, is_checkpoint_dir_pattern: list[str]) -> list[str]:
    """
    Discover the checkpoints in the given path, sorted by the last modified time. (Most recent last)
    Args:
        base_path:  Fsspec Path to the directory containing the checkpoints, possibly in nested directories.
        initial_glob_pattern:  Initial glob pattern to use to find the checkpoints.
        is_checkpoint_dir_pattern:  List of patterns to check if a directory is a checkpoint directory.
    """

    def _is_checkpoint_dir(path):
        for checkpoint_dir_pattern in is_checkpoint_dir_pattern:
            if not fsspec_exists(os.path.join(path, checkpoint_dir_pattern)):
                return False
        return True

    paths = fsspec_glob(os.path.join(base_path, initial_glob_pattern))
    paths.sort(key=lambda path: fsspec_mtime(path))
    checkpoint_paths = [os.path.dirname(path) for path in paths if _is_checkpoint_dir(os.path.dirname(path))]
    return checkpoint_paths


def discover_hf_checkpoints(base_path: str):
    """
    Discover the Hugging Face checkpoints in the given path, sorted by the last modified time. (Most recent last)
    Args:
        base_path:  Fsspec Path to the directory containing the checkpoints, possibly in nested directories.
    Returns:
        List of paths to the checkpoints, sorted by the last modified time.
    """

    return discover_checkpoints(base_path, "**/config.json", ["config.json", "tokenizer_config.json"])


def discover_levanter_checkpoints(base_path: str):
    """
    Discover the Levanter checkpoints in the given path, sorted by the last modified time. (Most recent last)
    Args:
        base_path:  Fsspec Path to the directory containing the checkpoints, possibly in nested directories.
    Returns:
        List of paths to the checkpoints, sorted by the last modified time.
    """

    return discover_checkpoints(base_path, "**/metadata.json", ["metadata.json", "model"])
