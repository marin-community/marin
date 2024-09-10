import time
from typing import Dict, Optional
import os
import psutil
import subprocess

from fsspec.implementations.local import LocalFileSystem
import fsspec
import yaml


def authenticate_with_hf(hf_auth_token: Optional[str]) -> None:
    """Authenticates with the Hugging Face API using the given token."""
    from huggingface_hub import login

    login(token=hf_auth_token)
    print("Logged in with Hugging Face.")


def is_remote_path(path: str) -> bool:
    """
    Checks if the given path is a remote path, e.g., Google Cloud Storage (GCS) path.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    return not isinstance(fs, LocalFileSystem)


def download_from_gcs(gcs_path: str, destination_path: str) -> None:
    """
    Downloads the folder at `gcs_path` to `destination_path`.
    """
    print(f"Downloading {gcs_path} from GCS.")
    start_time: float = time.time()
    fs = fsspec.filesystem("gcs")
    if not fs.exists(gcs_path):
        raise FileNotFoundError(f"{gcs_path} does not exist in GCS.")

    # The slash is needed to download the contents of the folder to `destination_path`
    os.makedirs(destination_path, exist_ok=True)
    fs.get(gcs_path + "/", destination_path, recursive=True)
    elapsed_time_seconds: float = time.time() - start_time
    print(f"Downloaded {gcs_path} to {destination_path} ({elapsed_time_seconds}s).")


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Uploads a folder `local_path` to Google Cloud Storage (GCS).
    """
    print(f"Uploading {local_path}.")
    fs = fsspec.filesystem("gcs")
    # The slash is needed to upload the contents of the folder to `gcs_path`
    fs.put(local_path + "/", gcs_path, recursive=True)
    print(f"Uploaded {local_path} to {gcs_path}.")


def run_bash_command(command: str, check: bool = True) -> None:
    """Runs a bash command."""
    print(command)
    start_time: float = time.time()
    subprocess.run(command, shell=True, check=check)
    elapsed_time_seconds: float = time.time() - start_time
    print(f"Completed: {command} ({elapsed_time_seconds}s)")


def write_yaml(content: Dict, output_path: str) -> None:
    """Writes the given content to a YAML file."""
    with open(output_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)


def kill_process_on_port(port):
    # Iterate over all running processes
    for proc in psutil.process_iter(['pid', 'name', 'connections']):
        # Check for connections the process has
        for conn in proc.info['connections']:
            if conn.laddr.port == port:
                try:
                    print(f"Killing process {proc.info['name']} (PID {proc.info['pid']}) on port {port}")
                    proc.kill()
                except psutil.NoSuchProcess:
                    print(f"Process {proc.info['pid']} no longer exists.")
                except Exception as e:
                    print(f"Error killing process: {e}")
