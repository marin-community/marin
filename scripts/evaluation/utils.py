from typing import Dict, Optional
import os
import subprocess

import fsspec
import yaml


def authenticate_with_hf(hf_auth_token: Optional[str]) -> None:
    """Authenticates with the Hugging Face API using the given token."""
    from huggingface_hub import login

    login(token=hf_auth_token)
    print("Logged in with Hugging Face.")


def is_gcs_path(path: str) -> bool:
    """
    Checks if the given path is a Google Cloud Storage (GCS) path.
    """
    return path.startswith("gs://") or path.startswith("gcs://")


def download_from_gcs(gcs_path: str, destination_path: str) -> None:
    """
    Downloads the folder at `gcs_path` to `destination_path`.
    """
    os.makedirs(destination_path, exist_ok=True)
    fs, _ = fsspec.core.url_to_fs(gcs_path)
    fs.get(gcs_path + "/", destination_path, recursive=True)
    print(f"Downloaded {gcs_path} to {destination_path}.")


def upload_to_gcs(local_path: str, gcs_path: str) -> None:
    """
    Uploads a folder `local_path` to Google Cloud Storage (GCS).
    """
    fs = fsspec.filesystem("gcs")
    fs.put(local_path, gcs_path, recursive=True)
    print(f"Uploaded {local_path} to {gcs_path}.")


def run_bash_command(command: str, check: bool = True) -> None:
    """Runs a bash command."""
    print(command)
    subprocess.run(command, shell=True, check=check)


def write_yaml(content: Dict, output_path: str) -> None:
    """Writes the given content to a YAML file."""
    with open(output_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)
