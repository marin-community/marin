import logging
import os
import subprocess
import time

import fsspec
import psutil
import yaml
from fsspec.implementations.local import LocalFileSystem

from marin.utils import fsspec_exists, fsspec_glob, fsspec_mtime

logger = logging.getLogger(__name__)


def authenticate_with_hf(hf_auth_token: str | None) -> None:
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
    logger.info(f"Uploaded {local_path} to {gcs_path}.")


def run_bash_command(command: list[str], check: bool = True, verbose: bool = True) -> None:
    """Runs a bash command."""
    command_str: str = " ".join(command)
    # TODO: logger doesn't print with ray, need to use print
    print(f"Running command: {command_str}", flush=True)
    start_time: float = time.time()

    try:
        if verbose:
            # Use Popen for real-time output streaming when verbose is True
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output in real-time
            while True:
                stdout_line = process.stdout.readline() if process.stdout else ""
                stderr_line = process.stderr.readline() if process.stderr else ""

                if stdout_line:
                    print(f"STDOUT: {stdout_line.rstrip()}", flush=True)
                if stderr_line:
                    print(f"STDERR: {stderr_line.rstrip()}", flush=True)

                # Break if process has finished and no more output
                if process.poll() is not None and not stdout_line and not stderr_line:
                    break

            # Get return code and handle errors
            return_code = process.poll()
            elapsed_time_seconds: float = time.time() - start_time
            print(f"COMPLETED: {command_str} ({elapsed_time_seconds}s)", flush=True)

            if check and return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
        else:
            # Original behavior when verbose is False
            result = subprocess.run(command, check=check, text=True, capture_output=True)
            elapsed_time_seconds: float = time.time() - start_time
            logger.info(f"COMPLETED: {command_str} ({elapsed_time_seconds}s)")
            logger.info(f"STDOUT:\n{result.stdout}")
            logger.info(f"STDERR:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FAILED: {command_str}")
        logger.error(f"STDOUT:\n{e.stdout}")
        logger.error(f"STDERR:\n{e.stderr}")
        raise


def write_yaml(content: dict, output_path: str) -> None:
    """Writes the given content to a YAML file."""
    with open(output_path, "w") as file:
        yaml.dump(content, file, default_flow_style=False)


def kill_process_on_port(port: int) -> None:
    # Iterate over all running processes
    for proc in psutil.process_iter(["pid", "name", "connections"]):
        # Check for connections the process has
        for conn in proc.info["connections"]:
            if conn.laddr.port == port:
                try:
                    print(f"Killing process {proc.info['name']} (PID {proc.info['pid']}) on port {port}")
                    proc.kill()
                except psutil.NoSuchProcess:
                    print(f"Process {proc.info['pid']} no longer exists.")
                except Exception as e:
                    print(f"Error killing process: {e}")


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


def discover_hf_checkpoints(base_path: str):
    """
    Discover the Hugging Face checkpoints in the given path, sorted by the last modified time. (Most recent last)
    Args:
        base_path:  Fsspec Path to the directory containing the checkpoints, possibly in nested directories.
    Returns:
        List of paths to the checkpoints, sorted by the last modified time.
    """

    def _is_checkpoint_dir(path):
        return fsspec_exists(os.path.join(path, "config.json")) and fsspec_exists(
            os.path.join(path, "tokenizer_config.json")
        )

    paths = fsspec_glob(os.path.join(base_path, "**/config.json"))

    # sort by modified time
    paths.sort(key=lambda path: fsspec_mtime(path))

    # Filter out the paths that are not checkpoint directories
    checkpoint_paths = [os.path.dirname(path) for path in paths if _is_checkpoint_dir(os.path.dirname(path))]

    return checkpoint_paths
