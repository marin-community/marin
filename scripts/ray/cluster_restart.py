"""
This script is used to download the working directories and commands for all running ray jobs, so that we can restart
them after a cluster restart. Variables like `LOCAL_PATH` and `DASHBOARD_PORT` can be modified as needed.
Typical workflow:
1. python scripts/ray/cluster_restart.py --config infra/marin-us-central2.yaml --stage before
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DASHBOARD_PORT = 8265
DASHBOARD_API_URL_TEMPLATE = "http://localhost:{port}/api/jobs/"
LOCAL_PATH = "/tmp/ray_cluster_restart/working_dirs"


def start_ray_dashboard(cluster_config: str, port: int = DASHBOARD_PORT) -> subprocess.Popen:
    """
    Start the Ray Dashboard in a non-blocking subprocess.

    Args:
        cluster_config (str): Path to the Ray cluster configuration file.
        port (int): Port for the Ray Dashboard.

    Returns:
        subprocess.Popen: The process running the Ray Dashboard.
    """
    logger.info("Starting Ray Dashboard...")
    dashboard_process = subprocess.Popen(
        ["ray", "dashboard", cluster_config], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return dashboard_process


def fetch_jobs(port: int = DASHBOARD_PORT) -> List[Any]:
    """
    Fetch the list of jobs using the Ray Jobs API.

    Args:
        port (int): Port for the Ray Dashboard API.

    Returns:
         List[Any]: The response JSON containing job details for all jobs.
    """
    dashboard_api_url = DASHBOARD_API_URL_TEMPLATE.format(port=port)
    response = requests.get(dashboard_api_url)
    if response.status_code != 200:
        logger.error(f"Failed to fetch jobs: {response.status_code}")
        return []
    return response.json()


def download_working_directory(
    cluster_config: str, job_id: str, working_dir: str, remote_working_dir: str, local_path: str
) -> str:
    """
    Download the working directory by executing a remote command.

    Args:
        cluster_config (str): Path to the Ray cluster configuration file.
        job_id (str): The ID of the job.
        working_dir (str): The local directory to save the job files.
        remote_working_dir (str): The remote directory to download files from.
        local_path (str): Local path to store job data.

    Returns:
        str: Path to the downloaded directory.
    """
    dest_dir = os.path.join(local_path, job_id, working_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_dir = os.path.join(dest_dir, "")  # Add trailing slash for rsync

    rsync_command = ["ray", "rsync-down", cluster_config, remote_working_dir, dest_dir]
    subprocess.run(rsync_command, check=True)

    logger.info(f"Working directory for job {remote_working_dir} saved to {dest_dir}")
    return dest_dir


def save_runtime_env_entrypoint(job_details: Dict[str, Any], job_id: str, local_path: str) -> Dict[str, Any]:
    """
    Save the runtime environment and entrypoint for the job.

    Args:
        job_details (Dict[str, Any]): Job details from the API response.
        job_id (str): The ID of the job.
        local_path (str): Local path to store job data.

    Returns:
        Dict[str, Any]: The saved runtime environment.
    """
    runtime_env = job_details.get("runtime_env", {})
    runtime_env.pop("working_dir", None)  # Remove unnecessary fields
    runtime_env.pop("_ray_commit", None)
    runtime_env["entrypoint"] = job_details.get("entrypoint")

    env_file = os.path.join(local_path, job_id, "runtime_env.json")
    with open(env_file, "w") as f:
        json.dump(runtime_env, f, indent=4)
    return runtime_env


def resubmit_job(
    job_id: str, entrypoint: str, working_dir: str, runtime_env: Optional[Dict[str, Any]], raise_errors: bool
) -> None:
    """
    Resubmit the job using the working directory and runtime environment.

    Args:
        job_id (str): The ID of the job.
        entrypoint (str): The entrypoint command for the job.
        working_dir (str): The local working directory.
        runtime_env (Optional[Dict[str, Any]]): Runtime environment for the job.
        raise_errors (bool): Raise errors instead of just logging them.
    """
    runtime_env_args = ["--runtime-env", json.dumps(runtime_env)] if runtime_env else []

    logger.info(f"Resubmitting job {job_id}...")
    job_array = ["ray", "job", "submit", "--working-dir", working_dir, *runtime_env_args, "--", entrypoint]
    job_str = " ".join(job_array)
    logger.info(f"Submitting the job: {job_str}")

    result = subprocess.run(job_array, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info(f"Successfully resubmitted job {job_id}")
    else:
        logger.error(f"Failed to resubmit job {job_id}: {result.stderr}")
        if raise_errors:
            raise ValueError(f"Failed to resubmit job {job_id}")


def before_cluster_restart(cluster_config: str, local_path: str, raise_errors: bool) -> None:
    """
    Perform the 'before' stage actions: download job data.

    Args:
        cluster_config (str): Path to the Ray cluster configuration file.
        local_path (str): Local path to store job data.
        raise_errors (bool): Raise errors instead of just logging them.
    """
    logger.info("Fetching jobs from Ray Jobs API...")
    jobs_data = fetch_jobs()
    if not jobs_data:
        logger.info("No jobs found.")
        return

    for job_details in jobs_data:
        job_id = job_details.get("job_id")
        logger.info(f"Processing job {job_id}...")
        status = job_details.get("status")
        if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
            logger.info(f"Skipping job {job_id} with status {status}.")
            continue
        runtime_env = job_details.get("runtime_env")
        working_dir = runtime_env.get("working_dir").split("/")[-1][:-4]
        remote_working_dir = f"/tmp/ray/session_latest/runtime_resources/working_dir_files/{working_dir}/"

        try:
            download_working_directory(cluster_config, job_id, working_dir, remote_working_dir, local_path)
            save_runtime_env_entrypoint(job_details, job_id, local_path)
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            if raise_errors:
                raise e

    logger.info("All jobs backed up.")


def restart_jobs_after_restart(local_path: str, raise_errors: bool) -> None:
    """
    Perform the 'after' stage actions: resubmit jobs.

    Args:
        local_path (str): Local path where job data is stored.
    """
    backup_dir = Path(local_path)
    if not backup_dir.exists():
        logger.error("No backup data found. Run the script with `--stage before` first.")
        return

    for job_id_dir in backup_dir.iterdir():
        if not job_id_dir.is_dir():
            continue

        files = list(job_id_dir.iterdir())
        if len(files) != 2:
            logger.warning(f"Incomplete backup for job {job_id_dir.name}. Skipping.")
            continue
        working_dir = next((f for f in files if f.is_dir()), None)
        runtime_env_file = job_id_dir / "runtime_env.json"

        if not working_dir or not runtime_env_file.exists():
            logger.warning(f"Incomplete backup for job {job_id_dir.name}. Skipping.")
            continue

        with open(runtime_env_file, "r") as f:
            runtime_env = json.load(f)

        entrypoint = runtime_env.pop("entrypoint", None)
        if not entrypoint:
            logger.error(f"No entrypoint found for job {job_id_dir.name}. Skipping.")
            if raise_errors:
                raise ValueError(f"No entrypoint found for job {job_id_dir.name}.")
            continue

        resubmit_job(job_id_dir.name, entrypoint, str(working_dir), runtime_env, raise_errors)

    logger.info("All jobs resubmitted.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Ray jobs around cluster restarts.")
    parser.add_argument("--config", required=True, help="Path to the Ray cluster configuration file.")
    parser.add_argument("--local-path", default=LOCAL_PATH, help="Local path to store job data.")
    parser.add_argument("--raise-errors", action="store_true", help="Raise errors instead of " "just logging them.")
    parser.add_argument("--stage", required=True, choices=["before", "after"], help="Specify the stage.")
    args = parser.parse_args()

    dashboard_process = start_ray_dashboard(args.config)

    try:
        if args.stage == "before":
            local_path = Path(args.local_path)
            if local_path.exists():
                logger.warning(f"Local path: {local_path} is not empty. Please clear it before running the script.")
                return
            before_cluster_restart(args.config, args.local_path, args.raise_errors)
        elif args.stage == "after":
            restart_jobs_after_restart(args.local_path, args.raise_errors)
    finally:
        logger.info("Terminating Ray Dashboard process...")
        dashboard_process.terminate()
        dashboard_process.wait()
        logger.info("Ray Dashboard process terminated.")


if __name__ == "__main__":
    main()
