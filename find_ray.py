import argparse
import subprocess
import os
from ray.job_submission import JobSubmissionClient

from fray.v1.cluster.ray.dashboard import DashboardConfig, ray_dashboard


def list_and_stop_vlm_jobs(ray_address: str):
    """List and optionally stop VLM jobs at the given Ray address."""
    client = JobSubmissionClient(ray_address)

    jobs = client.list_jobs()
    vlm_jobs = [job for job in jobs if
                (job.job_id and "UNIFIED" in job.job_id.upper()) or
                (job.entrypoint and "UNIFIED" in job.entrypoint.upper())]

    # Only show RUNNING jobs
    running_jobs = [job for job in vlm_jobs if job.status == "RUNNING"]
    print(f"Found {len(running_jobs)} RUNNING VLM jobs:\n")
    for job in running_jobs:
        # Extract EXP_NAME from runtime_env
        job_name = 'N/A'
        if job.runtime_env and isinstance(job.runtime_env, dict):
            env_vars = job.runtime_env.get('env_vars', {})
            if isinstance(env_vars, dict):
                job_name = env_vars.get('EXP_NAME', 'N/A')
        print(f"job_id: {job.job_id}, name: {job_name}, submission_id: {job.submission_id}, Status: {job.status}")

    # Stop all RUNNING jobs
    print("\n--- Stopping jobs ---")
    env = os.environ.copy()
    env["RAY_ADDRESS"] = ray_address

    for job in running_jobs:
        # Prefer submission_id, otherwise use job_id
        stop_id = job.submission_id or job.job_id
        job_name = 'N/A'
        if job.runtime_env and isinstance(job.runtime_env, dict):
            env_vars = job.runtime_env.get('env_vars', {})
            if isinstance(env_vars, dict):
                job_name = env_vars.get('EXP_NAME', 'N/A')
        if stop_id:
            print(f"Stopping {stop_id} (name: {job_name})...")
            confirm = input(f"  Are you sure you want to stop {stop_id} (name: {job_name})? (y/n): ")
            if confirm.lower() != 'y':
                print(f"  Skipped.")
                continue
            try:
                result = subprocess.run(
                    ["ray", "job", "stop", stop_id],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                print(f"  stdout: {result.stdout.strip()}")
                if result.stderr:
                    print(f"  stderr: {result.stderr.strip()}")
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"Cannot stop job with no ID (entrypoint: {job.entrypoint})")


def main():
    parser = argparse.ArgumentParser(description="List and stop VLM jobs on a Ray cluster")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", help="Ray cluster config file (e.g., infra/marin-eu-west4.yaml)")
    group.add_argument("--address", help="Direct Ray address (e.g., http://10.164.0.176:8265)")
    args = parser.parse_args()

    if args.config:
        with ray_dashboard(DashboardConfig.from_cluster(args.config)):
            ray_address = os.environ["RAY_ADDRESS"]
            list_and_stop_vlm_jobs(ray_address)
    else:
        list_and_stop_vlm_jobs(args.address)


if __name__ == "__main__":
    main()
