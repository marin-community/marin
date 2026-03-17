"""List and stop Iris jobs matching a filter (default: VLM jobs)."""

import argparse
import subprocess
import sys


def list_and_stop_jobs(config: str, filter_str: str = "vlm"):
    """List and optionally stop jobs matching filter on an Iris cluster."""
    # List running jobs as JSON
    result = subprocess.run(
        ["uv", "run", "iris", "--config", config, "job", "list", "--state", "running", "--state", "pending", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        print(f"Error listing jobs: {result.stderr.strip()}")
        sys.exit(1)

    import json

    jobs = json.loads(result.stdout)

    # Filter jobs by keyword
    filter_upper = filter_str.upper()
    matched = [
        j for j in jobs
        if filter_upper in j.get("job_id", "").upper()
        or filter_upper in json.dumps(j.get("entrypoint", {})).upper()
    ]

    if not matched:
        print(f"No running jobs matching '{filter_str}' found.")
        return

    print(f"Found {len(matched)} running job(s) matching '{filter_str}':\n")
    for j in matched:
        job_id = j.get("job_id", "N/A")
        state = j.get("state", "N/A").replace("JOB_STATE_", "").lower()
        submitted = j.get("submitted_at", {}).get("epoch_ms", "N/A")
        resources = j.get("resources", {})
        print(f"  job_id: {job_id}")
        print(f"  state:  {state}")
        print(f"  submitted: {submitted}")
        if resources:
            print(f"  resources: {json.dumps(resources)}")
        print()

    # Stop jobs
    print("--- Stopping jobs ---")
    for j in matched:
        job_id = j.get("job_id", "")
        if not job_id:
            print("  Skipping job with no ID")
            continue

        confirm = input(f"  Stop {job_id}? (y/n): ")
        if confirm.lower() != "y":
            print("  Skipped.")
            continue

        print(f"  Stopping {job_id}...")
        stop_result = subprocess.run(
            ["uv", "run", "iris", "--config", config, "job", "stop", job_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if stop_result.stdout.strip():
            print(f"  stdout: {stop_result.stdout.strip()}")
        if stop_result.stderr.strip():
            print(f"  stderr: {stop_result.stderr.strip()}")
        if stop_result.returncode != 0:
            print(f"  Stop failed (exit code {stop_result.returncode})")
        else:
            print(f"  Stopped.")


def main():
    parser = argparse.ArgumentParser(description="List and stop Iris jobs matching a filter")
    parser.add_argument("--config", default="lib/iris/examples/marin.yaml", help="Iris config file (default: lib/iris/examples/marin.yaml)")
    parser.add_argument("--filter", default="vlm", help="Filter keyword for job IDs/entrypoints (default: vlm)")
    args = parser.parse_args()

    list_and_stop_jobs(args.config, args.filter)


if __name__ == "__main__":
    main()
