import argparse
import json
import subprocess
import sys


IRIS_CONFIG = "lib/iris/examples/marin.yaml"


def run_iris_cmd(args: list[str], timeout: int = 30) -> str:
    """Run an iris CLI command and return stdout."""
    cmd = ["uv", "run", "iris", "--config", IRIS_CONFIG] + args
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout


def list_and_stop_jobs(keyword: str, state: str = "running", dry_run: bool = False):
    """List and optionally stop Iris jobs matching a keyword."""
    stdout = run_iris_cmd(["job", "list", "--state", state, "--json"])

    # Parse JSON (skip log lines before the JSON array)
    json_start = stdout.find("[")
    if json_start == -1:
        print("No jobs found.")
        return
    jobs = json.loads(stdout[json_start:])

    # Filter by keyword (case-insensitive)
    keyword_upper = keyword.upper()
    matched = [
        job for job in jobs
        if keyword_upper in job.get("job_id", "").upper()
        or keyword_upper in job.get("name", "").upper()
    ]

    if not matched:
        print(f"No {state} jobs matching '{keyword}'.")
        return

    print(f"Found {len(matched)} {state} job(s) matching '{keyword}':\n")
    for job in matched:
        job_id = job.get("job_id", "N/A")
        job_state = job.get("state", "N/A")
        task_counts = job.get("task_state_counts", {})
        print(f"  job_id: {job_id}")
        print(f"  state:  {job_state}")
        print(f"  tasks:  {task_counts}")
        print()

    if dry_run:
        print("(dry run — not stopping any jobs)")
        return

    # Stop jobs
    print("--- Stopping jobs ---")
    for job in matched:
        job_id = job.get("job_id")
        if not job_id:
            continue
        confirm = input(f"Stop {job_id}? (y/n): ")
        if confirm.lower() != "y":
            print("  Skipped.")
            continue
        try:
            cmd = ["uv", "run", "iris", "--config", IRIS_CONFIG, "job", "stop", job_id]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            print(f"  stdout: {result.stdout.strip()}")
            if result.stderr:
                # Filter out iris log lines
                stderr_lines = [
                    line for line in result.stderr.strip().split("\n")
                    if not line.startswith("I2") and not line.startswith("W2")
                ]
                if stderr_lines:
                    print(f"  stderr: {chr(10).join(stderr_lines)}")
        except Exception as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="List and stop Iris jobs by keyword")
    parser.add_argument("keyword", nargs="?", default="unified", help="Keyword to filter jobs (default: 'unified')")
    parser.add_argument("--state", default="running", help="Job state filter (default: 'running')")
    parser.add_argument("--dry-run", action="store_true", help="Only list, don't stop")
    args = parser.parse_args()

    list_and_stop_jobs(args.keyword, state=args.state, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
