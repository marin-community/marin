import argparse
import asyncio
import json
import logging
import os
import shlex

import toml
from ray.job_submission import JobSubmissionClient

from marin.run.vars import ENV_VARS, PIP_DEPS, REMOTE_DASHBOARD_URL

# Setup logger
logger = logging.getLogger("ray")


def get_dependencies_from_toml(toml_file: str) -> list:
    """Extract dependencies from the pyproject.toml file and return them as a list."""
    try:
        parsed_toml = toml.load(toml_file)
        dependencies = parsed_toml.get("project", {}).get("dependencies", [])
        logger.info(f"Dependencies extracted: {dependencies}")
        return dependencies
    except FileNotFoundError:
        logger.error(f"File {toml_file} not found.")
        return []
    except toml.TomlDecodeError:
        logger.error(f"Failed to parse {toml_file}.")
        return []


async def submit_and_track_job(entrypoint: str, dependencies: list, env_vars: dict, no_wait: bool):
    """Submit a job to Ray and optionally track logs."""

    current_dir = os.getcwd()
    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)

    # Submit the job with runtime environment and entrypoint
    submission_id = client.submit_job(
        entrypoint=entrypoint, runtime_env={"pip": dependencies, "working_dir": current_dir, "env_vars": env_vars}
    )
    logger.info(f"Job submitted with ID: {submission_id}")

    if no_wait:
        return

    # Stream logs asynchronously
    async for lines in client.tail_job_logs(submission_id):
        print(lines, end="")


def main():
    """Parse command-line arguments and submit the job."""
    parser = argparse.ArgumentParser(description="Submit Ray jobs using the command-line.")
    parser.add_argument("--no_wait", action="store_true", help="Do not wait for the job to finish.")
    parser.add_argument("--env_vars", type=str, help="Environment variables to set for the job (JSON format).")
    parser.add_argument("--pip_deps", type=list, help="List of pip dependencies to install before running.")
    parser.add_argument("cmd", help="The command to run in the Ray cluster.", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Combine the remaining arguments to form the full command
    full_cmd = " ".join(shlex.quote(arg) for arg in args.cmd).strip()
    if not full_cmd.startswith("--"):
        logger.error("Command must start with '--'.")
        exit(1)
    full_cmd = full_cmd[2:]

    # Load and merge environment variables
    env_vars = {}
    if args.env_vars:
        try:
            env_vars = json.loads(args.env_vars)
        except json.JSONDecodeError:
            logger.error("Invalid JSON format for --env_vars.")
            exit(1)

    env_vars = {**ENV_VARS, **env_vars}

    # Convert pyproject.toml to requirements.txt before submission
    pyproject_toml = "pyproject.toml"
    dependencies = get_dependencies_from_toml(pyproject_toml)
    dependencies += PIP_DEPS

    # Submit the job and track it asynchronously
    asyncio.run(submit_and_track_job(full_cmd, dependencies, env_vars, args.no_wait))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
