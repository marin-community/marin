import argparse
import asyncio
import json
import logging
import os
import re
import shlex

import toml
from ray.job_submission import JobSubmissionClient

from marin.run.vars import ENV_VARS, PIP_DEPS, REMOTE_DASHBOARD_URL

# Setup logger
logger = logging.getLogger("ray")


def generate_pythonpath(base_dir="submodules"):
    # List to hold all the paths
    paths = []

    if not os.path.exists(base_dir):
        logger.warning(f"Base directory {base_dir} does not exist.")
        return ""

    # Iterate through the directories inside submodules
    for submodule in os.listdir(base_dir):
        submodule_path = os.path.join(base_dir, submodule)

        # Check if it's a directory
        if os.path.isdir(submodule_path):
            # Add both submodule and submodule/src paths
            paths.append(submodule_path)
            src_path = os.path.join(submodule_path, "src")
            if os.path.isdir(src_path):
                paths.append(src_path)

    # Add "." for the current directory to make sure the imports are working properly
    # Join the paths with ':'
    pythonpath = ":".join(paths)
    return pythonpath


def get_dependencies_from_toml(toml_file: str) -> list:
    """Extract dependencies from the pyproject.toml file and return them as a list."""
    try:
        parsed_toml = toml.load(toml_file)
        dependencies = parsed_toml.get("project", {}).get("dependencies", [])
        dependencies = _remove_problematic_deps(dependencies)
        logger.info(f"Dependencies extracted: {dependencies}")
        return dependencies
    except FileNotFoundError:
        logger.error(f"File {toml_file} not found.")
        return []
    except toml.TomlDecodeError:
        logger.error(f"Failed to parse {toml_file}.")
        return []


def _remove_problematic_deps(dependencies: list[str]):
    out: list[str] = []
    # remove ray from dependencies. We do this because Ray gets mad if you try to install another version of Ray
    expr = re.compile(r"^\s*ray([^a-zA-Z0-9_]|$)")
    for dep in dependencies:
        if not expr.match(dep):
            out.append(dep)
        else:
            logger.debug(f"Skipping dependency: {dep}")
    return out


async def submit_and_track_job(entrypoint: str, dependencies: list, env_vars: dict, no_wait: bool):
    """Submit a job to Ray and optionally track logs."""

    current_dir = os.getcwd()
    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)
    runtime_dict = {"pip": dependencies, "working_dir": current_dir, "env_vars": env_vars}

    logger.info(f"Submitting job with entrypoint: {entrypoint}")
    logger.info(f"Dependencies: {json.dumps(dependencies, indent=4)}")
    logger.info(f"env_vars: {json.dumps(env_vars, indent=4)}")

    logger.info(
        f"Terminal command: \n" f"ray job submit " f"--runtime-env-json '{json.dumps(runtime_dict)}'" f" -- {entrypoint}"
    )
    # Submit the job with runtime environment and entrypoint
    submission_id = client.submit_job(entrypoint=entrypoint, runtime_env=runtime_dict)
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
    parser.add_argument(
        "--env_vars",
        "-e",
        action="append",
        nargs="+",
        metavar=("KEY", "VALUE"),
        help="Set environment variables for the job. If only a KEY is provided, "
        "the VALUE will be set to an empty string.",
    )
    parser.add_argument(
        "--pip_deps", type=lambda x: x.split(","), help="List of pip dependencies to " "install before running."
    )
    parser.add_argument("cmd", help="The command to run in the Ray cluster.", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Combine the remaining arguments to form the full command
    full_cmd = " ".join(shlex.quote(arg) for arg in args.cmd).strip()
    if not full_cmd.startswith("--"):
        logger.error("Command must start with '--'.")
        exit(1)
    full_cmd = full_cmd[2:]

    # Load and merge environment variables from multiple -e options
    env_vars = {}
    if args.env_vars:
        for item in args.env_vars:
            if len(item) > 2:
                logger.error(
                    f"Too many values provided for environment variable: {' '.join(item)}. "
                    f"Expected at most 2 (KEY VALUE)."
                )
                exit(1)
            elif len(item) == 1:
                # If only the key is provided, set its value to an empty string
                if "=" in item[0]:
                    logger.error(
                        f"Invalid key provided for environment variable: {' '.join(item)}. "
                        f"Key should not contain '='.\n\n"
                        f"You probably meant to do '-e {' '.join(item[0].split('='))}'."
                    )
                    exit(1)
                env_vars[item[0]] = ""
            elif len(item) == 2:
                # If both key and value are provided, store them as a key-value pair
                if "=" in item[0]:
                    logger.error(
                        f"Invalid key provided for environment variable: {' '.join(item)}. "
                        f"Key should not contain '='.\n\n"
                        f"You probably meant to do '-e {' '.join(item[0].split('='))}'."
                    )
                    exit(1)
                env_vars[item[0]] = item[1]

    # Now env_vars is a dictionary with the environment variables set using -e

    env_vars = {**ENV_VARS, **env_vars}

    env_vars["PYTHONPATH"] = generate_pythonpath() + ":" + env_vars.get("PYTHONPATH", "")

    # Convert pyproject.toml to requirements.txt before submission
    pyproject_toml = "pyproject.toml"
    dependencies = get_dependencies_from_toml(pyproject_toml)
    dependencies += PIP_DEPS
    dependencies += args.pip_deps if args.pip_deps else []

    # Submit the job and track it asynchronously
    asyncio.run(submit_and_track_job(full_cmd, dependencies, env_vars, args.no_wait))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
