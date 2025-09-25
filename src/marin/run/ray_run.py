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

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shlex
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ray.job_submission import JobSubmissionClient

from marin.run.ray_deps import build_runtime_env_for_packages
from marin.run.vars import REMOTE_DASHBOARD_URL

if TYPE_CHECKING:  # pragma: no cover - imported only for type checking
    from levanter.infra.cli_helpers import CliConfig

logger = logging.getLogger(__name__)


def parse_user_command_line(command: str) -> dict[str, str | None]:
    """Parse command line to extract experiment name and prefix for submission ID."""
    experiment = re.search(r"experiments/([^ ]+)", command)

    try:
        if experiment:
            experiment = experiment.group(1).split("/")[-1].split(".")[0]
    except Exception:
        experiment = ""

    return {
        "experiment": experiment,
    }


def generate_submission_id(command: str) -> str:
    """Generate a nice submission ID based on the inferred experiment."""
    parsed = parse_user_command_line(command)
    timestamp_micros = int(time.time() * 1_000_000)
    parts = ["ray-run"]
    parts.append(f"experiment-{parsed.get('experiment', 'unknown')}")
    # parts.append(f"prefix-{parsed.get('prefix', 'unknown')}")
    parts.append(str(timestamp_micros))

    return "-".join(parts)


def tpus_per_node(tpu_type: str) -> int:
    """Return the number of TPU chips per node for a given TPU type."""
    if tpu_type in {"v4-8", "v5p-8"}:
        return 4
    match = re.search(r"-(\d+)$", tpu_type)
    if not match:
        raise ValueError(f"Cannot parse TPU type: {tpu_type}")
    chips = int(match.group(1))
    if chips > 8:
        raise ValueError("Only single tpu nodes are supported with the CLI")
    return chips


def _load_cli_config() -> "CliConfig" | None:
    """Load the Levanter CLI configuration if it is available."""
    try:
        from levanter.infra import cli_helpers
    except ImportError:
        logger.debug("levanter.infra.cli_helpers is not available; skipping .levanter.yaml.")
        return None

    try:
        return cli_helpers.load_config()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load .levanter.yaml: %s", exc)
        return None


def load_env_vars_from_cli_config(tpu_type: str | None = None) -> dict[str, str]:
    """Return environment variables defined in `.levanter.yaml`.

    If the configuration defines accelerator specific overrides and a TPU type
    is provided, those overrides are merged with the base environment.
    """

    config = _load_cli_config()
    if config is None:
        return {}

    try:
        env = config.env_for_accel(tpu_type) if tpu_type else dict(config.env)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to resolve accelerator specific env for %s: %s", tpu_type, exc)
        env = dict(config.env)

    # Filter out unset values and coerce everything to strings for Ray.
    return {key: str(value) for key, value in env.items() if value is not None}


def _collect_gitignore_patterns(working_dir: str) -> list[str]:
    """Return `.gitignore` patterns relative to ``working_dir``.

    Ray normally respects every `.gitignore` it encounters while walking the
    directory tree. When we disable that behaviour we must replicate it by
    collecting the patterns manually and providing them through the runtime
    environment. Patterns from nested directories are converted to be relative
    to ``working_dir`` so that Ray's own gitwildmatch handling produces the same
    results.
    """

    patterns: list[str] = []

    for current_dir, dirs, _ in os.walk(working_dir):
        dirs.sort()
        if ".git" in dirs:
            dirs.remove(".git")

        gitignore_file = os.path.join(current_dir, ".gitignore")
        if not os.path.isfile(gitignore_file):
            continue

        rel_dir = os.path.relpath(current_dir, working_dir)
        is_root = rel_dir in (".", "")
        dir_prefix = "" if is_root else rel_dir.replace(os.sep, "/") + "/"

        with open(gitignore_file, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\r\n")
                if not line.strip():
                    continue
                if line.lstrip().startswith("#"):
                    continue

                negated = line.startswith("!")
                pattern_body = line[1:] if negated else line

                if not pattern_body:
                    continue

                pattern_body = pattern_body.replace("\\", "/")

                anchored = pattern_body.startswith("/")
                if anchored:
                    pattern_body = pattern_body.lstrip("/")

                directory_only = pattern_body.endswith("/")
                if directory_only:
                    pattern_body = pattern_body.rstrip("/")
                    if not pattern_body:
                        continue

                if is_root:
                    pattern = f"/{pattern_body}" if anchored else pattern_body
                else:
                    if not pattern_body:
                        continue

                    if anchored or "/" in pattern_body:
                        pattern = f"{dir_prefix}{pattern_body}"
                    else:
                        pattern = f"{dir_prefix}**/{pattern_body}"

                if directory_only:
                    pattern = pattern.rstrip("/") + "/"

                if negated:
                    pattern = f"!{pattern}"

                patterns.append(pattern)

    return patterns


def maybe_include_levanter_config(runtime_env: dict, working_dir: str) -> tuple[dict, bool]:
    """Ensure `.levanter.yaml` (or the legacy `.config`) is uploaded with the job.

    Returns the possibly-updated runtime environment and a boolean indicating
    whether Ray's implicit `.gitignore` handling should be disabled while
    packaging the runtime environment. When this is ``True`` the caller must set
    the ``RAY_RUNTIME_ENV_IGNORE_GITIGNORE`` environment variable while
    submitting the job.
    """

    present_configs: list[str] = []
    for filename in (".levanter.yaml", ".config"):
        abs_path = os.path.join(working_dir, filename)
        if os.path.isfile(abs_path):
            present_configs.append(filename)

    if not present_configs:
        return runtime_env, False

    patterns = _collect_gitignore_patterns(working_dir)

    if patterns:
        for config_name in present_configs:
            negated = f"!{config_name}"
            if negated not in patterns:
                patterns.append(negated)

        excludes = list(runtime_env.get("excludes", []))
        for pattern in patterns:
            if pattern not in excludes:
                excludes.append(pattern)
        runtime_env["excludes"] = excludes

        logger.debug(
            "Applied .gitignore patterns to runtime env excludes and re-included %s.",
            ", ".join(present_configs),
        )

        return runtime_env, True

    return runtime_env, False


@contextmanager
def _temporarily_ignore_gitignore(should_ignore: bool) -> Iterator[None]:
    """Temporarily disable Ray's automatic `.gitignore` handling."""

    if not should_ignore:
        yield
        return

    env_var = "RAY_RUNTIME_ENV_IGNORE_GITIGNORE"
    previous = os.environ.get(env_var)
    os.environ[env_var] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = previous


async def submit_and_track_job(
    entrypoint: str,
    extra: str,
    env_vars: dict,
    no_wait: bool,
    *,
    entrypoint_num_cpus: float | None = None,
    entrypoint_num_gpus: float | None = None,
    entrypoint_memory: int | None = None,
    entrypoint_resources: dict | None = None,
):
    """Submit a job to Ray and optionally track logs."""

    current_dir = os.getcwd()
    client = JobSubmissionClient(REMOTE_DASHBOARD_URL)

    logger.info(f"Submitting job with entrypoint: {entrypoint}")
    logger.info(f"Extras: {extra}")
    logger.info(f"env_vars: {json.dumps(env_vars, indent=4)}")
    submission_id = generate_submission_id(entrypoint)

    runtime_dict = {
        "working_dir": current_dir,
        "config": {"setup_timeout_seconds": 1800},
        "excludes": [".git", "tests/"],
    }

    # add the TPU dependency for cluster jobs.
    runtime_dict = build_runtime_env_for_packages(extra=[*extra.split(","), "tpu"], env_vars=env_vars) | runtime_dict
    runtime_dict, ignore_gitignore = maybe_include_levanter_config(runtime_dict, current_dir)

    logger.info(
        f"Terminal command: \n"
        f"ray job submit "
        f"--runtime-env-json '{json.dumps(runtime_dict)}' "
        f"--submission-id '{submission_id} "
        f" -- {entrypoint}"
    )

    # Submit the job with runtime environment and entrypoint
    with _temporarily_ignore_gitignore(ignore_gitignore):
        submission_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env=runtime_dict,
            entrypoint_num_cpus=entrypoint_num_cpus,
            entrypoint_num_gpus=entrypoint_num_gpus,
            entrypoint_memory=entrypoint_memory,
            entrypoint_resources=entrypoint_resources,
            submission_id=submission_id,
        )
    logger.info(f"Job submitted with ID: {submission_id}")
    logger.info(f"Job URL: http://localhost:8265/#/jobs/{submission_id}")

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
        "--extra",
        type=str,
        default="",
        help="List of pip dependencies to install before running.",
    )
    parser.add_argument(
        "--entrypoint-num-cpus",
        type=float,
        default=None,
        help="Number of CPUs to reserve for the entrypoint command.",
    )
    parser.add_argument(
        "--entrypoint-num-gpus",
        type=float,
        default=None,
        help="Number of GPUs to reserve for the entrypoint command.",
    )
    parser.add_argument(
        "--entrypoint-memory",
        type=int,
        default=None,
        help="Amount of memory to reserve for the entrypoint command.",
    )
    parser.add_argument(
        "--entrypoint-resources",
        type=json.loads,
        default=None,
        help="JSON dictionary describing resources to reserve for the entrypoint command.",
    )
    parser.add_argument(
        "--tpu",
        type=str,
        default=None,
        help="TPU type to reserve for the entrypoint (e.g. v4-8)",
    )
    parser.add_argument("cmd", help="The command to run in the Ray cluster.", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Combine the remaining arguments to form the full command
    full_cmd = " ".join(shlex.quote(arg) for arg in args.cmd).strip()
    if not full_cmd.startswith("--"):
        logger.error("Command must start with '--'.")
        exit(1)
    full_cmd = full_cmd[2:]

    # Load and merge environment variables from configuration and CLI flags
    env_vars = load_env_vars_from_cli_config(args.tpu)

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

    entrypoint_resources = args.entrypoint_resources
    if args.tpu:
        try:
            chips = tpus_per_node(args.tpu)
        except ValueError as e:
            logger.error(str(e))
            exit(1)
        tpu_res = {f"TPU-{args.tpu}-head": 1, "TPU": chips}
        entrypoint_resources = (entrypoint_resources or {}) | tpu_res

    # Submit the job and track it asynchronously
    asyncio.run(
        submit_and_track_job(
            full_cmd,
            args.extra,
            env_vars,
            args.no_wait,
            entrypoint_num_cpus=args.entrypoint_num_cpus,
            entrypoint_num_gpus=args.entrypoint_num_gpus,
            entrypoint_memory=args.entrypoint_memory,
            entrypoint_resources=entrypoint_resources,
        )
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
