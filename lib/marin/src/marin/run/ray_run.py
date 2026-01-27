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
# ruff: noqa

import argparse
import asyncio
import getpass
import json
import logging
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml

os.environ.setdefault("RAY_AUTH_MODE", "token")
_default_token_path = Path.home() / ".ray" / "auth_token"
if _default_token_path.exists() and "RAY_AUTH_TOKEN_PATH" not in os.environ:
    os.environ["RAY_AUTH_TOKEN_PATH"] = str(_default_token_path)


from ray.job_submission import JobSubmissionClient

from marin.cluster.config import find_config_by_region
from fray.cluster.ray import DashboardConfig, ray_dashboard
from fray.cluster.ray.deps import build_runtime_env_for_packages, accelerator_type_from_extra, AcceleratorType

logger = logging.getLogger(__name__)

REMOTE_DASHBOARD_URL = "http://127.0.0.1:8265"


def _maybe_enable_ray_token_auth(*, require_token: bool) -> None:
    """Enable token auth client-side.

    Ray only adds auth headers when `RAY_AUTH_MODE=token` is set in the client
    process.
    """
    os.environ.setdefault("RAY_AUTH_MODE", "token")

    if os.environ.get("RAY_AUTH_TOKEN"):
        return

    token_path = os.environ.get("RAY_AUTH_TOKEN_PATH")
    if token_path and Path(token_path).expanduser().exists():
        return

    if (Path.home() / ".ray" / "auth_token").exists():
        return

    if require_token:
        raise RuntimeError(
            "Ray token authentication is enabled but no local token was found. "
            "Create a token file at ~/.ray/auth_token, or set RAY_AUTH_TOKEN_PATH / RAY_AUTH_TOKEN."
        )


def parse_user_command_line(command: str) -> dict[str, str]:
    """Extract interesting parts from a user command line."""
    parts = command.strip().split()
    entrypoint = None
    for part in parts:
        if Path(part).exists() and "/python" not in part:
            entrypoint = Path(part).name.split(".")[0]
            return {"entrypoint": entrypoint}

    if parts and entrypoint is None:
        entrypoint = parts[0]
    else:
        entrypoint = "unknown"

    return {"entrypoint": entrypoint}


def generate_submission_id(command: str) -> str:
    """Generate a nice submission ID based on the inferred experiment."""
    parsed = parse_user_command_line(command)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    parts = ["ray-run", getpass.getuser(), parsed["entrypoint"], timestamp]
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


def make_client() -> JobSubmissionClient:
    """Create a JobSubmissionClient based on environment variables."""
    address = os.environ.get("RAY_ADDRESS", REMOTE_DASHBOARD_URL)
    # Always pass an explicit HTTP dashboard URL. If Ray has to infer the Jobs
    # API endpoint (e.g. from a `ray://...` address), it can resolve to the head
    # node's internal `webui_url`, which isn't reachable from a developer laptop
    # when using SSH port forwarding.
    return JobSubmissionClient(address)


async def submit_and_track_job(
    entrypoint: str,
    extra: str,
    env_vars: dict,
    no_wait: bool,
    submission_id: str,
    *,
    entrypoint_num_cpus: float | None = None,
    entrypoint_num_gpus: float | None = None,
    entrypoint_memory: int | None = None,
    entrypoint_resources: dict | None = None,
):
    """Submit a job to Ray and optionally track logs."""
    client = make_client()
    current_dir = os.getcwd()

    # Inject GIT_COMMIT into the environment for logging
    env_vars["GIT_COMMIT"] = subprocess.getoutput("git rev-parse HEAD")

    # Tell Fray to use Ray cluster for job execution
    env_vars["FRAY_CLUSTER_SPEC"] = "ray"

    logger.info(f"Submitting job with entrypoint: {entrypoint}")
    logger.info(f"Extras: {extra}")
    logger.info(f"env_vars: {json.dumps(env_vars, indent=4)}")

    runtime_dict = {
        "working_dir": current_dir,
        "config": {"setup_timeout_seconds": 1800},
        "excludes": [".git", "docs/", "**/*.pack", "lib/levanter/docs"],
    }

    # add the TPU dependency for cluster jobs.
    extra_list = extra.split(",") if extra else []
    if accelerator_type_from_extra(extra_list) == AcceleratorType.NONE:
        extra_list.append("tpu")

    runtime_dict = build_runtime_env_for_packages(extra=[*extra_list], env_vars=env_vars) | runtime_dict

    logger.info(
        f"Terminal command: \n"
        f"ray job submit "
        f"--runtime-env-json '{json.dumps(runtime_dict)}' "
        f"--submission-id '{submission_id} "
        f" -- {entrypoint}"
    )

    # Submit the job with runtime environment and entrypoint
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
    logger.info(f"Job URL: {client.get_address()}/#/jobs/{submission_id}")

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
    parser.add_argument(
        "--cluster",
        type=str,
        default=None,
        help="Cluster name or config file path to submit job to",
    )
    parser.add_argument(
        "--submission-id",
        type=str,
        default=None,
        help="Custom submission ID for the job. If not provided, a default ID will be generated.",
    )
    parser.add_argument(
        "--auto-stop",
        action="store_true",
        help="Automatically stop the submitted job on exit (including ctrl+c interrupt).",
    )
    parser.add_argument("cmd", help="The command to run in the Ray cluster.", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Combine the remaining arguments to form the full command
    full_cmd = " ".join(shlex.quote(arg) for arg in args.cmd).strip()
    if not full_cmd.startswith("--"):
        logger.error("Command must start with '--'.")
        exit(1)
    full_cmd = full_cmd[2:]

    # Auto-load env defaults from .marin.yaml if present, then merge -e overrides
    env_vars = {}
    marin_yaml = Path(".marin.yaml")
    if marin_yaml.exists():
        try:
            with open(marin_yaml, "r") as f:
                marin_cfg = yaml.safe_load(f) or {}
            if isinstance(marin_cfg.get("env"), dict):
                for k, v in marin_cfg["env"].items():
                    env_vars[str(k)] = "" if v is None else str(v)
        except Exception as e:
            logger.warning(f"Failed to parse {marin_yaml}: {e}")

    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        if key not in env_vars and os.environ.get(key) is not None:
            env_vars[key] = os.environ[key]

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

    # Resolve cluster config (required)
    cluster_config = None
    if args.cluster:
        if args.cluster.endswith(".yaml") or os.path.exists(args.cluster):
            cluster_config = args.cluster
        else:
            cluster_config = find_config_by_region(args.cluster)

    _maybe_enable_ray_token_auth(require_token=cluster_config is None)

    # Submit the job and track it asynchronously
    if args.submission_id:
        submission_id = args.submission_id
    else:
        submission_id = generate_submission_id(full_cmd)

    async def run_job():
        try:
            await submit_and_track_job(
                full_cmd,
                args.extra,
                env_vars,
                args.no_wait,
                submission_id=submission_id,
                entrypoint_num_cpus=args.entrypoint_num_cpus,
                entrypoint_num_gpus=args.entrypoint_num_gpus,
                entrypoint_memory=args.entrypoint_memory,
                entrypoint_resources=entrypoint_resources,
            )
        except KeyboardInterrupt:
            pass
        except asyncio.CancelledError:
            logger.info("Job tracking cancelled by user.")
            pass
        except Exception as e:
            logger.error(f"Error submitting or tracking job: {e}")
            raise

    try:
        if cluster_config:
            with ray_dashboard(DashboardConfig.from_cluster(cluster_config)):
                asyncio.run(run_job())
        else:
            asyncio.run(run_job())
    except Exception:
        logger.error("Failed to run job", exc_info=True)
    finally:
        if args.auto_stop:
            logger.info(f"Auto-stopping job {submission_id}...")
            # Open a fresh connection for cleanup
            if cluster_config:
                with ray_dashboard(DashboardConfig.from_cluster(cluster_config)):
                    client = make_client()
                    client.stop_job(submission_id)
            else:
                client = make_client()
                client.stop_job(submission_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
