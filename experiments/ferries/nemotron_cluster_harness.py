# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Restart the Nemotron Iris cluster, wipe temp outputs, and launch the data job.

Usage:
    uv run python experiments/ferries/nemotron_cluster_harness.py
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
from enum import StrEnum
from pathlib import Path

from iris.cluster.config import IrisConfig
from iris.cluster.providers.k8s.controller import configure_client_s3
from marin.utils import fsspec_rm

from experiments.ferries.nemotron_data import S3_PREFIX

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "lib/iris/examples/coreweave-rav-nt.yaml"
DEFAULT_COMMAND = ("python", "experiments/ferries/nemotron_data.py")
R2_ACCESS_KEY_ID_ENV = "R2_ACCESS_KEY_ID"
R2_SECRET_ACCESS_KEY_ENV = "R2_SECRET_ACCESS_KEY"


class RestartScope(StrEnum):
    NONE = "none"
    CONTROLLER = "controller"
    CLUSTER = "cluster"


class CleanupMode(StrEnum):
    KEEP = "keep"
    WIPE = "wipe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Iris config used for restart and submission.",
    )
    parser.add_argument(
        "--restart",
        type=RestartScope,
        choices=list(RestartScope),
        default=RestartScope.CONTROLLER,
        help="Cluster lifecycle action before submission.",
    )
    parser.add_argument(
        "--cleanup",
        type=CleanupMode,
        choices=list(CleanupMode),
        default=CleanupMode.WIPE,
        help="Whether to wipe the temporary output prefix before submission.",
    )
    parser.add_argument(
        "--output-prefix",
        default=S3_PREFIX,
        help="Temporary output prefix to wipe before the run.",
    )
    parser.add_argument(
        "--job-name",
        default=None,
        help="Optional Iris job name override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the actions without executing them.",
    )
    return parser.parse_args()


def require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(f"{var_name} must be set in the local environment")
    return value


def run_command(
    args: list[str],
    *,
    dry_run: bool,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    logger.info("$ %s", shlex.join(args))
    if dry_run:
        return None

    result = subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=capture_output,
        check=False,
    )

    if capture_output and result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
    if capture_output and result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.info("  %s", line)

    result.check_returncode()
    return result


def restart_cluster(config_path: Path, restart_scope: RestartScope, *, dry_run: bool) -> None:
    if restart_scope is RestartScope.NONE:
        logger.info("Skipping cluster restart")
        return

    if restart_scope is RestartScope.CLUSTER:
        run_command(["uv", "run", "iris", "--config", str(config_path), "cluster", "restart"], dry_run=dry_run)
    else:
        run_command(["uv", "run", "iris", "--config", str(config_path), "cluster", "stop"], dry_run=dry_run)
        run_command(["uv", "run", "iris", "--config", str(config_path), "cluster", "start"], dry_run=dry_run)

    run_command(["uv", "run", "iris", "--config", str(config_path), "cluster", "status"], dry_run=dry_run)
    stop_leftover_running_jobs(config_path, dry_run=dry_run)


def stop_leftover_running_jobs(config_path: Path, *, dry_run: bool) -> None:
    result = run_command(
        ["uv", "run", "iris", "--config", str(config_path), "job", "list", "--state", "running", "--json"],
        dry_run=dry_run,
        capture_output=True,
    )
    if result is None:
        logger.info("Would terminate any leftover running jobs after restart")
        return

    jobs = json.loads(result.stdout)
    job_ids = [job["job_id"] for job in jobs]
    if not job_ids:
        logger.info("No leftover running jobs found after restart")
        return

    logger.info("Terminating %d leftover running job(s) after restart", len(job_ids))
    run_command(
        ["uv", "run", "iris", "--config", str(config_path), "job", "stop", *job_ids],
        dry_run=dry_run,
    )


def wipe_outputs(config_path: Path, output_prefix: str, cleanup_mode: CleanupMode, *, dry_run: bool) -> None:
    if cleanup_mode is CleanupMode.KEEP:
        logger.info("Keeping temporary outputs under %s", output_prefix)
        return

    configure_client_s3(IrisConfig.load(config_path).proto)
    logger.info("Wiping temporary outputs under %s", output_prefix)
    if dry_run:
        return

    removed = fsspec_rm(output_prefix)
    if removed:
        logger.info("Removed %s", output_prefix)
    else:
        logger.info("Nothing to remove under %s", output_prefix)


def submit_job(config_path: Path, job_name: str | None, *, dry_run: bool) -> str | None:
    object_storage_key = require_env(R2_ACCESS_KEY_ID_ENV)
    object_storage_secret = require_env(R2_SECRET_ACCESS_KEY_ENV)
    object_storage_endpoint = IrisConfig.load(config_path).proto.platform.coreweave.object_storage_endpoint

    command = [
        "uv",
        "run",
        "iris",
        "--config",
        str(config_path),
        "job",
        "run",
        "-e",
        "MARIN_PREFIX",
        "s3://marin-na/marin/",
        "-e",
        "AWS_ACCESS_KEY_ID",
        object_storage_key,
        "-e",
        "AWS_SECRET_ACCESS_KEY",
        object_storage_secret,
        "-e",
        "AWS_ENDPOINT_URL",
        object_storage_endpoint,
        "--no-wait",
    ]
    if job_name:
        command.extend(["--job-name", job_name])
    command.extend(["--", *DEFAULT_COMMAND])

    result = run_command(command, dry_run=dry_run, capture_output=True)
    if result is None:
        return None
    return result.stdout.strip().splitlines()[-1]


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)

    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    require_env(R2_ACCESS_KEY_ID_ENV)
    require_env(R2_SECRET_ACCESS_KEY_ENV)

    restart_cluster(config_path, args.restart, dry_run=args.dry_run)
    wipe_outputs(config_path, args.output_prefix, args.cleanup, dry_run=args.dry_run)
    job_id = submit_job(config_path, args.job_name, dry_run=args.dry_run)

    if args.dry_run:
        logger.info("Dry run complete")
        return 0

    if not job_id:
        raise RuntimeError("Job submission did not return a job id")

    logger.info("Submitted Nemotron data job: %s", job_id)
    logger.info(
        "Follow with: %s",
        shlex.join(["uv", "run", "iris", "--config", str(config_path), "job", "logs", "-f", job_id]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
