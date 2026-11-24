# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the lane's driver as an in-region Iris CPU job (Phase 2b).

Long-running generation runs as an Iris job, never on the Mac. The launcher resolves the
Ortet relay's current in-cluster address from the endpoint registry on every submit — the
address is never hardcoded, and relay pod moves are recovered by relaunching, which the
per-task event logs make free (recorded model calls replay instead of regenerating).

The relay job name resolves only on the cluster that hosts it, so the driver job receives
``GLM_BASE_URL`` (the resolved address) instead of a relay job name. The bulk-pool bearer
token travels as a secret file (``GLM_API_KEY``) that the driver reads with
``rigging.secrets.require_secret``, which fails fast when the credential is missing or
empty — the tool-conversion fleet's ``Illegal header value b'Bearer '`` failure mode.

Usage (from the repo root, with CW_KEY_ID/CW_KEY_SECRET/GLM_BULK_TOKEN exported)::

    python -m experiments.post_training.task_curation.balanced_reasoning.launch \\
        --selection s3://.../pilot/selection.json \\
        --run-root s3://.../pilot --concurrency 16
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]

JOB_TIMEOUT_SECONDS = 259_200
"""Three days: the Phase 2b task timeout."""

DEFAULT_TARGET_CLUSTER = "cw-us-east-08a"
DEFAULT_RELAY_CLUSTER = "cw-us-east-02a"
DEFAULT_RELAY_ENDPOINT = "/muchanem/glm53-relay/glm-5.3"
DEFAULT_CONCURRENCY = 16

REQUIRED_ENV = ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_BULK_TOKEN")


def parse_endpoint_address(listing: str, endpoint: str) -> str:
    """The ADDRESS column of the matching row in an ``iris endpoints list`` output."""
    for line in listing.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0] == endpoint:
            return fields[3]
    raise ValueError(f"The endpoint {endpoint!r} has no registered address")


def resolve_relay_base_url(cluster: str, endpoint: str) -> str:
    """The relay pod's current in-cluster address, resolved from the endpoint registry."""
    command = [
        "uv",
        "run",
        "--frozen",
        "iris",
        f"--cluster={cluster}",
        "endpoints",
        "list",
        endpoint,
        "--exact",
    ]
    listing = subprocess.run(command, cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    address = parse_endpoint_address(listing, endpoint)
    return address if address.endswith("/v1") else f"{address}/v1"


def driver_command(arguments: argparse.Namespace, base_url: str) -> list[str]:
    """The full iris job-run invocation for the lane's driver."""
    command = [
        "uv",
        "run",
        "--frozen",
        "iris",
        "--cluster=marin",
        "job",
        "run",
        "--no-wait",
        "--enable-extra-resources",
        "--job-name",
        arguments.job_name,
        "--target-cluster",
        arguments.target_cluster,
        "--cpu",
        str(arguments.cpu),
        "--memory",
        arguments.memory,
        "--disk",
        arguments.disk,
        "--timeout",
        str(JOB_TIMEOUT_SECONDS),
        "--max-retries",
        str(arguments.max_retries),
        "--priority",
        arguments.priority,
        "--sync-package",
        "marin-core",
        "--sync-package",
        "marin-taskforge",
        "--extra",
        "cpu",
        # By reference only: Iris delivers these as secret files and records this argv
        # in MARIN_PROVENANCE; the token value never appears on the command line.
        "--secret",
        "CW_KEY_ID",
        "env:CW_KEY_ID",
        "--secret",
        "CW_KEY_SECRET",
        "env:CW_KEY_SECRET",
        "--secret",
        "GLM_API_KEY",
        "env:GLM_BULK_TOKEN",
        "-e",
        "GLM_BASE_URL",
        base_url,
        "--",
        "python",
        "-m",
        "experiments.post_training.task_curation.balanced_reasoning.cli",
        "run",
        "--selection",
        arguments.selection,
        "--run-root",
        arguments.run_root,
        "--concurrency",
        str(arguments.concurrency),
        "--pool",
        arguments.pool,
        "--allow-live",
    ]
    return command


def check_credentials() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Export {', '.join(missing)} (source ~/Documents/secrets.env); never print their values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, help="selection.json URI for the driver")
    parser.add_argument("--run-root", required=True, help="Run root URI (event logs, store bundle)")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--target-cluster", default=DEFAULT_TARGET_CLUSTER)
    parser.add_argument("--relay-cluster", default=DEFAULT_RELAY_CLUSTER)
    parser.add_argument("--relay-endpoint", default=DEFAULT_RELAY_ENDPOINT)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--memory", default="16GB")
    parser.add_argument("--disk", default="50GB")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--priority", default="production")
    parser.add_argument("--pool", default="bulk")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the relay and print the submit command without submitting",
    )
    arguments = parser.parse_args()

    check_credentials()
    base_url = resolve_relay_base_url(arguments.relay_cluster, arguments.relay_endpoint)
    print(f"RELAY {arguments.relay_endpoint} -> {base_url}", file=sys.stderr)
    command = driver_command(arguments, base_url)
    if arguments.dry_run:
        print(subprocess.list2cmdline(command))
        return 0
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
