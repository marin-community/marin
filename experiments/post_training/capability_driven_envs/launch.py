# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit sized Iris drivers over stable SHA256 proposal-key buckets.

--drivers N partitions 4096 buckets; --driver-range a:b launches only that half-open
range of drivers. To resize the last quarter of 32 drivers to 16, cancel 24:32 and
launch --drivers 64 --driver-range 48:64 under a new job prefix.
"""

import argparse
import os
import subprocess
from collections.abc import Mapping

BUCKETS = 4096
"""Passed to every driver as ``--buckets``; fixed across relaunches so that each proposal keeps its bucket."""
SECRETS = ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_API_KEY", "PARALLEL_API_KEY")
"""Read by the driver with rigging.secrets.get_secret; PARALLEL_API_KEY is optional."""
THREAD_PINS = ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "CARGO_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL")
"""Pinned to ``--cpu``: CPU is request-only, so a tool sizing itself from ``nproc`` sees the node's cores."""


def relay_job(value: str) -> str:
    """A non-blank relay job, so that a quoted unset variable fails before any job is submitted."""
    if not value.strip():
        raise argparse.ArgumentTypeError(f"must name an Iris relay job, got {value!r}")
    return value


def base_url(value: str) -> str:
    if not value.endswith("/v1"):
        raise argparse.ArgumentTypeError(f"must be a URL ending in /v1, got {value!r}")
    return value


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("proposals", "output", "policy", "job-prefix"):
        parser.add_argument(f"--{name}", required=True)
    for name in ("drivers", "concurrency", "cpu"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--driver-range", type=lambda value: tuple(map(int, value.split(":"))))
    parser.add_argument("--memory", default="8GB")
    parser.add_argument("--disk", default="50GB")
    parser.add_argument("--target-cluster", default="cw-rno2a")
    parser.add_argument("--ranks")
    parser.add_argument("--resume-held", action="store_true", help="Resume published held workpieces")
    gateway = parser.add_mutually_exclusive_group(required=True)
    gateway.add_argument("--glm-relay-job", type=relay_job, help="The Iris relay job that resolves the GLM gateway")
    gateway.add_argument("--glm-base-url", type=base_url, help="A direct GLM gateway endpoint ending in /v1")
    return parser


def commands(args: argparse.Namespace, environ: Mapping[str, str]) -> list[list[str]]:
    """One ``iris job run`` argv per driver in the requested range; secrets travel by name, never by value."""
    start, end = args.driver_range or (0, args.drivers)
    if min(args.concurrency, args.cpu) < 1 or not 0 <= start < end <= args.drivers <= BUCKETS:
        raise ValueError(f"Positive concurrency/CPU and nonempty driver range within 1..{BUCKETS} drivers required")
    if missing := [key for key in SECRETS[:3] if not environ.get(key)]:
        raise ValueError(f"Export {', '.join(missing)}")
    # By reference only: Iris delivers them as secret files and records this argv in MARIN_PROVENANCE.
    credentials = [item for key in SECRETS if environ.get(key) for item in ("--secret", key, f"env:{key}")]
    options = {
        "target-cluster": args.target_cluster,
        "cpu": str(args.cpu),
        "memory": args.memory,
        "disk": args.disk,
        "max-retries": "3",
        "sync-package": "marin-taskforge",
    }
    names = ("proposals", "output", "policy", "concurrency", "glm_relay_job", "glm_base_url")
    driver = {name.replace("_", "-"): str(getattr(args, name)) for name in names if getattr(args, name) is not None}
    driver["buckets"] = str(BUCKETS)
    if args.ranks:
        driver["ranks"] = args.ranks
    pins = [item for key in THREAD_PINS for item in ("-e", key, str(args.cpu))]
    result = []
    for index in range(start, end):
        options["job-name"] = f"{args.job_prefix}-{index}"
        driver["bucket-range"] = f"{BUCKETS * index // args.drivers}:{BUCKETS * (index + 1) // args.drivers}"
        command = "uv run --frozen iris --cluster=marin job run --no-wait --enable-extra-resources".split()
        command += [item for key, value in options.items() for item in (f"--{key}", value)]
        command += [*credentials, *pins, "--", "python", "-m", "taskforge.driver", "run"]
        command += [item for key, value in driver.items() for item in (f"--{key}", value)]
        result.append([*command, *(["--resume-held"] if args.resume_held else [])])
    return result


def main() -> None:
    for command in commands(parser().parse_args(), os.environ):
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
