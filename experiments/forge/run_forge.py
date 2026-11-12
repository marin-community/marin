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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposals", "output", "policy", "job-prefix"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", choices=("S", "T"), required=True)
    for name in ("drivers", "concurrency", "cpu"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--driver-range", type=lambda value: tuple(map(int, value.split(":"))))
    parser.add_argument("--memory", default="8GB")
    parser.add_argument("--disk", default="50GB")
    parser.add_argument("--local-sandbox", action="store_true", help="THROWAWAY: bwrap sandboxes in a privileged driver")
    parser.add_argument("--target-cluster", default="cw-rno2a")
    parser.add_argument("--ranks")
    parser.add_argument("--resume-held", action="store_true", help="Resume published held workpieces")
    parser.add_argument("--revise", action="store_true", help="--proposals is a run root of accepted workpieces")
    args = parser.parse_args()
    start, end = args.driver_range or (0, args.drivers)
    if min(args.concurrency, args.cpu) < 1 or not 0 <= start < end <= args.drivers <= 4096:
        raise ValueError("Positive concurrency/CPU and nonempty driver range within 1..4096 drivers required")
    endpoint = tuple(key for key in ("GLM_RELAY_JOB", "GLM_BASE_URL") if os.environ.get(key))
    if len(endpoint) != 1:
        raise ValueError("Set exactly one of GLM_RELAY_JOB and GLM_BASE_URL")
    keys = ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_API_KEY", *endpoint)
    credentials = [item for key in keys for item in ("-e", key, os.environ[key])]
    if "PARALLEL_API_KEY" in os.environ:
        credentials += ["-e", "PARALLEL_API_KEY", os.environ["PARALLEL_API_KEY"]]
    options = {
        "target-cluster": args.target_cluster,
        "cpu": str(args.cpu),
        "memory": args.memory,
        "disk": args.disk,
        "max-retries": "3",
        "sync-package": "marin-forge",
    }
    if args.local_sandbox:
        options["container-profile"] = "container_profile_privileged"
    driver = {name: str(getattr(args, name)) for name in ("proposals", "output", "policy", "arm", "concurrency")}
    driver["buckets"] = "4096"
    if args.ranks:
        driver["ranks"] = args.ranks
    pins = [
        item
        for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "CARGO_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL")
        for item in ("-e", key, str(args.cpu))
    ]
    if os.environ.get("ROUTING_KEY"):  # THROWAWAY: per-session gateway routing key
        pins += ["-e", "FORGE_ROUTING_KEY", "1"]
    if args.local_sandbox:
        pins += ["-e", "FORGE_LOCAL_SANDBOX", "1"]
        if os.environ.get("BWRAP_RLIMIT_GB"):  # THROWAWAY: per-command RLIMIT_DATA inside bwrap machines
            pins += ["-e", "FORGE_LOCAL_RLIMIT_DATA_GB", os.environ["BWRAP_RLIMIT_GB"]]
    for index in range(start, end):
        options["job-name"] = f"{args.job_prefix}-{index}"
        driver["bucket-range"] = f"{4096 * index // args.drivers}:{4096 * (index + 1) // args.drivers}"
        command = "uv run --frozen iris --cluster=marin job run --no-wait --enable-extra-resources".split()
        command += [item for key, value in options.items() for item in (f"--{key}", value)]
        command += [*credentials, *pins, "--", "python", "-m", "forge.driver", "run"]
        command += [item for key, value in driver.items() for item in (f"--{key}", value)]
        if args.resume_held:
            command.append("--resume-held")
        if args.revise:
            command.append("--revise")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
