# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one explicitly sized Iris driver per disjoint proposal shard."""

import argparse
import os
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("proposals", "output", "policy", "job-prefix"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--arm", choices=("S", "T"), required=True)
    for name in ("shards", "concurrency", "cpu"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--memory", default="8GB")
    parser.add_argument("--disk", default="50GB")
    parser.add_argument("--target-cluster", default="cw-rno2a")
    parser.add_argument("--ranks")
    args = parser.parse_args()
    if min(args.shards, args.concurrency, args.cpu) < 1:
        raise ValueError("Shard count, concurrency and CPU must be positive")
    credentials = [
        item
        for key in ("CW_KEY_ID", "CW_KEY_SECRET", "GLM_BASE_URL", "GLM_API_KEY")
        for item in ("-e", key, os.environ[key])
    ]
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
    driver = {name: str(getattr(args, name)) for name in ("proposals", "output", "policy", "arm", "concurrency")}
    driver["shard-count"] = str(args.shards)
    if args.ranks:
        driver["ranks"] = args.ranks
    pins = [
        item
        for key in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "CARGO_BUILD_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL")
        for item in ("-e", key, str(args.cpu))
    ]
    for shard in range(args.shards):
        options["job-name"], driver["shard-index"] = f"{args.job_prefix}-{shard}", str(shard)
        command = "uv run --frozen iris --cluster=marin job run --no-wait --enable-extra-resources".split()
        command += [item for key, value in options.items() for item in (f"--{key}", value)]
        command += [*credentials, *pins, "--", "python", "-m", "forge.driver", "run"]
        command += [item for key, value in driver.items() for item in (f"--{key}", value)]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
