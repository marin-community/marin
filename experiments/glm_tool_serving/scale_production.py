# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Feed retained source shards through conversion with durable batch cursors."""

import argparse
from pathlib import Path

from experiments.glm_tool_serving.fleet import FleetConfig, run_fleet

DESTINATION = "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization/h100-scale-20261103"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=2)
    parser.add_argument("--catalog", default="scale-catalog.json")
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--predecessor", help="Read-only previous run prefix; used only when no new cursor exists")
    parser.add_argument("--slots", type=int, default=16)
    parser.add_argument("--artifact-budget", type=int, default=250_000_000_000)
    parser.add_argument("--hours", type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.replica <= args.replicas <= 8 or args.slots * args.replicas > 32:
        raise ValueError("Invalid replica ownership or aggregate admission allocation")
    run_fleet(
        FleetConfig(
            replica=args.replica,
            replicas=args.replicas,
            catalog=args.catalog,
            destination=args.destination,
            predecessor=args.predecessor,
            slots=args.slots,
            artifact_budget=args.artifact_budget,
            hours=args.hours,
            root=Path(f"/tmp/tool-scale-r{args.replica}"),
            config_file=f"config-{args.replica}.json",
            recovery_file=f"recovery-{args.replica}.json",
            generation=f"{args.destination.rstrip('/').rsplit('/', 1)[-1]}-r{args.replica}",
            capacity_hours=30,
            tail_catalog="ortet-24-catalog.json",
        )
    )


if __name__ == "__main__":
    main()
