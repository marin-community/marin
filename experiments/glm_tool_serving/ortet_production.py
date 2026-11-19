# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Feed Ortet-assigned tail shards through the shared GLM relay."""

import argparse
from pathlib import Path

from experiments.glm_tool_serving.fleet import FleetConfig, run_fleet

DESTINATION = "s3://marin-us-east-08a/marin/users/benfeuer/tool-normalization/ortet-24-20261112"
MAX_REPLICAS = 24


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica", type=int, required=True)
    parser.add_argument("--replicas", type=int, default=MAX_REPLICAS)
    parser.add_argument("--catalog", default="ortet-24-catalog.json")
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--predecessor", help="Read-only previous run prefix; used only when no new cursor exists")
    parser.add_argument("--recovery-file", help="Recovery JSON for one held batch; see reconciliations/ records")
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--artifact-budget", type=int, default=100_000_000_000)
    parser.add_argument("--hours", type=float, default=72)
    args = parser.parse_args()
    if not 1 <= args.replica <= args.replicas <= MAX_REPLICAS:
        raise ValueError("Invalid replica ownership for the Ortet run")
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
            root=Path(f"/tmp/tool-ortet-r{args.replica}"),
            config_file="config-1.json",
            recovery_file=args.recovery_file,
            generation=f"ortet-24-r{args.replica}-20261112",
            capacity_hours=80,
            tail_catalog=None,
        )
    )


if __name__ == "__main__":
    main()
