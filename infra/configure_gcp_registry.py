#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure the 30d cleanup policy on a GCP Artifact Registry repository.

A registry repo's "location" is the GCP region. The canonical region list is
sourced from `config/marin.yaml` so this script
never drifts from the runtime view of the fleet (us-central1, us-central2,
us-east1, us-east5, us-west4, europe-west4).

Usage:
    uv run infra/configure_gcp_registry.py marin --region=us-central2
    uv run infra/configure_gcp_registry.py marin --all-regions
    uv run infra/configure_gcp_registry.py marin --all-regions --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

from rigging.filesystem import load_cluster_config

_MARIN_CONFIG = load_cluster_config("marin")

CLEANUP_POLICY_30D = [
    {
        "name": "delete-older-than-30d",
        "action": {"type": "Delete"},
        "condition": {
            "olderThan": "30d",
            "tagState": "ANY",
        },
    },
    {
        "name": "keep-16",
        "action": {"type": "Keep"},
        "mostRecentVersions": {
            "keepCount": 16,
        },
    },
]


def run(argv):
    try:
        return subprocess.check_output(argv, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e.output.decode(), file=sys.stderr)
        raise


def apply_cleanup_policy(repository: str, region: str, project: str | None, dry_run: bool) -> None:
    """Apply the 30d cleanup policy to *repository* in *region*.

    Writes the policy to a temp file and invokes ``gcloud artifacts
    repositories set-cleanup-policies``. With *dry_run* set, prints the gcloud
    command that would run instead of executing it.
    """
    project_args = ["--project", project] if project else []

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(CLEANUP_POLICY_30D, f, indent=2)
        policy_path = f.name

    try:
        argv = [
            "gcloud",
            "artifacts",
            "repositories",
            "set-cleanup-policies",
            f"--location={region}",
            *project_args,
            f"--policy={policy_path}",
            repository,
        ]
        if dry_run:
            print(f"[dry-run] Would set 30d cleanup policy for '{repository}' in '{region}':")
            print(f"  {' '.join(argv)}")
            return
        print(f"Setting 30d cleanup policy for repository '{repository}' in region '{region}'...")
        run(argv)
        print("Policy applied successfully.")
    finally:
        os.remove(policy_path)


def main():
    parser = argparse.ArgumentParser(description="Configure GCP Artifact Registry cleanup policy (keep 30d)")
    parser.add_argument("repository", help="Name of the Artifact Registry repository")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--region", help="GCP region (location of the registry repo)")
    target.add_argument(
        "--all-regions",
        action="store_true",
        help="Apply to every region in config/marin.yaml.",
    )
    parser.add_argument("--project", default=None, help="GCP project ID (default: current gcloud project)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud command(s) that would run, per region, without executing.",
    )
    args = parser.parse_args()

    if args.all_regions:
        regions = list(_MARIN_CONFIG.region_buckets.keys())
    elif args.region:
        regions = [args.region]
    else:
        parser.error("one of --region or --all-regions is required")

    for region in regions:
        apply_cleanup_policy(args.repository, region, args.project, args.dry_run)


if __name__ == "__main__":
    main()
