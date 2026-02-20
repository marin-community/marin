#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import sys
import tempfile

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


def main():
    parser = argparse.ArgumentParser(description="Configure GCP Artifact Registry cleanup policy (keep 30d)")
    parser.add_argument("repository", help="Name of the Artifact Registry repository")
    parser.add_argument("--region", help="GCP region")
    parser.add_argument("--project", default=None, help="GCP project ID (default: current gcloud project)")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
        json.dump(CLEANUP_POLICY_30D, f, indent=2)
        policy_path = f.name

    try:
        # Optionally add --project
        project_args = ["--project", args.project] if args.project else []

        print(f"Setting 30d cleanup policy for repository '{args.repository}' in region '{args.region}'...")
        run(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "set-cleanup-policies",
                f"--location={args.region}",
                *project_args,
                f"--policy={policy_path}",
                args.repository,
            ]
        )
        print("Policy applied successfully.")
    finally:
        os.remove(policy_path)


if __name__ == "__main__":
    main()
