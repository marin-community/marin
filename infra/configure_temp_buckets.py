#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provision marin-tmp-* scratch buckets and configure TTL-based lifecycle rules.

This script is the sole owner of lifecycle configuration for marin-tmp-* buckets.
Running it overwrites all existing lifecycle rules on each bucket.

Usage:
    uv run infra/configure_temp_buckets.py              # apply to all buckets
    uv run infra/configure_temp_buckets.py --dry-run    # preview without applying
    uv run infra/configure_temp_buckets.py --bucket marin-tmp-us-central2
"""

import json
import logging
import os
import subprocess
import sys
import tempfile

import click

logger = logging.getLogger(__name__)

# Dedicated tmp buckets, one per region.
BUCKETS: dict[str, str] = {
    "marin-tmp-asia-northeast-1": "asia-northeast1",
    "marin-tmp-us-central1": "us-central1",
    "marin-tmp-us-central2": "us-central2",
    "marin-tmp-eu-west4": "europe-west4",
    "marin-tmp-us-west4": "us-west4",
    "marin-tmp-us-east1": "us-east1",
    "marin-tmp-us-east5": "us-east5",
}

# Each TTL value N generates a lifecycle rule that deletes objects under the
# prefix "ttl=Nd/" after N days.
TTLS: list[int] = [1, 2, 3, 4, 5, 6, 7, 14, 30]

PROJECT = "hai-gcp-models"


def run(argv: list[str], *, raise_on_error: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute a subprocess, logging stderr on failure.

    By default, this raises subprocess.CalledProcessError when the command
    exits with a non-zero status. Callers that intentionally inspect the
    return code can pass raise_on_error=False.
    """
    result = subprocess.run(argv, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Command failed: %s\nstderr: %s", " ".join(argv), result.stderr.strip())
        if raise_on_error:
            raise subprocess.CalledProcessError(
                result.returncode,
                argv,
                output=result.stdout,
                stderr=result.stderr,
            )
    return result


def bucket_exists(bucket: str) -> bool:
    result = run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket}"],
        raise_on_error=False,
    )
    return result.returncode == 0


def create_bucket(bucket: str, region: str) -> None:
    run(
        [
            "gcloud",
            "storage",
            "buckets",
            "create",
            f"gs://{bucket}",
            f"--project={PROJECT}",
            f"--location={region}",
            "--uniform-bucket-level-access",
            "--default-storage-class=STANDARD",
            "--soft-delete-duration=0",
        ]
    )


def build_ttl_rules() -> list[dict]:
    """Return lifecycle Delete rules, one per TTL value.

    Each rule matches the prefix ``ttl=Nd/`` and deletes objects older than N days.
    """
    rules = []
    for n in TTLS:
        rules.append(
            {
                "action": {"type": "Delete"},
                "condition": {"age": n, "matchesPrefix": [f"ttl={n}d/"]},
            }
        )
    return rules


def apply_lifecycle(bucket: str, rules: list[dict]) -> None:
    """Write lifecycle JSON to a temp file and apply it to the bucket.

    This overwrites all existing lifecycle rules — this script is the sole owner
    of lifecycle configuration for marin-tmp-* buckets.
    """
    lifecycle = {"rule": rules}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(lifecycle, f, indent=2)
        f.flush()
        temp_path = f.name

    try:
        run(["gcloud", "storage", "buckets", "update", f"gs://{bucket}", f"--lifecycle-file={temp_path}"])
    finally:
        os.remove(temp_path)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would happen without executing.")
@click.option("--bucket", type=str, default=None, help="Only configure this bucket (must be a key in BUCKETS).")
def main(dry_run: bool, bucket: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    targets = BUCKETS
    if bucket is not None:
        if bucket not in BUCKETS:
            logger.error("Unknown bucket %r. Known buckets: %s", bucket, ", ".join(sorted(BUCKETS)))
            sys.exit(1)
        targets = {bucket: BUCKETS[bucket]}

    rules = build_ttl_rules()

    for name, region in sorted(targets.items()):
        logger.info("=== %s (region=%s) ===", name, region)

        exists = bucket_exists(name)
        if exists:
            logger.info("Bucket already exists, skipping creation.")
        else:
            if dry_run:
                logger.info("[dry-run] Would create bucket gs://%s in %s", name, region)
            else:
                logger.info("Creating bucket gs://%s in %s ...", name, region)
                create_bucket(name, region)

        if dry_run:
            logger.info("[dry-run] Would apply lifecycle rules:\n%s", json.dumps({"rule": rules}, indent=2))
        else:
            logger.info("Applying lifecycle rules ...")
            apply_lifecycle(name, rules)

    logger.info("Done.")


if __name__ == "__main__":
    main()
