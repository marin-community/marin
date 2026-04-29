#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure soft-delete and TTL lifecycle rules on each ``marin-{region}`` bucket.

This script owns lifecycle rules whose ``matchesPrefix`` is exactly
``["tmp/ttl=Nd/"]`` for ``N`` in :data:`rigging.filesystem.ALLOWED_TTL_DAYS`.
Rules with any other prefix (or any other shape) are preserved as-is, so it is
safe to re-run alongside hand-curated lifecycle rules.

It also enforces that soft-delete is disabled on each bucket — Marin churns a
lot of multi-gigabyte ephemeral data, and soft-delete retention quickly
explodes storage cost.

Usage:
    uv run infra/configure_main_buckets.py              # apply to all buckets
    uv run infra/configure_main_buckets.py --dry-run    # preview without applying
    uv run infra/configure_main_buckets.py --bucket marin-us-central2
"""

import json
import logging
import os
import subprocess
import sys
import tempfile

import click

from rigging.filesystem import ALLOWED_TTL_DAYS, REGION_TO_DATA_BUCKET, TEMP_PATH_PREFIX

logger = logging.getLogger(__name__)

# Region from which each bucket is served. Built from the canonical
# REGION_TO_DATA_BUCKET map so this script never drifts from the runtime view.
BUCKETS: dict[str, str] = {bucket: region for region, bucket in REGION_TO_DATA_BUCKET.items()}

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


def describe_bucket(bucket: str) -> dict | None:
    """Return parsed ``gcloud storage buckets describe`` JSON, or ``None`` if missing."""
    result = run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket}", "--format=json"],
        raise_on_error=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def _ttl_prefix(n: int) -> str:
    return f"{TEMP_PATH_PREFIX}/ttl={n}d/"


def build_ttl_rules() -> list[dict]:
    """Return Delete rules for every TTL value, scoped to ``tmp/ttl=Nd/``."""
    return [
        {
            "action": {"type": "Delete"},
            "condition": {"age": n, "matchesPrefix": [_ttl_prefix(n)]},
        }
        for n in ALLOWED_TTL_DAYS
    ]


def _is_marin_ttl_rule(rule: dict) -> bool:
    """Return True iff *rule* looks like a rule that this script owns.

    We only claim ownership of Delete rules whose ``matchesPrefix`` is exactly
    ``["tmp/ttl=Nd/"]`` for some allowed ``N``. Every other rule shape — even
    one that happens to match the same prefix in combination with other
    conditions — belongs to whoever wrote it, and we leave it alone.
    """
    if rule.get("action", {}).get("type") != "Delete":
        return False
    condition = rule.get("condition", {})
    if set(condition.keys()) != {"age", "matchesPrefix"}:
        return False
    prefixes = condition["matchesPrefix"]
    if not isinstance(prefixes, list) or len(prefixes) != 1:
        return False
    return prefixes[0] in {_ttl_prefix(n) for n in ALLOWED_TTL_DAYS}


def merge_lifecycle_rules(existing: list[dict], owned: list[dict]) -> list[dict]:
    """Drop our prior TTL rules from *existing* and append the canonical *owned* set.

    Foreign rules (anything we don't recognize as ours) survive untouched.
    """
    foreign = [rule for rule in existing if not _is_marin_ttl_rule(rule)]
    return foreign + owned


def soft_delete_enabled(bucket_info: dict) -> bool:
    """Return True if the bucket has a non-zero soft-delete retention policy."""
    policy = bucket_info.get("soft_delete_policy") or bucket_info.get("softDeletePolicy") or {}
    duration = policy.get("retentionDurationSeconds") or policy.get("retention_duration_seconds")
    if duration is None:
        return False
    try:
        return int(duration) > 0
    except (TypeError, ValueError):
        return False


def apply_lifecycle(bucket: str, rules: list[dict]) -> None:
    """Write the merged lifecycle JSON and apply it to the bucket."""
    lifecycle = {"rule": rules}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(lifecycle, f, indent=2)
        f.flush()
        temp_path = f.name

    try:
        run(["gcloud", "storage", "buckets", "update", f"gs://{bucket}", f"--lifecycle-file={temp_path}"])
    finally:
        os.remove(temp_path)


def clear_soft_delete(bucket: str) -> None:
    run(["gcloud", "storage", "buckets", "update", f"gs://{bucket}", "--clear-soft-delete"])


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would happen without executing.")
@click.option(
    "--bucket",
    type=str,
    default=None,
    help="Only configure this bucket (must be a key in REGION_TO_DATA_BUCKET).",
)
def main(dry_run: bool, bucket: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    targets = BUCKETS
    if bucket is not None:
        if bucket not in BUCKETS:
            logger.error("Unknown bucket %r. Known buckets: %s", bucket, ", ".join(sorted(BUCKETS)))
            sys.exit(1)
        targets = {bucket: BUCKETS[bucket]}

    owned = build_ttl_rules()

    for name, region in sorted(targets.items()):
        logger.info("=== %s (region=%s) ===", name, region)

        info = describe_bucket(name)
        if info is None:
            logger.error(
                "Bucket gs://%s does not exist or is inaccessible — main buckets must be created out-of-band; skipping.",
                name,
            )
            continue

        if soft_delete_enabled(info):
            if dry_run:
                logger.info("[dry-run] Would clear soft-delete on gs://%s", name)
            else:
                logger.info("Clearing soft-delete on gs://%s ...", name)
                clear_soft_delete(name)
        else:
            logger.info("Soft-delete already disabled.")

        existing_rules = (info.get("lifecycle_config") or info.get("lifecycleConfig") or {}).get("rule", []) or []
        merged = merge_lifecycle_rules(existing_rules, owned)

        if dry_run:
            logger.info(
                "[dry-run] Would apply lifecycle rules (kept %d foreign + %d owned):\n%s",
                len(merged) - len(owned),
                len(owned),
                json.dumps({"rule": merged}, indent=2),
            )
        else:
            logger.info(
                "Applying lifecycle rules (kept %d foreign + %d owned).",
                len(merged) - len(owned),
                len(owned),
            )
            apply_lifecycle(name, merged)

    logger.info("Done.")


if __name__ == "__main__":
    main()
