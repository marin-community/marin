#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure TTL lifecycle rules on each marin data bucket (GCS and R2).

This script owns the lifecycle rules that delete objects under ``tmp/ttl=Nd/``
after ``N`` days, for every ``N`` in `config/marin.yaml`.
Rules it does not recognize as its own are preserved untouched, so it is safe to
re-run alongside hand-curated lifecycle rules.

Two backends are configured:

* **GCS** ``marin-{region}`` buckets, via ``gcloud storage``. The owned rules are
  ``Delete`` rules whose ``matchesPrefix`` is exactly ``["tmp/ttl=Nd/"]``. This
  backend also enforces that soft-delete is disabled — Marin churns a lot of
  multi-gigabyte ephemeral data, and soft-delete retention quickly explodes
  storage cost.
* **Cloudflare R2** buckets (:data:`rigging.filesystem.R2_DATA_BUCKETS`), via the
  S3 lifecycle API (botocore). The owned rules are ``Expiration`` rules whose
  ``ID`` starts with ``marin-ttl-``. R2 has no soft-delete to manage.

R2 access needs credentials (``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY``,
or their ``AWS_*`` equivalents) and an endpoint (``AWS_ENDPOINT_URL_S3`` /
``AWS_ENDPOINT_URL`` / ``R2_ENDPOINT_URL``, defaulting to
:data:`R2_ENDPOINT_URL`). Target a specific GCS bucket with ``--bucket`` to
configure GCS without R2 credentials on hand.

Usage:
    uv run infra/configure_buckets.py                  # apply to all buckets
    uv run infra/configure_buckets.py --dry-run        # preview without applying
    uv run infra/configure_buckets.py --bucket marin-us-central2
    uv run infra/configure_buckets.py --bucket marin-na  # R2 only
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable

import botocore.client
import botocore.config
import botocore.session
import click
from botocore.exceptions import ClientError
from rigging.filesystem import R2_DATA_BUCKETS, load_cluster_config

logger = logging.getLogger(__name__)

_MARIN_CONFIG = load_cluster_config("marin")

# Region from which each bucket is served. Built from the canonical
# region_buckets map so this script never drifts from the runtime view.
BUCKETS: dict[str, str] = {bucket: region for region, bucket in _MARIN_CONFIG.region_buckets.items()}

PROJECT = "hai-gcp-models"

# Default Cloudflare R2 S3 API endpoint (the account-level endpoint, already
# committed in the CoreWeave CI configs). Overridable via the AWS_ENDPOINT_URL*
# / R2_ENDPOINT_URL env vars.
R2_ENDPOINT_URL = "https://74981a43be0de7712369306c7b19133d.r2.cloudflarestorage.com"

# Lifecycle rules this script owns on R2 buckets carry an ``ID`` with this
# prefix; everything else is treated as a foreign rule and left alone.
R2_RULE_ID_PREFIX = "marin-ttl-"


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


def _ttl_prefix(n: int) -> str:
    return f"{_MARIN_CONFIG.temp_path}/ttl={n}d/"


def _apply_or_preview_lifecycle(
    merged: list[dict],
    owned: list[dict],
    dry_run: bool,
    *,
    preview_doc: dict,
    apply: Callable[[], None],
) -> None:
    """Log and (unless *dry_run*) apply a merged lifecycle rule set.

    *preview_doc* is the backend-shaped document rendered in dry-run output
    (``{"rule": ...}`` for GCS, ``{"Rules": ...}`` for R2); *apply* performs the
    actual backend write.
    """
    foreign_count = len(merged) - len(owned)
    if dry_run:
        logger.info(
            "[dry-run] Would apply lifecycle rules (kept %d foreign + %d owned):\n%s",
            foreign_count,
            len(owned),
            json.dumps(preview_doc, indent=2),
        )
    else:
        logger.info("Applying lifecycle rules (kept %d foreign + %d owned).", foreign_count, len(owned))
        apply()


# ---------------------------------------------------------------------------
# GCS backend (gcloud storage)
# ---------------------------------------------------------------------------


def describe_bucket(bucket: str) -> dict | None:
    """Return parsed ``gcloud storage buckets describe`` JSON, or ``None`` if missing."""
    result = run(
        ["gcloud", "storage", "buckets", "describe", f"gs://{bucket}", "--format=json"],
        raise_on_error=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def build_gcs_ttl_rules() -> list[dict]:
    """Return Delete rules for every TTL value, scoped to ``tmp/ttl=Nd/``."""
    return [
        {
            "action": {"type": "Delete"},
            "condition": {"age": n, "matchesPrefix": [_ttl_prefix(n)]},
        }
        for n in _MARIN_CONFIG.ttl_days
    ]


def _is_marin_gcs_ttl_rule(rule: dict) -> bool:
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
    return prefixes[0] in {_ttl_prefix(n) for n in _MARIN_CONFIG.ttl_days}


def merge_lifecycle_rules(existing: list[dict], owned: list[dict], is_owned: Callable[[dict], bool]) -> list[dict]:
    """Drop our prior TTL rules from *existing* and append the canonical *owned* set.

    Foreign rules (anything *is_owned* does not recognize as ours) survive untouched.
    """
    foreign = [rule for rule in existing if not is_owned(rule)]
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


def apply_gcs_lifecycle(bucket: str, rules: list[dict]) -> None:
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


def configure_gcs_bucket(bucket: str, region: str, owned: list[dict], dry_run: bool) -> None:
    """Disable soft-delete and apply the owned TTL rules to one GCS bucket."""
    logger.info("=== %s (region=%s) ===", bucket, region)

    info = describe_bucket(bucket)
    if info is None:
        logger.error(
            "Bucket gs://%s does not exist or is inaccessible — main buckets must be created out-of-band; skipping.",
            bucket,
        )
        return

    if soft_delete_enabled(info):
        if dry_run:
            logger.info("[dry-run] Would clear soft-delete on gs://%s", bucket)
        else:
            logger.info("Clearing soft-delete on gs://%s ...", bucket)
            clear_soft_delete(bucket)
    else:
        logger.info("Soft-delete already disabled.")

    existing_rules = (info.get("lifecycle_config") or info.get("lifecycleConfig") or {}).get("rule", []) or []
    merged = merge_lifecycle_rules(existing_rules, owned, _is_marin_gcs_ttl_rule)

    _apply_or_preview_lifecycle(
        merged,
        owned,
        dry_run,
        preview_doc={"rule": merged},
        apply=lambda: apply_gcs_lifecycle(bucket, merged),
    )


# ---------------------------------------------------------------------------
# R2 backend (S3 lifecycle API via botocore)
# ---------------------------------------------------------------------------


def make_r2_client() -> botocore.client.BaseClient:
    """Build a botocore S3 client for R2, failing fast if credentials are missing.

    Credentials come from ``R2_ACCESS_KEY_ID`` / ``R2_SECRET_ACCESS_KEY`` (or
    their ``AWS_*`` equivalents); the endpoint from ``AWS_ENDPOINT_URL_S3`` /
    ``AWS_ENDPOINT_URL`` / ``R2_ENDPOINT_URL``, defaulting to
    :data:`R2_ENDPOINT_URL`. R2 ignores the AWS region scheme, so we sign with
    ``region_name="auto"`` and force virtual-host addressing.
    """
    key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise click.ClickException(
            "R2 credentials are required to configure R2 buckets. Set R2_ACCESS_KEY_ID and "
            "R2_SECRET_ACCESS_KEY (or their AWS_* equivalents), or target a specific GCS bucket "
            "with --bucket to skip R2."
        )
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("AWS_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT_URL")
        or R2_ENDPOINT_URL
    )
    session = botocore.session.get_session()
    return session.create_client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=botocore.config.Config(s3={"addressing_style": "virtual"}),
    )


def build_r2_ttl_rules() -> list[dict]:
    """Return S3 Expiration rules for every TTL value, scoped to ``tmp/ttl=Nd/``."""
    return [
        {
            "ID": f"{R2_RULE_ID_PREFIX}{n}d",
            "Filter": {"Prefix": _ttl_prefix(n)},
            "Expiration": {"Days": n},
            "Status": "Enabled",
        }
        for n in _MARIN_CONFIG.ttl_days
    ]


def _is_marin_r2_ttl_rule(rule: dict) -> bool:
    """Return True iff *rule* is one this script owns (``ID`` starts with ``marin-ttl-``)."""
    return str(rule.get("ID", "")).startswith(R2_RULE_ID_PREFIX)


def get_r2_lifecycle_rules(client: botocore.client.BaseClient, bucket: str) -> list[dict]:
    """Return the bucket's existing lifecycle rules, or ``[]`` if none are set."""
    try:
        resp = client.get_bucket_lifecycle_configuration(Bucket=bucket)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchLifecycleConfiguration":
            return []
        raise
    return resp.get("Rules", [])


def apply_r2_lifecycle(client: botocore.client.BaseClient, bucket: str, rules: list[dict]) -> None:
    client.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})


def configure_r2_bucket(client: botocore.client.BaseClient, bucket: str, owned: list[dict], dry_run: bool) -> None:
    """Apply the owned TTL rules to one R2 bucket (R2 has no soft-delete to manage)."""
    logger.info("=== %s (R2) ===", bucket)

    existing_rules = get_r2_lifecycle_rules(client, bucket)
    merged = merge_lifecycle_rules(existing_rules, owned, _is_marin_r2_ttl_rule)

    _apply_or_preview_lifecycle(
        merged,
        owned,
        dry_run,
        preview_doc={"Rules": merged},
        apply=lambda: apply_r2_lifecycle(client, bucket, merged),
    )


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would happen without executing.")
@click.option(
    "--bucket",
    type=str,
    default=None,
    help=("Only configure this bucket (a GCS bucket in config/marin.yaml " "or an R2 bucket in R2_DATA_BUCKETS)."),
)
def main(dry_run: bool, bucket: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    gcs_targets = BUCKETS
    r2_targets = R2_DATA_BUCKETS
    if bucket is not None:
        if bucket in BUCKETS:
            gcs_targets, r2_targets = {bucket: BUCKETS[bucket]}, frozenset()
        elif bucket in R2_DATA_BUCKETS:
            gcs_targets, r2_targets = {}, frozenset({bucket})
        else:
            known = ", ".join(sorted([*BUCKETS, *R2_DATA_BUCKETS]))
            logger.error("Unknown bucket %r. Known buckets: %s", bucket, known)
            sys.exit(1)

    if gcs_targets:
        owned_gcs = build_gcs_ttl_rules()
        for name, region in sorted(gcs_targets.items()):
            configure_gcs_bucket(name, region, owned_gcs, dry_run)

    if r2_targets:
        client = make_r2_client()
        owned_r2 = build_r2_ttl_rules()
        for name in sorted(r2_targets):
            configure_r2_bucket(client, name, owned_r2, dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()
