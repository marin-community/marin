#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure TTL lifecycle rules on each marin data bucket (GCS, R2, CoreWeave).

This script owns the lifecycle rules that delete objects under ``tmp/ttl=Nd/``
after ``N`` days, for every ``N`` in `config/marin.yaml`.
Rules it does not recognize as its own are preserved untouched, so it is safe to
re-run alongside hand-curated lifecycle rules.

Three backends are configured:

* **GCS** ``marin-{region}`` buckets, via ``gcloud storage``. The owned rules are
  ``Delete`` rules whose ``matchesPrefix`` is exactly ``["tmp/ttl=Nd/"]``. This
  backend also enforces that soft-delete is disabled — Marin churns a lot of
  multi-gigabyte ephemeral data, and soft-delete retention quickly explodes
  storage cost.
* **Cloudflare R2** and **CoreWeave AI Object Storage** buckets, via the S3
  lifecycle API (botocore). Both are enumerated from
  :func:`rigging.filesystem.s3_data_buckets` — the R2/CoreWeave buckets declared
  in ``config/*.yaml`` by their ``store`` type — so the set lives in config, not
  here. The owned rules are ``Expiration`` rules whose ``ID`` starts with
  ``marin-ttl-``; neither backend has soft-delete to manage. CoreWeave AOS is
  virtual-host only and signs with each bucket's CoreWeave region; R2 is
  region-agnostic (``region_name="auto"``).

Credentials and endpoints come from :mod:`rigging.filesystem.s3_compat`, so this
script and the runtime filesystems read the same variables: ``CW_KEY_ID`` /
``CW_KEY_SECRET`` for CoreWeave and ``R2_KEY_ID`` / ``R2_KEY_SECRET`` for R2, each
falling back to the generic ``AWS_*`` pair, with ``CW_S3_ENDPOINT`` /
``R2_S3_ENDPOINT`` overriding the default endpoints. The default all-buckets run
configures R2 and CoreWeave together and they have distinct keys, so set the
namespaced pairs there; the ``AWS_*`` fallback suits single-backend runs. Target a
specific GCS bucket with ``--bucket`` to configure GCS without S3 credentials on
hand.

Usage:
    uv run infra/configure_buckets.py                  # GCS + R2 + CoreWeave (all buckets)
    uv run infra/configure_buckets.py --dry-run        # preview without applying
    uv run infra/configure_buckets.py --bucket marin-us-central2    # one GCS bucket
    uv run infra/configure_buckets.py --bucket marin-na             # R2 only
    uv run infra/configure_buckets.py --bucket marin-us-east-02a    # one CoreWeave bucket
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
import rigging.filesystem.s3_compat
from botocore.exceptions import ClientError
from rigging.filesystem import BucketSpec, StoreType, load_cluster_config, s3_data_buckets

logger = logging.getLogger(__name__)

_MARIN_CONFIG = load_cluster_config("marin")

# GCS bucket -> serving region, from the marin cluster's region_buckets so this
# script never drifts from the runtime view. The R2/CoreWeave (S3) buckets come
# from s3_data_buckets(), which reads their `store` type from config/*.yaml.
BUCKETS: dict[str, str] = {
    spec.name: region for region, spec in _MARIN_CONFIG.region_buckets.items() if spec.store == StoreType.GCS
}

PROJECT = "hai-gcp-models"

# Lifecycle rules this script owns on S3-compatible buckets (R2 and CoreWeave)
# carry an ``ID`` with this prefix; everything else is treated as a foreign rule
# and left alone.
S3_RULE_ID_PREFIX = "marin-ttl-"


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
# S3-compatible backend (R2 and CoreWeave, via the S3 lifecycle API / botocore)
# ---------------------------------------------------------------------------


def _s3_client(store: StoreType, region: str) -> botocore.client.BaseClient:
    """Build a botocore S3 client for *store*, signing in *region*.

    Both backends are virtual-host only. R2 ignores the AWS region scheme (sign
    ``"auto"``); CoreWeave routes on the bucket's own region (e.g. ``US-EAST-02A``).
    """
    credentials = rigging.filesystem.s3_compat.s3_credentials(store)
    if credentials is None:
        raise click.ClickException(
            f"{store} credentials are required: "
            f"{rigging.filesystem.s3_compat.credentials_hint(store)}, or target a different "
            f"bucket with --bucket to skip this backend."
        )
    key, secret = credentials
    session = botocore.session.get_session()
    return session.create_client(
        "s3",
        endpoint_url=rigging.filesystem.s3_compat.s3_endpoint(store),
        region_name=region,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=botocore.config.Config(s3={"addressing_style": "virtual"}),
    )


def build_s3_ttl_rules() -> list[dict]:
    """Return S3 Expiration rules for every TTL value, scoped to ``tmp/ttl=Nd/``."""
    return [
        {
            "ID": f"{S3_RULE_ID_PREFIX}{n}d",
            "Filter": {"Prefix": _ttl_prefix(n)},
            "Expiration": {"Days": n},
            "Status": "Enabled",
        }
        for n in _MARIN_CONFIG.ttl_days
    ]


def _is_marin_s3_ttl_rule(rule: dict) -> bool:
    """Return True iff *rule* is one this script owns (``ID`` starts with ``marin-ttl-``)."""
    return str(rule.get("ID", "")).startswith(S3_RULE_ID_PREFIX)


def get_s3_lifecycle_rules(client: botocore.client.BaseClient, bucket: str) -> list[dict]:
    """Return the bucket's existing lifecycle rules, or ``[]`` if none are set."""
    try:
        resp = client.get_bucket_lifecycle_configuration(Bucket=bucket)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchLifecycleConfiguration":
            return []
        raise
    return resp.get("Rules", [])


def apply_s3_lifecycle(client: botocore.client.BaseClient, bucket: str, rules: list[dict]) -> None:
    client.put_bucket_lifecycle_configuration(Bucket=bucket, LifecycleConfiguration={"Rules": rules})


def configure_s3_bucket(
    client: botocore.client.BaseClient, bucket: str, owned: list[dict], dry_run: bool, *, label: str
) -> None:
    """Apply the owned TTL rules to one S3-compatible bucket (R2 or CoreWeave).

    Neither backend has soft-delete to manage. *label* names the backend in the
    log header (e.g. ``"R2"`` or ``"CoreWeave US-EAST-02A"``).
    """
    logger.info("=== %s (%s) ===", bucket, label)

    existing_rules = get_s3_lifecycle_rules(client, bucket)
    merged = merge_lifecycle_rules(existing_rules, owned, _is_marin_s3_ttl_rule)

    _apply_or_preview_lifecycle(
        merged,
        owned,
        dry_run,
        preview_doc={"Rules": merged},
        apply=lambda: apply_s3_lifecycle(client, bucket, merged),
    )


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would happen without executing.")
@click.option(
    "--bucket",
    type=str,
    default=None,
    help="Only configure this bucket (a GCS bucket in config/marin.yaml, or an R2/CoreWeave bucket in config/*.yaml).",
)
def main(dry_run: bool, bucket: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Default run configures every backend; each S3 backend reads its own
    # namespaced credentials (R2_* / CW_*, AWS_* fallback) in its client factory.
    s3_buckets = s3_data_buckets()
    gcs_targets: dict[str, str] = BUCKETS
    s3_targets: dict[str, BucketSpec] = dict(s3_buckets)
    if bucket is not None:
        if bucket in BUCKETS:
            gcs_targets, s3_targets = {bucket: BUCKETS[bucket]}, {}
        elif bucket in s3_buckets:
            gcs_targets, s3_targets = {}, {bucket: s3_buckets[bucket]}
        else:
            known = ", ".join(sorted([*BUCKETS, *s3_buckets]))
            logger.error("Unknown bucket %r. Known buckets: %s", bucket, known)
            sys.exit(1)

    if gcs_targets:
        owned_gcs = build_gcs_ttl_rules()
        for name, region in sorted(gcs_targets.items()):
            configure_gcs_bucket(name, region, owned_gcs, dry_run)

    if s3_targets:
        configure_s3_buckets(s3_targets, build_s3_ttl_rules(), dry_run)

    logger.info("Done.")


def configure_s3_buckets(targets: dict[str, BucketSpec], owned: list[dict], dry_run: bool) -> None:
    """Configure lifecycle rules on R2/CoreWeave buckets, one client per backend.

    The R2 client is shared across all R2 buckets; CoreWeave signs per bucket
    region, so each gets its own client.
    """
    r2_client: botocore.client.BaseClient | None = None
    for name, spec in sorted(targets.items()):
        if spec.store == StoreType.R2:
            r2_client = r2_client or _s3_client(StoreType.R2, "auto")
            configure_s3_bucket(r2_client, name, owned, dry_run, label="R2")
        elif spec.store == StoreType.COREWEAVE:
            assert spec.signing_region is not None, f"CoreWeave bucket {name!r} missing signing_region"
            region = spec.signing_region
            configure_s3_bucket(
                _s3_client(StoreType.COREWEAVE, region), name, owned, dry_run, label=f"CoreWeave {region}"
            )
        else:
            raise click.ClickException(f"bucket {name!r} has non-S3 store {spec.store!r}")


if __name__ == "__main__":
    main()
