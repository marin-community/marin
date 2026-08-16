#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configure TTL lifecycle rules on Marin's S3-compatible data buckets.

This script owns the lifecycle rules that delete objects under ``tmp/ttl=Nd/``
after ``N`` days, for every ``N`` in `config/marin.yaml`.
Rules it does not recognize as its own are preserved untouched, so it is safe to
re-run alongside hand-curated lifecycle rules. Regional GCS buckets and their
lifecycle policies are managed by ``infra/pulumi``.

Two backends are configured:

* **Cloudflare R2** and **CoreWeave AI Object Storage** buckets, via the S3
  lifecycle API (botocore). Both are enumerated from
  :func:`rigging.filesystem.cluster_config.s3_data_buckets` — the R2/CoreWeave buckets declared
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
specific bucket with ``--bucket`` to avoid needing the other backend's credentials.

Usage:
    uv run infra/configure_buckets.py                  # R2 + CoreWeave (all buckets)
    uv run infra/configure_buckets.py --dry-run        # preview without applying
    uv run infra/configure_buckets.py --bucket marin-na             # R2 only
    uv run infra/configure_buckets.py --bucket marin-us-east-02a    # one CoreWeave bucket
"""

import json
import logging
import sys

import botocore.client
import botocore.config
import botocore.session
import click
import rigging.filesystem.s3_compat
from botocore.exceptions import ClientError
from rigging.filesystem.cluster_config import BucketSpec, StoreType, load_cluster_config, s3_data_buckets

logger = logging.getLogger(__name__)

_MARIN_CONFIG = load_cluster_config("marin")

# Lifecycle rules this script owns on S3-compatible buckets (R2 and CoreWeave)
# carry an ``ID`` with this prefix; everything else is treated as a foreign rule
# and left alone.
S3_RULE_ID_PREFIX = "marin-ttl-"


def _ttl_prefix(n: int) -> str:
    return f"{_MARIN_CONFIG.temp_path}/ttl={n}d/"


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
    foreign = [rule for rule in existing_rules if not _is_marin_s3_ttl_rule(rule)]
    merged = foreign + owned
    if dry_run:
        logger.info(
            "[dry-run] Would apply lifecycle rules (kept %d foreign + %d owned):\n%s",
            len(foreign),
            len(owned),
            json.dumps({"Rules": merged}, indent=2),
        )
        return
    logger.info("Applying lifecycle rules (kept %d foreign + %d owned).", len(foreign), len(owned))
    apply_s3_lifecycle(client, bucket, merged)


@click.command()
@click.option("--dry-run", is_flag=True, help="Print what would happen without executing.")
@click.option(
    "--bucket",
    type=str,
    default=None,
    help="Only configure this R2/CoreWeave bucket from config/*.yaml.",
)
def main(dry_run: bool, bucket: str | None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Each backend reads its own
    # namespaced credentials (R2_* / CW_*, AWS_* fallback) in its client factory.
    s3_buckets = s3_data_buckets()
    s3_targets: dict[str, BucketSpec] = dict(s3_buckets)
    if bucket is not None:
        if bucket not in s3_buckets:
            known = ", ".join(sorted(s3_buckets))
            logger.error("Unknown bucket %r. Known buckets: %s", bucket, known)
            sys.exit(1)
        s3_targets = {bucket: s3_buckets[bucket]}

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
