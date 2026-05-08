# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for cluster-region / data-region validation helpers.

These helpers used to live in ``scripts/debug/inspect_data.py`` (the Ray-only
``inspect_data`` CLI). That script was deleted when Ray operator tooling was
retired; the helpers were small enough that moving them here — the only
importer — was simpler than finding a new home for them.
"""

import logging

import click
import pytest


def _normalize_cluster_region(cluster: str) -> str:
    """Extract the canonical region string from a --cluster argument.

    Accepts region names like ``us-central2`` or ``marin-us-central2`` and
    strips the ``marin-`` prefix if present.
    """
    region = cluster
    if region.startswith("marin-"):
        region = region[len("marin-") :]
    return region


def _validate_data_region(wandb_dict: dict, cluster: str) -> None:
    """Ensure every component cache_dir lives in the same region as *cluster*.

    Raises ``click.ClickException`` if any component's GCS bucket is in a
    different region, preventing accidental cross-region egress charges.
    """
    from rigging.filesystem import REGION_TO_DATA_BUCKET

    logger = logging.getLogger(__name__)

    cluster_region = _normalize_cluster_region(cluster)

    # Build reverse mapping: bucket name -> region
    bucket_to_region: dict[str, str] = {bucket: region for region, bucket in REGION_TO_DATA_BUCKET.items()}

    mismatches: list[str] = []
    for name, cache_dir in wandb_dict.get("components", {}).items():
        if not cache_dir or not cache_dir.startswith("gs://"):
            continue
        # Extract bucket name from gs://bucket/path
        bucket_name = cache_dir.split("/")[2]
        data_region = bucket_to_region.get(bucket_name)
        if data_region is None:
            # Unknown bucket — can't verify, warn but allow
            logger.warning(
                "Cannot determine region for bucket %r (component %r); skipping region check.",
                bucket_name,
                name,
            )
            continue
        if data_region != cluster_region:
            mismatches.append(
                f"  - component {name!r}: bucket {bucket_name!r} is in {data_region}, "
                f"but cluster is in {cluster_region}"
            )

    if mismatches:
        detail = "\n".join(mismatches)
        raise click.ClickException(
            f"Cross-region data access detected! The data lives in a different region "
            f"than the target cluster ({cluster_region}).\n"
            f"This would incur significant egress charges.\n\n"
            f"{detail}\n\n"
            f"Please use --cluster <region> matching your data's region."
        )


def test_normalize_cluster_region_plain():
    assert _normalize_cluster_region("us-central2") == "us-central2"


def test_normalize_cluster_region_with_marin_prefix():
    assert _normalize_cluster_region("marin-us-central2") == "us-central2"


def test_validate_data_region_same_region(monkeypatch):
    """No error when data and cluster are in the same region."""
    wandb_dict = {
        "components": {
            "pile": "gs://marin-us-central2/data/pile",
            "wiki": "gs://marin-us-central2/data/wiki",
        }
    }
    # Should not raise
    _validate_data_region(wandb_dict, "us-central2")


def test_validate_data_region_mismatch(monkeypatch):
    """Error when data is in a different region than the cluster."""
    wandb_dict = {
        "components": {
            "pile": "gs://marin-us-central1/data/pile",
        }
    }
    with pytest.raises(click.ClickException, match="Cross-region data access detected"):
        _validate_data_region(wandb_dict, "us-central2")


def test_validate_data_region_unknown_bucket_warns(caplog):
    """Unknown buckets produce a warning but don't block."""
    wandb_dict = {
        "components": {
            "custom": "gs://my-custom-bucket/data/stuff",
        }
    }
    with caplog.at_level(logging.WARNING):
        # Should not raise
        _validate_data_region(wandb_dict, "us-central2")
    assert "Cannot determine region" in caplog.text


def test_validate_data_region_non_gcs_skipped():
    """Non-GCS paths are silently skipped."""
    wandb_dict = {
        "components": {
            "local": "/tmp/data/stuff",
        }
    }
    # Should not raise
    _validate_data_region(wandb_dict, "us-central2")


def test_validate_data_region_empty_components():
    """No error when there are no components."""
    _validate_data_region({"components": {}}, "us-central2")
    _validate_data_region({}, "us-central2")
