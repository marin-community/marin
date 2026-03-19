# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for region validation in inspect_data.py."""

import click
import logging

import pytest

from scripts.debug.inspect_data import _normalize_cluster_region, _validate_data_region


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
