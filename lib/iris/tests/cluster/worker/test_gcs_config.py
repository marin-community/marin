# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCS configuration."""

import os
from unittest.mock import patch

import pytest

from iris.cluster.worker.gcs_config import get_iris_log_prefix


def test_explicit_prefix_takes_precedence(monkeypatch):
    """Test that IRIS_WORKER_PREFIX environment variable takes precedence."""
    monkeypatch.setenv("IRIS_WORKER_PREFIX", "gs://my-bucket/logs")

    prefix = get_iris_log_prefix()
    assert prefix == "gs://my-bucket/logs"


def test_region_inference(monkeypatch):
    """Test that region is inferred from GCP metadata when env var not set."""
    # Remove IRIS_WORKER_PREFIX if set
    monkeypatch.delenv("IRIS_WORKER_PREFIX", raising=False)

    # Mock get_vm_region to return a test region
    with patch("iris.cluster.worker.gcs_config.get_vm_region") as mock_get_region:
        mock_get_region.return_value = "us-central2"

        prefix = get_iris_log_prefix()
        assert prefix == "gs://marin-tmp-us-central2/ttl=30d/iris-logs"


def test_returns_none_when_not_on_gcp(monkeypatch):
    """Test that None is returned when not on GCP and no env var set."""
    # Remove IRIS_WORKER_PREFIX if set
    monkeypatch.delenv("IRIS_WORKER_PREFIX", raising=False)

    # Mock get_vm_region to raise (simulating not on GCP)
    with patch("iris.cluster.worker.gcs_config.get_vm_region") as mock_get_region:
        mock_get_region.side_effect = ValueError("Not on GCP")

        prefix = get_iris_log_prefix()
        assert prefix is None
