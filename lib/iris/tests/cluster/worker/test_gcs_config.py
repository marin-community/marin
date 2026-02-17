# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCS configuration."""

import os
from unittest.mock import patch

import pytest

from iris.cluster.worker.gcs_config import get_iris_log_prefix


def test_explicit_prefix():
    """Test that explicit IRIS_WORKER_PREFIX is used."""
    with patch.dict(os.environ, {"IRIS_WORKER_PREFIX": "gs://my-bucket/logs"}):
        prefix = get_iris_log_prefix()
        assert prefix == "gs://my-bucket/logs"


def test_inferred_prefix():
    """Test that prefix is inferred from GCP region."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("marin.utilities.gcs_utils.get_vm_region", return_value="us-west1"):
            prefix = get_iris_log_prefix()
            assert prefix == "gs://marin-tmp-us-west1/ttl=30d/iris-logs"


def test_no_prefix_available():
    """Test that None is returned when no prefix can be determined."""
    with patch.dict(os.environ, {}, clear=True):
        with patch("marin.utilities.gcs_utils.get_vm_region", side_effect=ValueError("Not on GCP")):
            prefix = get_iris_log_prefix()
            assert prefix is None


def test_explicit_takes_precedence():
    """Test that explicit env var takes precedence over inference."""
    with patch.dict(os.environ, {"IRIS_WORKER_PREFIX": "gs://explicit/logs"}):
        with patch("marin.utilities.gcs_utils.get_vm_region", return_value="us-central1"):
            prefix = get_iris_log_prefix()
            assert prefix == "gs://explicit/logs"
