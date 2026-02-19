# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

import pytest

from marin.utilities.gcs_utils import get_temp_bucket_path, get_vm_region

# The canonical temp-bucket tests live in lib/iris/tests/test_temp_buckets.py.
# These tests verify the marin re-exports and the get_vm_region wrapper.


def test_get_temp_bucket_path_via_marin_reexport():
    """get_temp_bucket_path re-exported from iris works through marin."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central2-b"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        result = get_temp_bucket_path(ttl_days=30, prefix="compilation-cache")
        assert result == "gs://marin-tmp-us-central2/ttl=30d/compilation-cache"


def test_get_temp_bucket_path_marin_prefix_fallback():
    """MARIN_PREFIX fallback works through the marin re-export."""
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert (
            get_temp_bucket_path(ttl_days=3, prefix="compilation-cache")
            == "gs://marin-tmp-us-east1/ttl=3d/compilation-cache"
        )


def test_get_vm_region_returns_region():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central2-b"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert get_vm_region() == "us-central2"


def test_get_vm_region_raises_when_unavailable():
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        pytest.raises(ValueError),
    ):
        get_vm_region()
