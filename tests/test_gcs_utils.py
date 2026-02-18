# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

from marin.utilities.gcs_utils import REGION_TO_TMP_BUCKET, get_temp_bucket_path


def test_get_temp_bucket_path_known_region():
    with patch("marin.utilities.gcs_utils.get_vm_region", return_value="us-central2"):
        result = get_temp_bucket_path(ttl_days=30, prefix="compilation-cache")
        assert result == "gs://marin-tmp-us-central2/ttl=30d/compilation-cache"


def test_get_temp_bucket_path_europe_region():
    """europe-west4 maps to marin-tmp-eu-west4 (abbreviated name)."""
    with patch("marin.utilities.gcs_utils.get_vm_region", return_value="europe-west4"):
        result = get_temp_bucket_path(ttl_days=7, prefix="checkpoints")
        assert result == "gs://marin-tmp-eu-west4/ttl=7d/checkpoints"


def test_get_temp_bucket_path_unknown_region():
    with patch("marin.utilities.gcs_utils.get_vm_region", return_value="antarctica-south1"):
        assert get_temp_bucket_path(ttl_days=30) is None


def test_get_temp_bucket_path_non_gcp_no_marin_prefix():
    with (
        patch("marin.utilities.gcs_utils.get_vm_region", side_effect=ValueError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert get_temp_bucket_path(ttl_days=30) is None


def test_get_temp_bucket_path_falls_back_to_marin_prefix():
    """Falls back to MARIN_PREFIX when metadata is unavailable."""
    with (
        patch("marin.utilities.gcs_utils.get_vm_region", side_effect=ValueError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert (
            get_temp_bucket_path(ttl_days=3, prefix="compilation-cache")
            == "gs://marin-tmp-us-east1/ttl=3d/compilation-cache"
        )


def test_get_temp_bucket_path_marin_prefix_unknown_region():
    """MARIN_PREFIX with an unmapped region returns None."""
    with (
        patch("marin.utilities.gcs_utils.get_vm_region", side_effect=ValueError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-antarctica-south1/scratch"}),
    ):
        assert get_temp_bucket_path(ttl_days=30) is None


def test_get_temp_bucket_path_no_prefix():
    with patch("marin.utilities.gcs_utils.get_vm_region", return_value="us-east1"):
        result = get_temp_bucket_path(ttl_days=14)
        assert result == "gs://marin-tmp-us-east1/ttl=14d"


def test_get_temp_bucket_path_strips_prefix_slashes():
    with patch("marin.utilities.gcs_utils.get_vm_region", return_value="us-central1"):
        result = get_temp_bucket_path(ttl_days=3, prefix="/foo/bar/")
        assert result == "gs://marin-tmp-us-central1/ttl=3d/foo/bar"


def test_region_mapping_covers_all_configured_buckets():
    """Verify our mapping covers all regions from configure_temp_buckets.py."""
    # The canonical BUCKETS dict is bucket_name -> region. Ours is region -> bucket_name.
    # Hard-code the expected set to detect drift.
    expected_regions = {
        "asia-northeast1",
        "us-central1",
        "us-central2",
        "europe-west4",
        "eu-west4",
        "us-west4",
        "us-east1",
        "us-east5",
    }
    assert set(REGION_TO_TMP_BUCKET.keys()) == expected_regions
