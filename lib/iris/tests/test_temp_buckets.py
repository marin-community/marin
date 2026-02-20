# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import MagicMock, patch

from iris.temp_buckets import get_temp_bucket_path, region_from_marin_prefix, region_from_metadata


def test_region_from_metadata_parses_zone():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central2-b"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert region_from_metadata() == "us-central2"


def test_region_from_metadata_returns_none_on_failure():
    with patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")):
        assert region_from_metadata() is None


def test_region_from_marin_prefix():
    with patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}):
        assert region_from_marin_prefix() == "us-east1"


def test_region_from_marin_prefix_not_set():
    with patch.dict(os.environ, {}, clear=True):
        assert region_from_marin_prefix() is None


def test_get_temp_bucket_path_from_metadata():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central2-b"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert get_temp_bucket_path(ttl_days=30, prefix="compilation-cache") == (
            "gs://marin-tmp-us-central2/ttl=30d/compilation-cache"
        )


def test_get_temp_bucket_path_falls_back_to_marin_prefix():
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert get_temp_bucket_path(ttl_days=3, prefix="zephyr") == "gs://marin-tmp-us-east1/ttl=3d/zephyr"


def test_get_temp_bucket_path_none_when_unresolvable():
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert get_temp_bucket_path(ttl_days=30) is None


def test_get_temp_bucket_path_unknown_region():
    with (
        patch("iris.temp_buckets.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-antarctica-south1/scratch"}),
    ):
        assert get_temp_bucket_path(ttl_days=30) is None


def test_get_temp_bucket_path_no_prefix():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-east1-c"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert get_temp_bucket_path(ttl_days=14) == "gs://marin-tmp-us-east1/ttl=14d"


def test_get_temp_bucket_path_strips_prefix_slashes():
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"projects/12345/zones/us-central1-a"
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None

    with patch("iris.temp_buckets.urllib.request.urlopen", return_value=mock_resp):
        assert get_temp_bucket_path(ttl_days=3, prefix="/foo/bar/") == "gs://marin-tmp-us-central1/ttl=3d/foo/bar"
