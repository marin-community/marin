# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.marin_fs import (
    check_gcs_paths_same_region,
    marin_prefix,
    marin_region,
    marin_temp_bucket,
    region_from_metadata,
    region_from_prefix,
)


def _mock_urlopen(zone_bytes: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = zone_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None
    return mock_resp


def test_region_from_metadata_parses_zone():
    with patch(
        "iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b")
    ):
        assert region_from_metadata() == "us-central2"


def test_region_from_metadata_returns_none_on_failure():
    with patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")):
        assert region_from_metadata() is None


def test_region_from_prefix():
    assert region_from_prefix("gs://marin-us-east1/scratch") == "us-east1"


def test_region_from_prefix_not_a_marin_bucket():
    assert region_from_prefix("gs://other-bucket/foo") is None


def test_region_from_prefix_empty():
    assert region_from_prefix("") is None


def test_marin_prefix_from_env():
    with patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1"}):
        assert marin_prefix() == "gs://marin-us-central1"


def test_marin_prefix_from_metadata():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_prefix() == "gs://marin-us-central2"


def test_marin_prefix_falls_back_to_local():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_prefix() == "/tmp/marin"


def test_marin_region_from_metadata():
    with patch("iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c")):
        assert marin_region() == "us-east1"


def test_marin_region_from_env_prefix():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-west4/scratch"}),
    ):
        assert marin_region() == "us-west4"


def test_marin_region_none_when_unresolvable():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_region() is None


def test_marin_temp_bucket_from_metadata():
    with patch(
        "iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b")
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="compilation-cache") == (
            "gs://marin-tmp-us-central2/ttl=30d/compilation-cache"
        )


def test_marin_temp_bucket_from_env_prefix():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="zephyr") == "gs://marin-tmp-us-east1/ttl=3d/zephyr"


def test_marin_temp_bucket_falls_back_to_marin_prefix_when_no_region():
    # Unknown region in MARIN_PREFIX → no entry in REGION_TO_TMP_BUCKET → falls back to marin_prefix/tmp
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-antarctica-south1/scratch"}),
    ):
        result = marin_temp_bucket(ttl_days=30)
        assert result == "gs://marin-antarctica-south1/scratch/tmp"


def test_marin_temp_bucket_local_fallback_when_unresolvable():
    with (
        patch("iris.marin_fs.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="iris-logs") == "/tmp/marin/tmp/iris-logs"


def test_marin_temp_bucket_no_prefix():
    with patch("iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c")):
        assert marin_temp_bucket(ttl_days=14) == "gs://marin-tmp-us-east1/ttl=14d"


def test_marin_temp_bucket_strips_prefix_slashes():
    with patch(
        "iris.marin_fs.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central1-a")
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="/foo/bar/") == "gs://marin-tmp-us-central1/ttl=3d/foo/bar"


def test_check_gcs_paths_same_region_accepts_matching_region():
    config = {"cache_dir": "gs://bucket/path"}

    check_gcs_paths_same_region(
        config,
        local_ok=False,
        region="us-central1",
        path_checker=lambda _key, _path, _region, _local_ok: None,
    )


def test_check_gcs_paths_same_region_raises_for_mismatch():
    config = {"cache_dir": Path("gs://bucket/path")}

    def checker(_key: str, _path: str, _region: str, _local_ok: bool) -> None:
        raise ValueError("not in the same region")

    with pytest.raises(ValueError, match="not in the same region"):
        check_gcs_paths_same_region(
            config,
            local_ok=False,
            region="us-central1",
            path_checker=checker,
        )


def test_check_gcs_paths_same_region_skips_train_source_urls():
    config = {"train_urls": ["gs://bucket/path"], "validation_urls": ["gs://bucket/path"]}

    def checker(_key: str, _path: str, _region: str, _local_ok: bool) -> None:
        raise AssertionError("source URLs should be skipped")

    check_gcs_paths_same_region(
        config,
        local_ok=False,
        region="us-central1",
        path_checker=checker,
    )


def test_check_gcs_paths_same_region_allows_unknown_region_for_local_runs():
    def fail_region_lookup() -> str | None:
        return None

    check_gcs_paths_same_region(
        {"cache_dir": "gs://bucket/path"},
        local_ok=True,
        region_getter=fail_region_lookup,
    )
