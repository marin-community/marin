# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for region resolution and temp-bucket routing (the cluster_config region
aspect). See test_cluster_config.py for DataConfig loading and resolved_root(),
test_cross_region.py for the transfer budget and guard, and test_factory.py for the
guarded entry points."""

import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rigging.filesystem.cluster_config import (
    check_gcs_paths_same_region,
    collect_gcs_paths,
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
        "rigging.filesystem.cluster_config.urllib.request.urlopen",
        return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b"),
    ):
        assert region_from_metadata() == "us-central2"


def test_region_from_metadata_returns_none_on_failure():
    with patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")):
        assert region_from_metadata() is None


@pytest.mark.parametrize(
    "prefix, expected",
    [
        ("gs://marin-us-east1/scratch", "us-east1"),
        ("gs://marin-us-central2/data", "us-central2"),
        # Abbreviated bucket name normalizes to canonical GCP region.
        ("gs://marin-eu-west4/tokenized", "europe-west4"),
        ("gs://other-bucket/foo", None),
        ("", None),
    ],
)
def test_region_from_prefix(prefix, expected):
    assert region_from_prefix(prefix) == expected


def test_marin_region_from_metadata():
    with patch(
        "rigging.filesystem.cluster_config.urllib.request.urlopen",
        return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
    ):
        assert marin_region() == "us-east1"


def test_marin_region_from_env_prefix():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-west4/scratch"}),
    ):
        assert marin_region() == "us-west4"


def test_marin_region_normalizes_eu_west4():
    """Regression test: marin-eu-west4 bucket must resolve to europe-west4."""
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-eu-west4/tokenized"}),
    ):
        assert marin_region() == "europe-west4"


def test_marin_region_none_when_unresolvable():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_region() is None


def test_marin_temp_bucket_from_metadata():
    with patch(
        "rigging.filesystem.cluster_config.urllib.request.urlopen",
        return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b"),
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="compilation-cache") == (
            "gs://marin-us-central2/tmp/ttl=30d/compilation-cache"
        )


def test_marin_temp_bucket_from_env_prefix():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="zephyr") == "gs://marin-us-east1/tmp/ttl=3d/zephyr"


def test_marin_temp_bucket_eu_west4_uses_main_bucket_alias():
    """eu-west4 region resolves to the canonical europe-west4 marin-eu-west4 bucket."""
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-eu-west4/scratch"}),
    ):
        assert marin_temp_bucket(ttl_days=1, prefix="ferry") == "gs://marin-eu-west4/tmp/ttl=1d/ferry"


def test_marin_temp_bucket_uses_source_prefix_region():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        assert marin_temp_bucket(
            ttl_days=14,
            prefix="checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints",
            source_prefix="gs://marin-us-east5/experiments/grug/run",
        ) == ("gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints")


def test_marin_temp_bucket_uses_source_prefix_region_from_local_launcher():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_temp_bucket(
            ttl_days=14,
            prefix="checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints",
            source_prefix="gs://marin-us-east5/experiments/grug/run",
        ) == ("gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints")


def test_marin_temp_bucket_r2_uses_bucket_root_ttl_path():
    """An R2 marin prefix resolves temp to the bucket root, dropping the marin/ subdir."""
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "s3://marin-na/marin"}),
    ):
        assert marin_temp_bucket(ttl_days=1, prefix="zephyr") == "s3://marin-na/tmp/ttl=1d/zephyr"
        assert marin_temp_bucket(ttl_days=14) == "s3://marin-na/tmp/ttl=14d"


def test_marin_temp_bucket_r2_uses_source_prefix_bucket():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert (
            marin_temp_bucket(ttl_days=3, prefix="ferry", source_prefix="s3://marin-na/experiments/grug")
            == "s3://marin-na/tmp/ttl=3d/ferry"
        )


def test_marin_temp_bucket_r2_source_prefix_overrides_gcs_launcher():
    """An explicit R2 source_prefix wins over a gs:// MARIN_PREFIX and GCP metadata."""
    with (
        patch(
            "rigging.filesystem.cluster_config.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-central1-a"),
        ),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        assert (
            marin_temp_bucket(ttl_days=7, prefix="out", source_prefix="s3://marin-na/experiments/grug")
            == "s3://marin-na/tmp/ttl=7d/out"
        )


def test_marin_temp_bucket_unknown_s3_bucket_falls_back_to_flat_path():
    """Unknown S3 buckets have no lifecycle rules, so they get the flat non-TTL fallback."""
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "s3://some-other-bucket/marin"}),
    ):
        assert marin_temp_bucket(ttl_days=1, prefix="x") == "s3://some-other-bucket/marin/tmp/x"


def test_marin_temp_bucket_falls_back_to_marin_prefix_when_no_region():
    # Unknown region in MARIN_PREFIX → no entry in DataConfig.region_buckets → falls back to marin_prefix/tmp
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-antarctica-south1/scratch"}),
    ):
        result = marin_temp_bucket(ttl_days=30)
        assert result == "gs://marin-antarctica-south1/scratch/tmp"


def test_marin_temp_bucket_local_fallback_when_unresolvable():
    with (
        patch("rigging.filesystem.cluster_config.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="iris-logs") == "file:///tmp/marin/tmp/iris-logs"


def test_marin_temp_bucket_no_prefix():
    with patch(
        "rigging.filesystem.cluster_config.urllib.request.urlopen",
        return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
    ):
        assert marin_temp_bucket(ttl_days=14) == "gs://marin-us-east1/tmp/ttl=14d"


def test_marin_temp_bucket_strips_prefix_slashes():
    with patch(
        "rigging.filesystem.cluster_config.urllib.request.urlopen",
        return_value=_mock_urlopen(b"projects/12345/zones/us-central1-a"),
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="/foo/bar/") == "gs://marin-us-central1/tmp/ttl=3d/foo/bar"


def test_marin_temp_bucket_rounds_up_unsupported_ttl(caplog):
    """ttl_days values between configured points round up to the next one with a warning."""
    with (
        patch(
            "rigging.filesystem.cluster_config.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
        ),
        caplog.at_level("WARNING", logger="rigging.filesystem.cluster_config"),
    ):
        # 10 → 14, 15 → 30
        assert marin_temp_bucket(ttl_days=10, prefix="zephyr") == "gs://marin-us-east1/tmp/ttl=14d/zephyr"
        assert marin_temp_bucket(ttl_days=15) == "gs://marin-us-east1/tmp/ttl=30d"
    assert any("rounding up to 14" in rec.message for rec in caplog.records)
    assert any("rounding up to 30" in rec.message for rec in caplog.records)


def test_marin_temp_bucket_clamps_above_max_ttl(caplog):
    """ttl_days above the configured maximum clamp to the max with a warning."""
    with (
        patch(
            "rigging.filesystem.cluster_config.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
        ),
        caplog.at_level("WARNING", logger="rigging.filesystem.cluster_config"),
    ):
        assert marin_temp_bucket(ttl_days=100) == "gs://marin-us-east1/tmp/ttl=30d"
    assert any("clamping to 30" in rec.message for rec in caplog.records)


def test_marin_temp_bucket_rejects_non_positive_ttl():
    with pytest.raises(ValueError, match="must be positive"):
        marin_temp_bucket(ttl_days=0)


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


@dataclass
class _PathHolder:
    path: str


def test_collect_gcs_paths_recurses_and_skips_prefixes():
    payload = {
        "cache_dir": "gs://bucket/path",
        "train_urls": ["gs://bucket/source"],
        "nested": _PathHolder(path=Path("gs://bucket/nested")),
        "set_field": {"gs://bucket/from_set"},
    }
    paths = collect_gcs_paths(payload, path_prefix="config", skip_if_prefix_contains=("train_urls",))
    assert sorted(paths) == [
        ("config.cache_dir", "gs://bucket/path"),
        ("config.nested.path", "gs://bucket/nested"),
        ("config.set_field[0]", "gs://bucket/from_set"),
    ]
