# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.marin_fs import (
    CROSS_REGION_READ_THRESHOLD_BYTES,
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    CrossRegionGuardedFS,
    CrossRegionReadError,
    _fs_is_gcs,
    check_gcs_paths_same_region,
    guarded_open,
    guarded_url_to_fs,
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
        assert marin_temp_bucket(ttl_days=30, prefix="iris-logs") == "file:///tmp/marin/tmp/iris-logs"


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


# ---------------------------------------------------------------------------
# CrossRegionGuardedFS tests
# ---------------------------------------------------------------------------


class _FakeGCSFS:
    """Minimal fake GCS filesystem for testing the cross-region guard.

    Stores files as {path: bytes} and reports protocol ``gs``.
    """

    protocol = "gs"

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def add_file(self, path: str, data: bytes) -> None:
        self._files[path] = data

    def size(self, path: str) -> int | None:
        data = self._files.get(path)
        return len(data) if data is not None else None

    def open(self, path: str, mode: str = "rb", **kwargs):
        return self._files.get(path, b"")

    def cat_file(self, path: str, start=None, end=None, **kwargs) -> bytes:
        return self._files.get(path, b"")

    def cat(self, path, recursive=False, on_error="raise", **kwargs):
        if isinstance(path, str):
            return self._files.get(path, b"")
        return {p: self._files.get(p, b"") for p in path}

    def get_file(self, rpath: str, lpath: str, **kwargs) -> None:
        pass

    def exists(self, path: str) -> bool:
        return path in self._files


class _FakeLocalFS:
    """Minimal fake local filesystem (protocol ``file``)."""

    protocol = "file"

    def open(self, path: str, mode: str = "rb", **kwargs):
        return b""

    def size(self, path: str) -> int:
        return 999_999_999


def test_fs_is_gcs_detects_gs_protocol():
    assert _fs_is_gcs(_FakeGCSFS()) is True
    assert _fs_is_gcs(_FakeLocalFS()) is False


def test_guarded_fs_blocks_large_cross_region_read():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with pytest.raises(CrossRegionReadError, match="Cross-region read blocked"):
        guarded.open("remote-bucket/big-file.bin", "rb")


def test_guarded_fs_allows_small_cross_region_read():
    fs = _FakeGCSFS()
    small_data = b"x" * 1024  # 1 KB
    fs.add_file("remote-bucket/small-file.bin", small_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    # Should not raise
    guarded.open("remote-bucket/small-file.bin", "rb")


def test_guarded_fs_allows_same_region_large_read():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("local-bucket/big-file.bin", large_data)

    # cross_region_checker returns False → same region
    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: False)

    guarded.open("local-bucket/big-file.bin", "rb")


def test_guarded_fs_override_env_allows_large_cross_region_read():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with patch.dict(os.environ, {MARIN_CROSS_REGION_OVERRIDE_ENV: "testuser"}):
        guarded.open("remote-bucket/big-file.bin", "rb")


def test_guarded_fs_skips_non_gcs_filesystem():
    fs = _FakeLocalFS()
    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    # Should not raise even though cross_region_checker returns True,
    # because the filesystem is not GCS.
    guarded.open("/some/local/path", "rb")


def test_guarded_fs_allows_write_mode():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    # Write mode should not trigger the guard
    guarded.open("remote-bucket/big-file.bin", "wb")


def test_guarded_fs_cat_file_blocked():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with pytest.raises(CrossRegionReadError):
        guarded.cat_file("remote-bucket/big-file.bin")


def test_guarded_fs_cat_list_blocked():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with pytest.raises(CrossRegionReadError):
        guarded.cat(["remote-bucket/big-file.bin"])


def test_guarded_fs_get_file_blocked():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with pytest.raises(CrossRegionReadError):
        guarded.get_file("remote-bucket/big-file.bin", "/tmp/local")


def test_guarded_fs_custom_threshold():
    fs = _FakeGCSFS()
    data = b"x" * 500  # 500 bytes
    fs.add_file("remote-bucket/file.bin", data)

    # Threshold of 100 bytes → should block
    guarded = CrossRegionGuardedFS(
        fs,
        threshold_bytes=100,
        cross_region_checker=lambda _bucket: True,
    )

    with pytest.raises(CrossRegionReadError):
        guarded.open("remote-bucket/file.bin", "rb")


def test_guarded_fs_delegates_non_read_methods():
    fs = _FakeGCSFS()
    fs.add_file("bucket/file.txt", b"hello")

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    # exists() is not intercepted, should delegate transparently
    assert guarded.exists("bucket/file.txt") is True
    assert guarded.exists("bucket/nope.txt") is False


def test_guarded_url_to_fs_returns_guarded_fs(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    guarded_fs, _path = guarded_url_to_fs(str(test_file))
    assert isinstance(guarded_fs, CrossRegionGuardedFS)


def test_guarded_open_local_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    # Local files should work fine
    result = guarded_open(str(test_file), "r")
    with result as f:
        assert f.read() == "hello"


def test_cross_region_read_error_message():
    fs = _FakeGCSFS()
    large_data = b"x" * (CROSS_REGION_READ_THRESHOLD_BYTES + 1)
    fs.add_file("remote-bucket/big-file.bin", large_data)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _bucket: True)

    with pytest.raises(CrossRegionReadError, match=MARIN_CROSS_REGION_OVERRIDE_ENV):
        guarded.open("remote-bucket/big-file.bin", "rb")
