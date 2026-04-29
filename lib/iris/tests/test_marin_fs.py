# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rigging.filesystem import (
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    CrossRegionGuardedFS,
    TransferBudget,
    TransferBudgetExceeded,
    _regions_match,
    check_gcs_paths_same_region,
    collect_gcs_paths,
    filesystem,
    marin_prefix,
    marin_region,
    marin_temp_bucket,
    open_url,
    region_from_metadata,
    region_from_prefix,
    url_to_fs,
)


def _mock_urlopen(zone_bytes: bytes) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.read.return_value = zone_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None
    return mock_resp


def test_region_from_metadata_parses_zone():
    with patch(
        "rigging.filesystem.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b")
    ):
        assert region_from_metadata() == "us-central2"


def test_region_from_metadata_returns_none_on_failure():
    with patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")):
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


def test_marin_prefix_from_env():
    with patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1"}):
        assert marin_prefix() == "gs://marin-us-central1"


def test_marin_prefix_from_metadata():
    with (
        patch(
            "rigging.filesystem.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b"),
        ),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_prefix() == "gs://marin-us-central2"


def test_marin_prefix_falls_back_to_local():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_prefix() == "/tmp/marin"


def test_marin_region_from_metadata():
    with patch(
        "rigging.filesystem.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c")
    ):
        assert marin_region() == "us-east1"


def test_marin_region_from_env_prefix():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-west4/scratch"}),
    ):
        assert marin_region() == "us-west4"


def test_marin_region_normalizes_eu_west4():
    """Regression test: marin-eu-west4 bucket must resolve to europe-west4."""
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-eu-west4/tokenized"}),
    ):
        assert marin_region() == "europe-west4"


def test_marin_region_none_when_unresolvable():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_region() is None


def test_marin_temp_bucket_from_metadata():
    with patch(
        "rigging.filesystem.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central2-b")
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="compilation-cache") == (
            "gs://marin-us-central2/tmp/ttl=30d/compilation-cache"
        )


def test_marin_temp_bucket_from_env_prefix():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-east1/scratch"}),
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="zephyr") == "gs://marin-us-east1/tmp/ttl=3d/zephyr"


def test_marin_temp_bucket_eu_west4_uses_main_bucket_alias():
    """eu-west4 region resolves to the canonical europe-west4 marin-eu-west4 bucket."""
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-eu-west4/scratch"}),
    ):
        assert marin_temp_bucket(ttl_days=1, prefix="ferry") == "gs://marin-eu-west4/tmp/ttl=1d/ferry"


def test_marin_temp_bucket_uses_source_prefix_region():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-us-central1/scratch"}),
    ):
        assert marin_temp_bucket(
            ttl_days=14,
            prefix="checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints",
            source_prefix="gs://marin-us-east5/experiments/grug/run",
        ) == ("gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints")


def test_marin_temp_bucket_uses_source_prefix_region_from_local_launcher():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_temp_bucket(
            ttl_days=14,
            prefix="checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints",
            source_prefix="gs://marin-us-east5/experiments/grug/run",
        ) == ("gs://marin-us-east5/tmp/ttl=14d/" "checkpoints-temp/marin-us-east5/experiments/grug/run/checkpoints")


def test_marin_temp_bucket_falls_back_to_marin_prefix_when_no_region():
    # Unknown region in MARIN_PREFIX → no entry in REGION_TO_DATA_BUCKET → falls back to marin_prefix/tmp
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {"MARIN_PREFIX": "gs://marin-antarctica-south1/scratch"}),
    ):
        result = marin_temp_bucket(ttl_days=30)
        assert result == "gs://marin-antarctica-south1/scratch/tmp"


def test_marin_temp_bucket_local_fallback_when_unresolvable():
    with (
        patch("rigging.filesystem.urllib.request.urlopen", side_effect=OSError("not on GCP")),
        patch.dict(os.environ, {}, clear=True),
    ):
        assert marin_temp_bucket(ttl_days=30, prefix="iris-logs") == "file:///tmp/marin/tmp/iris-logs"


def test_marin_temp_bucket_no_prefix():
    with patch(
        "rigging.filesystem.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c")
    ):
        assert marin_temp_bucket(ttl_days=14) == "gs://marin-us-east1/tmp/ttl=14d"


def test_marin_temp_bucket_strips_prefix_slashes():
    with patch(
        "rigging.filesystem.urllib.request.urlopen", return_value=_mock_urlopen(b"projects/12345/zones/us-central1-a")
    ):
        assert marin_temp_bucket(ttl_days=3, prefix="/foo/bar/") == "gs://marin-us-central1/tmp/ttl=3d/foo/bar"


def test_marin_temp_bucket_rounds_up_unsupported_ttl(caplog):
    """ttl_days values between configured points round up to the next one with a warning."""
    with (
        patch(
            "rigging.filesystem.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
        ),
        caplog.at_level("WARNING", logger="rigging.filesystem"),
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
            "rigging.filesystem.urllib.request.urlopen",
            return_value=_mock_urlopen(b"projects/12345/zones/us-east1-c"),
        ),
        caplog.at_level("WARNING", logger="rigging.filesystem"),
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


# ---------------------------------------------------------------------------
# _regions_match tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vm_region, bucket_location, expected",
    [
        ("us-central1", "us-central1", True),
        ("US-Central1", "us-central1", True),
        ("europe-west4", "europe-west4", True),
        ("us-central1", "eu-west4", False),
        ("us-central1", "us", True),
        ("us-east1", "us", True),
        ("europe-west4", "eu", True),
        ("asia-northeast1", "asia", True),
        ("eu-west4", "us", False),
        ("us-central1", "asia", False),
    ],
)
def test_regions_match(vm_region, bucket_location, expected):
    assert _regions_match(vm_region, bucket_location) is expected


# ---------------------------------------------------------------------------
# TransferBudget tests
# ---------------------------------------------------------------------------


def test_budget_records_and_blocks():
    budget = TransferBudget(limit_bytes=1000)
    budget.record(400, "a")
    budget.record(400, "b")
    assert budget.bytes_used == 800

    with pytest.raises(TransferBudgetExceeded, match="transfer budget exceeded"):
        budget.record(300, "c")

    # Counter unchanged on failure.
    assert budget.bytes_used == 800


def test_budget_thread_safety():
    budget = TransferBudget(limit_bytes=10 * 1024 * 1024)

    def record_batch():
        for _ in range(100):
            budget.record(1, "x")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: record_batch(), range(8)))

    assert budget.bytes_used == 800


# ---------------------------------------------------------------------------
# CrossRegionGuardedFS tests
# ---------------------------------------------------------------------------


class _FakeGCSFS:
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

    def get(self, rpath, lpath, recursive=False, **kwargs) -> None:
        pass

    def exists(self, path: str) -> bool:
        return path in self._files


@pytest.fixture()
def budget():
    return TransferBudget(limit_bytes=1024)


def test_guarded_fs_charges_budget_for_cross_region_reads(budget):
    fs = _FakeGCSFS()
    for i in range(3):
        fs.add_file(f"remote-bucket/f{i}.bin", b"x" * 400)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: True, budget=budget)

    guarded.open("remote-bucket/f0.bin", "rb")
    guarded.open("remote-bucket/f1.bin", "rb")
    assert budget.bytes_used == 800

    with pytest.raises(TransferBudgetExceeded):
        guarded.open("remote-bucket/f2.bin", "rb")


def test_guarded_fs_skips_same_region(budget):
    fs = _FakeGCSFS()
    fs.add_file("local-bucket/big.bin", b"x" * 9999)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: False, budget=budget)
    guarded.open("local-bucket/big.bin", "rb")
    assert budget.bytes_used == 0


def test_guarded_fs_override_env_bypasses_budget(budget):
    fs = _FakeGCSFS()
    fs.add_file("remote-bucket/big.bin", b"x" * 2000)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: True, budget=budget)

    with patch.dict(os.environ, {MARIN_CROSS_REGION_OVERRIDE_ENV: "testuser"}):
        guarded.open("remote-bucket/big.bin", "rb")

    assert budget.bytes_used == 0


def test_guarded_fs_write_mode_skips_guard(budget):
    fs = _FakeGCSFS()
    fs.add_file("remote-bucket/big.bin", b"x" * 2000)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: True, budget=budget)
    guarded.open("remote-bucket/big.bin", "wb")
    assert budget.bytes_used == 0


@pytest.mark.parametrize(
    "method, args",
    [
        ("cat_file", ("remote-bucket/f.bin",)),
        ("cat", (["remote-bucket/f.bin"],)),
        ("get_file", ("remote-bucket/f.bin", "/tmp/local")),
        ("get", ("remote-bucket/f.bin", "/tmp/local")),
    ],
    ids=["cat_file", "cat_list", "get_file", "get"],
)
def test_guarded_fs_all_read_methods_charge_budget(budget, method, args):
    fs = _FakeGCSFS()
    fs.add_file("remote-bucket/f.bin", b"x" * 100)

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: True, budget=budget)
    getattr(guarded, method)(*args)
    assert budget.bytes_used == 100


def test_guarded_fs_delegates_non_read_methods():
    fs = _FakeGCSFS()
    fs.add_file("bucket/file.txt", b"hello")

    guarded = CrossRegionGuardedFS(fs, cross_region_checker=lambda _: True)
    assert guarded.exists("bucket/file.txt") is True
    assert guarded.exists("bucket/nope.txt") is False


# ---------------------------------------------------------------------------
# Guarded entry point tests
# ---------------------------------------------------------------------------


def test_url_to_fs_does_not_wrap_local(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    fs, _path = url_to_fs(str(test_file))
    assert not isinstance(fs, CrossRegionGuardedFS)


def test_open_url_local_file(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello")

    result = open_url(str(test_file), "r")
    with result as f:
        assert f.read() == "hello"


def test_filesystem_local():
    fs = filesystem("file")
    assert not isinstance(fs, CrossRegionGuardedFS)
