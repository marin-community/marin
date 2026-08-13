# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import os
import pickle
from unittest.mock import patch

import pytest
import rigging.filesystem.cross_region as rfs
from rigging.filesystem.cross_region import (
    MARIN_CROSS_REGION_OVERRIDE_ENV,
    CrossRegionGuardedFS,
    TransferBudget,
    TransferBudgetExceeded,
    _bucket_from_gcs_url,
    _regions_match,
    record_transfer,
)


@pytest.mark.parametrize(
    ("url", "bucket"),
    [
        ("gs://bucket/path", "bucket"),
        ("gcs://bucket/path", "bucket"),
        ("gs://bucket", "bucket"),
        ("s3://bucket/path", None),
        ("/local/path", None),
    ],
)
def test_bucket_from_gcs_url(url, bucket):
    assert _bucket_from_gcs_url(url) == bucket


@pytest.fixture()
def patched_regions(monkeypatch):
    """VM is in us-central1; bucket lookup is faked from the URL."""
    monkeypatch.setattr(rfs, "cached_marin_region", lambda: "us-central1")
    monkeypatch.setattr(
        rfs,
        "_cached_bucket_location",
        lambda bucket: {
            "marin-us-central1": "us-central1",
            "marin-us-central2": "us-central2",
        }.get(bucket),
    )


def test_record_transfer_charges_cross_region(patched_regions):
    budget = TransferBudget(limit_bytes=1024 * 1024)
    record_transfer(1000, "gs://marin-us-central2/checkpoint", budget=budget)
    assert budget.bytes_used == 1000


def test_record_transfer_skips_same_region_and_local(patched_regions):
    budget = TransferBudget(limit_bytes=1024 * 1024)
    record_transfer(1000, "gs://marin-us-central1/checkpoint", budget=budget)
    record_transfer(1000, "/tmp/checkpoint", budget=budget)
    assert budget.bytes_used == 0


def test_record_transfer_raises_when_budget_exceeded(patched_regions):
    budget = TransferBudget(limit_bytes=500)
    with pytest.raises(TransferBudgetExceeded):
        record_transfer(1000, "gs://marin-us-central2/checkpoint", budget=budget)


def test_transfer_budget_exceeded_round_trips_through_pickle():
    # The exception crosses process boundaries (Zephyr ships shard/coordinator
    # errors via cloudpickle). It must revive without a constructor TypeError.
    original = TransferBudgetExceeded(bytes_used=9_960, attempted=400, limit=10_000, path="gs://marin-us-east5/x")
    revived = pickle.loads(pickle.dumps(original))
    assert (revived.bytes_used, revived.attempted, revived.limit, revived.path) == (
        9_960,
        400,
        10_000,
        "gs://marin-us-east5/x",
    )
    assert str(revived) == str(original)
    assert "Cross-region transfer budget exceeded" in str(revived)


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
