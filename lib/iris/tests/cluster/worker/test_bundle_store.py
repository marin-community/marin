# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for worker-side BundleStore behavior."""

import hashlib
import io
import zipfile

import pytest

from iris.cluster.bundle import BundleStore


class _FakeResponse:
    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return self._data


def _make_zip(entries: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return output.getvalue()


@pytest.fixture
def store(tmp_path):
    return BundleStore(
        storage_dir=tmp_path / "bundles",
        controller_address="http://controller.internal",
        max_items=2,
    )


def test_extract_bundle_fetches_on_demand(monkeypatch, store, tmp_path):
    """extract_bundle_to should fetch from controller on cache miss."""
    bundle_zip = _make_zip({"main.py": b"print('hello')", "src/module.py": b"def f():\n  return 1\n"})
    bundle_id = hashlib.sha256(bundle_zip).hexdigest()

    def fake_urlopen(url: str, timeout: int):
        assert url == f"http://controller.internal/bundles/{bundle_id}.zip"
        return _FakeResponse(bundle_zip)

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)

    extract_dir = tmp_path / "extract"
    store.extract_bundle_to(bundle_id, extract_dir)
    assert (extract_dir / "main.py").exists()
    assert (extract_dir / "src/module.py").exists()


def test_extract_bundle_uses_cache_on_hit(store, tmp_path):
    """extract_bundle_to should use local cache without hitting the network."""
    bundle_zip = _make_zip({"cached.txt": b"cached data"})
    bundle_id = store.write_zip(bundle_zip)

    extract_dir = tmp_path / "extract"
    store.extract_bundle_to(bundle_id, extract_dir)
    assert (extract_dir / "cached.txt").read_bytes() == b"cached data"


def test_extract_bundle_hash_verification_failure(monkeypatch, store, tmp_path):
    bad_zip = _make_zip({"a.txt": b"A"})
    wrong_id = "a" * 64

    def fake_urlopen(url: str, timeout: int):
        assert url == f"http://controller.internal/bundles/{wrong_id}.zip"
        return _FakeResponse(bad_zip)

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)
    with pytest.raises(ValueError, match="Bundle hash mismatch"):
        store.extract_bundle_to(wrong_id, tmp_path / "extract")


def test_lru_eviction_by_item_count(store):
    bundles = []
    for i in range(3):
        bundle_zip = _make_zip({"test.txt": f"bundle {i}".encode()})
        bundle_id = hashlib.sha256(bundle_zip).hexdigest()
        bundles.append((bundle_id, bundle_zip))
        store.write_zip(bundle_zip)

    with pytest.raises(FileNotFoundError, match="Bundle not found"):
        store.get_zip(bundles[0][0])
    assert store.get_zip(bundles[1][0]) == bundles[1][1]
    assert store.get_zip(bundles[2][0]) == bundles[2][1]
