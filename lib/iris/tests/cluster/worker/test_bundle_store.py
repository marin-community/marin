# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for LocalBundleStore."""

import hashlib
import io
import os
import zipfile

import pytest
from iris.cluster.bundle import LocalBundleStore


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
def temp_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


def test_download_and_extract_bundle(monkeypatch, temp_cache_dir):
    bundle_zip = _make_zip({"main.py": b"print('hello')", "src/module.py": b"def f():\n  return 1\n"})
    bundle_id = hashlib.sha256(bundle_zip).hexdigest()

    def fake_urlopen(url: str, timeout: int):
        assert url == f"http://controller.internal/bundles/{bundle_id}.zip"
        return _FakeResponse(bundle_zip)

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)
    cache = LocalBundleStore(temp_cache_dir, controller_address="http://controller.internal")

    extract_path = cache.get_bundle(bundle_id)
    assert (extract_path / "main.py").exists()
    assert (extract_path / "src/module.py").exists()


def test_caching_behavior(monkeypatch, temp_cache_dir):
    bundle_zip = _make_zip({"a.txt": b"A"})
    bundle_id = hashlib.sha256(bundle_zip).hexdigest()
    calls = 0

    def fake_urlopen(url: str, timeout: int):
        nonlocal calls
        calls += 1
        assert url == f"http://controller.internal/bundles/{bundle_id}.zip"
        return _FakeResponse(bundle_zip)

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)
    cache = LocalBundleStore(temp_cache_dir, controller_address="http://controller.internal")

    path1 = cache.get_bundle(bundle_id)
    path2 = cache.get_bundle(bundle_id)

    assert path1 == path2
    assert calls == 1


def test_hash_verification_failure(monkeypatch, temp_cache_dir):
    bad_zip = _make_zip({"a.txt": b"A"})
    wrong_id = "a" * 64

    def fake_urlopen(url: str, timeout: int):
        assert url == f"http://controller.internal/bundles/{wrong_id}.zip"
        return _FakeResponse(bad_zip)

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)
    cache = LocalBundleStore(temp_cache_dir, controller_address="http://controller.internal")

    with pytest.raises(ValueError, match="Bundle hash mismatch"):
        cache.get_bundle(wrong_id)


def test_lru_eviction(monkeypatch, temp_cache_dir):
    cache = LocalBundleStore(temp_cache_dir, controller_address="http://controller.internal", max_bundles=2)
    bundles = []
    for i in range(3):
        bundle_zip = _make_zip({"test.txt": f"bundle {i}".encode()})
        bundle_id = hashlib.sha256(bundle_zip).hexdigest()
        bundles.append((bundle_id, bundle_zip))

    def fake_urlopen(url: str, timeout: int):
        bundle_id = url.rsplit("/", 1)[1].removesuffix(".zip")
        for b_id, blob in bundles:
            if b_id == bundle_id:
                return _FakeResponse(blob)
        raise AssertionError(f"unexpected bundle fetch: {url}")

    monkeypatch.setattr("iris.cluster.bundle.urlopen", fake_urlopen)

    paths = []
    for bundle_id, _ in bundles:
        for j, p in enumerate(paths):
            os.utime(p, (j, j))
        paths.append(cache.get_bundle(bundle_id))

    assert not paths[0].exists()
    assert paths[1].exists()
    assert paths[2].exists()
    assert len(list((temp_cache_dir / "extracts").iterdir())) == 2
