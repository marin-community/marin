"""Pinned files remain reusable after concurrent or failed staging attempts."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event

import fsspec
import pytest

from eval import data_sources

_REPO = "test-owner/benchmark"
_FIRST_REVISION = "a" * 40
_SECOND_REVISION = "b" * 40


@pytest.fixture
def hub_files(tmp_path, monkeypatch):
    versions = {}
    for revision, content in ((_FIRST_REVISION, b"first version"), (_SECOND_REVISION, b"second version")):
        path = tmp_path / revision
        path.write_bytes(content)
        versions[revision] = path

    def download(repo_id, filename, *, repo_type, revision):
        assert (repo_id, filename, repo_type) == (_REPO, "data/test.jsonl", "dataset")
        return str(versions[revision])

    monkeypatch.setattr(data_sources, "hf_hub_download", download)
    return versions


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_staged_files_are_revision_isolated_and_work_without_hub(storage, tmp_path, monkeypatch, hub_files):
    cache = str(tmp_path / "cache") if storage == "local" else f"memory://{tmp_path.name}"
    first = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION, cache_prefix=cache)
    second = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_SECOND_REVISION, cache_prefix=cache)

    def offline(*args, **kwargs):
        raise ConnectionError("Hugging Face unavailable")

    monkeypatch.setattr(data_sources, "hf_hub_download", offline)
    for revision, expected in ((_FIRST_REVISION, b"first version"), (_SECOND_REVISION, b"second version")):
        path = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=revision, cache_prefix=cache)
        with fsspec.open(path, "rb") as source:
            assert source.read() == expected
    assert first != second


def test_no_shared_prefix_returns_native_hub_cache_file(hub_files):
    path = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION)
    assert Path(path).read_bytes() == b"first version"
    assert Path(path) == hub_files[_FIRST_REVISION]


def test_failed_partial_stage_is_removed_and_can_be_retried(tmp_path, monkeypatch, hub_files):
    cache = tmp_path / "cache"
    filesystem = fsspec.filesystem("file")
    original_open = filesystem.open

    class InterruptedWriter:
        def __init__(self, stream):
            self.stream = stream

        def write(self, data):
            self.stream.write(data[:2])
            raise OSError("Upload interrupted")

    @contextmanager
    def failing_open(path, mode="rb", **kwargs):
        with original_open(path, mode, **kwargs) as stream:
            yield InterruptedWriter(stream) if mode == "wb" else stream

    with monkeypatch.context() as patch:
        patch.setattr(filesystem, "open", failing_open)
        with pytest.raises(OSError, match="Upload interrupted"):
            data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION, cache_prefix=str(cache))
    assert not list(cache.rglob("*.tmp.*"))
    assert not list(cache.rglob("*.complete"))

    path = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION, cache_prefix=str(cache))
    assert Path(path).read_bytes() == b"first version"


def test_concurrent_staging_only_exposes_complete_files(tmp_path, monkeypatch, hub_files):
    cache = str(tmp_path / "cache")
    start = Barrier(8)
    downloading = Event()
    finish_download = Event()

    def download(*args, **kwargs):
        downloading.set()
        assert finish_download.wait(timeout=10)
        return str(hub_files[_FIRST_REVISION])

    monkeypatch.setattr(data_sources, "hf_hub_download", download)

    def fetch():
        start.wait(timeout=10)
        path = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION, cache_prefix=cache)
        return path, Path(path).read_bytes()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch) for _ in range(8)]
        try:
            assert downloading.wait(timeout=10)
            assert not list(Path(cache).rglob("*.complete"))
            finish_download.set()
            results = [future.result(timeout=10) for future in futures]
        finally:
            finish_download.set()

    assert len({path for path, _ in results}) == 1
    assert all(content == b"first version" for _, content in results)
