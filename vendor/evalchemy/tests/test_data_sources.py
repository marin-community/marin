"""Pinned files remain reusable after concurrent or failed staging attempts."""

import io
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Barrier, Event, Thread

import fsspec
import huggingface_hub.constants
import pytest

from eval import data_sources

_REPO = "test-owner/benchmark"
_FIRST_REVISION = "a" * 40
_SECOND_REVISION = "b" * 40
_URL_CONTENT = b'{"question": "cached benchmark"}\n'
_URL_VERSION = "2026-01-15"


class _HttpResponse(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.headers = HTTPMessage()


@pytest.fixture
def http_origin():
    interrupted = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(_URL_CONTENT)))
            self.end_headers()
            if self.path == "/interrupted" and not interrupted.is_set():
                interrupted.set()
                self.wfile.write(_URL_CONTENT[:2])
                self.close_connection = True
                return
            self.wfile.write(_URL_CONTENT)

        def log_message(self, *args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/data.json?token=private", server
        finally:
            server.shutdown()
            thread.join(timeout=10)


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


def test_native_hub_snapshot_remains_readable_without_origin(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    def download(repo_id, filename, *, repo_type, revision):
        snapshot = cache / "datasets--test-owner--benchmark" / "snapshots" / revision / filename
        snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(b"native cached data")
        return str(snapshot)

    monkeypatch.setattr(data_sources, "hf_hub_download", download)
    first = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION)

    def offline(*args, **kwargs):
        raise ConnectionError("Hugging Face unavailable")

    monkeypatch.setattr(data_sources, "hf_hub_download", offline)
    second = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION)
    assert first == second
    assert Path(second).read_bytes() == b"native cached data"


def test_cold_shared_cache_stages_native_hub_snapshot_without_origin(tmp_path, monkeypatch):
    native_cache = tmp_path / "hub"
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(native_cache))
    snapshot = native_cache / "datasets--test-owner--benchmark" / "snapshots" / _FIRST_REVISION / "data/test.jsonl"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"native cached data")

    def offline(*args, **kwargs):
        raise ConnectionError("Hugging Face unavailable")

    monkeypatch.setattr(data_sources, "hf_hub_download", offline)
    shared_cache = tmp_path / "shared"
    path = data_sources.fetch_hf(_REPO, "data/test.jsonl", revision=_FIRST_REVISION, cache_prefix=str(shared_cache))

    assert Path(path) == shared_cache / _REPO / _FIRST_REVISION / "data/test.jsonl"
    assert Path(path).read_bytes() == b"native cached data"
    assert Path(f"{path}.complete").exists()


@pytest.mark.parametrize("storage", ["local", "memory"])
def test_url_cache_remains_readable_without_origin(storage, tmp_path, http_origin):
    url, server = http_origin
    cache = str(tmp_path / "cache") if storage == "local" else f"memory://{tmp_path.name}"
    first = data_sources.fetch_url(url, version=_URL_VERSION, cache_prefix=cache)
    server.shutdown()
    server.server_close()

    second = data_sources.fetch_url(url, version=_URL_VERSION, cache_prefix=cache)
    assert first == second
    assert "private" not in first
    with fsspec.open(second, "rb") as reader:
        assert reader.read() == _URL_CONTENT


def test_url_cache_isolates_sources_and_publisher_versions(tmp_path, monkeypatch):
    cache = str(tmp_path / "cache")
    contents = iter([b"first release", b"second release", b"other source"])

    def download(*args, **kwargs):
        return _HttpResponse(next(contents))

    monkeypatch.setattr(data_sources, "urlopen", download)
    first = data_sources.fetch_url("https://example.invalid/a", version="v1", cache_prefix=cache)
    second = data_sources.fetch_url("https://example.invalid/a", version="v2", cache_prefix=cache)
    other = data_sources.fetch_url("https://example.invalid/b", version="v1", cache_prefix=cache)

    assert Path(first).read_bytes() == b"first release"
    assert Path(second).read_bytes() == b"second release"
    assert Path(other).read_bytes() == b"other source"


def test_truncated_http_download_is_unpublished_and_can_be_retried(tmp_path, http_origin):
    _, server = http_origin
    url = f"http://127.0.0.1:{server.server_port}/interrupted"
    cache = tmp_path / "cache"

    with pytest.raises(OSError):
        data_sources.fetch_url(url, version=_URL_VERSION, cache_prefix=str(cache))
    assert not list(cache.rglob("*.tmp.*"))
    assert not list(cache.rglob("*.complete"))
    assert not list(cache.rglob("content"))

    path = data_sources.fetch_url(url, version=_URL_VERSION, cache_prefix=str(cache))
    assert Path(path).read_bytes() == _URL_CONTENT
    assert Path(f"{path}.complete").exists()


def test_concurrent_url_staging_only_exposes_complete_files(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    start = Barrier(4)
    downloading = Event()
    finish_download = Event()

    def download(*args, **kwargs):
        downloading.set()
        assert finish_download.wait(timeout=10)
        return _HttpResponse(_URL_CONTENT)

    monkeypatch.setattr(data_sources, "urlopen", download)

    def fetch():
        start.wait(timeout=10)
        path = data_sources.fetch_url("https://example.invalid/data", version=_URL_VERSION, cache_prefix=str(cache))
        return path, Path(path).read_bytes()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch) for _ in range(4)]
        try:
            assert downloading.wait(timeout=10)
            assert not list(cache.rglob("*.complete"))
            assert not list(cache.rglob("content"))
            finish_download.set()
            results = [future.result(timeout=10) for future in futures]
        finally:
            finish_download.set()

    assert len({path for path, _ in results}) == 1
    assert all(content == _URL_CONTENT for _, content in results)


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
