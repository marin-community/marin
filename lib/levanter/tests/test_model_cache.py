# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the distributed-locked model snapshot cache."""

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import fsspec
from huggingface_hub import RepoFile

import pytest

import levanter.model_cache as model_cache
from levanter.model_cache import (
    DEFAULT_COMPLETE_MARKER,
    cache_hf_model,
    cache_to_prefix,
    resolve_cached_model_path,
)


def _repo_file(path: str, contents: bytes) -> RepoFile:
    return RepoFile(path=path, size=len(contents), oid="0" * 40)


class _FakeHub:
    """In-memory stand-in for ``HfFileSystem`` that records which repo files were opened."""

    def __init__(self, contents: dict[str, bytes], *, revision: str | None = None):
        self.contents = contents
        self.revision = revision
        self.opened: list[str] = []
        self.fail_on: set[str] = set()

    def open(self, path, mode="rb", revision=None, block_size=None):
        assert mode == "rb"
        assert revision == self.revision
        repo, _, filename = path.partition("org/model/")
        assert repo == ""
        self.opened.append(filename)
        if filename in self.fail_on:
            raise OSError(f"interrupted download of {filename}")
        return io.BytesIO(self.contents[filename])


def _serve_repo(monkeypatch, hub: _FakeHub, sizes: dict[str, bytes] | None = None) -> None:
    """List *sizes* (default: the hub's contents) as the repo tree and serve files from *hub*."""
    listed = hub.contents if sizes is None else sizes
    monkeypatch.setattr(
        model_cache,
        "list_repo_tree",
        lambda *_args, **_kwargs: [_repo_file(path, data) for path, data in listed.items()],
    )
    monkeypatch.setattr(model_cache, "HfFileSystem", lambda: hub)


def _make_populate(call_count: list[int], lock: threading.Lock, *, delay: float = 0.0):
    """Return a populate callback that records invocations and streams a payload to the cache."""

    def populate(fs: fsspec.AbstractFileSystem, dest: str) -> None:
        with lock:
            call_count.append(1)
        if delay:
            time.sleep(delay)
        with fs.open(f"{dest}/weights.bin", "w") as handle:
            handle.write("payload")

    return populate


def test_concurrent_callers_populate_once(tmp_path):
    """Under N concurrent callers only one populate runs; all get the cached path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    # A non-trivial populate window forces losers to actually block on the lock
    # instead of all winning the marker fast-path on the first check.
    populate = _make_populate(calls, count_lock, delay=0.3)

    def run() -> str:
        return cache_to_prefix(cache_path, populate, poll_interval=0.02)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: run(), range(8)))

    assert len(calls) == 1, f"expected a single populate, got {len(calls)}"
    assert Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()
    # Every caller returns the cache path holding the populated snapshot.
    for path in results:
        assert path == cache_path.rstrip("/")
        assert Path(path, "weights.bin").read_text() == "payload"


def test_cache_hit_skips_populate(tmp_path):
    """A second call after the marker exists returns the cache path without re-populating."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    populate = _make_populate(calls, count_lock)

    first = cache_to_prefix(cache_path, populate)
    second = cache_to_prefix(cache_path, populate)

    assert len(calls) == 1
    assert first == second == cache_path.rstrip("/")
    assert Path(first, "weights.bin").read_text() == "payload"


def test_custom_complete_marker(tmp_path):
    """The completion marker name is configurable and gates the cache-hit fast path."""
    cache_path = str(tmp_path / "cache" / "model")
    calls: list[int] = []
    count_lock = threading.Lock()
    populate = _make_populate(calls, count_lock)

    cache_to_prefix(cache_path, populate, complete_marker=".done")
    assert Path(cache_path, ".done").exists()

    cache_to_prefix(cache_path, populate, complete_marker=".done")
    assert len(calls) == 1


def test_cache_hf_model_mirrors_every_repo_file(tmp_path, monkeypatch):
    contents = {
        name: f"contents of {name}".encode() for name in ("config.json", "model.safetensors", "sub/tokenizer.json")
    }
    hub = _FakeHub(contents)
    _serve_repo(monkeypatch, hub)

    cache_path = str(tmp_path / "cache" / "model")
    result = cache_hf_model(cache_path, "org/model")

    assert result == cache_path
    for filename, expected in contents.items():
        assert Path(cache_path, filename).read_bytes() == expected
    assert json.loads(Path(cache_path, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": None}


def test_cache_hf_model_does_not_copy_repository_completion_marker(tmp_path, monkeypatch):
    """An interrupted mirror must not look complete because the repo contains Marin's marker."""
    revision = "a" * 40
    cache_path = str(tmp_path / "cache" / "model")
    hub = _FakeHub(
        {DEFAULT_COMPLETE_MARKER: b'{"source_revision": "parent"}', "model.safetensors": b"weights"},
        revision=revision,
    )
    hub.fail_on.add("model.safetensors")
    _serve_repo(monkeypatch, hub)

    with pytest.raises(OSError, match="interrupted download"):
        cache_hf_model(cache_path, "org/model", revision=revision)

    assert not Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()

    hub.fail_on.clear()
    cache_hf_model(cache_path, "org/model", revision=revision)

    assert DEFAULT_COMPLETE_MARKER not in hub.opened
    assert Path(cache_path, "model.safetensors").read_text() == "weights"
    assert json.loads(Path(cache_path, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": revision}


def test_cache_hf_model_resumes_complete_files_and_replaces_partial_files(tmp_path, monkeypatch):
    revision = "a" * 40
    contents = {
        "model.safetensors": b"weight shard",
        "config.json": b'{"model": 1}',
        "tokenizer.json": b'{"vocab": 2}',
    }
    cache_path = str(tmp_path / "cache" / "model")
    fs = fsspec.filesystem("file")
    hub = _FakeHub(contents, revision=revision)
    hub.fail_on.add("tokenizer.json")
    monkeypatch.setattr(model_cache, "url_to_fs", lambda path: (fs, path))
    _serve_repo(monkeypatch, hub)

    with pytest.raises(OSError, match="interrupted download"):
        cache_hf_model(cache_path, "org/model", revision=revision)

    shard = Path(cache_path, "model.safetensors")
    config = Path(cache_path, "config.json")
    assert shard.read_bytes() == contents["model.safetensors"]
    assert config.read_bytes() == contents["config.json"]
    assert not Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()

    os.utime(shard, ns=(1_000_000_000, 1_000_000_000))
    config.write_bytes(b"partial")
    real_open = fs.open

    def reject_cached_reads(path, mode="rb", *args, **kwargs):
        if path.startswith(cache_path) and "r" in mode:
            pytest.fail(f"unexpected cache read: {path}")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(fs, "open", reject_cached_reads)
    hub.opened.clear()
    hub.fail_on.clear()

    cache_hf_model(cache_path, "org/model", revision=revision)

    assert sorted(hub.opened) == ["config.json", "tokenizer.json"]
    assert shard.stat().st_mtime_ns == 1_000_000_000
    for filename, expected in contents.items():
        assert Path(cache_path, filename).read_bytes() == expected
    assert json.loads(Path(cache_path, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": revision}


def test_cache_hf_model_rejects_truncated_stream_before_completion(tmp_path, monkeypatch):
    """A Hub stream that ends early leaves a short file, which must fail the mirror."""
    contents = b"valid weights"
    cache_path = str(tmp_path / "cache" / "model")
    hub = _FakeHub({"model.safetensors": contents[:-1]}, revision="a" * 40)
    _serve_repo(monkeypatch, hub, sizes={"model.safetensors": contents})

    with pytest.raises(OSError, match="wrong size"):
        cache_hf_model(cache_path, "org/model", revision="a" * 40)

    assert not Path(cache_path, DEFAULT_COMPLETE_MARKER).exists()


@pytest.mark.parametrize(
    "path",
    [
        "gs://bucket/snapshot",  # object store
        "s3://bucket/snapshot",  # object store
        "hf://org/model",  # explicit fsspec HF URL
        "/local/checkpoint/dir",  # absolute local path
        "./relative/dir",  # relative local path
    ],
)
def test_resolve_loads_non_repo_paths_in_place(path, monkeypatch):
    """A path that already names a snapshot is returned unchanged, never mirrored.

    Mirroring it would feed the path to ``cache_hf_model`` as if it were a repo id and fail.
    """
    monkeypatch.setattr(model_cache, "cache_hf_model", lambda *a, **k: pytest.fail("must not mirror"))
    assert resolve_cached_model_path(path, cache_ttl_days=30, cache_prefix="models") == path


def test_resolve_disabled_ttl_skips_mirror(monkeypatch):
    monkeypatch.setattr(model_cache, "cache_hf_model", lambda *a, **k: pytest.fail("must not mirror"))
    assert resolve_cached_model_path("org/model", cache_ttl_days=0, cache_prefix="models") == "org/model"


def test_resolve_pinned_commit_skips_hf_lookup(tmp_path, monkeypatch):
    """An immutable ref can use an existing cache while Hugging Face is unavailable."""
    commit = "a" * 40

    monkeypatch.setattr(model_cache, "model_info", lambda *args, **kwargs: pytest.fail("must not query HF"))
    _serve_repo(monkeypatch, _FakeHub({"config.json": commit.encode()}, revision=commit))
    monkeypatch.setattr(model_cache, "marin_temp_bucket", lambda _ttl_days, prefix: str(tmp_path / prefix))

    first = resolve_cached_model_path(f"org/model@{commit}", cache_ttl_days=7, cache_prefix="models")
    monkeypatch.setattr(model_cache, "list_repo_tree", lambda *_args, **_kwargs: pytest.fail("cache miss"))
    monkeypatch.setattr(model_cache, "HfFileSystem", lambda: pytest.fail("cache miss"))
    second = resolve_cached_model_path(f"org/model@{commit}", cache_ttl_days=7, cache_prefix="models")

    assert first == second
    assert Path(second, "config.json").read_text() == commit
    assert json.loads(Path(second, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": commit}


def test_resolve_keeps_distinct_refs_in_distinct_cache_dirs(monkeypatch):
    """Two refs that a lossy slug would collide (``org/model_a`` vs ``org/model@a``) must mirror
    to different cache dirs, so a hit on one never loads the other's snapshot."""
    mirrored: dict[str, tuple[str, str | None]] = {}
    revisions = {
        ("org/model_a", None): "a" * 40,
        ("org/model", "a"): "b" * 40,
    }

    monkeypatch.setattr(model_cache, "marin_temp_bucket", lambda ttl_days, prefix: f"gs://temp/{prefix}")
    monkeypatch.setattr(
        model_cache,
        "model_info",
        lambda model_id, *, revision=None, token=None: SimpleNamespace(sha=revisions[(model_id, revision)]),
    )

    def fake_cache(cache_path, repo, *, revision=None, complete_marker):
        mirrored[cache_path] = (repo, revision)
        return cache_path

    monkeypatch.setattr(model_cache, "cache_hf_model", fake_cache)

    a = resolve_cached_model_path("org/model_a", cache_ttl_days=7, cache_prefix="models")
    b = resolve_cached_model_path("org/model@a", cache_ttl_days=7, cache_prefix="models")

    assert a != b
    assert mirrored[a] == ("org/model_a", revisions[("org/model_a", None)])
    assert mirrored[b] == ("org/model", revisions[("org/model", "a")])


def test_resolve_tracks_mutable_hf_revision(tmp_path, monkeypatch):
    """A branch update produces a new cache snapshot pinned to the new commit."""
    first_commit = "a" * 40
    second_commit = "b" * 40
    current_commit = first_commit
    repo_files = ["config.json", "generation_config.json", "tokenizer_config.json"]

    def fake_model_info(model_id, *, revision=None, token=None):
        assert model_id == "org/model"
        assert revision is None
        return SimpleNamespace(sha=current_commit)

    def fake_list_repo_tree(model_id, revision=None, recursive=False):
        assert model_id == "org/model"
        assert revision == current_commit
        assert recursive
        return [_repo_file(filename, current_commit.encode()) for filename in repo_files]

    class RevisionHub:
        def open(self, path, mode="rb", revision=None, block_size=None):
            assert path.startswith("org/model/")
            return io.BytesIO(revision.encode())

    monkeypatch.setattr(model_cache, "model_info", fake_model_info)
    monkeypatch.setattr(model_cache, "list_repo_tree", fake_list_repo_tree)
    monkeypatch.setattr(model_cache, "HfFileSystem", RevisionHub)
    monkeypatch.setattr(model_cache, "marin_temp_bucket", lambda _ttl_days, prefix: str(tmp_path / prefix))

    first_path = resolve_cached_model_path("org/model", cache_ttl_days=7, cache_prefix="models")
    current_commit = second_commit
    second_path = resolve_cached_model_path("org/model", cache_ttl_days=7, cache_prefix="models")

    assert first_path != second_path
    assert json.loads(Path(first_path, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": first_commit}
    assert json.loads(Path(second_path, DEFAULT_COMPLETE_MARKER).read_text()) == {"source_revision": second_commit}
    for filename in repo_files:
        assert Path(first_path, filename).read_text() == first_commit
        assert Path(second_path, filename).read_text() == second_commit
