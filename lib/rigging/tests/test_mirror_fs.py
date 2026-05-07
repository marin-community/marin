# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import threading

import fsspec
import pytest
from rigging.filesystem import (
    MirrorFileSystem,
    TransferBudget,
    TransferBudgetExceeded,
    _mirror_remote_prefixes,
    resolve_tree,
    to_mirror_url,
)


@pytest.fixture()
def mirror_env(tmp_path):
    """Local directories mimicking marin regional buckets."""
    local_dir = tmp_path / "marin-local"
    local_dir.mkdir()
    remote1 = tmp_path / "marin-us-central2"
    remote1.mkdir()
    remote2 = tmp_path / "marin-eu-west4"
    remote2.mkdir()
    return {
        "local_dir": local_dir,
        "remote1": remote1,
        "remote2": remote2,
        "local_prefix": str(local_dir),
        "remote_prefixes": [str(remote1), str(remote2)],
    }


@pytest.fixture()
def mirror_fs(mirror_env, tmp_path):
    """MirrorFileSystem backed by local directories with an isolated budget."""
    fs = MirrorFileSystem.__new__(MirrorFileSystem)
    fsspec.AbstractFileSystem.__init__(fs)
    fs._local_prefix = mirror_env["local_prefix"]
    fs._remote_prefixes = mirror_env["remote_prefixes"]
    fs._budget = TransferBudget(limit_bytes=10 * 1024 * 1024 * 1024)
    fs._worker_id = "test-holder"
    fs._inproc_copy_locks = {}
    fs._inproc_copy_locks_mutex = threading.Lock()
    lock_dir = str(tmp_path / "locks")
    fs._lock_path_for = lambda path: os.path.join(lock_dir, f"{path.replace('/', '_')}.lock")
    return fs


def _write_file(base_dir, rel_path, data):
    full = os.path.join(str(base_dir), rel_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(data)


def test_read_from_local(mirror_fs, mirror_env):
    _write_file(mirror_env["local_dir"], "models/ckpt.bin", b"local-data")
    assert mirror_fs.cat_file("models/ckpt.bin") == b"local-data"


def test_read_copies_from_remote(mirror_fs, mirror_env):
    _write_file(mirror_env["remote1"], "models/ckpt.bin", b"remote-data")

    assert mirror_fs.cat_file("models/ckpt.bin") == b"remote-data"
    # Should now exist locally
    local_path = os.path.join(str(mirror_env["local_dir"]), "models/ckpt.bin")
    with open(local_path, "rb") as f:
        assert f.read() == b"remote-data"


def test_file_not_found_raises(mirror_fs):
    with pytest.raises(FileNotFoundError, match="not found in any marin bucket"):
        mirror_fs.cat_file("nonexistent/file.bin")


def test_copy_budget_raises_when_exceeded(mirror_fs, mirror_env):
    mirror_fs._budget.reset(limit_bytes=500)
    _write_file(mirror_env["remote1"], "data/big.bin", b"x" * 1000)

    with pytest.raises(TransferBudgetExceeded):
        mirror_fs.cat_file("data/big.bin")


def test_copy_budget_cumulative(mirror_fs, mirror_env):
    """Budget accumulates across separate tree copies.

    Each ``cat_file`` materializes the file's enclosing directory; placing
    the two files in separate parent dirs is the only way to drive
    independent copies in tree-on-open semantics.
    """
    mirror_fs._budget.reset(limit_bytes=1500)
    _write_file(mirror_env["remote1"], "tree-a/file.bin", b"x" * 800)
    _write_file(mirror_env["remote1"], "tree-b/file.bin", b"y" * 800)

    mirror_fs.cat_file("tree-a/file.bin")
    assert mirror_fs.bytes_copied == 800

    with pytest.raises(TransferBudgetExceeded):
        mirror_fs.cat_file("tree-b/file.bin")


def test_second_read_uses_local_cache(mirror_fs, mirror_env):
    _write_file(mirror_env["remote1"], "data/file.bin", b"data")

    mirror_fs.cat_file("data/file.bin")
    assert mirror_fs.bytes_copied == 4

    # Remove from remote to prove local is used
    os.remove(os.path.join(str(mirror_env["remote1"]), "data/file.bin"))

    assert mirror_fs.cat_file("data/file.bin") == b"data"
    assert mirror_fs.bytes_copied == 4


def test_read_finds_file_in_second_remote(mirror_fs, mirror_env):
    _write_file(mirror_env["remote2"], "data/remote2.txt", b"from-remote2")
    assert mirror_fs.cat_file("data/remote2.txt") == b"from-remote2"


# ---------------------------------------------------------------------------
# ls / glob tests
# ---------------------------------------------------------------------------


def test_ls_returns_local_entries(mirror_fs, mirror_env):
    _write_file(mirror_env["local_dir"], "data/a.jsonl", b"a")
    _write_file(mirror_env["local_dir"], "data/b.jsonl", b"b")

    entries = mirror_fs.ls("data", detail=False)
    assert sorted(entries) == ["data/a.jsonl", "data/b.jsonl"]


def test_ls_discovers_remote_only_entries(mirror_fs, mirror_env):
    _write_file(mirror_env["remote1"], "data/remote.jsonl", b"r")

    entries = mirror_fs.ls("data", detail=False)
    assert "data/remote.jsonl" in entries


def test_ls_unions_local_and_remote(mirror_fs, mirror_env):
    _write_file(mirror_env["local_dir"], "data/local.jsonl", b"l")
    _write_file(mirror_env["remote1"], "data/remote.jsonl", b"r")

    entries = mirror_fs.ls("data", detail=False)
    assert sorted(entries) == ["data/local.jsonl", "data/remote.jsonl"]


def test_ls_local_takes_precedence(mirror_fs, mirror_env):
    _write_file(mirror_env["local_dir"], "data/file.jsonl", b"local-version")
    _write_file(mirror_env["remote1"], "data/file.jsonl", b"remote-version")

    entries = mirror_fs.ls("data", detail=True)
    assert len(entries) == 1
    assert entries[0]["name"] == "data/file.jsonl"


def test_ls_unions_multiple_remotes(mirror_fs, mirror_env):
    _write_file(mirror_env["remote1"], "data/from_r1.jsonl", b"r1")
    _write_file(mirror_env["remote2"], "data/from_r2.jsonl", b"r2")

    entries = mirror_fs.ls("data", detail=False)
    assert sorted(entries) == ["data/from_r1.jsonl", "data/from_r2.jsonl"]


def test_ls_empty_when_path_missing_everywhere(mirror_fs):
    entries = mirror_fs.ls("nonexistent/path", detail=False)
    assert entries == []


def test_glob_discovers_remote_files(mirror_fs, mirror_env):
    _write_file(mirror_env["remote1"], "docs/a.jsonl.gz", b"a")
    _write_file(mirror_env["remote1"], "docs/b.jsonl.gz", b"b")
    _write_file(mirror_env["remote1"], "docs/skip.txt", b"skip")

    matched = mirror_fs.glob("docs/*.jsonl.gz")
    assert sorted(matched) == ["docs/a.jsonl.gz", "docs/b.jsonl.gz"]


# ---------------------------------------------------------------------------
# mirror_budget context manager tests
# ---------------------------------------------------------------------------


def test_mirror_budget_context_manager(mirror_fs, mirror_env):
    """Transfer budget set via context manager is used for copies."""
    from rigging.filesystem import mirror_budget

    _write_file(mirror_env["remote1"], "data/big.bin", b"x" * 1000)

    with mirror_budget(budget_gb=0.001):  # ~1MB
        assert mirror_fs.cat_file("data/big.bin") == b"x" * 1000


def test_mirror_budget_context_manager_blocks_over_budget(mirror_fs, mirror_env):
    from rigging.filesystem import mirror_budget

    mirror_fs._budget.reset(limit_bytes=10 * 1024 * 1024 * 1024)  # high instance budget
    _write_file(mirror_env["remote1"], "data/big.bin", b"x" * 2000)

    with mirror_budget(budget_gb=0.000001):  # ~1KB — too small
        with pytest.raises(TransferBudgetExceeded):
            mirror_fs.cat_file("data/big.bin")


# ---------------------------------------------------------------------------
# Remote-prefix gating tests (GCS fallback must not fire on non-GCS prefixes)
# ---------------------------------------------------------------------------


def test_mirror_remote_prefixes_empty_for_s3():
    """Non-GCS local prefix must not list GCS buckets as mirror fallbacks.

    Regression test for marin-community/marin#4656: CoreWeave CI runs with
    ``MARIN_PREFIX=s3://...`` but the mirror filesystem unconditionally fell
    back to ``gs://marin-*`` buckets, causing noisy gcsfs 401s.
    """
    assert _mirror_remote_prefixes("s3://marin-na/marin") == []


def test_mirror_remote_prefixes_empty_for_local_path():
    assert _mirror_remote_prefixes("/tmp/marin") == []
    assert _mirror_remote_prefixes("file:///tmp/marin") == []


def test_mirror_remote_prefixes_populated_for_gcs():
    prefixes = _mirror_remote_prefixes("gs://marin-us-central1")
    assert prefixes
    assert all(p.startswith("gs://") for p in prefixes)
    # The local prefix itself must not appear in the remote list.
    assert "gs://marin-us-central1" not in prefixes
    # Other marin data buckets should still be present.
    assert "gs://marin-us-central2" in prefixes


def test_mirror_filesystem_init_skips_gcs_on_s3_prefix(monkeypatch):
    """Full __init__ path: no GCS remote prefixes when MARIN_PREFIX is s3://.

    Without the fix, this instance would attempt to contact GCS on every
    ``exists``/``ls`` call and spam anonymous-caller 401 errors.
    """
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-na/marin")
    fs = MirrorFileSystem()
    assert fs._local_prefix == "s3://marin-na/marin"
    assert fs._remote_prefixes == []


# ---------------------------------------------------------------------------
# Tree-on-open semantics: opening any file materializes its enclosing dir.
# ---------------------------------------------------------------------------


def test_open_one_file_copies_entire_enclosing_dir(mirror_fs, mirror_env):
    """Reading one file in a remote-only dir brings every sibling local.

    Regression guard: if mirror only copied the file you opened, a later
    consumer that walked the directory locally would see only that one file
    even though the source has many.  Tree-on-open closes that gap.
    """
    _write_file(mirror_env["remote1"], "step-200/metadata.json", b"{}")
    _write_file(mirror_env["remote1"], "step-200/shard-0", b"shard0-bytes")
    _write_file(mirror_env["remote1"], "step-200/shard-1", b"shard1-bytes")

    # Open exactly one file.
    assert mirror_fs.cat_file("step-200/metadata.json") == b"{}"

    # All siblings must now be local.
    local_dir = os.path.join(mirror_env["local_dir"], "step-200")
    assert sorted(os.listdir(local_dir)) == ["metadata.json", "shard-0", "shard-1"]


def test_open_subdir_file_copies_recursive_subtree(mirror_fs, mirror_env):
    """Opening a deep file materializes the whole containing subtree.

    The mirror walks the union ls recursively, so files in nested
    directories under the file's parent come along too.
    """
    _write_file(mirror_env["remote1"], "step-200/metadata.json", b"{}")
    _write_file(mirror_env["remote1"], "step-200/zarr/0/manifest", b"m0")
    _write_file(mirror_env["remote1"], "step-200/zarr/0/data", b"d0")
    _write_file(mirror_env["remote1"], "step-200/zarr/1/manifest", b"m1")

    mirror_fs.cat_file("step-200/metadata.json")

    base = os.path.join(mirror_env["local_dir"], "step-200")
    assert os.path.exists(os.path.join(base, "metadata.json"))
    assert os.path.exists(os.path.join(base, "zarr/0/manifest"))
    assert os.path.exists(os.path.join(base, "zarr/0/data"))
    assert os.path.exists(os.path.join(base, "zarr/1/manifest"))


def test_open_unions_files_split_across_regions(mirror_fs, mirror_env):
    """A tree split across multiple regions is reassembled locally.

    Different files of the same logical tree live in different regions —
    each pulls from whichever region carries it.
    """
    _write_file(mirror_env["remote1"], "tree/a", b"AAA")
    _write_file(mirror_env["remote2"], "tree/b", b"BBB")

    mirror_fs.cat_file("tree/a")

    base = os.path.join(mirror_env["local_dir"], "tree")
    with open(os.path.join(base, "a"), "rb") as f:
        assert f.read() == b"AAA"
    with open(os.path.join(base, "b"), "rb") as f:
        assert f.read() == b"BBB"


def test_info_does_not_copy(mirror_fs, mirror_env):
    """``info`` is probe-only; it must not auto-materialize the tree.

    Tree materialization happens on read (open / cat_file) or via
    ``resolve_tree``.  ``info`` is metadata-only and goes through the
    remote when the file isn't local yet.
    """
    _write_file(mirror_env["remote1"], "tree/file.bin", b"x" * 10)

    info = mirror_fs.info("tree/file.bin")
    assert info["size"] == 10

    # Nothing should have been copied locally.
    assert not os.path.exists(os.path.join(mirror_env["local_dir"], "tree"))


# ---------------------------------------------------------------------------
# Concurrency: per-file lock + in-process mutex prevent redundant transfers.
# ---------------------------------------------------------------------------


def test_concurrent_opens_only_copy_each_file_once(mirror_fs, mirror_env):
    """Racing readers in one process pay exactly one transfer per file.

    Without the in-process mutex, threads sharing a worker id all
    "successfully" reacquire the distributed lock and each fetch the same
    file — wasting bandwidth.  With it, only the first thread copies and
    the rest see the file already local.
    """
    _write_file(mirror_env["remote1"], "tree/a.bin", b"A" * 100)
    _write_file(mirror_env["remote1"], "tree/b.bin", b"B" * 200)

    def worker():
        mirror_fs.cat_file("tree/a.bin")
        mirror_fs.cat_file("tree/b.bin")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert mirror_fs.bytes_copied == 300


# ---------------------------------------------------------------------------
# resolve_tree
# ---------------------------------------------------------------------------


def test_resolve_tree_passthrough_for_non_mirror():
    """Non-``mirror://`` URLs round-trip unchanged in both modes."""
    assert resolve_tree("gs://some-bucket/path", mode="read") == "gs://some-bucket/path"
    assert resolve_tree("/tmp/foo", mode="write") == "/tmp/foo"
    assert resolve_tree("file:///tmp/foo", mode="read") == "file:///tmp/foo"


def test_resolve_tree_write_returns_local_url(mirror_fs, mirror_env):
    """``mode="write"`` mkdirs the local destination and returns its concrete URL."""
    out = mirror_fs.resolve_tree("mirror://step-200", mode="write")
    expected = os.path.join(mirror_env["local_dir"], "step-200")
    assert out == expected
    assert os.path.isdir(expected)


def test_resolve_tree_read_materializes_remote_tree(mirror_fs, mirror_env):
    """``mode="read"`` copies every file under the tree from remote regions."""
    _write_file(mirror_env["remote1"], "step-200/metadata.json", b"{}")
    _write_file(mirror_env["remote1"], "step-200/zarr/0/data", b"d0")
    _write_file(mirror_env["remote2"], "step-200/zarr/1/data", b"d1")

    out = mirror_fs.resolve_tree("mirror://step-200", mode="read")

    expected = os.path.join(mirror_env["local_dir"], "step-200")
    assert out == expected
    assert os.path.exists(os.path.join(expected, "metadata.json"))
    assert os.path.exists(os.path.join(expected, "zarr/0/data"))
    assert os.path.exists(os.path.join(expected, "zarr/1/data"))


def test_resolve_tree_read_raises_when_tree_missing(mirror_fs):
    with pytest.raises(FileNotFoundError, match="no marin region carries this tree"):
        mirror_fs.resolve_tree("mirror://nope", mode="read")


def test_resolve_tree_invalid_mode(mirror_fs):
    with pytest.raises(ValueError, match="invalid mode"):
        mirror_fs.resolve_tree("mirror://x", mode="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# to_mirror_url
# ---------------------------------------------------------------------------


def test_to_mirror_url_converts_marin_bucket():
    assert to_mirror_url("gs://marin-us-central1/run-x/checkpoints") == "mirror://run-x/checkpoints"
    assert to_mirror_url("gs://marin-eu-west4/foo/bar") == "mirror://foo/bar"


def test_to_mirror_url_passthrough_non_marin():
    """Non-marin GCS buckets and non-GCS URLs round-trip unchanged."""
    assert to_mirror_url("gs://some-other-bucket/path") == "gs://some-other-bucket/path"
    assert to_mirror_url("s3://x/y") == "s3://x/y"
    assert to_mirror_url("/local/path") == "/local/path"
    # Already a mirror URL — leave alone.
    assert to_mirror_url("mirror://run-x") == "mirror://run-x"


def test_to_mirror_url_skips_bare_bucket():
    """A URL with no key inside the bucket has no tree to mirror; passthrough."""
    assert to_mirror_url("gs://marin-us-central1") == "gs://marin-us-central1"
    assert to_mirror_url("gs://marin-us-central1/") == "gs://marin-us-central1/"
