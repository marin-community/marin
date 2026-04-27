# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os

import fsspec
import pytest

from rigging.filesystem import (
    REGION_TO_DATA_BUCKET,
    REGION_TO_TMP_BUCKET,
    MirrorFileSystem,
    MirrorTmpFileSystem,
    TransferBudget,
    TransferBudgetExceeded,
    _mirror_local_prefix,
    _mirror_remote_prefixes,
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
    mirror_fs._budget.reset(limit_bytes=1500)
    _write_file(mirror_env["remote1"], "data/a.bin", b"x" * 800)
    _write_file(mirror_env["remote1"], "data/b.bin", b"y" * 800)

    mirror_fs.cat_file("data/a.bin")
    assert mirror_fs.bytes_copied == 800

    with pytest.raises(TransferBudgetExceeded):
        mirror_fs.cat_file("data/b.bin")


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
# MirrorTmpFileSystem (mirrortmp://) — bucket-family + remote-prefix tests
# ---------------------------------------------------------------------------


def test_mirrortmp_remote_prefixes_use_tmp_bucket_map():
    """mirrortmp:// remote prefixes come from REGION_TO_TMP_BUCKET, never from data buckets."""
    prefixes = _mirror_remote_prefixes("gs://marin-tmp-us-central1", family="tmp")
    assert prefixes
    tmp_bucket_urls = {f"gs://{b}" for b in REGION_TO_TMP_BUCKET.values()}
    assert set(prefixes) <= tmp_bucket_urls
    # Local prefix itself must be excluded.
    assert "gs://marin-tmp-us-central1" not in prefixes
    # Another tmp bucket should be present.
    assert "gs://marin-tmp-eu-west4" in prefixes
    # No primary data buckets should leak in.
    data_bucket_urls = {f"gs://{b}" for b in REGION_TO_DATA_BUCKET.values()}
    assert not (set(prefixes) & data_bucket_urls)


def test_mirrortmp_remote_prefixes_empty_for_non_gcs():
    """Regression for marin-community/marin#4656 applied to the tmp family."""
    assert _mirror_remote_prefixes("s3://marin-na/marin", family="tmp") == []
    assert _mirror_remote_prefixes("/tmp/marin", family="tmp") == []
    assert _mirror_remote_prefixes("file:///tmp/marin", family="tmp") == []


def test_mirrortmp_local_prefix_uses_region_tmp_bucket(monkeypatch):
    """On a GCS marin prefix with a known region, the local prefix is the matching tmp bucket."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    monkeypatch.setattr("rigging.filesystem.region_from_metadata", lambda: "us-east5")
    assert _mirror_local_prefix("tmp") == "gs://marin-tmp-us-east5"


def test_mirrortmp_local_prefix_falls_back_for_non_gcs(monkeypatch):
    """Non-GCS marin prefix → ``<marin_prefix>/tmp`` (mirrors marin_temp_bucket fallback)."""
    monkeypatch.setenv("MARIN_PREFIX", "/var/marin")
    assert _mirror_local_prefix("tmp") == "file:///var/marin/tmp"
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-na/marin")
    assert _mirror_local_prefix("tmp") == "s3://marin-na/marin/tmp"


def test_mirrortmp_filesystem_init_on_s3(monkeypatch):
    """Full __init__ on a non-GCS prefix: empty remote prefixes, local prefix routes to tmp/."""
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-na/marin")
    fs = MirrorTmpFileSystem()
    assert fs._local_prefix == "s3://marin-na/marin/tmp"
    assert fs._remote_prefixes == []


# ---------------------------------------------------------------------------
# stage_to_local — recursive directory staging
# ---------------------------------------------------------------------------


@pytest.fixture()
def stage_fs(mirror_env, tmp_path):
    """A MirrorFileSystem (data family) with hand-set local + remote prefixes for staging tests."""
    fs = MirrorFileSystem.__new__(MirrorFileSystem)
    fsspec.AbstractFileSystem.__init__(fs)
    fs._local_prefix = mirror_env["local_prefix"]
    fs._remote_prefixes = mirror_env["remote_prefixes"]
    fs._budget = TransferBudget(limit_bytes=10 * 1024 * 1024 * 1024)
    fs._worker_id = "test-holder"
    lock_dir = str(tmp_path / "stage_locks")
    fs._lock_path_for = lambda path: os.path.join(lock_dir, f"{path.replace('/', '_')}.lock")
    return fs


def test_stage_to_local_copies_directory_recursively(stage_fs, mirror_env):
    _write_file(mirror_env["remote1"], "ckpt/run-A/step-1/params.shard", b"P" * 100)
    _write_file(mirror_env["remote1"], "ckpt/run-A/step-1/metadata.json", b'{"step": 1}')

    local_url = stage_fs.stage_to_local("ckpt/run-A/step-1")

    assert local_url == os.path.join(mirror_env["local_prefix"], "ckpt/run-A/step-1")
    assert os.path.exists(os.path.join(local_url, "params.shard"))
    assert os.path.exists(os.path.join(local_url, "metadata.json"))
    with open(os.path.join(local_url, "params.shard"), "rb") as f:
        assert f.read() == b"P" * 100


def test_stage_to_local_complete_local_no_remote_returns_local(stage_fs, mirror_env):
    """When local is fully populated and no remote source exists, return local without erroring."""
    _write_file(mirror_env["local_dir"], "ckpt/local-only/file.bin", b"L")

    bytes_before = stage_fs.bytes_copied
    local_url = stage_fs.stage_to_local("ckpt/local-only")

    assert local_url == os.path.join(mirror_env["local_prefix"], "ckpt/local-only")
    assert stage_fs.bytes_copied == bytes_before  # no remote found ⇒ no debit


def test_stage_to_local_no_source_anywhere_raises(stage_fs):
    with pytest.raises(FileNotFoundError, match="not found in any marin bucket"):
        stage_fs.stage_to_local("ckpt/never-existed")


def test_stage_to_local_charges_budget_once(stage_fs, mirror_env):
    _write_file(mirror_env["remote1"], "ckpt/run-B/step-2/a.bin", b"A" * 250)
    _write_file(mirror_env["remote1"], "ckpt/run-B/step-2/b.bin", b"B" * 250)

    bytes_before = stage_fs.bytes_copied
    stage_fs.stage_to_local("ckpt/run-B/step-2")
    assert stage_fs.bytes_copied == bytes_before + 500


def test_stage_to_local_budget_exceeded_no_partial_copy(stage_fs, mirror_env):
    stage_fs._budget.reset(limit_bytes=100)
    _write_file(mirror_env["remote1"], "ckpt/big/step-3/a.bin", b"A" * 250)
    _write_file(mirror_env["remote1"], "ckpt/big/step-3/b.bin", b"B" * 250)

    with pytest.raises(TransferBudgetExceeded):
        stage_fs.stage_to_local("ckpt/big/step-3")

    # No partial files should have been written locally.
    assert not os.path.exists(os.path.join(mirror_env["local_prefix"], "ckpt/big/step-3/a.bin"))
    assert not os.path.exists(os.path.join(mirror_env["local_prefix"], "ckpt/big/step-3/b.bin"))


def test_stage_to_local_repairs_partial_local_directory(stage_fs, mirror_env):
    """Partial prior stage left local dir present but missing a file — repair, don't trust local."""
    _write_file(mirror_env["remote1"], "ckpt/partial/step-4/a.bin", b"A" * 50)
    _write_file(mirror_env["remote1"], "ckpt/partial/step-4/b.bin", b"B" * 70)
    # Simulate a crashed prior stage: one file present locally, the other missing.
    _write_file(mirror_env["local_dir"], "ckpt/partial/step-4/a.bin", b"A" * 50)

    bytes_before = stage_fs.bytes_copied
    stage_fs.stage_to_local("ckpt/partial/step-4")

    # Both files now present locally.
    assert os.path.exists(os.path.join(mirror_env["local_prefix"], "ckpt/partial/step-4/a.bin"))
    assert os.path.exists(os.path.join(mirror_env["local_prefix"], "ckpt/partial/step-4/b.bin"))
    # Only the missing file's bytes were charged — the already-present 50 bytes aren't double-counted.
    assert stage_fs.bytes_copied == bytes_before + 70


def test_stage_to_local_repairs_size_mismatch(stage_fs, mirror_env):
    """Local file present but with wrong size → re-copy."""
    _write_file(mirror_env["remote1"], "ckpt/mismatch/step-5/data.bin", b"R" * 100)
    _write_file(mirror_env["local_dir"], "ckpt/mismatch/step-5/data.bin", b"OLD")  # 3 bytes

    bytes_before = stage_fs.bytes_copied
    stage_fs.stage_to_local("ckpt/mismatch/step-5")

    with open(os.path.join(mirror_env["local_prefix"], "ckpt/mismatch/step-5/data.bin"), "rb") as f:
        assert f.read() == b"R" * 100
    assert stage_fs.bytes_copied == bytes_before + 100


def test_stage_to_local_returns_concrete_local_url(stage_fs, mirror_env):
    """Returned URL is the local concrete path (not mirror://...) so tensorstore can read it."""
    _write_file(mirror_env["remote1"], "ckpt/scheme/step-6/x.bin", b"x")
    url = stage_fs.stage_to_local("ckpt/scheme/step-6")
    assert "mirror" not in url
    assert url.startswith(mirror_env["local_prefix"])
