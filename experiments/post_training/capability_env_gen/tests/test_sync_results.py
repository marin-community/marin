import argparse
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

# This unit exercises local omission/finalization behavior only. Stub optional
# S3 imports so it runs in the project test environment without fsspec.
sys.modules.setdefault("fsspec", types.SimpleNamespace())
rigging = sys.modules.setdefault("rigging", types.ModuleType("rigging"))
filesystem = sys.modules.setdefault(
    "rigging.filesystem", types.ModuleType("rigging.filesystem")
)
s3_compat = sys.modules.setdefault(
    "rigging.filesystem.s3_compat", types.ModuleType("rigging.filesystem.s3_compat")
)
s3_compat.configure_coreweave_s3 = lambda: None
rigging.filesystem = filesystem
filesystem.s3_compat = s3_compat

SPEC = importlib.util.spec_from_file_location(
    "sync_results", Path(__file__).resolve().parents[1] / "scripts/sync_results.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FS:
    def __init__(self):
        self.piped = {}
        self.puts = {}

    def exists(self, path):
        return path in self.piped

    def pipe(self, path, data):
        self.piped[path] = data

    def put(self, source, dest):
        self.puts[dest] = Path(source).read_bytes()

    def find(self, path):
        return [key for key in {**self.piped, **self.puts} if key.startswith(path.rstrip("/") + "/")]

    def cat(self, path):
        data = {**self.piped, **self.puts}.get(path)
        if data is None:
            raise FileNotFoundError(path)
        return data


class FailingFS(FS):
    def put(self, source, dest):
        if "/_objects/" in dest:
            raise OSError("object store unavailable")
        super().put(source, dest)


@pytest.mark.parametrize("item_prefix", ["items", "construction/items"])
def test_final_sync_rejects_protected_item_without_uploading_bytes(
    tmp_path, monkeypatch, item_prefix
):
    source = tmp_path / "results"
    secret = source / item_prefix / "x" / "workspace" / "secrets.json"
    secret.parent.mkdir(parents=True)
    secret.write_bytes(b"do-not-upload")
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source),
        state=str(tmp_path / "state"),
        destination="s3://ignored",
        final=True,
    )
    assert MOD.sync(args) == 1
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert manifest["final"] is False
    assert (
        manifest["omitted"][f"{item_prefix}/x/workspace/secrets.json"]
        == "credential_like_component"
    )
    assert b"do-not-upload" not in b"".join(fs.piped.values())


def test_declared_c07_mutant_reports_upload_without_weakening_prefix_filter(
    tmp_path, monkeypatch
):
    source = tmp_path / "results"
    declared = [
        "items/c07.experiments-health.experimental-design.ab-analysis-3-7ab4ffa1a034/workspace/build/mutants/reports/token_case_preserved.json",
        "quality/c07.experiments-health.experimental-design.ab-analysis-3-7ab4ffa1a034/attempt-1/input/workspace/build/mutants/reports/token_case_preserved.json",
    ]
    for relative in declared:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"killed")
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source),
        state=str(tmp_path / "state"),
        destination="s3://ignored",
        final=True,
    )
    assert MOD.sync(args) == 0
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert manifest["final"] is True
    assert set(manifest["files"]) == set(declared)
    assert b"killed" in fs.puts.values()


def test_parallel_object_failure_never_publishes_a_manifest(tmp_path, monkeypatch):
    source = tmp_path / "results"
    (source / "items/x/result.json").parent.mkdir(parents=True)
    (source / "items/x/result.json").write_text("{}")
    fs = FailingFS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 1
    assert "bucket/run/_manifests/latest.json" not in fs.puts


def test_fresh_state_skips_objects_already_stored_without_per_object_checks(tmp_path, monkeypatch):
    source = tmp_path / "results"
    (source / "items/x").mkdir(parents=True)
    (source / "items/x/old.json").write_text("restored")
    (source / "items/x/new.json").write_text("written after restore")
    fs = FS()
    fs.piped[f"bucket/run/_objects/{MOD.digest(source / 'items/x/old.json')}"] = b"restored"
    fs.exists = lambda path: pytest.fail(f"per-object check after a listing: {path}")
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    # With a listing in hand only the genuinely new object is uploaded; exists() is still
    # consulted for it by _upload_one, so allow exactly that path.
    new_key = f"bucket/run/_objects/{MOD.digest(source / 'items/x/new.json')}"
    fs.exists = lambda path: False if path == new_key else pytest.fail(f"unexpected check: {path}")
    assert MOD.sync(args) == 0
    uploaded = [k for k in fs.puts if "/_objects/" in k]
    assert uploaded == [new_key]
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert set(manifest["files"]) == {"items/x/old.json", "items/x/new.json"}


def test_growing_log_publishes_hash_of_immutable_captured_bytes(tmp_path, monkeypatch):
    source = tmp_path / "results"
    source.mkdir()
    log = source / "worker.log"
    log.write_bytes(b"before\n")
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    original_spool = MOD._spool_snapshot

    def spool(path, cache):
        path.write_bytes(b"before\nafter\n")
        captured = original_spool(path, cache)
        path.write_bytes(b"even later\n")
        return captured

    monkeypatch.setattr(MOD, "_spool_snapshot", spool)
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 0
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    expected = hashlib.sha256(b"before\nafter\n").hexdigest()
    assert manifest["files"]["worker.log"] == expected
    assert fs.puts[f"bucket/run/_objects/{expected}"] == b"before\nafter\n"
    assert manifest["final"] is False


def test_content_allowlist_is_rechecked_after_source_changes(tmp_path, monkeypatch):
    source = tmp_path / "results"
    token, _ = _pygments_paths(source)
    token.parent.mkdir(parents=True)
    token.write_bytes(b"verified-token")
    monkeypatch.setattr(MOD, "_PYGMENTS_221_TOKEN_SHA256", MOD.digest(token))
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    original_spool = MOD._spool_snapshot

    def spool(path, cache):
        path.write_bytes(b"changed-protected-content")
        return original_spool(path, cache)

    monkeypatch.setattr(MOD, "_spool_snapshot", spool)
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 1
    assert fs.puts == {}


def test_credential_names_remain_excluded_outside_declared_artifacts():
    for relative in [
        "items/x/workspace/token.json",
        "items/x/workspace/tokens",
        "items/x/workspace/secrets.yaml",
        "items/x/workspace/Secret.txt",
        "items/x/workspace/secrets/config.json",
        "items/x/workspace/.env",
        "items/x/workspace/.env.local",
        "items/x/workspace/pygments/token.py",
    ]:
        assert MOD.omission_reason(relative) == "credential_like_component"


def test_words_that_merely_start_with_credential_names_are_task_content():
    for relative in [
        "items/x/workspace/task/evidence/battery/token_governance/attempt-1.json",
        "items/x/workspace/build/mutants/reports/token_case_preserved.json",
        "items/x/workspace/token_backup.json",
        "items/x/workspace/secret_backup.json",
        "items/x/workspace/token-production/config.json",
        "items/x/workspace/tokenizer.py",
    ]:
        assert MOD.omission_reason(relative) is None


def _pygments_paths(source: Path) -> tuple[Path, Path]:
    package = (
        source / "items/x/workspace/build/venv/lib/python3.11/site-packages/pygments"
    )
    return package / "token.py", package / "__pycache__/token.cpython-311.pyc"


def test_verified_pygments_token_source_uploads_and_its_cache_is_intentionally_ignored(
    tmp_path, monkeypatch
):
    source = tmp_path / "results"
    token, cache = _pygments_paths(source)
    token.parent.mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    token.write_bytes(b"verified-token")
    cache.write_bytes(b"regenerable-bytecode")
    monkeypatch.setattr(MOD, "_PYGMENTS_221_TOKEN_SHA256", MOD.digest(token))
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source),
        state=str(tmp_path / "state"),
        destination="s3://ignored",
        final=True,
    )
    assert MOD.sync(args) == 0
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    token_rel = token.relative_to(source).as_posix()
    cache_rel = cache.relative_to(source).as_posix()
    assert token_rel in manifest["files"]
    assert manifest["omitted"][cache_rel] == "regenerable_verified_pygments_cache"
    assert cache.read_bytes() not in fs.piped.values()


def test_verified_pygments_clean_toolchain_source_survives_generate_sync(
    tmp_path, monkeypatch
):
    source = tmp_path / "results"
    token = (
        source
        / "construction/items/x/workspace/clean-inputs/toolchain/lib/pygments/token.py"
    )
    token.parent.mkdir(parents=True)
    token.write_bytes(b"verified-token")
    monkeypatch.setattr(MOD, "_PYGMENTS_221_TOKEN_SHA256", MOD.digest(token))
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source), state=str(tmp_path / "state"),
        destination="s3://ignored", final=True,
    )
    assert MOD.sync(args) == 0
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert manifest["final"] is True
    assert manifest["files"][token.relative_to(source).as_posix()] == MOD.digest(token)

    token.write_bytes(b"changed-protected-content")
    assert MOD.sync(args) == 1
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert manifest["omitted"][token.relative_to(source).as_posix()] == "credential_like_component"


def test_mutated_pygments_token_and_cache_remain_credential_omissions(
    tmp_path, monkeypatch
):
    source = tmp_path / "results"
    token, cache = _pygments_paths(source)
    token.parent.mkdir(parents=True)
    cache.parent.mkdir(parents=True)
    token.write_bytes(b"mutated-token")
    cache.write_bytes(b"cache")
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source),
        state=str(tmp_path / "state"),
        destination="s3://ignored",
        final=True,
    )
    assert MOD.sync(args) == 1
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert (
        manifest["omitted"][token.relative_to(source).as_posix()]
        == "credential_like_component"
    )
    assert (
        manifest["omitted"][cache.relative_to(source).as_posix()]
        == "credential_like_component"
    )


def test_pygments_cache_without_verified_source_remains_credential_omission(
    tmp_path, monkeypatch
):
    source = tmp_path / "results"
    _, cache = _pygments_paths(source)
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b"cache")
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(
        source=str(source),
        state=str(tmp_path / "state"),
        destination="s3://ignored",
        final=True,
    )
    assert MOD.sync(args) == 1
    manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
    assert (
        manifest["omitted"][cache.relative_to(source).as_posix()]
        == "credential_like_component"
    )


def test_verified_pygments_token_survives_every_stage_root(tmp_path, monkeypatch):
    """A new phase root must not silently lose the content-bound exemption."""
    for stage in ("items", "construction/items", "checkpoint-revalidation/items"):
        source = tmp_path / stage.replace("/", "_") / "results"
        token = (
            source / stage
            / "x/workspace/clean-inputs/toolchain/lib/pygments/token.py"
        )
        token.parent.mkdir(parents=True)
        token.write_bytes(b"verified-token")
        monkeypatch.setattr(MOD, "_PYGMENTS_221_TOKEN_SHA256", MOD.digest(token))
        fs = FS()
        monkeypatch.setattr(MOD, "resolve", lambda _, fs=fs: (fs, "bucket/run"))
        args = argparse.Namespace(
            source=str(source), state=str(source.parent / "state"),
            destination="s3://ignored", final=True,
        )
        assert MOD.sync(args) == 0, stage
        manifest = json.loads(fs.puts["bucket/run/_manifests/latest.json"])
        relative = token.relative_to(source).as_posix()
        assert manifest["omitted"] == {}, stage
        assert manifest["files"][relative] == MOD.digest(token), stage


def _publish(fs, files):
    fs.puts["bucket/run/_manifests/latest.json"] = json.dumps({"files": files}).encode()


@pytest.mark.parametrize("final", [False, True])
def test_partial_tree_never_replaces_a_fuller_snapshot(tmp_path, monkeypatch, final):
    """2026-09-29 shard-071/075: a half-restored tree was published over the snapshot."""
    source = tmp_path / "results"
    (source / "items/a").mkdir(parents=True)
    (source / "items/a/status.json").write_text("{}")
    (source / "attack-adjudication/x.json").parent.mkdir(parents=True)
    (source / "attack-adjudication/x.json").write_text("{}")
    fs = FS()
    _publish(fs, {"items/a/status.json": "0" * 64, "items/b/status.json": "1" * 64,
                  "quality/a/attempt-1/result.json": "2" * 64})
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=final)
    before = fs.puts["bucket/run/_manifests/latest.json"]
    assert MOD.sync(args) == 1
    assert fs.puts["bucket/run/_manifests/latest.json"] == before
    assert not (tmp_path / "state/uploaded.json").exists()  # the next round re-checks


def test_whole_tree_publishes_and_keeps_hourly_history(tmp_path, monkeypatch):
    source = tmp_path / "results"
    for item in ("a", "b"):
        (source / f"items/{item}").mkdir(parents=True)
        (source / f"items/{item}/status.json").write_text(item)
    fs = FS()
    _publish(fs, {"items/a/status.json": "0" * 64, "items/b/status.json": "1" * 64})
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 0
    history = [key for key in fs.puts if "/_manifests/history/" in key]
    assert len(history) == 1
    (source / "items/c").mkdir()
    (source / "items/c/status.json").write_text("c")
    assert MOD.sync(args) == 0  # same hour: no second history object
    assert [key for key in fs.puts if "/_manifests/history/" in key] == history


def test_unreadable_published_manifest_refuses_a_fresh_publish(tmp_path, monkeypatch):
    source = tmp_path / "results"
    (source / "items/a").mkdir(parents=True)
    (source / "items/a/status.json").write_text("{}")
    fs = FS()
    fs.puts["bucket/run/_manifests/latest.json"] = b"not json"
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 1


def _snapshot(fs, files, stamps):
    for name, value in stamps.items():
        data = json.dumps(value).encode()
        sha = hashlib.sha256(data).hexdigest()
        fs.piped[f"bucket/run/_objects/{sha}"] = data
        files[name] = sha
    fs.puts["bucket/run/_manifests/latest.json"] = json.dumps({"files": files}).encode()


def test_restore_waits_for_a_still_publishing_writer(tmp_path, monkeypatch):
    fs = FS()
    status = b'{"state": "quality_accepted"}'
    status_sha = hashlib.sha256(status).hexdigest()
    fs.piped[f"bucket/run/_objects/{status_sha}"] = status
    # The old job's last incremental snapshot: started, never finished.
    _snapshot(fs, {}, {"submission.json": {"started_utc": "2026-09-29T20:32:00Z"}})
    sleeps = []

    def final_sync_lands(_seconds):
        sleeps.append(_seconds)
        _snapshot(fs, {"items/a/status.json": status_sha},
                  {"submission.json": {"started_utc": "2026-09-29T20:32:00Z"},
                   "terminal.json": {"finished_utc": "2026-09-29T21:10:53Z"}})

    monkeypatch.setattr(MOD.time, "sleep", final_sync_lands)
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source="s3://ignored", destination=str(tmp_path / "dest"), new_run=False)
    assert MOD.restore(args) == 0
    assert sleeps == [15]
    assert (tmp_path / "dest/items/a/status.json").read_bytes() == status


def test_restore_of_a_finished_writer_does_not_wait(tmp_path, monkeypatch):
    fs = FS()
    _snapshot(fs, {}, {"submission.json": {"started_utc": "2026-09-29T20:32:00Z"},
                       "terminal.json": {"finished_utc": "2026-09-29T21:10:53Z"}})
    monkeypatch.setattr(MOD.time, "sleep", lambda _s: pytest.fail("waited for a finished writer"))
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source="s3://ignored", destination=str(tmp_path / "dest"), new_run=False)
    assert MOD.restore(args) == 0


def test_conveyor_view_is_published_at_a_fixed_key_only_when_it_changes(tmp_path, monkeypatch):
    source = tmp_path / "results"
    source.mkdir()
    (source / "conveyor.json").write_text('{"items": {}}')
    fs = FS()
    monkeypatch.setattr(MOD, "resolve", lambda _: (fs, "bucket/run"))
    args = argparse.Namespace(source=str(source), state=str(tmp_path / "state"), destination="s3://ignored", final=False)
    assert MOD.sync(args) == 0
    assert fs.puts["bucket/run/_live/conveyor.json"] == b'{"items": {}}'
    del fs.puts["bucket/run/_live/conveyor.json"]
    assert MOD.sync(args) == 0
    assert "bucket/run/_live/conveyor.json" not in fs.puts  # unchanged: not re-sent
    (source / "conveyor.json").write_text('{"items": {"a": {}}}')
    assert MOD.sync(args) == 0
    assert fs.puts["bucket/run/_live/conveyor.json"] == b'{"items": {"a": {}}}'
