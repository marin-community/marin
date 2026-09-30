import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "pull_snapshot", Path(__file__).resolve().parents[1] / "scripts/pull_snapshot.py"
)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


class FS:
    def __init__(self, files):
        self.reads = []
        self.objects = {}
        self.publish("initial", files)

    def publish(self, identity, files):
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        self.manifest = json.dumps({"snapshot_id": identity, "files": hashes, "final": False}).encode()
        self.objects.update({sha: files[name] for name, sha in hashes.items()})

    def cat(self, path):
        self.reads.append(path)
        return self.manifest if path.endswith("latest.json") else self.objects[path.rsplit("/", 1)[-1]]


def test_resume_uses_captured_manifest_not_mutated_latest(tmp_path):
    fs = FS({"items/one/result.json": b"old", "items/two/result.json": b"other"})
    first = MOD.pull_snapshot(fs, "bucket/run", "s3://bucket/run", tmp_path, prefixes=("items/one",))
    assert first["complete_snapshot"] is False
    fs.publish("new", {"items/one/result.json": b"new"})
    (tmp_path / "items/one/result.json").write_bytes(b"corrupt-local-copy")
    second = MOD.pull_snapshot(fs, "bucket/run", "s3://bucket/run", tmp_path, prefixes=("items/one",))
    assert second["remote_snapshot_id"] == "initial"
    assert (tmp_path / "items/one/result.json").read_bytes() == b"old"
    assert sum(p.endswith("latest.json") for p in fs.reads) == 1
    with pytest.raises(MOD.SnapshotError, match="different source or selection"):
        MOD.pull_snapshot(fs, "bucket/other", "s3://bucket/other", tmp_path)


def test_remote_corruption_never_writes_file_or_success_receipt(tmp_path):
    fs = FS({"result.json": b"expected"})
    fs.objects = {key: b"corrupt" for key in fs.objects}
    with pytest.raises(MOD.SnapshotError, match="checksum mismatch"):
        MOD.pull_snapshot(fs, "bucket/run", "source", tmp_path)
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "pull-manifest.json").exists()
    assert (tmp_path / "snapshot-capture.json").exists()


def test_published_manifest_with_omissions_is_not_complete_snapshot(tmp_path):
    fs = FS({"result.json": b"data"})
    manifest = json.loads(fs.manifest)
    manifest["omitted"] = {"items/x/token.py": "credential_like_component"}
    fs.manifest = json.dumps(manifest).encode()
    receipt = MOD.pull_snapshot(fs, "bucket/run", "source", tmp_path)
    assert receipt["complete_manifest"] is True
    assert receipt["complete_snapshot"] is False
    assert receipt["omitted"] == manifest["omitted"]


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a//b", "a/../b", "pull-manifest.json"])
def test_unsafe_remote_names_rejected_before_capture(tmp_path, name):
    with pytest.raises(MOD.SnapshotError):
        MOD.pull_snapshot(FS({name: b"x"}), "bucket/run", "source", tmp_path)
    assert not list(tmp_path.iterdir())


def test_symlink_on_resume_cannot_write_outside_destination(tmp_path):
    destination = tmp_path / "download"
    fs = FS({"items/result.json": b"data"})
    MOD.pull_snapshot(fs, "bucket/run", "source", destination)
    (destination / "items/result.json").unlink()
    (destination / "items").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "items").symlink_to(outside, target_is_directory=True)
    with pytest.raises(MOD.SnapshotError, match="symlink"):
        MOD.pull_snapshot(fs, "bucket/run", "source", destination)
    assert not list(outside.iterdir())
