import hashlib
import json
import shutil

import pytest
from test_checkpoint_revalidation import _bundle, _dump, _files, _sha

from capability_pipeline.inference import digest
from scripts.build_checkpoint_revalidation import build_bundle


def _inputs(tmp_path):
    seed_bundle, item = _bundle(tmp_path)
    checkpoint = tmp_path / "checkpoint"
    shutil.copytree(seed_bundle / "restore-seed", checkpoint)
    source = tmp_path / "historical-source"
    package = source / "capability_pipeline"
    package.mkdir(parents=True)
    (package / "synthesis.py").write_text("# historical controller; evidence only\n")
    for name in ("source.lock.json", "composite_extension.lock.json"):
        _dump(source / "vendor/task_spec" / name, {"revision": "historical"})
        shutil.copy2(source / "vendor/task_spec" / name,
                     checkpoint / "construction/items" / item / "contract" / name)
    _dump(checkpoint / "input-pilot.json", {"capabilities": []})
    _dump(checkpoint / "proposal/accepted.json", json.loads((seed_bundle / "accepted.json").read_text()))
    identity = {"pilot_sha256": _sha(checkpoint / "input-pilot.json"),
                "settings": {"controller_package_sha256": digest(_files(package)),
                             "taskcompendium_lock_sha256": _sha(source / "vendor/task_spec/source.lock.json")}}
    identity["identity_sha256"] = digest(identity)
    _dump(checkpoint / "generate-run.json", identity)
    files = _files(checkpoint)
    remote = {"snapshot_id": "terminal-snapshot", "final": True, "files": files, "omitted": {}}
    raw = json.dumps(remote)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    _dump(checkpoint / "snapshot-capture.json", {
        "remote_manifest_json": raw, "remote_manifest_sha256": sha,
    })
    _dump(checkpoint / "pull-manifest.json", {
        "schema_version": "capability-snapshot-pull-v1", "files": files,
        "remote_manifest_sha256": sha, "remote_snapshot_id": "terminal-snapshot",
        "remote_final": True, "complete_manifest": True, "complete_snapshot": True,
    })
    return checkpoint, source, item


def test_build_binds_full_snapshot_and_historical_controller(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    output = tmp_path / "prepared"
    result = build_bundle(checkpoint, source, item, output)
    assert result["state"] == "prepared"
    assert result["source_snapshot_id"] == "terminal-snapshot"
    assert (output / "source-controller/capability_pipeline/synthesis.py").read_bytes() == (
        source / "capability_pipeline/synthesis.py"
    ).read_bytes()
    assert (output / "restore-seed/generate-run.json").read_bytes() == (
        checkpoint / "generate-run.json"
    ).read_bytes()


def test_build_rejects_different_historical_controller_before_copy(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    (source / "capability_pipeline/synthesis.py").write_text("# different\n")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="historical controller differs"):
        build_bundle(checkpoint, source, item, output)
    assert not output.exists()


def test_build_rejects_tampered_checkpoint_before_copy(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    (checkpoint / "construction" / "items" / item / "workspace/artifact.txt").write_text("changed")
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="hash|fingerprint|checksum"):
        build_bundle(checkpoint, source, item, output)
    assert not output.exists()


def _record_omission(checkpoint, name, *, final=True):
    remote = json.loads(json.loads((checkpoint / "snapshot-capture.json").read_text())["remote_manifest_json"])
    remote["omitted"] = {name: "credential_like_component"}
    remote["final"] = final
    raw = json.dumps(remote)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    _dump(checkpoint / "snapshot-capture.json", {
        "remote_manifest_json": raw, "remote_manifest_sha256": sha,
    })
    pull = json.loads((checkpoint / "pull-manifest.json").read_text())
    pull.update({"omitted": remote["omitted"], "remote_manifest_sha256": sha,
                 "remote_final": final, "complete_manifest": True,
                 "complete_snapshot": False})
    _dump(checkpoint / "pull-manifest.json", pull)


def test_build_retains_unrelated_item_omission_without_claiming_full_snapshot(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    omission = "construction/items/another-task/workspace/research/venv/tokenizer.py"
    _record_omission(checkpoint, omission)
    output = tmp_path / "prepared"

    result = build_bundle(checkpoint, source, item, output)

    assert result["state"] == "prepared"
    pull = json.loads((output / "source-metadata/pull-manifest.json").read_text())
    assert pull["omitted"] == {omission: "credential_like_component"}
    assert pull["complete_snapshot"] is False


@pytest.mark.parametrize("omission", [
    "construction/items/{item}/workspace/research/venv/tokenizer.py",
    "construction/repairs/{item}/attempt-1/status.json",
    "construction/repair-history/{item}/attempt-1/status.json",
    "construction/quality/{item}/result.json",
    "construction/report.json",
    "proposal/accepted.json",
])
def test_build_rejects_selected_or_shared_omission(tmp_path, omission):
    checkpoint, source, item = _inputs(tmp_path)
    _record_omission(checkpoint, omission.format(item=item))
    with pytest.raises(ValueError, match="omission affects selected or shared evidence"):
        build_bundle(checkpoint, source, item, tmp_path / "prepared")


def test_build_rejects_nonterminal_snapshot_with_unrelated_omission(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    _record_omission(checkpoint, "construction/items/another-task/workspace/token.py", final=False)
    with pytest.raises(ValueError, match="terminal snapshot"):
        build_bundle(checkpoint, source, item, tmp_path / "prepared")


def test_build_rejects_incomplete_manifest_even_with_unrelated_omission(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    _record_omission(checkpoint, "construction/items/another-task/workspace/token.py")
    pull = json.loads((checkpoint / "pull-manifest.json").read_text())
    pull["complete_manifest"] = False
    _dump(checkpoint / "pull-manifest.json", pull)
    with pytest.raises(ValueError, match="verified terminal snapshot"):
        build_bundle(checkpoint, source, item, tmp_path / "prepared")


def test_unrelated_omission_does_not_excuse_missing_declared_file(tmp_path):
    checkpoint, source, item = _inputs(tmp_path)
    _record_omission(checkpoint, "construction/items/another-task/workspace/token.py")
    (checkpoint / "construction/items" / item / "workspace/artifact.txt").unlink()
    with pytest.raises(ValueError, match="fingerprint differs"):
        build_bundle(checkpoint, source, item, tmp_path / "prepared")
