import hashlib
import json
import os
import shutil

import pytest

from capability_pipeline.fixed_grading_capture import (
    load_capture,
    materialize_workspace,
    write_capture,
)
from capability_pipeline.grading_input import grading_input_fingerprint


def _capture_inputs(workspace):
    specification = b'{"specification":"frozen"}'
    protocol = b'{"submission":{"path":"out/a"}}'
    response = "answer"
    transcript = ({"role": "assistant", "content": "answer"},)
    payload = json.dumps(
        {
            "specification": json.loads(specification),
            "protocol": json.loads(protocol),
            "step_index": 0,
            "attempt": {"response": response, "transcript": transcript},
        }
    ).encode()
    fingerprint = grading_input_fingerprint(
        specification,
        json.loads(protocol),
        response,
        workspace,
        transcript,
        payload=payload,
    )
    return {
        "specification": specification,
        "protocol": protocol,
        "response": response,
        "transcript": transcript,
        "payload": payload,
        "workspace": workspace,
        "fingerprint": fingerprint,
        "source_specification_sha256": "b" * 64,
        "source_renderings_sha256": "c" * 64,
    }


def test_capture_binds_full_workspace_and_rejects_mutation(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "empty").mkdir(parents=True)
    (workspace / "out").mkdir()
    (workspace / "out/a").write_bytes(b"a")
    root = tmp_path / "capture"
    receipt = write_capture(root, **_capture_inputs(workspace))
    loaded = load_capture(root, expected_manifest_sha256=receipt["manifest_sha256"])
    assert loaded["manifest_sha256"] == receipt["manifest_sha256"]
    assert (loaded["workspace"] / "empty").is_dir() and loaded["response"] == "answer"
    (root / "workspace/out/a").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed|differs"):
        load_capture(root)


def test_capture_rejects_links_and_overwrite(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    values = _capture_inputs(workspace)
    (workspace / "link").symlink_to(tmp_path)
    root = tmp_path / "capture"
    with pytest.raises(ValueError, match="symlink|unsafe"):
        write_capture(root, **values)
    assert root.exists()
    with pytest.raises(FileExistsError):
        write_capture(root, **values)


def test_capture_rejects_unlisted_file_linked_manifest_and_bad_payload(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "capture"
    write_capture(root, **_capture_inputs(workspace))
    (root / "unexpected.txt").write_text("not in manifest")
    with pytest.raises(ValueError, match="tree differs"):
        load_capture(root)

    clean = tmp_path / "clean"
    receipt = write_capture(clean, **_capture_inputs(workspace))
    manifest = clean / "manifest.json"
    copied = clean / "manifest-copy.json"
    manifest.rename(copied)
    manifest.symlink_to(copied.name)
    with pytest.raises(ValueError, match="manifest is absent or linked"):
        load_capture(clean)
    assert receipt["manifest_sha256"] == hashlib.sha256(copied.read_bytes()).hexdigest()

    invalid = tmp_path / "invalid"
    values = _capture_inputs(workspace)
    values["payload"] = values["payload"].replace(b"answer", b"other", 1)
    with pytest.raises(ValueError, match="payload response"):
        write_capture(invalid, **values)


def test_capture_rejects_manifest_identity_and_noncanonical_member_spelling(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    root = tmp_path / "capture"
    write_capture(root, **_capture_inputs(workspace))
    with pytest.raises(ValueError, match="expected identity"):
        load_capture(root, expected_manifest_sha256="0" * 64)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["members"][0]["path"] = "workspace//bad"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unsafe"):
        load_capture(root)


def test_capture_survives_byte_only_restore_and_reconstructs_logical_tree(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "empty").mkdir(parents=True)
    (workspace / "bin").mkdir()
    executable = workspace / "bin/run"
    executable.write_text("#!/bin/sh\n")
    os.chmod(workspace / "bin", 0o711)
    os.chmod(executable, 0o755)
    root = tmp_path / "capture"
    receipt = write_capture(root, **_capture_inputs(workspace))

    # This models the S3 synchronizer: files arrive, but their source modes
    # normalize and empty directories do not survive.
    restored = tmp_path / "restored"
    shutil.copytree(root, restored, copy_function=shutil.copyfile)
    (restored / "workspace/empty").rmdir()
    for path in restored.rglob("*"):
        if path.is_file() and not path.is_symlink():
            os.chmod(path, 0o644)
    captured = load_capture(
        restored, expected_manifest_sha256=receipt["manifest_sha256"]
    )
    logical = tmp_path / "logical"
    materialize_workspace(captured, logical)
    assert (logical / "empty").is_dir()
    assert (logical / "bin").stat().st_mode & 0o777 == 0o711
    assert (logical / "bin/run").stat().st_mode & 0o777 == 0o755

    manifest = json.loads((restored / "manifest.json").read_text())
    next(member for member in manifest["members"] if member["path"] == "workspace/bin")[
        "mode"
    ] = 0o700
    (restored / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected identity"):
        load_capture(restored, expected_manifest_sha256=receipt["manifest_sha256"])
