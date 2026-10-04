# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import subprocess
import tarfile
from dataclasses import replace

import pytest
from rigging.runtime_bundle import RuntimeBundle, install_runtime_bundle


def test_runtime_bundle_installs_verified_executable_and_rejects_changed_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", os.environ["PATH"])
    source = tmp_path / "source" / "russell-rsi-runtime"
    source.mkdir(parents=True)
    tool = source / "probe"
    tool.write_text("#!/bin/sh\necho runtime-ready\n")
    tool.chmod(0o755)
    archive = tmp_path / "runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname=source.name)
    parent = tmp_path / "installed"
    manifest = {
        "directory_name": source.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": {"probe": hashlib.sha256(tool.read_bytes()).hexdigest()},
        "host_packages": {},
        "tools": {"probe": str(parent / source.name / "probe")},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config = RuntimeBundle(
        str(manifest_path),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        str(archive),
        manifest["archive_sha256"],
        str(parent),
    )
    result = install_runtime_bundle(config)
    assert subprocess.check_output([result["tools"]["probe"]], text=True) == "runtime-ready\n"
    installed = parent / source.name / "probe"
    installed.write_text("changed")
    with pytest.raises(ValueError, match="file hash mismatch"):
        install_runtime_bundle(config)
    assert installed.read_text() == "changed"
    fresh_parent = tmp_path / "bad-archive"
    archive.write_bytes(archive.read_bytes() + b"changed archive")
    with pytest.raises(ValueError, match="archive hash mismatch"):
        install_runtime_bundle(replace(config, installation_parent=str(fresh_parent)))
    assert not (fresh_parent / source.name).exists()
