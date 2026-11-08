# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from iris.cluster.node_agent.runsc_logs import protected_runtime_ids, prune, redirected_config


def test_redirected_config_preserves_provider_runtime_options():
    source = """# Provider-owned configuration
log_path = "/var/log/runsc/%ID%/shim.log"
log_level = "info"
binary_name = "/usr/local/bin/nvidia-container-runtime-gvisor"
[runsc_config]
systemd-cgroup = "true"
nvproxy = "true"
debug = "false"
debug-log = "/var/log/runsc/%ID%/gvisor.%COMMAND%.log"
"""
    updated = redirected_config(source, Path("/mnt/local/iris-runtime/runsc"))
    settings = tomllib.loads(updated)
    assert settings == {
        "log_path": "/mnt/local/iris-runtime/runsc/%ID%/shim.log",
        "log_level": "info",
        "binary_name": "/usr/local/bin/nvidia-container-runtime-gvisor",
        "runsc_config": {
            "systemd-cgroup": "true",
            "nvproxy": "true",
            "debug": "false",
            "debug-log": "/mnt/local/iris-runtime/runsc/%ID%/gvisor.%COMMAND%.log",
        },
    }
    assert redirected_config(updated, Path("/mnt/local/iris-runtime/runsc")) == updated


def test_prune_preserves_live_recent_and_unrecognized_logs(tmp_path):
    root = tmp_path / "runsc"
    root.mkdir()
    directories = {}
    for index, name in enumerate(("orphan", "live", "recent", "recent-file")):
        directory = root / f"{index:064x}"
        directory.mkdir()
        (directory / "shim.log").write_text(name)
        os.utime(directory / "shim.log", (100, 100))
        os.utime(directory, (100, 100))
        directories[name] = directory
    os.utime(directories["recent"], (300, 300))
    os.utime(directories["recent-file"] / "shim.log", (300, 300))
    unknown = root / "%ID%"
    unknown.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.log").write_text("keep")
    (root / ("f" * 64)).symlink_to(outside, target_is_directory=True)
    known = {directories["live"].name}

    count, allocated = prune((root,), known, cutoff=200, apply=False)
    assert count == 1
    assert allocated > 0
    assert (directories["orphan"] / "shim.log").read_text() == "orphan"
    assert prune((root,), known, cutoff=200, apply=True) == (count, allocated)
    assert not directories["orphan"].exists()
    assert (directories["live"] / "shim.log").read_text() == "live"
    assert (directories["recent"] / "shim.log").read_text() == "recent"
    assert (directories["recent-file"] / "shim.log").read_text() == "recent-file"
    assert unknown.is_dir()
    assert (outside / "keep.log").read_text() == "keep"


def test_protected_runtime_ids_include_cri_processes_and_open_logs(tmp_path, monkeypatch):
    def crictl(command, **kwargs):
        key, identifier = ("items", "1" * 64) if command[1] == "pods" else ("containers", "2" * 64)
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({key: [{"id": identifier}]}))

    monkeypatch.setattr(subprocess, "run", crictl)
    proc = tmp_path / "proc"
    process = proc / "123"
    (process / "fd").mkdir(parents=True)
    (process / "cmdline").write_bytes(b"shim\x00-id\x00" + b"3" * 64 + b"\x00")
    root = tmp_path / "logs"
    (process / "fd" / "1").symlink_to(root / ("4" * 64) / "shim.log")
    assert protected_runtime_ids((root,), proc) == {str(index) * 64 for index in range(1, 5)}


def test_failed_cri_inventory_aborts_cleanup(tmp_path, monkeypatch):
    def crictl(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="runtime unavailable")

    monkeypatch.setattr(subprocess, "run", crictl)
    with pytest.raises(subprocess.CalledProcessError):
        protected_runtime_ids((tmp_path,))
