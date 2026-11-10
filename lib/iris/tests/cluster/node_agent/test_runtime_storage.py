# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tomllib
from pathlib import Path

import pytest
from iris.cluster.node_agent.runtime_storage import configure


def test_configure_redirects_logs_and_preserves_backup_and_permissions(tmp_path, monkeypatch):
    config = tmp_path / "etc/containerd/runsc.toml"
    config.parent.mkdir(parents=True)
    source = 'log_path = "/var/log/runsc/%ID%/shim.log"\n[runsc_config]\ndebug-log = "/var/log/runsc/%ID%/gvisor.log"\n'
    config.write_text(source)
    config.chmod(0o640)
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    host_stat = Path.stat

    def separate_nvme_device(path, **kwargs):
        metadata = host_stat(path, **kwargs)
        if path.is_relative_to(nvme):
            values = list(metadata)
            values[2] += 1
            return os.stat_result(values)
        return metadata

    monkeypatch.setattr(Path, "stat", separate_nvme_device)
    configure(config, nvme)
    updated = config.read_text()
    assert tomllib.loads(updated)["log_path"] == f"{nvme}/iris-runtime/runsc/%ID%/shim.log"
    assert config.stat().st_mode & 0o777 == 0o640
    backups = list((nvme / "iris-runtime").glob("runsc.toml.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text() == source
    configure(config, nvme)
    assert config.read_text() == updated
    assert backups[0].read_text() == source


def test_configure_refuses_to_place_runtime_logs_on_host_root(tmp_path):
    config = tmp_path / "runsc.toml"
    source = 'log_path = "/var/log/runsc/%ID%/shim.log"\n[runsc_config]\ndebug-log = "/var/log/runsc/%ID%/gvisor.log"\n'
    config.write_text(source)
    nvme = tmp_path / "unmounted-nvme"
    nvme.mkdir()
    with pytest.raises(ValueError, match="different filesystem"):
        configure(config, nvme)
    assert config.read_text() == source
    assert not (nvme / "iris-runtime").exists()
