# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tomllib
from pathlib import Path

import pytest
from iris.cluster.node_agent.runtime_storage import configure

# runsc.toml as shipped by CoreWeave node images that log sandboxes to the root filesystem.
LOGGING_CONFIG = """\
log_path = "/var/log/runsc/%ID%/shim.log"
log_level = "info"
binary_name = "/usr/local/bin/runsc"

[runsc_config]
  systemd-cgroup = "true"
  nvproxy = "false"
  debug = "false"
  debug-log = "/var/log/runsc/%ID%/gvisor.%COMMAND%.log"
"""

# runsc.toml as shipped by newer CoreWeave node images, which disable sandbox logging.
LOGGING_DISABLED_CONFIG = """\
# Shim configuration for the gvisor runtime class, rendered once
# per entry in gvisor_runtime_classes -- see defaults/main.yml. Every class
# shares one runsc and one shim binary; this file is what makes them behave
# differently, so a setting added here lands on exactly one class.
#
# Sandbox logging is off: with log_path set, the shim writes to
# /var/log/runsc/<container-id>/ on the tmpfs root and creates a directory per
# sandbox that nothing ever removes. Shim logs go to containerd's own output instead,
# which journald already caps: journalctl -u containerd.
#
# Debugging a sandbox means uncommenting log_path and debug-log below (%ID% is
# the container ID) and turning debug on. Anything written there is unmanaged,
# so undo it when finished.
#log_path = "/var/log/runsc/%ID%/shim.log"
log_level = "info"
binary_name = "/usr/local/bin/runsc"

[runsc_config]
  systemd-cgroup = "true"
  nvproxy = "false"
  debug = "false"
  #debug-log = "/var/log/runsc/%ID%/gvisor.%COMMAND%.log"
  # Inert with nvproxy off, and emitted for every class regardless so that
  # nvproxy is the single line of difference between them.
  nvproxy-allow-unsupported-driver = "true"
"""


def test_configure_redirects_logs_and_preserves_backup_and_permissions(tmp_path, monkeypatch):
    config = tmp_path / "etc/containerd/runsc.toml"
    config.parent.mkdir(parents=True)
    source = LOGGING_CONFIG
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
    settings = tomllib.loads(updated)
    assert settings["log_path"] == f"{nvme}/iris-runtime/runsc/%ID%/shim.log"
    assert settings["runsc_config"]["debug-log"] == f"{nvme}/iris-runtime/runsc/%ID%/gvisor.%COMMAND%.log"
    assert config.stat().st_mode & 0o777 == 0o640
    backups = list((nvme / "iris-runtime").glob("runsc.toml.*.bak"))
    assert len(backups) == 1
    assert backups[0].read_text() == source
    configure(config, nvme)
    assert config.read_text() == updated
    assert backups[0].read_text() == source


@pytest.mark.parametrize(
    "source",
    [
        LOGGING_CONFIG,
        LOGGING_CONFIG.replace("log_path =", "#log_path ="),
        LOGGING_CONFIG.replace("debug-log =", "#debug-log ="),
    ],
    ids=["both-paths", "debug-log-only", "log-path-only"],
)
def test_configure_refuses_to_place_runtime_logs_on_host_root(tmp_path, source):
    config = tmp_path / "runsc.toml"
    config.write_text(source)
    nvme = tmp_path / "unmounted-nvme"
    nvme.mkdir()
    with pytest.raises(ValueError, match="different filesystem"):
        configure(config, nvme)
    assert config.read_text() == source
    assert not (nvme / "iris-runtime").exists()


@pytest.mark.parametrize("source", [LOGGING_DISABLED_CONFIG, 'log_level = "info"\n'])
def test_configure_leaves_config_with_logging_disabled_untouched(tmp_path, source):
    config = tmp_path / "etc/containerd/runsc.toml"
    config.parent.mkdir(parents=True)
    config.write_text(source)
    # Same filesystem as the config: setup must not need NVMe when there is nothing to redirect.
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    configure(config, nvme)
    assert config.read_text() == source
    assert not (nvme / "iris-runtime").exists()
