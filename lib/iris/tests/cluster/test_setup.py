# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup-script construction and resolution at the submit boundary."""

from iris.cluster.setup import default_setup_script
from iris.cluster.types import EnvironmentSpec


def test_default_script_syncs_all_packages_by_default():
    script = default_setup_script()
    assert "uv sync" in script
    assert "--all-packages" in script
    # cloudpickle/py-spy/memray are always installed so callable entrypoints and
    # the profiler attach paths work.
    assert "cloudpickle" in script
    assert "py-spy" in script
    assert "memray" in script


def test_default_script_includes_extras():
    script = default_setup_script(extras=["gpu", "mypackage:data"])
    # The package prefix is dropped; the extra name is what uv receives.
    assert "--extra gpu" in script
    assert "--extra data" in script


def test_default_script_scopes_sync_to_packages():
    script = default_setup_script(packages=["marin-core", "iris"], extras=["tpu"])
    assert "--all-packages" not in script
    assert "--package marin-core" in script
    assert "--package iris" in script
    assert "--extra tpu" in script


def test_default_script_pins_python_version():
    assert "--python 3.12" in default_setup_script(python_version="3.12")
    assert "--python" not in default_setup_script()


def test_default_script_uses_workdir_and_pip_packages():
    script = default_setup_script(pip_packages=["torch>=2.0"])
    assert 'cd "$IRIS_WORKDIR"' in script
    assert "torch>=2.0" in script


def test_to_proto_builds_default_when_setup_script_unset():
    cfg = EnvironmentSpec(extras=["tpu"], sync_packages=["marin-core"]).to_proto()
    assert "uv sync" in cfg.setup_script
    assert "--package marin-core" in cfg.setup_script
    assert "--extra tpu" in cfg.setup_script


def test_to_proto_uses_custom_script_verbatim():
    cfg = EnvironmentSpec(setup_script="echo hi\n", extras=["tpu"]).to_proto()
    # extras are ignored once a script is supplied.
    assert cfg.setup_script == "echo hi\n"


def test_to_proto_empty_script_means_no_setup():
    cfg = EnvironmentSpec(setup_script="").to_proto()
    assert cfg.setup_script == ""
