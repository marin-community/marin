# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for setup-script construction and resolution."""

import pytest
from google.protobuf import json_format
from iris.cluster.runtime.setup import default_setup_script, resolve_setup_script
from iris.rpc import job_pb2


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


def test_resolve_default_mode_builds_script():
    env = job_pb2.EnvironmentConfig(extras=["tpu"], sync_packages=["marin-core"])
    script = resolve_setup_script(env)
    assert "uv sync" in script
    assert "--package marin-core" in script
    assert "--extra tpu" in script


def test_resolve_custom_mode_uses_script_verbatim():
    env = job_pb2.EnvironmentConfig(
        setup_mode=job_pb2.SETUP_MODE_CUSTOM,
        setup_script="echo hello\npip install thing\n",
        extras=["tpu"],  # ignored in custom mode
    )
    script = resolve_setup_script(env)
    assert script == "echo hello\npip install thing\n"
    assert "uv sync" not in script


def test_resolve_custom_empty_means_no_setup():
    env = job_pb2.EnvironmentConfig(setup_mode=job_pb2.SETUP_MODE_CUSTOM, setup_script="")
    assert resolve_setup_script(env) == ""


@pytest.mark.parametrize("setup_script", ["", "echo bring-your-own-env"])
def test_custom_mode_survives_json_round_trip(setup_script):
    # The controller persists EnvironmentConfig as proto JSON; an explicit CUSTOM
    # choice (especially the empty BYO script) must not decay back to DEFAULT.
    env = job_pb2.EnvironmentConfig(setup_mode=job_pb2.SETUP_MODE_CUSTOM, setup_script=setup_script)
    restored = json_format.ParseDict(
        json_format.MessageToDict(env, preserving_proto_field_name=True), job_pb2.EnvironmentConfig()
    )
    assert restored.setup_mode == job_pb2.SETUP_MODE_CUSTOM
    assert resolve_setup_script(restored) == setup_script
