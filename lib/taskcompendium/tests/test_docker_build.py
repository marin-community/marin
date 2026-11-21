# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original Docker build contexts survive explicit machine selection."""

import json

import pytest
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

from taskcompendium.models import BuildDirectory, CommandSemantics, DockerBuildContext, EnvironmentRequirements
from taskcompendium.runtime.docker_build import materialize_docker_build, require_original_docker_build
from taskcompendium.runtime.environment import validate_machine_spec
from taskcompendium.runtime.resources import inline_resource


class SelectedFactory:
    def check_environment(self, requirements, spec):
        assert isinstance(spec.source, DockerfileSource)


def test_declared_context_preserves_build_bytes_metadata_and_rejects_drift(tmp_path):
    build = DockerBuildContext(
        files=(
            inline_resource("Dockerfile", b"FROM ubuntu:24.04\nCOPY input /input\n"),
            inline_resource("input", b"\x00original\xff").model_copy(
                update={"mode": "0755", "mtime_ns": 1234567890000000000}
            ),
        )
    )
    source = materialize_docker_build(build, tmp_path / "context")
    environment = EnvironmentRequirements(command_semantics=CommandSemantics.LINUX_PROCESS, docker_build=build)
    factory = SelectedFactory()
    validate_machine_spec(environment, factory, MachineSpec(source))
    assert (source.context / "input").read_bytes() == b"\x00original\xff"
    assert (source.context / "input").stat().st_mode & 0o777 == 0o755
    assert (source.context / "input").stat().st_mtime_ns == 1234567890000000000
    (source.context / "input").write_bytes(b"changed")
    with pytest.raises(UnsupportedMachineSpec, match="input changed"):
        validate_machine_spec(environment, factory, MachineSpec(source))
    with pytest.raises(UnsupportedMachineSpec, match="Unresolved Docker build"):
        validate_machine_spec(environment, factory, MachineSpec(RegistryImage("unrelated:latest")))


def test_explicit_empty_build_directory_survives_materialization_and_detects_loss(tmp_path):
    original = DockerBuildContext(files=(inline_resource("Dockerfile", b"FROM scratch\nCOPY task_file /app\n"),))
    frozen = original.model_dump_json()
    assert "empty_directories" not in json.loads(frozen)
    build = original.model_copy(
        update={"empty_directories": (BuildDirectory(path="task_file", mode="0755", mtime_ns=0),)}
    )
    restored = DockerBuildContext.model_validate_json(build.model_dump_json())
    source = materialize_docker_build(restored, tmp_path / "build")
    require_original_docker_build(restored, source)
    assert (source.context / "task_file").is_dir()
    assert list((source.context / "task_file").iterdir()) == []
    assert (source.context / "Dockerfile").read_bytes() == b"FROM scratch\nCOPY task_file /app\n"
    (source.context / "task_file").rmdir()
    with pytest.raises(UnsupportedMachineSpec, match="directories differ"):
        require_original_docker_build(restored, source)
    assert DockerBuildContext.model_validate_json(frozen).model_dump_json() == frozen
