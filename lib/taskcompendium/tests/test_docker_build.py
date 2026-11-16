# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Original Docker build contexts survive explicit machine selection."""

import pytest
from shellbox.image import DockerfileSource, RegistryImage
from shellbox.machine import MachineSpec, UnsupportedMachineSpec

from taskcompendium.models import CommandSemantics, DockerBuildContext, EnvironmentRequirements
from taskcompendium.runtime.docker_build import materialize_docker_build
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
