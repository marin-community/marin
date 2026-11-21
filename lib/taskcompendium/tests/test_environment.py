# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Environment values passed to task and verifier machines."""

import pytest
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.image import RegistryImage
from shellbox.machine import Backend, CommandSemantics, MachineSpec

from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.environment import prepare_machine_spec, resolve_env_vars


def test_task_environment_resolves_host_values_and_defaults() -> None:
    host_environment = {"TASK_TOKEN": "host-token"}
    assert resolve_env_vars(
        {
            "TOKEN": "${TASK_TOKEN}",
            "REGION": "${TASK_REGION:-us-east-1}",
            "LITERAL": "prefix-${TASK_TOKEN}",
        },
        host_environment,
    ) == {"TOKEN": "host-token", "REGION": "us-east-1", "LITERAL": "prefix-${TASK_TOKEN}"}

    with pytest.raises(ValueError, match="TASK_REGION"):
        resolve_env_vars({"REGION": "${TASK_REGION}"}, host_environment)


def test_local_preparation_preserves_memory_for_capacity_admission() -> None:
    image = RegistryImage("fixture@sha256:" + "a" * 64)
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image=image.reference
    )
    selected = MachineSpec(image, cpus=1, memory_mb=8192)
    factory = DockerMachineFactory()
    factory.backend = Backend.LOCAL

    prepared = prepare_machine_spec(requirements, factory, selected, {})

    assert prepared.memory_mb == selected.memory_mb
    assert prepared.cpus == selected.cpus
