# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Factory admission routes software and capabilities without acquiring machines."""

from dataclasses import dataclass

import pytest
from shellbox.backends.daytona.support import check_environment as check_daytona_environment
from shellbox.backends.local.support import check_local_environment
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.context import MachineRoute, ShellboxContext, declared_spec
from shellbox.machine import (
    Backend,
    CommandSemantics,
    NetworkPolicy,
    PackageSpec,
    UnsupportedMachineSpec,
    task_requirements,
)

PINNED_IMAGE = "ghcr.io/example/task@sha256:" + "0" * 64
UNPINNED_IMAGE = "ghcr.io/example/task:latest"

SIMULATOR = CommandSemantics.SHELL_SIMULATOR
LINUX = CommandSemantics.LINUX_PROCESS


@dataclass(frozen=True)
class Requirements:
    command_semantics: CommandSemantics
    capabilities: tuple[str, ...] = ()
    docker_image: str | None = None
    packages_lock: str | None = None
    package_spec: PackageSpec | None = None
    network: NetworkPolicy = NetworkPolicy.DENY
    working_directory: str | None = None
    workspace: str = "/workspace"

    requirements = property(task_requirements)


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements, spec):
        check_local_environment(requirements, spec)

    async def create(self, spec):
        raise AssertionError("Offline admission must not create a machine")


class DaytonaAdmissionFactory:
    backend = Backend.DAYTONA

    def check_environment(self, requirements, spec):
        check_daytona_environment(requirements, spec)

    async def create(self, spec):
        raise AssertionError("Offline admission must not create a machine")


@pytest.mark.parametrize(
    "requirements, expected",
    [
        (
            Requirements(command_semantics=SIMULATOR, capabilities=("shell", "filesystem")),
            (Backend.SHELLSIM, Backend.LOCAL),
        ),
        (
            Requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem"),
                package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
            ),
            (Backend.SHELLSIM, Backend.LOCAL),
        ),
        (
            Requirements(
                command_semantics=LINUX, capabilities=("shell", "filesystem"), packages_lock="s3://bucket/task/uv.lock"
            ),
            (Backend.LOCAL,),
        ),
        (
            Requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem"),
                package_spec=PackageSpec(pypi_packages=("numpy==2.2.3",)),
            ),
            (Backend.LOCAL,),
        ),
        (
            Requirements(command_semantics=LINUX, capabilities=("shell", "filesystem"), docker_image=PINNED_IMAGE),
            (Backend.LOCAL, Backend.DAYTONA),
        ),
        (
            Requirements(command_semantics=LINUX, docker_image=PINNED_IMAGE),
            (Backend.LOCAL, Backend.DAYTONA),
        ),
        (
            Requirements(command_semantics=LINUX, capabilities=("shell", "filesystem"), docker_image=UNPINNED_IMAGE),
            (Backend.DAYTONA,),
        ),
        (
            Requirements(command_semantics=LINUX, capabilities=("shell",)),
            (),
        ),
        (
            Requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem", "process"),
                package_spec=PackageSpec(),
            ),
            (Backend.LOCAL,),
        ),
        (
            Requirements(
                command_semantics=SIMULATOR,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.ALLOW,
                package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
            ),
            (Backend.LOCAL,),
        ),
        (
            Requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.ALLOW,
                package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
            ),
            (Backend.LOCAL,),
        ),
        (
            Requirements(
                command_semantics=LINUX,
                capabilities=("shell", "filesystem", "network"),
                network=NetworkPolicy.DENY,
                docker_image=PINNED_IMAGE,
            ),
            (),
        ),
        (
            Requirements(
                command_semantics=LINUX,
                capabilities=("shell", "nemotron-agent:swe_agent"),
                docker_image=PINNED_IMAGE,
            ),
            (),
        ),
    ],
)
def test_declared_requirements_route_to_the_satisfying_tiers(requirements: Requirements, expected: tuple):
    factories = (ShellSimMachineFactory(), LocalAdmissionFactory(), DaytonaAdmissionFactory())
    try:
        spec = declared_spec(requirements, requirements.network)
        routes = ShellboxContext(tuple(MachineRoute(factory, spec) for factory in factories)).routes(
            requirements.requirements
        )
    except UnsupportedMachineSpec:
        routes = ()
    assert tuple(route.backend for route in routes) == expected


@pytest.mark.parametrize(
    "requirements, expected",
    [
        (Requirements(LINUX, docker_image=PINNED_IMAGE), ""),
        (Requirements(LINUX, docker_image=PINNED_IMAGE, working_directory="/app"), "/app"),
        (Requirements(LINUX, packages_lock="s3://bucket/lock", workspace="/app"), "/app"),
        (Requirements(LINUX, packages_lock="s3://bucket/lock"), "/workspace"),
        (Requirements(LINUX, package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",))), "/workspace"),
    ],
    ids=["image_default", "image_declared", "lock_workspace", "lock_default", "packages_default"],
)
def test_declared_spec_binds_the_working_directory_once(requirements: Requirements, expected: str):
    spec = declared_spec(requirements, requirements.network, workspace=requirements.workspace)
    assert spec.workdir == expected
