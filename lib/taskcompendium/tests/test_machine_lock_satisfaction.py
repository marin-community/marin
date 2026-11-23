# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A lock-declaring grader environment accepts its own composed toolchain image.

The wave resolves a frozen lock-only grader environment to its composed image, which
bakes the environment's pinned packages (and, for the Lean graders, the Lean toolchain
the grader probes for). ``prepare_machine_spec`` must run that machine from the image
instead of demanding a HostImage to mount a lock runtime over.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
from shellbox.machine import (
    Backend,
    CommandSemantics,
    ExecutionRequirements,
    HostImage,
    MachineSpec,
    RegistryImage,
    UnsupportedMachineSpec,
)

from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.environment import prepare_machine_spec

PINNED = "ghcr.io/marin-community/rollout-images@sha256:" + "e" * 64
LOCK = "s3://marin-us-east-02a/marin/images/env-e26fe3f8a6933204/2026.10.08/requirements.lock"


class RecordingFactory:
    """Records the bound spec so the test can assert what the factory would receive."""

    backend: Backend = Backend.LOCAL

    def __init__(self) -> None:
        self.checked: list[MachineSpec] = []

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        self.checked.append(spec)

    async def create(self, spec: MachineSpec):
        raise NotImplementedError("prepare_machine_spec never creates machines")


def lock_environment() -> EnvironmentRequirements:
    return EnvironmentRequirements(packages_lock=LOCK, command_semantics=CommandSemantics.LINUX_PROCESS)


def test_lock_satisfied_machine_runs_from_its_composed_image() -> None:
    factory = RecordingFactory()
    spec = MachineSpec(RegistryImage(PINNED), lock_satisfied_by_image=True)
    prepared = prepare_machine_spec(lock_environment(), factory, spec, {})
    assert prepared.source == RegistryImage(PINNED)
    # No lock runtime mounts or bin dirs over the image.
    assert prepared.source == spec.source
    assert factory.checked and factory.checked[0].source == RegistryImage(PINNED)


def test_unmarked_lock_machine_still_requires_a_host_image(monkeypatch) -> None:
    factory = RecordingFactory()
    spec = MachineSpec(RegistryImage(PINNED))
    with pytest.raises(UnsupportedMachineSpec, match="Package locks require a HostImage source"):
        prepare_machine_spec(lock_environment(), factory, spec, {})


def test_lock_satisfied_flag_requires_a_registry_image() -> None:
    factory = RecordingFactory()
    spec = MachineSpec(HostImage(), lock_satisfied_by_image=True)
    with pytest.raises(UnsupportedMachineSpec, match="A lock-satisfied machine must run from its registry image"):
        prepare_machine_spec(lock_environment(), factory, spec, {})


def test_host_image_lock_machine_keeps_its_runtime_mount(monkeypatch) -> None:
    @dataclass(frozen=True)
    class FakeRuntime:
        root: Path
        bin_dir: Path
        mounts: tuple
        env: dict

        def ensure_built(self) -> None: ...

    runtime = FakeRuntime(root=Path("/tmp/runtime"), bin_dir=Path("/tmp/runtime/bin"), mounts=(), env={"X": "1"})
    monkeypatch.setattr("taskcompendium.runtime.environment.local_runtime", lambda _lock: runtime)
    factory = RecordingFactory()
    spec = MachineSpec(HostImage(read_only=(Path("/srv/data"),)))
    prepared = prepare_machine_spec(lock_environment(), factory, spec, {})
    assert isinstance(prepared.source, HostImage)
    assert Path("/tmp/runtime") in prepared.source.read_only
    assert Path("/tmp/runtime/bin") in prepared.source.bin_dirs
    assert prepared.env["X"] == "1"
