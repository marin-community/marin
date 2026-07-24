# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe.train import _localize_automatic_jaxpp_input_shardings


class _FakeClient:
    platform = "gpu"

    def process_index(self):
        return 0


class _FakeMemory:
    def __init__(self, kind: str, device):
        self.kind = kind
        self.device = device

    def __eq__(self, other):
        if isinstance(other, _FakeMemory):
            return self.kind == other.kind
        return self.kind == other

    def __hash__(self):
        return hash(self.kind)

    def __str__(self):
        return self.kind


class _FakeDevice:
    platform = "gpu"
    device_kind = "fake gpu"
    core_count = 1
    client = _FakeClient()

    def __init__(self, device_id: int, process_index: int):
        self.id = device_id
        self.process_index = process_index
        self._memories = {kind: _FakeMemory(kind, self) for kind in ("device", "pinned_host")}

    def addressable_memories(self):
        return list(self._memories.values())

    def memory(self, kind: str):
        return self._memories[kind]

    def default_memory(self):
        return self.memory("device")


@dataclasses.dataclass(frozen=True)
class _InputInfo:
    in_shardings: tuple[NamedSharding, ...]


@dataclasses.dataclass(frozen=True)
class _CompiledFunction:
    in_info: _InputInfo


@dataclasses.dataclass(frozen=True)
class _MpmdMesh:
    jax_mesh: Mesh
    unstack: tuple[Mesh, ...]
    my_mpmd_axis_index: int

    def lowering_mesh(self) -> Mesh:
        return self.unstack[self.my_mpmd_axis_index]


def test_automatic_jaxpp_input_shardings_use_addressable_stage_mesh_without_changing_specs():
    local_device = _FakeDevice(0, process_index=0)
    remote_device = _FakeDevice(1, process_index=1)
    axis_names = ("pipeline", "replica_dcn", "data", "expert", "model")
    global_devices = np.asarray([local_device, remote_device], dtype=object).reshape(2, 1, 1, 1, 1)
    global_mesh = Mesh(global_devices, axis_names)
    stage_meshes = (
        Mesh(np.asarray([local_device], dtype=object).reshape(1, 1, 1, 1, 1), axis_names),
        Mesh(np.asarray([remote_device], dtype=object).reshape(1, 1, 1, 1, 1), axis_names),
    )
    global_sharding = NamedSharding(global_mesh, P("expert", "data", "model")).with_memory_kind("pinned_host")
    compiled = _CompiledFunction(_InputInfo((global_sharding,)))
    mpmd_mesh = _MpmdMesh(global_mesh, stage_meshes, my_mpmd_axis_index=0)

    localized = _localize_automatic_jaxpp_input_shardings(compiled, mpmd_mesh)
    (localized_sharding,) = localized.in_info.in_shardings

    assert not global_sharding.is_fully_addressable
    assert localized_sharding.is_fully_addressable
    assert localized_sharding.mesh == stage_meshes[0]
    assert localized_sharding.spec == global_sharding.spec
    assert localized_sharding.memory_kind == global_sharding.memory_kind
    assert compiled.in_info.in_shardings == (global_sharding,)


def test_automatic_jaxpp_input_shardings_do_not_change_single_process_compilation():
    device = _FakeDevice(0, process_index=0)
    mesh = Mesh(np.asarray([device], dtype=object), ("pipeline",))
    compiled = _CompiledFunction(_InputInfo((NamedSharding(mesh, P()),)))
    mpmd_mesh = _MpmdMesh(mesh, (mesh,), my_mpmd_axis_index=0)

    localized = _localize_automatic_jaxpp_input_shardings(compiled, mpmd_mesh)

    assert localized is compiled
