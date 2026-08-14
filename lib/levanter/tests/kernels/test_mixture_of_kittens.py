# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import draccus
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from levanter.kernels.mixture_of_kittens import (
    MokLikeBackwardPeerStorage,
    MokLikeBuildConfig,
    MokLikeConfig,
    MokLikeForwardXStorage,
    MokLikeWorkspaceTransport,
    initialize_mok_like_runtime,
    mok_like_reference,
    mok_like_preflight_status,
    mok_like_runtime_initialized,
    validate_mok_like_expert_groups,
    validate_mok_like_inputs,
)
from levanter.kernels.mixture_of_kittens import availability, build as mok_build, runtime
from levanter.kernels.mixture_of_kittens.api import _failure_agreement_axes, _failure_outputs
from levanter.kernels.mixture_of_kittens.collective_memory_probe import (
    collective_memory_ring_u32,
    memory_space_frontend_attributes,
)
from levanter.kernels.mixture_of_kittens.ffi import MokLikeForwardContext, backward_bf16_local, fence_mok_like_failure
from levanter.kernels.mixture_of_kittens.runtime import MokLikeRuntimeHandle
from levanter.kernels.mixture_of_kittens.schedule import build_schedule, schedule_capacity


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _reference_schedule(
    top_experts: np.ndarray,
    *,
    num_local_experts: int,
    rank: int,
    expert_padding: int,
    capacity: int,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    world_size, _, top_k = top_experts.shape
    first_expert = rank * num_local_experts
    peer_rank = np.full(capacity, -1, dtype=np.int32)
    peer_token_idx = np.full(capacity, -1, dtype=np.int32)
    tokens_per_expert = np.zeros(num_local_experts, dtype=np.int32)
    output_offset = 0

    for local_expert in range(num_local_experts):
        global_expert = first_expert + local_expert
        peer_assignments: list[list[int]] = []
        for peer in range(world_size):
            assignments = [
                token * top_k + route
                for token in range(top_experts.shape[1])
                for route in range(top_k)
                if top_experts[peer, token, route] == global_expert
            ]
            peer_assignments.append(assignments)

        real_count = sum(len(assignments) for assignments in peer_assignments)
        padded_count = _round_up(real_count, expert_padding)
        tokens_per_expert[local_expert] = padded_count
        for ordinal in range(max((len(assignments) for assignments in peer_assignments), default=0)):
            for peer, assignments in enumerate(peer_assignments):
                if ordinal >= len(assignments):
                    continue
                destination = output_offset + sum(min(len(other), ordinal) for other in peer_assignments)
                destination += sum(len(peer_assignments[earlier]) > ordinal for earlier in range(peer))
                if destination < capacity:
                    peer_rank[destination] = peer
                    peer_token_idx[destination] = assignments[ordinal]
        output_offset += padded_count

    return peer_rank, peer_token_idx, output_offset, tokens_per_expert


@pytest.mark.parametrize(
    ("top_experts", "rank"),
    [
        (
            [
                [[0, 3], [1, 2], [0, 4]],
                [[1, 4], [0, 3], [1, 5]],
                [[0, 2], [0, 5], [1, 4]],
            ],
            0,
        ),
        (
            [
                [[0, 2], [3, 5], [2, 4]],
                [[2, 5], [4, 3], [2, 0]],
                [[5, 1], [3, 0], [4, 2]],
            ],
            1,
        ),
    ],
)
def test_schedule_matches_peer_interleaving_and_padding(
    top_experts: Sequence[Sequence[Sequence[int]]],
    rank: int,
) -> None:
    routes = np.asarray(top_experts, dtype=np.int32)
    expected = _reference_schedule(
        routes,
        num_local_experts=2,
        rank=rank,
        expert_padding=4,
        capacity=24,
    )

    actual = jax.jit(
        lambda values, destination: build_schedule(
            values,
            num_local_experts=2,
            schedule_capacity=24,
            rank=destination,
            expert_padding=4,
        )
    )(jnp.asarray(routes), jnp.asarray(rank, dtype=jnp.int32))

    np.testing.assert_array_equal(actual.peer_rank, expected[0])
    np.testing.assert_array_equal(actual.peer_token_idx, expected[1])
    assert int(actual.num_tokens) == expected[2]
    np.testing.assert_array_equal(actual.tokens_per_expert, expected[3])
    assert int(actual.dropped_assignments) == 0
    assert not bool(actual.overflow)


def test_schedule_preserves_zero_token_expert_segment() -> None:
    routes = jnp.zeros((4, 3, 1), dtype=jnp.int32)

    schedule = build_schedule(
        routes,
        num_local_experts=2,
        schedule_capacity=20,
        rank=jnp.asarray(0, dtype=jnp.int32),
        expert_padding=4,
    )

    np.testing.assert_array_equal(schedule.tokens_per_expert, np.array([12, 0], dtype=np.int32))
    np.testing.assert_array_equal(schedule.peer_rank[12:], np.full(8, -1, dtype=np.int32))
    assert int(schedule.dropped_assignments) == 0


def test_schedule_reports_skewed_capacity_overflow() -> None:
    routes = jnp.zeros((2, 3, 1), dtype=jnp.int32)

    schedule = build_schedule(
        routes,
        num_local_experts=1,
        schedule_capacity=4,
        rank=jnp.asarray(0, dtype=jnp.int32),
        expert_padding=4,
    )

    assert int(schedule.num_tokens) == 4
    np.testing.assert_array_equal(schedule.tokens_per_expert, np.array([4], dtype=np.int32))
    assert int(schedule.dropped_assignments) == 2
    assert bool(schedule.overflow)


def test_schedule_capacity_includes_headroom_and_per_expert_padding() -> None:
    config = MokLikeConfig()
    capacity = schedule_capacity(65536, 4, 2, config)

    assert capacity >= 65536 * 4 * config.schedule_capacity_factor + 2 * 255
    assert capacity % config.minibatch_size == 0


def test_collective_memory_probe_preserves_shard_shape_and_dtype_during_abstract_evaluation() -> None:
    shard = jax.ShapeDtypeStruct((1024,), jnp.uint32)

    remote_read, remote_written = jax.eval_shape(collective_memory_ring_u32, shard)

    assert remote_read == shard
    assert remote_written == shard


@pytest.mark.parametrize(
    ("memory_space", "expected"),
    [
        (0, {"operands_memory_spaces": "{0:0}", "results_memory_spaces": "{0:0,1:0}"}),
        (1, {"operands_memory_spaces": "{0:1}", "results_memory_spaces": "{0:1,1:1}"}),
        (99, {"operands_memory_spaces": "{0:99}", "results_memory_spaces": "{0:99,1:99}"}),
    ],
)
def test_collective_memory_probe_uses_openxla_memory_space_wire_format(
    memory_space: int, expected: dict[str, str]
) -> None:
    assert memory_space_frontend_attributes(memory_space) == expected


@pytest.mark.parametrize(
    ("shape", "dtype", "error"),
    [
        ((2, 2), jnp.uint32, ValueError),
        ((2,), jnp.int32, TypeError),
        ((0,), jnp.uint32, ValueError),
    ],
)
def test_collective_memory_probe_rejects_unsupported_shards(
    shape: tuple[int, ...], dtype: jnp.dtype, error: type[Exception]
) -> None:
    shard = jnp.zeros(shape, dtype=dtype)
    with pytest.raises(error):
        collective_memory_ring_u32(shard)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_comm_sms": 3},
        {"bwd_num_comm_sms": 1},
        {"minibatch_size": 257},
        {"macrobatch_size": 5000},
        {"schedule_capacity_factor": 0.9},
        {"workspace_slots": 0},
        {"workspace_slots": 3},
        {"workspace_slots": True},
    ],
)
def test_config_rejects_unsupported_native_launch_shapes(kwargs: dict[str, int | float]) -> None:
    with pytest.raises(ValueError):
        MokLikeConfig(**kwargs)


def test_config_decodes_experimental_forward_x_storage_to_native_abi() -> None:
    config = draccus.decode(
        MokLikeConfig,
        {
            "forward_x_storage": "xla_peer_experimental",
            "backward_peer_storage": "xla_peer_experimental",
        },
    )

    assert config.forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL
    assert config.forward_x_storage.native_ffi_code == 1
    assert MokLikeForwardXStorage.RUNTIME_STAGED.native_ffi_code == 0
    assert config.backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL
    assert config.backward_peer_storage.native_ffi_code == 1
    assert MokLikeBackwardPeerStorage.RUNTIME_STAGED.native_ffi_code == 0
    hybrid = draccus.decode(
        MokLikeConfig,
        {"backward_peer_storage": "xla_peer_inputs_experimental"},
    )
    assert hybrid.backward_peer_storage is MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL
    assert hybrid.backward_peer_storage.native_ffi_code == 2


@pytest.mark.parametrize(
    ("field", "storage"),
    [
        ("forward_x_storage", MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL),
        ("backward_peer_storage", MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL),
    ],
)
def test_config_rejects_peer_xla_reads_on_a_cross_process_transport(field: str, storage: object) -> None:
    """A four-rank fabric group runs one process per GPU, so a peer's XLA buffer is unreachable.

    The group still fits on one node, so a check keyed on the device count passes it and the
    megakernel dereferences peer pointers the rendezvous never filled in.
    """
    with pytest.raises(ValueError, match="process-local peer mappings"):
        MokLikeConfig(
            num_devices=4,
            workspace_transport=MokLikeWorkspaceTransport.FABRIC_SYMMETRIC,
            **{field: storage},
        )


def test_config_allows_peer_xla_reads_on_a_process_local_transport() -> None:
    config = MokLikeConfig(
        num_devices=4,
        workspace_transport=MokLikeWorkspaceTransport.IN_PROCESS_PEER,
        forward_x_storage=MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL,
    )

    assert config.forward_x_storage is MokLikeForwardXStorage.XLA_PEER_EXPERIMENTAL


def test_config_rejects_untyped_forward_x_storage() -> None:
    with pytest.raises(TypeError, match="MokLikeForwardXStorage"):
        MokLikeConfig(forward_x_storage="xla_peer_experimental")  # type: ignore[arg-type]


def test_config_rejects_untyped_backward_peer_storage() -> None:
    with pytest.raises(TypeError, match="MokLikeBackwardPeerStorage"):
        MokLikeConfig(backward_peer_storage="xla_peer_experimental")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("storage", "expected_code"),
    [
        (MokLikeBackwardPeerStorage.RUNTIME_STAGED, 0),
        (MokLikeBackwardPeerStorage.XLA_PEER_EXPERIMENTAL, 1),
        (MokLikeBackwardPeerStorage.XLA_PEER_INPUTS_EXPERIMENTAL, 2),
    ],
)
def test_backward_peer_storage_reaches_typed_ffi_abi(storage: MokLikeBackwardPeerStorage, expected_code: int) -> None:
    class CompatibleRuntime:
        def require_compatible(self, *, num_tokens: int, hidden_dim: int, top_k: int, workspace_slots: int) -> None:
            del num_tokens, hidden_dim, top_k, workspace_slots

    def shaped(shape: tuple[int, ...], dtype: jnp.dtype) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct(shape, dtype)

    def bf16(shape: tuple[int, ...]) -> jax.ShapeDtypeStruct:
        return shaped(shape, jnp.bfloat16)

    def f32(shape: tuple[int, ...]) -> jax.ShapeDtypeStruct:
        return shaped(shape, jnp.float32)

    def i32(shape: tuple[int, ...]) -> jax.ShapeDtypeStruct:
        return shaped(shape, jnp.int32)

    context = MokLikeForwardContext(
        x_routed=bf16((256, 256)),
        gate_shared=bf16((256, 512)),
        gate_routed=bf16((256, 512)),
        up_shared=bf16((256, 512)),
        up_routed=bf16((256, 512)),
        hidden_shared=bf16((256, 512)),
        hidden_routed=bf16((256, 512)),
        stamp_slot=i32((1,)),
        stamp_generation_high=i32((1,)),
        stamp_generation_low=i32((1,)),
        stamp_runtime_epoch=i32((1,)),
    )
    arguments = (
        bf16((256, 256)),
        bf16((256, 256)),
        f32((256, 2)),
        bf16((512, 256)),
        bf16((2, 512, 256)),
        bf16((512, 256)),
        bf16((2, 512, 256)),
        bf16((256, 512)),
        bf16((2, 256, 512)),
        context,
        i32((1024,)),
        i32((1024,)),
        i32(()),
        i32((2,)),
    )
    config = MokLikeConfig(
        minibatch_size=256,
        macrobatch_size=256,
        backward_peer_storage=storage,
    )

    def traced_backward_with_id(collective_id: int, *values):
        return backward_bf16_local(
            *values,
            runtime=CompatibleRuntime(),
            config=config,
            collective_id=collective_id,
        )

    def traced_backward(*values):
        return traced_backward_with_id(7, *values)

    jaxpr = jax.make_jaxpr(traced_backward)(*arguments).jaxpr
    ffi_equation = next(equation for equation in jaxpr.eqns if equation.primitive.name == "ffi_call")
    attributes = dict(ffi_equation.params["attributes"])

    assert not jaxpr.effects
    assert ffi_equation.params["has_side_effect"] is False
    assert attributes["backward_peer_storage"] == np.int32(expected_code)
    assert attributes["collective_id"] == np.int64(7)

    routed_gradient_avals = tuple(variable.aval for variable in ffi_equation.outvars[2:5])
    assert tuple(aval.shape for aval in routed_gradient_avals) == (
        (2, 512, 256),
        (2, 512, 256),
        (2, 256, 512),
    )
    assert all(aval.dtype == jnp.float32 for aval in routed_gradient_avals)
    assert ffi_equation.outvars[-1].aval.shape == (1,)
    assert ffi_equation.outvars[-1].aval.dtype == jnp.int32
    assert all(equation.primitive.name != "reduce_sum" for equation in jaxpr.eqns)

    def failure_status_only(*values: jax.Array) -> jax.Array:
        return traced_backward(*values)[1]

    lowered = jax.jit(failure_status_only).lower(*arguments).as_text()
    assert "@levanter_mok_backward_bf16_4" in lowered
    assert "has_side_effect = true" not in lowered

    def distinct_call_sites(*values: jax.Array) -> jax.Array:
        first_status = traced_backward_with_id(10, *values)[1]
        second_status = traced_backward_with_id(11, *values)[1]
        return first_status + second_status

    distinct_lowering = jax.jit(distinct_call_sites).lower(*arguments).as_text()
    assert distinct_lowering.count("@levanter_mok_backward_bf16_4") == 2


def _canonical_shapes(dtype: jnp.dtype = jnp.bfloat16) -> tuple[jax.ShapeDtypeStruct, ...]:
    return (
        jax.ShapeDtypeStruct((8, 256), dtype),
        jax.ShapeDtypeStruct((8, 2), jnp.int32),
        jax.ShapeDtypeStruct((8, 2), jnp.float32),
        jax.ShapeDtypeStruct((4, 256, 256), dtype),
        jax.ShapeDtypeStruct((4, 256, 256), dtype),
        jax.ShapeDtypeStruct((4, 256, 256), dtype),
        jax.ShapeDtypeStruct((256, 256), dtype),
        jax.ShapeDtypeStruct((256, 256), dtype),
        jax.ShapeDtypeStruct((256, 256), dtype),
    )


def test_input_validation_rejects_noncanonical_weight_dtype() -> None:
    arguments = list(_canonical_shapes())
    arguments[3] = jax.ShapeDtypeStruct(arguments[3].shape, jnp.float32)

    with pytest.raises(TypeError, match="must be bfloat16"):
        validate_mok_like_inputs(*arguments)


def test_preflight_is_read_only_for_a_missing_explicit_source(tmp_path: Path) -> None:
    source_root = tmp_path / "does-not-exist"
    config = MokLikeBuildConfig(
        source_root=str(source_root),
        cache_root=str(tmp_path / "cache"),
        cuda_arch="sm_100a",
        clone_if_missing=True,
    )

    status = mok_like_preflight_status(config)

    assert not status.ok
    assert any("does not exist" in error for error in status.errors)
    assert not source_root.exists()
    assert not (tmp_path / "cache").exists()


def test_preflight_rejects_non_cuda_13_build_packages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(availability.importlib.metadata, "version", lambda _: "12.9.0")
    config = MokLikeBuildConfig(
        source_root=str(tmp_path / "missing-source"),
        cache_root=str(tmp_path / "cache"),
        cuda_arch="sm_100a",
    )

    status = mok_like_preflight_status(config)

    assert len([error for error in status.errors if "requires CUDA 13" in error]) == 5


def _fake_mok_like_devices(num_processes: int) -> np.ndarray:
    return np.asarray(
        [
            SimpleNamespace(
                id=process * 4 + local_id,
                process_index=process,
                local_hardware_id=None,
                platform="gpu",
            )
            for process in range(num_processes)
            for local_id in range(4)
        ],
        dtype=object,
    ).reshape(num_processes, 1, 4, 1)


def test_mok_like_expert_groups_are_process_local_on_a_multi_process_mesh() -> None:
    devices = _fake_mok_like_devices(2)

    validate_mok_like_expert_groups(
        devices,
        ("replica_dcn", "data", "expert", "model"),
        world_devices=tuple(devices.flat),
    )


@pytest.mark.parametrize("failure", ["cross_process", "device_id_order"])
def test_mok_like_expert_groups_reject_nonlocal_native_rank_layouts(failure: str) -> None:
    devices = _fake_mok_like_devices(2)
    world_devices = tuple(devices.flat)
    if failure == "cross_process":
        devices[0, 0, 3, 0], devices[1, 0, 0, 0] = devices[1, 0, 0, 0], devices[0, 0, 3, 0]
        message = "crosses JAX processes"
    else:
        devices[0, 0, 0, 0], devices[0, 0, 1, 0] = devices[0, 0, 1, 0], devices[0, 0, 0, 0]
        message = "process-local device order"

    with pytest.raises(ValueError, match=message):
        validate_mok_like_expert_groups(
            devices,
            ("replica_dcn", "data", "expert", "model"),
            world_devices=world_devices,
        )


def test_runtime_handle_allows_process_local_multi_process_world_and_owns_teardown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    build = MokLikeBuildConfig(
        source_root=str(tmp_path / "source"),
        cache_root=str(tmp_path / "cache"),
        cuda_arch="sm_100a",
    )
    library_path = tmp_path / "libmok.so"
    fake_library = object()
    fake_mesh = object()
    lifecycle: list[str] = []
    native_signatures: list[tuple[int, int, int, int, int]] = []
    topology_checks: list[object] = []
    monkeypatch.setattr(runtime, "mok_source_root", lambda _: tmp_path)
    monkeypatch.setattr(runtime, "require_mok_like_available", lambda _: None)
    monkeypatch.setattr(runtime.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        runtime.jax,
        "local_devices",
        lambda: tuple(SimpleNamespace(platform="gpu") for _ in range(4)),
    )
    monkeypatch.setattr(runtime, "validate_mok_like_mesh_topology", topology_checks.append)
    monkeypatch.setattr(runtime, "load_native_library", lambda _: (object(), fake_library, library_path))
    # The handler symbols are rank-suffixed, so registration has to name the count the library was
    # built for; recording it keeps a default from silently registering the wrong ones.
    registered_rank_counts: list[int] = []
    monkeypatch.setattr(
        runtime,
        "register_ffi_targets",
        lambda _library, num_devices: registered_rank_counts.append(num_devices),
    )
    monkeypatch.setattr(
        runtime,
        "initialize_native_runtime",
        lambda _library, signature: (native_signatures.append(signature), lifecycle.append("initialize")),
    )
    monkeypatch.setattr(runtime, "shutdown_native_runtime", lambda *_: lifecycle.append("shutdown"))
    monkeypatch.setattr(runtime, "_REGISTERED_LIBRARY_PATH", None)

    handle = initialize_mok_like_runtime(
        build_config=build,
        num_tokens=256,
        hidden_dim=256,
        top_k=2,
        workspace_slots=1,
        mesh=fake_mesh,  # type: ignore[arg-type]
    )
    assert mok_like_runtime_initialized(handle)
    handle.require_compatible(num_tokens=256, hidden_dim=256, top_k=2, workspace_slots=1)
    with pytest.raises(RuntimeError, match="requested"):
        handle.require_compatible(num_tokens=256, hidden_dim=256, top_k=2, workspace_slots=2)

    with pytest.raises(RuntimeError, match="setup failed"):
        with handle:
            raise RuntimeError("setup failed")
    handle.close()
    assert topology_checks == [fake_mesh]
    assert native_signatures == [(4, 256, 256, 2, 1)]
    assert registered_rank_counts == [4]
    assert lifecycle == ["initialize", "shutdown"]
    assert not mok_like_runtime_initialized(handle)


def test_native_runtime_init_uses_workspace_slot_abi_field() -> None:
    class FakeFunction:
        argtypes = None
        restype = None

        def __init__(self) -> None:
            self.calls: list[tuple[int, ...]] = []

        def __call__(self, *args: int) -> int:
            self.calls.append(args)
            return 0

    init = FakeFunction()
    library = SimpleNamespace(levanter_mok_init_runtime=init)

    runtime.initialize_native_runtime(library, (4, 65_536, 6_144, 4, 1))  # type: ignore[arg-type]

    assert init.calls == [(4, 65_536, 6_144, 4, 1)]
    assert init.argtypes == [runtime.ctypes.c_int] * 5


def test_runtime_call_counters_are_explicitly_resettable(tmp_path: Path) -> None:
    class FakeFunction:
        def __init__(self, result: int | None = None):
            self.result = result
            self.argtypes = None
            self.restype = None
            self.calls = 0

        def __call__(self) -> int | None:
            self.calls += 1
            return self.result

    reset = FakeFunction()
    forward = FakeFunction(4)
    backward = FakeFunction(4)
    fake_library = type(
        "FakeLibrary",
        (),
        {
            "levanter_mok_reset_call_counts": reset,
            "levanter_mok_forward_call_count": forward,
            "levanter_mok_backward_call_count": backward,
        },
    )()
    build = MokLikeBuildConfig(
        source_root=str(tmp_path / "source"),
        cache_root=str(tmp_path / "cache"),
        cuda_arch="sm_100a",
    )
    handle = MokLikeRuntimeHandle(
        build_config=build,
        signature=(4, 256, 256, 2, 2),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=fake_library,
    )

    handle.reset_call_counts()

    assert reset.calls == 1
    assert handle.call_counts() == (4, 4)


def test_runtime_failure_gate_uses_rank_phase_point_and_concurrency_abi(tmp_path: Path) -> None:
    class FakeFunction:
        argtypes = None
        restype = None

        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int]] = []

        def __call__(self, rank: int, phase: int, point: int, concurrent: int) -> int:
            self.calls.append((rank, phase, point, concurrent))
            return 0

    arm_failure = FakeFunction()
    handle = MokLikeRuntimeHandle(
        build_config=MokLikeBuildConfig(
            source_root=str(tmp_path / "source"),
            cache_root=str(tmp_path / "cache"),
            cuda_arch="sm_100a",
        ),
        signature=(4, 256, 256, 2, 2),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=SimpleNamespace(levanter_mok_arm_test_failure=arm_failure),  # type: ignore[arg-type]
    )

    handle.arm_test_failure(
        rank=2,
        phase=runtime.MokLikeTestFailurePhase.BACKWARD,
        point=runtime.MokLikeTestFailurePoint.BEFORE_COMPLETION,
        require_two_active_slots=True,
    )

    assert arm_failure.calls == [(2, 1, 1, 1)]
    assert arm_failure.argtypes == [runtime.ctypes.c_int] * 4


def test_reference_caps_ring_capacity_to_assignment_population() -> None:
    mesh = jax.sharding.AbstractMesh(
        (1, 1, 4, 1),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(jax.sharding.AxisType.Explicit,) * 4,
    )
    x = jax.ShapeDtypeStruct((2048, 256), jnp.bfloat16)
    selected_experts = jax.ShapeDtypeStruct((2048, 4), jnp.int32)
    combine_weights = jax.ShapeDtypeStruct((2048, 4), jnp.float32)
    routed_weight = jax.ShapeDtypeStruct((8, 256, 256), jnp.bfloat16)
    shared_weight = jax.ShapeDtypeStruct((256, 256), jnp.bfloat16)

    output = jax.eval_shape(
        lambda *arguments: mok_like_reference(
            *arguments,
            mesh=mesh,
            config=MokLikeConfig(schedule_capacity_factor=4, workspace_slots=2),
            fallback_implementation="ring",
        ),
        x,
        selected_experts,
        combine_weights,
        routed_weight,
        routed_weight,
        routed_weight,
        shared_weight,
        shared_weight,
        shared_weight,
    )

    assert output.shape == x.shape


def test_generated_peer_wait_validator_requires_cancellation_on_every_wait() -> None:
    source = """
system_generation_wait(const uint64_t *counter, const uint64_t *cancellation, uint64_t target) {}
system_generation_wait(peer_ready, cancellation, target);
system_generation_wait(peer_ready, target);
"""

    with pytest.raises(RuntimeError, match="cannot observe cancellation"):
        mok_build._validate_and_count_cancellable_generation_waits(source)


def test_generated_peer_wait_validator_counts_cancellable_waits() -> None:
    source = """
system_generation_wait(const uint64_t *counter, const uint64_t *cancellation, uint64_t target) {}
system_generation_wait(peer_ready, cancellation, target);
system_generation_wait(peer_ready, cancellation, target);
"""

    assert mok_build._validate_and_count_cancellable_generation_waits(source) == 2


def test_failure_agreement_excludes_only_the_process_local_expert_axis() -> None:
    mesh = jax.sharding.AbstractMesh(
        (8, 4, 4, 2),
        ("replica_dcn", "data", "expert", "model"),
    )

    agreement_axes = _failure_agreement_axes(mesh)
    jaxpr = jax.make_jaxpr(
        lambda status: jax.lax.pmax(status, agreement_axes),
        axis_env=tuple(zip(mesh.axis_names, mesh.axis_sizes, strict=True)),
    )(jnp.asarray(0, dtype=jnp.int32))

    assert agreement_axes == ("replica_dcn", "data", "model")
    pmax_equation = next(equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "pmax")
    assert pmax_equation.params["axes"] == agreement_axes


def test_failure_fence_is_data_returning_without_a_side_effect_token() -> None:
    marker = jnp.asarray(1, dtype=jnp.int32)
    jaxpr = jax.make_jaxpr(fence_mok_like_failure)(marker)

    fence_equation = next(equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "ffi_call")
    assert fence_equation.params["has_side_effect"] is False
    assert tuple(variable.aval.shape for variable in fence_equation.invars) == ((),)
    assert tuple(variable.aval.shape for variable in fence_equation.outvars) == ((),)
    assert jaxpr.jaxpr.outvars == fence_equation.outvars


def test_failure_marker_remains_in_lowered_branch_output_dataflow() -> None:
    marker = jnp.asarray(1, dtype=jnp.int32)
    output = jnp.zeros((2, 3), dtype=jnp.bfloat16)
    auxiliary = jnp.zeros((), dtype=jnp.int32)
    jaxpr = jax.make_jaxpr(_failure_outputs)(marker, (output, auxiliary))
    fence_equation = next(equation for equation in jaxpr.jaxpr.eqns if equation.primitive.name == "ffi_call")
    fence_output = fence_equation.outvars[0]

    marker_consumers = tuple(
        equation for equation in jaxpr.jaxpr.eqns if equation is not fence_equation and fence_output in equation.invars
    )
    assert marker_consumers
    assert fence_output in jaxpr.jaxpr.outvars

    def branch(status: jax.Array, value: jax.Array, extra: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.lax.cond(
            status == 0,
            lambda _: (value, extra),
            lambda _: _failure_outputs(status, (value, extra)),
            operand=None,
        )

    lowered = jax.jit(branch).lower(marker, output, auxiliary).as_text()
    failure_branch = lowered.split('"stablehlo.case"', maxsplit=1)[1].split("}, {", maxsplit=1)[0]
    assert "@levanter_mok_failure_fence" in lowered
    assert "stablehlo.return" in failure_branch
    assert "has_side_effect = true" not in lowered


def test_runtime_debug_counters_preserve_rank_and_phase_structure(tmp_path: Path) -> None:
    class FakeFunction:
        def __init__(self, function):
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    values = list(range(4 * 59))

    def read(output, count):
        assert count == len(values)
        for index, value in enumerate(values):
            output[index] = value
        return 0

    reset = FakeFunction(lambda: 0)
    fake_library = type(
        "FakeLibrary",
        (),
        {
            "levanter_mok_debug_counter_count": FakeFunction(lambda: len(values)),
            "levanter_mok_reset_debug_counters": reset,
            "levanter_mok_read_debug_counters": FakeFunction(read),
        },
    )()
    build = MokLikeBuildConfig(
        source_root=str(tmp_path / "source"),
        cache_root=str(tmp_path / "cache"),
        cuda_arch="sm_100a",
    )
    handle = MokLikeRuntimeHandle(
        build_config=build,
        signature=(4, 256, 256, 2, 2),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=fake_library,
    )

    handle.reset_debug_counters()
    counters = handle.debug_counters()

    assert counters.peer_ready_waits == (0, 59, 118, 177)
    assert counters.completion_waits == (1, 60, 119, 178)
    assert counters.generation_mismatches == (2, 61, 120, 179)
    assert counters.slot_reuse_failures == (3, 62, 121, 180)
    assert counters.slot_acquisitions == ((4, 5), (63, 64), (122, 123), (181, 182))
    assert counters.max_active_slots == (6, 65, 124, 183)
    assert counters.peer_wait_events[0] == (
        (7, 8, 9, 10),
        (11, 12, 13, 14),
        (15, 16, 17, 18),
        (19, 20, 21, 22),
    )
    assert counters.peer_wait_cycles[1][2] == (90, 91, 92, 93)
    assert counters.peer_wait_max_cycles[3][3] == (228, 229, 230, 231)
    assert counters.staging_copy_calls == ((55, 57), (114, 116), (173, 175), (232, 234))
    assert counters.staging_copy_bytes == ((56, 58), (115, 117), (174, 176), (233, 235))


def test_runtime_default_pool_trim_preserves_rank_telemetry_and_native_duration(tmp_path: Path) -> None:
    class FakeFunction:
        def __init__(self, function):
            self.function = function
            self.argtypes = None
            self.restype = None

        def __call__(self, *args):
            return self.function(*args)

    values = [
        value
        for rank in range(4)
        for value in (
            1000 + rank,
            700 + rank,
            800 + rank,
            700 + rank,
            10_000 + rank,
            20_000 + rank,
            11_000 + rank,
            20_000 + rank,
            500 + rank,
            400 + rank,
        )
    ]
    values.extend((0, 0, 125_000_000))

    def trim(output, count):
        assert count == len(values)
        for index, value in enumerate(values):
            output[index] = value
        return 0

    fake_library = SimpleNamespace(levanter_mok_trim_default_memory_pools=FakeFunction(trim))
    handle = MokLikeRuntimeHandle(
        build_config=MokLikeBuildConfig(
            source_root=str(tmp_path / "source"),
            cache_root=str(tmp_path / "cache"),
            cuda_arch="sm_100a",
        ),
        signature=(4, 256, 256, 2, 1),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=fake_library,  # type: ignore[arg-type]
    )

    telemetry = handle.trim_default_memory_pools()

    assert telemetry.active_reservations == 0
    assert telemetry.active_workspace_slots == 0
    assert telemetry.wall_time_seconds == 0.125
    assert telemetry.ranks[2].rank == 2
    assert telemetry.ranks[2].reserved_bytes_before == 1002
    assert telemetry.ranks[2].used_bytes_before == 702
    assert telemetry.ranks[2].reserved_bytes_after == 802
    assert telemetry.ranks[2].used_bytes_after == 702
    assert telemetry.ranks[2].device_free_bytes_before == 10_002
    assert telemetry.ranks[2].device_total_bytes_before == 20_002
    assert telemetry.ranks[2].device_free_bytes_after == 11_002
    assert telemetry.ranks[2].device_total_bytes_after == 20_002
    assert telemetry.ranks[2].graph_reserved_bytes_after == 502
    assert telemetry.ranks[2].graph_used_bytes_after == 402


def test_runtime_default_pool_trim_propagates_native_quiescence_failure(tmp_path: Path) -> None:
    class FakeFunction:
        argtypes = None
        restype = None

        def __init__(self, result):
            self.result = result

        def __call__(self, *args):
            return self.result

    fake_library = SimpleNamespace(
        levanter_mok_trim_default_memory_pools=FakeFunction(1),
        levanter_mok_last_error=FakeFunction(b"default memory-pool trim requires zero active workspace reservations"),
    )
    handle = MokLikeRuntimeHandle(
        build_config=MokLikeBuildConfig(
            source_root=str(tmp_path / "source"),
            cache_root=str(tmp_path / "cache"),
            cuda_arch="sm_100a",
        ),
        signature=(4, 256, 256, 2, 1),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=fake_library,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="zero active workspace reservations"):
        handle.trim_default_memory_pools()
