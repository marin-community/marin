# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any, cast

import pytest

from levanter.kernels.mixture_of_kittens.symmetric_memory import (
    MOK_LIKE_EP64_ARENA_ALIGNMENT,
    MOK_LIKE_EP64_ARENA_OFFSET_FIELDS,
    MOK_LIKE_EP64_ARENA_SCHEMA_VERSION,
    MokLikeSymmetricWorkspace,
    mok_like_symmetric_arena_layout,
)
from levanter.kernels.mixture_of_kittens import runtime
from levanter.kernels.mixture_of_kittens.config import MokLikeTopology
from levanter.kernels.mixture_of_kittens.source import MokLikeBuildConfig


def test_symmetric_arena_layout_covers_ep64_staging_and_stamp_regions() -> None:
    layout = mok_like_symmetric_arena_layout(num_tokens=256, hidden_dim=512, top_k=4)

    assert layout.sizes["x"] == 256 * 512 * 2
    assert layout.sizes["combine"] == 256 * 4 * 512 * 2
    assert layout.sizes["router_weights"] == 256 * 4 * 4
    for field_name in (
        "generation",
        "forward_input_ready",
        "backward_input_ready",
        "forward_completions",
        "backward_completions",
        "last_forward_completion",
        "cancellation",
    ):
        assert layout.sizes[field_name] == 64 * 8
    assert layout.sizes["debug_counters"] == 779 * 8

    for field_name, next_field_name in zip(
        MOK_LIKE_EP64_ARENA_OFFSET_FIELDS[:-1], MOK_LIKE_EP64_ARENA_OFFSET_FIELDS[1:]
    ):
        assert layout.offsets[field_name] % MOK_LIKE_EP64_ARENA_ALIGNMENT == 0
        assert layout.offsets[field_name] + layout.sizes[field_name] <= layout.offsets[next_field_name]
    assert layout.offsets[MOK_LIKE_EP64_ARENA_OFFSET_FIELDS[-1]] % MOK_LIKE_EP64_ARENA_ALIGNMENT == 0
    assert layout.total_bytes % MOK_LIKE_EP64_ARENA_ALIGNMENT == 0
    assert layout.offsets["debug_counters"] + layout.sizes["debug_counters"] <= layout.total_bytes

    assert layout.native_offset_table == (
        MOK_LIKE_EP64_ARENA_SCHEMA_VERSION,
        layout.total_bytes,
        *(layout.offsets[field_name] for field_name in MOK_LIKE_EP64_ARENA_OFFSET_FIELDS),
    )


class _FakeDistributed:
    def __init__(self, events: list[tuple[object, ...]]):
        self.events = events

    def barrier(self, *, group: object) -> None:
        self.events.append(("gloo_barrier", group))

    def destroy_process_group(self, group: object) -> None:
        self.events.append(("destroy_group", group))


def _fake_workspace(events: list[tuple[object, ...]]) -> MokLikeSymmetricWorkspace:
    pointers = tuple(0x100000 + rank * 0x10000 for rank in range(64))
    return MokLikeSymmetricWorkspace(
        rank=32,
        world_size=64,
        local_pointer=pointers[32],
        peer_pointers=pointers,
        layout=mok_like_symmetric_arena_layout(num_tokens=256, hidden_dim=256, top_k=2),
        num_tokens=256,
        hidden_dim=256,
        top_k=2,
        workspace_slots=1,
        backend="CUDA",
        _torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: events.append(("cuda_sync", device)))),
        _distributed=_FakeDistributed(events),
        _device="cuda:0",
        _group="ep64-group",
        _arena=object(),
        _handle=object(),
        _timeout=7.5,
    )


def test_symmetric_workspace_exposes_rank_ordered_native_arguments() -> None:
    workspace = _fake_workspace([])

    arguments = workspace.native_arguments

    assert arguments.rank == 32
    assert arguments.world_size == 64
    assert arguments.workspace_slots == 1
    assert len(arguments.peer_arena_pointers) == 64
    assert arguments.peer_arena_pointers[31:33] == (0x100000 + 31 * 0x10000, 0x100000 + 32 * 0x10000)
    assert arguments.peer_arena_pointers[63] == 0x100000 + 63 * 0x10000
    assert len(arguments.arena_offsets) == 16
    assert arguments.arena_offsets == workspace.layout.native_offset_table


def test_symmetric_workspace_close_uses_collective_reverse_order_once() -> None:
    events: list[tuple[object, ...]] = []
    workspace = _fake_workspace(events)

    workspace.close()
    workspace.close()

    assert events == [
        ("cuda_sync", "cuda:0"),
        ("gloo_barrier", "ep64-group"),
        ("cuda_sync", "cuda:0"),
        ("gloo_barrier", "ep64-group"),
        ("gloo_barrier", "ep64-group"),
        ("destroy_group", "ep64-group"),
    ]
    assert workspace.is_closed


class _FakeNativeFunction:
    argtypes: list[object]
    restype: object

    def __init__(self) -> None:
        self.arguments: tuple[object, ...] | None = None

    def __call__(self, *arguments: object) -> int:
        self.arguments = arguments
        return 0


def test_ep64_native_initialization_preserves_rank_and_pointer_order() -> None:
    workspace = _fake_workspace([])
    function = _FakeNativeFunction()
    library = SimpleNamespace(levanter_mok_init_runtime_ep64=function)

    runtime.initialize_native_runtime_ep64(library, workspace)

    assert function.arguments is not None
    assert len(function.argtypes) == len(function.arguments) == 10
    assert function.arguments[:6] == (32, 64, 256, 256, 2, 1)
    assert function.arguments[7] == 64
    assert function.arguments[9] == 16
    peer_pointers = function.arguments[6]
    arena_offsets = function.arguments[8]
    assert tuple(peer_pointers) == workspace.peer_pointers
    assert tuple(arena_offsets) == workspace.layout.native_offset_table


def test_ep64_debug_counter_layout_reports_one_local_rank_and_all_peers(tmp_path) -> None:
    values = list(range(779))

    def read_debug(output: Any, count: int) -> int:
        assert count == len(values)
        for index, value in enumerate(values):
            output[index] = value
        return 0

    def debug_count() -> int:
        return len(values)

    library = SimpleNamespace(
        levanter_mok_debug_counter_count=debug_count,
        levanter_mok_read_debug_counters=read_debug,
    )
    handle = runtime.MokLikeRuntimeHandle(
        build_config=MokLikeBuildConfig(
            source_root=str(tmp_path / "source"),
            cache_root=str(tmp_path / "cache"),
            cuda_arch="sm_100a",
        ),
        signature=(64, 256, 256, 2, 1),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=object(),
        _library=library,
        expert_parallel_size=64,
        topology=MokLikeTopology.NVLINK_EP64,
    )

    counters = handle.debug_counters()

    assert counters.peer_wait_events[0][0] == tuple(range(7, 71))
    assert counters.peer_wait_cycles[0][0] == tuple(range(263, 327))
    assert counters.peer_wait_max_cycles[0][0] == tuple(range(519, 583))
    assert counters.staging_copy_calls == ((775, 777),)
    assert counters.staging_copy_bytes == ((776, 778),)


class _FakeRollbackWorkspace:
    def __init__(
        self,
        events: list[str],
        *,
        rollback_errors: tuple[str | None, ...],
        quiesce_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.rollback_errors = rollback_errors
        self.quiesce_error = quiesce_error

    def quiesce(self) -> None:
        self.events.append("quiesce")
        if self.quiesce_error is not None:
            raise self.quiesce_error

    def gather_initialization_errors(self, error: BaseException | None) -> tuple[str | None, ...]:
        self.events.append(f"gather:{error}")
        return self.rollback_errors

    def close(self) -> None:
        self.events.append("close")


def test_ep64_mixed_init_success_rolls_back_native_before_symmetric_storage(monkeypatch) -> None:
    events: list[str] = []
    workspace = _FakeRollbackWorkspace(events, rollback_errors=(None,) * 64)
    monkeypatch.setattr(runtime, "shutdown_native_runtime", lambda _library: events.append("shutdown_native"))
    initialization_errors = (None, "RuntimeError: peer init failed", *(None for _ in range(62)))

    with pytest.raises(RuntimeError, match="peer ranks"):
        runtime._rollback_failed_ep64_initialization(
            library=cast(Any, object()),
            workspace=cast(MokLikeSymmetricWorkspace, workspace),
            native_error=None,
            initialization_errors=initialization_errors,
        )

    assert events == ["quiesce", "shutdown_native", "gather:None", "close"]


def test_ep64_init_rollback_preserves_primary_error_and_does_not_release_unsafe_storage(monkeypatch) -> None:
    events: list[str] = []
    primary_error = ValueError("rank 32 native init failed")
    workspace = _FakeRollbackWorkspace(
        events,
        rollback_errors=("RuntimeError: workspace quiesce failed", *(None for _ in range(63))),
        quiesce_error=RuntimeError("fabric barrier timed out"),
    )
    monkeypatch.setattr(runtime, "shutdown_native_runtime", lambda _library: events.append("shutdown_native"))

    with pytest.raises(ValueError, match="rank 32 native init failed") as raised:
        runtime._rollback_failed_ep64_initialization(
            library=cast(Any, object()),
            workspace=cast(MokLikeSymmetricWorkspace, workspace),
            native_error=primary_error,
            initialization_errors=("ValueError: rank 32 native init failed", *(None for _ in range(63))),
        )

    assert raised.value is primary_error
    assert events == ["quiesce", "gather:workspace quiesce failed: RuntimeError: fabric barrier timed out"]
    assert raised.value.__notes__ == ["EP64 initialization rollback was incomplete: " f"{workspace.rollback_errors}"]


def _runtime_handle(tmp_path, *, topology: MokLikeTopology) -> runtime.MokLikeRuntimeHandle:
    return runtime.MokLikeRuntimeHandle(
        build_config=MokLikeBuildConfig(
            source_root=str(tmp_path / "source"),
            cache_root=str(tmp_path / "cache"),
            cuda_arch="sm_100a",
        ),
        signature=(topology.expert_axis_size, 256, 256, 2, 1),
        library_path=tmp_path / "libmok.so",
        _cuda_driver=cast(Any, object()),
        _library=cast(Any, object()),
        expert_parallel_size=topology.expert_axis_size,
        topology=topology,
    )


def test_ep64_context_skips_collective_close_during_primary_failure(monkeypatch, tmp_path) -> None:
    handle = _runtime_handle(tmp_path, topology=MokLikeTopology.NVLINK_EP64)
    close_calls: list[None] = []
    monkeypatch.setattr(handle, "close", lambda: close_calls.append(None))

    with pytest.raises(ValueError, match="training failed"):
        with handle:
            raise ValueError("training failed")

    assert close_calls == []


def test_ep4_context_preserves_primary_failure_when_close_also_fails(monkeypatch, tmp_path) -> None:
    handle = _runtime_handle(tmp_path, topology=MokLikeTopology.LOCAL_EP4)

    def fail_close() -> None:
        raise RuntimeError("native shutdown failed")

    monkeypatch.setattr(handle, "close", fail_close)

    with pytest.raises(ValueError, match="training failed") as raised:
        with handle:
            raise ValueError("training failed")

    assert raised.value.__notes__ == ["MoK-like runtime close failed: RuntimeError: native shutdown failed"]
