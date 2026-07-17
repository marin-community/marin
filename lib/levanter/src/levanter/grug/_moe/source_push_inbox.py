# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private source-push inbox prototype for MGPU MoE permute_up.

This module contains the current Hopper source-push inbox kernel harness:

    source local GMEM -> source SMEM -> remote destination GMEM inbox

Each source rank sends a deterministic queue of token blocks to the configured destinations.
The destination owns bounded inbox slots, waits on remote-signaled full
semaphores, optionally computes W13 from the local inbox, and releases slots
back to the source.

The implementation is intentionally package-private while the source-push path
is still being stabilized. The benchmark script should be only a CLI wrapper;
new kernel-facing code should depend on this module or on
`source_push_inbox_profiles`, not on `scripts/bench`.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref, lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int
from levanter.grug._moe.common import _prepare_moe_dispatch_indices_with_assignment_ids
from levanter.grug._moe.source_push_plan import (
    SourcePushPlan,
    build_source_push_plan,
    pack_source_push_tokens,
    source_push_plan_row_stats,
    source_push_source_padded_row_bases,
)
from levanter.grug._moe.source_push_inbox_profiles import source_push_profile_defaults


AXIS = "expert"
KERNEL_NAME = "source_push_inbox"
DIAGNOSTIC_KERNEL_NAME = "source_push_inbox_diagnostic"
META_FIELDS = 4
BYTES_PER_BF16 = 2
SLOW_USEFUL_W13_TFLOPS_PER_RANK = 160.0
DIAGNOSTIC_VARIANT_FULL = "full"
DIAGNOSTIC_VARIANT_SEMAPHORE_ONLY = "semaphore_only"
DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY = "copy_release_only"
DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL = "compute_only_local"
DIAGNOSTIC_VARIANT_STORE_ZERO = "store_zero"
DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT = "wgmma_tiny_output"
DIAGNOSTIC_VARIANTS = (
    DIAGNOSTIC_VARIANT_FULL,
    DIAGNOSTIC_VARIANT_SEMAPHORE_ONLY,
    DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY,
    DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL,
    DIAGNOSTIC_VARIANT_STORE_ZERO,
    DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT,
)
DIAGNOSTIC_INPUT_MODE_SYNTHETIC_BLOCKS = "synthetic_blocks"
DIAGNOSTIC_INPUT_MODE_COMPACT_ROUTING = "compact_routing"
DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN = "source_push_plan"
DIAGNOSTIC_INPUT_MODES = (
    DIAGNOSTIC_INPUT_MODE_SYNTHETIC_BLOCKS,
    DIAGNOSTIC_INPUT_MODE_COMPACT_ROUTING,
    DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN,
)
ROW_START_MODE_SOURCE_PADDED = "source_padded_row_start"
ROW_START_MODE_EXACT_EXPERT_MAJOR = "exact_expert_major_row_start"
ROW_LAYOUT_SOURCE_PADDED_EXPERT_MAJOR = "source_padded_expert_major"
ROW_LAYOUT_EXACT_EXPERT_MAJOR = "exact_expert_major"
SOURCE_INPUT_PACKED_QUEUE = "packed_queue"
SOURCE_INPUT_RAW_TOKENS = "raw_tokens"
ROUTING_MODES = (
    "balanced",
    "roughly_balanced",
    "uniform",
    "tail_debug",
    "one_source_one_expert",
    "many_sources_one_expert",
    "many_sources_many_experts",
)
WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8
TINY_OUTPUT_COLUMNS_PER_N_TILE = 8


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _wgmma_transforms(shape: tuple[int, int], dtype: Any):
    swizzle_elems = WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "Lane WGMMA SMEM operands must be divisible by "
            f"tile=({WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return (
        mgpu.TilingTransform((WGMMA_TILE_M, swizzle_elems)),
        mgpu.SwizzleTransform(WGMMA_SWIZZLE_BYTES),
    )


def _wgmma_smem(shape: tuple[int, int], dtype: Any):
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=_wgmma_transforms(shape, dtype),
    )


def _validate_diagnostic_variant(diagnostic_variant: str) -> None:
    if diagnostic_variant not in DIAGNOSTIC_VARIANTS:
        raise ValueError(f"unknown diagnostic_variant={diagnostic_variant!r}; expected one of {DIAGNOSTIC_VARIANTS}")


def _validate_diagnostic_input_mode(input_mode: str) -> None:
    if input_mode not in DIAGNOSTIC_INPUT_MODES:
        raise ValueError(f"unknown diagnostic input_mode={input_mode!r}; expected one of {DIAGNOSTIC_INPUT_MODES}")


@dataclass(frozen=True)
class PushInboxConfig:
    ep_size: int = 8
    entries_per_rank: int = 2
    inbox_slots: int = 2
    hidden_dim: int = 2560
    intermediate_dim: int = 1280
    block_m: int = 64
    block_n: int = 128
    block_k: int = 128
    n_group: int = 1
    experts_per_rank: int = 32
    send_worker_programs_per_peer: int = 4
    worker_programs_per_peer: int = 16
    send_pipeline_depth: int = 1
    n_groups_per_job: int = 1
    routing: str = "balanced"
    tokens_per_rank: int = 32768
    topk: int = 4
    routing_seed: int = 0
    capacity_factor: float = 1.25

    def validate(self) -> None:
        if self.ep_size <= 1:
            raise ValueError(f"ep_size must be greater than 1, got {self.ep_size}")
        if self.entries_per_rank <= 0:
            raise ValueError(f"entries_per_rank must be positive, got {self.entries_per_rank}")
        if self.inbox_slots <= 0:
            raise ValueError(f"inbox_slots must be positive, got {self.inbox_slots}")
        if self.hidden_dim % self.block_k:
            raise ValueError(f"hidden_dim must be divisible by block_k, got {self.hidden_dim=} {self.block_k=}")
        if self.intermediate_dim % self.block_n:
            raise ValueError(
                f"intermediate_dim must be divisible by block_n, got {self.intermediate_dim=} {self.block_n=}"
            )
        if (self.intermediate_dim // self.block_n) % self.n_group:
            raise ValueError(
                "intermediate_dim / block_n must be divisible by n_group; "
                f"got {self.intermediate_dim=} {self.block_n=} {self.n_group=}"
            )
        if self.block_m <= 0 or self.block_k <= 0:
            raise ValueError(f"block_m and block_k must be positive, got {self.block_m=} {self.block_k=}")
        if self.block_n <= 0:
            raise ValueError(f"block_n must be positive, got {self.block_n}")
        if self.n_group not in (1, 2):
            raise ValueError(f"W13 inbox prototypes currently support n_group in (1, 2), got {self.n_group}")
        if self.experts_per_rank <= 0:
            raise ValueError(f"experts_per_rank must be positive, got {self.experts_per_rank}")
        if self.send_worker_programs_per_peer <= 0:
            raise ValueError(
                f"send_worker_programs_per_peer must be positive, got {self.send_worker_programs_per_peer}"
            )
        if self.worker_programs_per_peer <= self.send_worker_programs_per_peer:
            raise ValueError(
                "worker_programs_per_peer must leave at least one destination receiver program; "
                f"got {self.worker_programs_per_peer=} {self.send_worker_programs_per_peer=}"
            )
        if self.send_pipeline_depth not in (1, 2):
            raise ValueError(f"send_pipeline_depth must be in (1, 2), got {self.send_pipeline_depth}")
        if self.n_groups_per_job <= 0:
            raise ValueError(f"n_groups_per_job must be positive, got {self.n_groups_per_job}")
        if self.n_groups_per_job > self.intermediate_dim // self.block_n // self.n_group:
            raise ValueError(
                "n_groups_per_job must not exceed the number of N work groups; "
                f"got {self.n_groups_per_job=} with "
                f"{self.intermediate_dim=} {self.block_n=} {self.n_group=}"
            )
        if self.routing not in ROUTING_MODES:
            raise ValueError(f"unknown routing={self.routing!r}; expected one of {ROUTING_MODES}")
        if self.tokens_per_rank <= 0:
            raise ValueError(f"tokens_per_rank must be positive, got {self.tokens_per_rank}")
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {self.capacity_factor}")

    @property
    def traffic_fanout(self) -> int:
        return self.ep_size

    @property
    def hidden_rows_per_rank(self) -> int:
        return self.traffic_fanout * self.entries_per_rank * self.block_m

    @property
    def hidden_output_shape(self) -> tuple[int, int]:
        return (self.hidden_rows_per_rank, self.intermediate_dim)

    @property
    def h_output_shape(self) -> tuple[int, int]:
        return (self.hidden_rows_per_rank, 2 * self.intermediate_dim)


@dataclass(frozen=True)
class SourcePushInboxRunSettings:
    """Host-side settings for one source-push inbox benchmark run."""

    warmup: int = 1
    steps: int = 5
    repeat_runs: int = 1
    check: bool = True
    debug_exceptions: bool = False
    separate_compile: bool = False
    progress_events: bool = False


@dataclass(frozen=True)
class HostInputs:
    x: np.ndarray
    send_meta: np.ndarray
    recv_meta: np.ndarray
    expert_base: np.ndarray
    src_base_by_expert: np.ndarray
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool = False


@dataclass(frozen=True)
class DeviceInputs:
    x: jax.Array
    send_meta: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    w: jax.Array
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool


@dataclass(frozen=True)
class TimingResult:
    compile_time: float
    steady_state_times: list[float]
    output: Any
    lower_compile_time: float | None
    first_run_time: float | None


@dataclass(frozen=True)
class ValidationMetrics:
    max_abs_diff: float
    metadata_mismatches: int
    hidden_max_abs_diff: float | None
    hidden_mean_abs_diff: float | None
    hidden_all_max_abs_diff: float | None
    hidden_unwritten_max_abs: float | None


def source_push_inbox_profile(profile: str) -> tuple[PushInboxConfig, SourcePushInboxRunSettings]:
    """Return the package-private config and run settings for a named profile."""
    defaults = source_push_profile_defaults(profile)
    config_field_names = {field.name for field in fields(PushInboxConfig)}
    settings_field_names = {field.name for field in fields(SourcePushInboxRunSettings)}
    unknown = set(defaults) - config_field_names - settings_field_names
    if unknown:
        raise ValueError(f"profile {profile!r} has unknown source-push defaults: {sorted(unknown)}")
    config_kwargs = {name: defaults[name] for name in config_field_names if name in defaults}
    settings_kwargs = {name: defaults[name] for name in settings_field_names if name in defaults}
    return PushInboxConfig(**config_kwargs), SourcePushInboxRunSettings(**settings_kwargs)


def _hidden_output_shape_for_kernel(
    config: PushInboxConfig,
    diagnostic_variant: str,
    *,
    output_preactivation_h: bool = False,
) -> tuple[int, int]:
    if diagnostic_variant in (DIAGNOSTIC_VARIANT_SEMAPHORE_ONLY, DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY):
        return (1, 1)
    if diagnostic_variant == DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT:
        n_tiles = config.intermediate_dim // config.block_n
        return (config.hidden_rows_per_rank, n_tiles * TINY_OUTPUT_COLUMNS_PER_N_TILE)
    if output_preactivation_h:
        return config.h_output_shape
    return config.hidden_output_shape


def _make_kernel(
    config: PushInboxConfig,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
    *,
    use_exact_expert_major: bool = False,
    output_preactivation_h: bool = False,
    source_input_mode: str = SOURCE_INPUT_PACKED_QUEUE,
):
    _validate_diagnostic_variant(diagnostic_variant)
    if source_input_mode not in (SOURCE_INPUT_PACKED_QUEUE, SOURCE_INPUT_RAW_TOKENS):
        raise ValueError(
            "source_input_mode must be one of "
            f"{(SOURCE_INPUT_PACKED_QUEUE, SOURCE_INPUT_RAW_TOKENS)}, got {source_input_mode!r}"
        )
    k_tiles = config.hidden_dim // config.block_k
    n_tiles = config.intermediate_dim // config.block_n
    n_work_groups = n_tiles // config.n_group
    n_compute_jobs = (n_work_groups + config.n_groups_per_job - 1) // config.n_groups_per_job
    compute_worker_programs_per_peer = config.worker_programs_per_peer - config.send_worker_programs_per_peer
    rounds_per_slot = (config.entries_per_rank + config.inbox_slots - 1) // config.inbox_slots
    send_dst_offsets = tuple(range(config.ep_size))
    recv_src_offsets = tuple(range(config.ep_size))
    uses_remote_copy = diagnostic_variant in (
        DIAGNOSTIC_VARIANT_FULL,
        DIAGNOSTIC_VARIANT_COPY_RELEASE_ONLY,
        DIAGNOSTIC_VARIANT_STORE_ZERO,
        DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT,
    )
    uses_semaphores = diagnostic_variant != DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL
    computes_wgmma = diagnostic_variant in (
        DIAGNOSTIC_VARIANT_FULL,
        DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL,
        DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT,
    )
    stores_zero = diagnostic_variant == DIAGNOSTIC_VARIANT_STORE_ZERO
    stores_tiny_output = diagnostic_variant == DIAGNOSTIC_VARIANT_WGMMA_TINY_OUTPUT
    stores_preactivation_h = output_preactivation_h and diagnostic_variant in (
        DIAGNOSTIC_VARIANT_FULL,
        DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL,
    )

    def _body_impl(
        x_ref: Float[Ref, "... D"],
        token_ids_ref: Int[Ref, "DST Q M"],
        send_meta_ref: Int[Ref, "DST Q F"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        w_ref: Float[Ref, "E D twoI"],
        inbox_out_ref: Float[Ref, "SRC SLOTS M D"],
        hidden_ref: Float[Ref, "rows I"],
    ) -> None:
        inbox_ref = inbox_out_ref
        empty_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        full_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        done_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        rank = lax.axis_index(AXIS)
        peer_ordinal = pl.program_id(0)
        worker_program = pl.program_id(1)

        if uses_semaphores:

            @pl.when(worker_program < config.send_worker_programs_per_peer)
            def _send_worker() -> None:

                def _send_to_dst(dst_ordinal: int, dst_offset: int) -> None:
                    dst = (rank + dst_offset) % config.ep_size
                    remote_inbox_ref = None
                    if dst_offset != 0:
                        remote_inbox_ref = mgpu.remote_ref(inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                    @pl.loop(0, rounds_per_slot)
                    def _round_loop(round_i) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _slot_loop(slot_pos) -> None:
                            slot = slot_pos
                            send_task = dst_ordinal * config.inbox_slots + slot
                            should_send_slot = (send_task % config.send_worker_programs_per_peer) == worker_program

                            @pl.when(should_send_slot)
                            def _send_slot_entry() -> None:
                                entry = slot + round_i * config.inbox_slots

                                @pl.when(entry < config.entries_per_rank)
                                def _maybe_send_entry() -> None:
                                    valid_rows = send_meta_ref[dst_ordinal, entry, 3]

                                    @pl.when(valid_rows > 0)
                                    def _send_entry() -> None:
                                        pl.semaphore_wait(empty_sem_ref.at[dst, slot])

                                        def _copy_buffer(buffer_smem, k_start) -> None:
                                            if source_input_mode == SOURCE_INPUT_RAW_TOKENS:

                                                @pl.loop(0, config.block_m)
                                                def _row_loop(row) -> None:
                                                    @pl.when(row < valid_rows)
                                                    def _copy_valid_row() -> None:
                                                        token = token_ids_ref[dst_ordinal, entry, row]
                                                        buffer_smem[row, :] = x_ref[
                                                            token,
                                                            pl.ds(k_start, config.block_k),
                                                        ]

                                                    @pl.when(row >= valid_rows)
                                                    def _zero_invalid_row() -> None:
                                                        buffer_smem[row, :] = jnp.zeros(
                                                            (config.block_k,),
                                                            dtype=x_ref.dtype,
                                                        )

                                            else:
                                                buffer_smem[:, :] = x_ref[
                                                    dst_ordinal,
                                                    entry,
                                                    pl.ds(0, config.block_m),
                                                    pl.ds(k_start, config.block_k),
                                                ]
                                            mgpu.commit_smem()
                                            if dst_offset == 0:
                                                mgpu.copy_smem_to_gmem(
                                                    buffer_smem,
                                                    inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )
                                            else:
                                                mgpu.copy_smem_to_gmem(
                                                    buffer_smem,
                                                    remote_inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )

                                        if uses_remote_copy:
                                            if config.send_pipeline_depth == 1:

                                                def _copy_scope(tile_smem) -> None:
                                                    @pl.loop(0, k_tiles)
                                                    def _k_loop(kk) -> None:
                                                        k_start = kk * config.block_k
                                                        _copy_buffer(tile_smem, k_start)
                                                        mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                                pl.run_scoped(
                                                    _copy_scope,
                                                    tile_smem=mgpu.SMEM(
                                                        (config.block_m, config.block_k), dtype=x_ref.dtype
                                                    ),
                                                )
                                            elif config.send_pipeline_depth == 2:

                                                def _copy_scope(tile_smem, tile_smem_next) -> None:
                                                    @pl.loop(0, k_tiles)
                                                    def _k_loop(kk) -> None:
                                                        k_start = kk * config.block_k

                                                        @pl.when(kk >= 2)
                                                        def _wait_for_reusable_buffer() -> None:
                                                            mgpu.wait_smem_to_gmem(1, wait_read_only=False)

                                                        @pl.when((kk % 2) == 0)
                                                        def _copy_even_buffer() -> None:
                                                            _copy_buffer(tile_smem, k_start)

                                                        @pl.when((kk % 2) == 1)
                                                        def _copy_odd_buffer() -> None:
                                                            _copy_buffer(tile_smem_next, k_start)

                                                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                                pl.run_scoped(
                                                    _copy_scope,
                                                    tile_smem=mgpu.SMEM(
                                                        (config.block_m, config.block_k), dtype=x_ref.dtype
                                                    ),
                                                    tile_smem_next=mgpu.SMEM(
                                                        (config.block_m, config.block_k), dtype=x_ref.dtype
                                                    ),
                                                )
                                        if dst_offset == 0:
                                            pl.semaphore_signal(full_sem_ref.at[rank, slot])
                                        else:
                                            pl.semaphore_signal(
                                                full_sem_ref.at[rank, slot],
                                                device_id=dst,
                                                device_id_type=pl.DeviceIdType.LOGICAL,
                                            )

                def _switch_send_to_dst(dst_ordinal) -> None:
                    def _branch(static_dst_ordinal: int, static_dst_offset: int):
                        def _send_branch(_) -> None:
                            _send_to_dst(static_dst_ordinal, static_dst_offset)

                        return _send_branch

                    branches = tuple(
                        _branch(static_dst_ordinal, static_dst_offset)
                        for static_dst_ordinal, static_dst_offset in enumerate(send_dst_offsets)
                    )
                    lax.switch(dst_ordinal, branches, None)

                _switch_send_to_dst(peer_ordinal)

        def _init_empty_slots() -> None:
            def _init_empty_for_src(src_offset: int) -> None:
                src = (rank + src_offset) % config.ep_size

                @pl.loop(0, config.inbox_slots)
                def _init_empty_slot(slot) -> None:
                    if src_offset == 0:
                        pl.semaphore_signal(empty_sem_ref.at[rank, slot])
                    else:
                        pl.semaphore_signal(
                            empty_sem_ref.at[rank, slot],
                            device_id=src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )

            def _switch_init_empty_for_src(src_ordinal) -> None:
                def _branch(static_src_offset: int):
                    def _init_branch(_) -> None:
                        _init_empty_for_src(static_src_offset)

                    return _init_branch

                branches = tuple(_branch(static_src_offset) for static_src_offset in recv_src_offsets)
                lax.switch(src_ordinal, branches, None)

            _switch_init_empty_for_src(peer_ordinal)

        def _signal_empty_to_src(src, src_offset: int, slot) -> None:
            if src_offset == 0:
                pl.semaphore_signal(empty_sem_ref.at[rank, slot])
            else:
                pl.semaphore_signal(
                    empty_sem_ref.at[rank, slot],
                    device_id=src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

        def _hidden_row_start(src_ordinal, entry):
            if use_exact_expert_major:
                src_rank = recv_meta_ref[src_ordinal, entry, 0]
                expert = recv_meta_ref[src_ordinal, entry, 1]
                local_row_start = recv_meta_ref[src_ordinal, entry, 2]
                return expert_base_ref[expert] + src_base_by_expert_ref[src_rank, expert] + local_row_start
            return recv_meta_ref[src_ordinal, entry, 2]

        def _store_hidden(src_ordinal, entry, n_tile, hidden) -> None:
            dst_row_start = _hidden_row_start(src_ordinal, entry)
            if stores_tiny_output:
                output = hidden[:, :TINY_OUTPUT_COLUMNS_PER_N_TILE].astype(hidden_ref.dtype)
                idx = (
                    pl.ds(dst_row_start, config.block_m),
                    pl.ds(n_tile * TINY_OUTPUT_COLUMNS_PER_N_TILE, TINY_OUTPUT_COLUMNS_PER_N_TILE),
                )
                hidden_ref[idx] = output
            else:
                output = hidden.astype(hidden_ref.dtype)
                idx = (
                    pl.ds(dst_row_start, config.block_m),
                    pl.ds(n_tile * config.block_n, config.block_n),
                )
                hidden_ref[idx] = output

        def _store_h_tile(src_ordinal, entry, n_tile, gate, up) -> None:
            dst_row_start = _hidden_row_start(src_ordinal, entry)
            gate_idx = (
                pl.ds(dst_row_start, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            )
            up_idx = (
                pl.ds(dst_row_start, config.block_m),
                pl.ds((n_tile + n_tiles) * config.block_n, config.block_n),
            )
            hidden_ref[gate_idx] = gate.astype(hidden_ref.dtype)
            hidden_ref[up_idx] = up.astype(hidden_ref.dtype)

        def _store_zero_hidden(src_ordinal, entry, n_tile) -> None:
            dst_row_start = _hidden_row_start(src_ordinal, entry)
            idx = (
                pl.ds(dst_row_start, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            )
            zeros = jnp.zeros((config.block_m, config.block_n), dtype=hidden_ref.dtype)
            hidden_ref[idx] = zeros

        def _store_zero_n_group(src_ordinal, entry, n_group_i) -> None:
            if config.n_group == 1:
                _store_zero_hidden(src_ordinal, entry, n_group_i)
            else:
                n_tile = n_group_i * 2
                _store_zero_hidden(src_ordinal, entry, n_tile)
                _store_zero_hidden(src_ordinal, entry, n_tile + 1)

        def _copy_lhs_to_smem(lhs_smem, ready_barrier, src, src_ordinal, entry, slot, k_start) -> None:
            if diagnostic_variant == DIAGNOSTIC_VARIANT_COMPUTE_ONLY_LOCAL:
                mgpu.copy_gmem_to_smem(
                    x_ref.at[
                        src_ordinal,
                        entry,
                        pl.ds(0, config.block_m),
                        pl.ds(k_start, config.block_k),
                    ],
                    lhs_smem,
                    ready_barrier,
                )
            else:
                mgpu.copy_gmem_to_smem(
                    inbox_ref.at[
                        src,
                        slot,
                        pl.ds(0, config.block_m),
                        pl.ds(k_start, config.block_k),
                    ],
                    lhs_smem,
                    ready_barrier,
                )

        def _compute_hidden_n_tile(src, src_ordinal, entry, slot, expert, n_tile) -> None:
            def acc_scope(gate_acc_ref, up_acc_ref) -> jax.Array:
                def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier) -> None:
                    @pl.loop(0, k_tiles)
                    def _k_loop(kk) -> None:
                        k_start = kk * config.block_k
                        _copy_lhs_to_smem(lhs_smem, ready_barrier, src, src_ordinal, entry, slot, k_start)
                        mgpu.copy_gmem_to_smem(
                            w_ref.at[
                                expert,
                                pl.ds(k_start, config.block_k),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                            gate_smem,
                            ready_barrier,
                        )
                        mgpu.copy_gmem_to_smem(
                            w_ref.at[
                                expert,
                                pl.ds(k_start, config.block_k),
                                pl.ds((n_tile + n_tiles) * config.block_n, config.block_n),
                            ],
                            up_smem,
                            ready_barrier,
                        )
                        mgpu.barrier_wait(ready_barrier)
                        mgpu.commit_smem()
                        mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                        mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)
                        mgpu.wgmma_wait(0)

                pl.run_scoped(
                    smem_scope,
                    lhs_smem=_wgmma_smem((config.block_m, config.block_k), inbox_ref.dtype),
                    gate_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                    up_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                    ready_barrier=mgpu.Barrier(num_arrivals=3),
                )
                if stores_preactivation_h:
                    _store_h_tile(src_ordinal, entry, n_tile, gate_acc_ref[...], up_acc_ref[...])
                    return jnp.zeros((1,), dtype=hidden_ref.dtype)
                return _silu(gate_acc_ref[...]) * up_acc_ref[...]

            hidden = pl.run_scoped(
                acc_scope,
                gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
            )
            if not stores_preactivation_h:
                _store_hidden(src_ordinal, entry, n_tile, hidden)

        def _compute_hidden_n_group(src, src_ordinal, entry, slot, expert, n_group_i) -> None:
            if config.n_group == 1:
                _compute_hidden_n_tile(src, src_ordinal, entry, slot, expert, n_group_i)
            else:
                n_tile = n_group_i * 2

                def acc_scope(gate_n0_acc, up_n0_acc, gate_n1_acc, up_n1_acc) -> None:
                    def smem_scope(
                        lhs_smem,
                        gate_n0_smem,
                        up_n0_smem,
                        gate_n1_smem,
                        up_n1_smem,
                        ready_barrier,
                    ) -> None:
                        @pl.loop(0, k_tiles)
                        def _k_loop(kk) -> None:
                            k_start = kk * config.block_k
                            _copy_lhs_to_smem(lhs_smem, ready_barrier, src, src_ordinal, entry, slot, k_start)
                            mgpu.copy_gmem_to_smem(
                                w_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds(n_tile * config.block_n, config.block_n),
                                ],
                                gate_n0_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds((n_tile + n_tiles) * config.block_n, config.block_n),
                                ],
                                up_n0_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds((n_tile + 1) * config.block_n, config.block_n),
                                ],
                                gate_n1_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds((n_tile + 1 + n_tiles) * config.block_n, config.block_n),
                                ],
                                up_n1_smem,
                                ready_barrier,
                            )
                            mgpu.barrier_wait(ready_barrier)
                            mgpu.commit_smem()
                            mgpu.wgmma(gate_n0_acc, lhs_smem, gate_n0_smem)
                            mgpu.wgmma(up_n0_acc, lhs_smem, up_n0_smem)
                            mgpu.wgmma(gate_n1_acc, lhs_smem, gate_n1_smem)
                            mgpu.wgmma(up_n1_acc, lhs_smem, up_n1_smem)
                            mgpu.wgmma_wait(0)

                    pl.run_scoped(
                        smem_scope,
                        lhs_smem=_wgmma_smem((config.block_m, config.block_k), inbox_ref.dtype),
                        gate_n0_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                        up_n0_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                        gate_n1_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                        up_n1_smem=_wgmma_smem((config.block_k, config.block_n), w_ref.dtype),
                        ready_barrier=mgpu.Barrier(num_arrivals=5),
                    )
                    if stores_preactivation_h:
                        _store_h_tile(src_ordinal, entry, n_tile, gate_n0_acc[...], up_n0_acc[...])
                        _store_h_tile(src_ordinal, entry, n_tile + 1, gate_n1_acc[...], up_n1_acc[...])
                    else:
                        hidden_n0 = _silu(gate_n0_acc[...]) * up_n0_acc[...]
                        hidden_n1 = _silu(gate_n1_acc[...]) * up_n1_acc[...]
                        _store_hidden(src_ordinal, entry, n_tile, hidden_n0)
                        _store_hidden(src_ordinal, entry, n_tile + 1, hidden_n1)

                pl.run_scoped(
                    acc_scope,
                    gate_n0_acc=mgpu.ACC((config.block_m, config.block_n)),
                    up_n0_acc=mgpu.ACC((config.block_m, config.block_n)),
                    gate_n1_acc=mgpu.ACC((config.block_m, config.block_n)),
                    up_n1_acc=mgpu.ACC((config.block_m, config.block_n)),
                )

        if uses_semaphores:

            @pl.when(worker_program == config.send_worker_programs_per_peer)
            def _init_worker() -> None:
                _init_empty_slots()

        @pl.when(worker_program >= config.send_worker_programs_per_peer)
        def _n_tile_recv_worker() -> None:
            compute_worker = worker_program - config.send_worker_programs_per_peer

            def _recv_n_tile_src(src_ordinal: int, src_offset: int) -> None:
                src = (rank + src_offset) % config.ep_size

                def _recv_fixed_wait() -> None:

                    @pl.loop(0, rounds_per_slot)
                    def _round_loop(round_i) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _slot_loop(slot_pos) -> None:
                            slot = slot_pos
                            entry = slot + round_i * config.inbox_slots

                            @pl.when(entry < config.entries_per_rank)
                            def _maybe_recv_slot_entry() -> None:
                                valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                @pl.when(valid_rows > 0)
                                def _recv_slot_entry() -> None:
                                    @pl.loop(0, n_compute_jobs)
                                    def _job_loop(job_i) -> None:
                                        work_group = (src_ordinal * config.inbox_slots + slot) * n_compute_jobs + job_i
                                        should_compute = (
                                            work_group % compute_worker_programs_per_peer
                                        ) == compute_worker

                                        @pl.when(should_compute)
                                        def _recv_job() -> None:
                                            if uses_semaphores:
                                                pl.semaphore_wait(
                                                    full_sem_ref.at[src, slot],
                                                    value=round_i + 1,
                                                    decrement=False,
                                                )
                                            if computes_wgmma:
                                                expert = recv_meta_ref[src_ordinal, entry, 1]

                                                @pl.loop(0, config.n_groups_per_job)
                                                def _job_n_group_loop(group_offset) -> None:
                                                    n_group_i = job_i * config.n_groups_per_job + group_offset

                                                    @pl.when(n_group_i < n_work_groups)
                                                    def _job_n_group() -> None:
                                                        _compute_hidden_n_group(
                                                            src,
                                                            src_ordinal,
                                                            entry,
                                                            slot,
                                                            expert,
                                                            n_group_i,
                                                        )

                                            elif stores_zero:

                                                @pl.loop(0, config.n_groups_per_job)
                                                def _job_n_group_loop(group_offset) -> None:
                                                    n_group_i = job_i * config.n_groups_per_job + group_offset

                                                    @pl.when(n_group_i < n_work_groups)
                                                    def _job_n_group() -> None:
                                                        _store_zero_n_group(src_ordinal, entry, n_group_i)

                                            if uses_semaphores:
                                                pl.semaphore_signal(done_sem_ref.at[src, slot])

                                    if uses_semaphores:
                                        release_group = src_ordinal * config.inbox_slots + slot
                                        should_release = (
                                            release_group % compute_worker_programs_per_peer
                                        ) == compute_worker

                                        @pl.when(should_release)
                                        def _release_slot() -> None:
                                            pl.semaphore_wait(
                                                done_sem_ref.at[src, slot],
                                                value=(round_i + 1) * n_compute_jobs,
                                                decrement=False,
                                            )
                                            _signal_empty_to_src(src, src_offset, slot)

                _recv_fixed_wait()

            def _switch_recv_n_tile_src(src_ordinal) -> None:
                def _branch(static_src_ordinal: int, static_src_offset: int):
                    def _recv_branch(_) -> None:
                        _recv_n_tile_src(static_src_ordinal, static_src_offset)

                    return _recv_branch

                branches = tuple(
                    _branch(static_src_ordinal, static_src_offset)
                    for static_src_ordinal, static_src_offset in enumerate(recv_src_offsets)
                )
                lax.switch(src_ordinal, branches, None)

            _switch_recv_n_tile_src(peer_ordinal)

    if source_input_mode == SOURCE_INPUT_PACKED_QUEUE:

        def body(
            x_ref: Float[Ref, "DST Q M D"],
            send_meta_ref: Int[Ref, "DST Q F"],
            recv_meta_ref: Int[Ref, "SRC Q F"],
            expert_base_ref: Int[Ref, "E"],
            src_base_by_expert_ref: Int[Ref, "SRC E"],
            w_ref: Float[Ref, "E D twoI"],
            inbox_out_ref: Float[Ref, "SRC SLOTS M D"],
            hidden_ref: Float[Ref, "rows I"],
        ) -> None:
            _body_impl(
                x_ref,
                send_meta_ref,
                send_meta_ref,
                recv_meta_ref,
                expert_base_ref,
                src_base_by_expert_ref,
                w_ref,
                inbox_out_ref,
                hidden_ref,
            )

    else:

        def body(
            x_ref: Float[Ref, "T D"],
            token_ids_ref: Int[Ref, "DST Q M"],
            send_meta_ref: Int[Ref, "DST Q F"],
            recv_meta_ref: Int[Ref, "SRC Q F"],
            expert_base_ref: Int[Ref, "E"],
            src_base_by_expert_ref: Int[Ref, "SRC E"],
            w_ref: Float[Ref, "E D twoI"],
            inbox_out_ref: Float[Ref, "SRC SLOTS M D"],
            hidden_ref: Float[Ref, "rows I"],
        ) -> None:
            _body_impl(
                x_ref,
                token_ids_ref,
                send_meta_ref,
                recv_meta_ref,
                expert_base_ref,
                src_base_by_expert_ref,
                w_ref,
                inbox_out_ref,
                hidden_ref,
            )

    inbox_shape = (config.ep_size, config.inbox_slots, config.block_m, config.hidden_dim)
    grid = (len(send_dst_offsets), config.worker_programs_per_peer)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    out_shape = [
        jax.ShapeDtypeStruct(inbox_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(
            _hidden_output_shape_for_kernel(
                config,
                diagnostic_variant,
                output_preactivation_h=output_preactivation_h,
            ),
            jnp.bfloat16,
        ),
    ]

    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=grid,
        grid_names=("peer_phase", "worker_program"),
        compiler_params=compiler_params,
    )


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,))


def _destination_ranks(config: PushInboxConfig):
    return range(config.ep_size)


def _dst_ordinal(config: PushInboxConfig, src: int, dst: int) -> int:
    return (dst - src) % config.ep_size


def _recv_src_ordinal(config: PushInboxConfig, dst: int, src: int) -> int:
    return (src - dst) % config.ep_size


def _make_weights(config: PushInboxConfig):
    w_host = np.empty(
        (config.ep_size, config.experts_per_rank, config.hidden_dim, 2 * config.intermediate_dim),
        dtype=np.float32,
    )
    gate_scale = (((np.arange(config.experts_per_rank, dtype=np.float32) % 4) + 1.0) / float(config.hidden_dim))[
        None, :, None, None
    ]
    up_scale = (((np.arange(config.ep_size, dtype=np.float32) % 4) + 1.0) / float(2 * config.hidden_dim))[
        :, None, None, None
    ]
    w_host[:, :, :, : config.intermediate_dim] = gate_scale
    w_host[:, :, :, config.intermediate_dim :] = up_scale
    return jnp.asarray(w_host, dtype=jnp.bfloat16)


def _recv_meta_from_send_meta(config: PushInboxConfig, send_meta: np.ndarray) -> np.ndarray:
    recv_meta = np.zeros_like(send_meta)
    for dst in range(config.ep_size):
        for src in _destination_ranks(config):
            send_dst_ordinal = _dst_ordinal(config, src, dst)
            recv_src_ordinal = _recv_src_ordinal(config, dst, src)
            recv_meta[dst, recv_src_ordinal, :, :] = send_meta[src, send_dst_ordinal, :, :]
    return recv_meta


def _queue_stats(config: PushInboxConfig, send_meta: np.ndarray) -> dict[str, Any]:
    valid_rows = send_meta[:, :, :, 3]
    live = valid_rows > 0
    live_entries_per_rank = np.sum(live, axis=(1, 2))
    direct_self_entries_per_rank = np.zeros((config.ep_size,), dtype=np.int64)
    send_entries_per_rank = live_entries_per_rank - direct_self_entries_per_rank
    valid_rows_per_rank = np.sum(valid_rows, axis=(1, 2))
    rounded_rows_per_rank = live_entries_per_rank * config.block_m
    masked_rows = live.astype(np.int64) * config.block_m - valid_rows
    masked_rows_per_rank = np.sum(masked_rows, axis=(1, 2))
    direct_self_masked_rows_per_rank = np.zeros((config.ep_size,), dtype=np.int64)
    send_rounded_rows_per_rank = send_entries_per_rank * config.block_m
    send_masked_rows_per_rank = masked_rows_per_rank - direct_self_masked_rows_per_rank
    entries_per_pair = np.sum(live, axis=2)
    entries_by_dst = np.zeros((config.ep_size,), dtype=np.int64)
    entries_by_local_expert = np.zeros((config.experts_per_rank,), dtype=np.int64)
    entries_by_global_expert = np.zeros((config.ep_size * config.experts_per_rank,), dtype=np.int64)
    entries_by_source_destination_expert = np.zeros(
        (config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int64
    )
    max_slot_reuse_depth = 0
    for src in range(config.ep_size):
        for dst in _destination_ranks(config):
            dst_ordinal = _dst_ordinal(config, src, dst)
            pair_live_entries = int(entries_per_pair[src, dst_ordinal])
            entries_by_dst[dst] += pair_live_entries
            for slot in range(config.inbox_slots):
                if pair_live_entries > slot:
                    max_slot_reuse_depth = max(
                        max_slot_reuse_depth,
                        1 + (pair_live_entries - 1 - slot) // config.inbox_slots,
                    )
            for entry in range(config.entries_per_rank):
                if not live[src, dst_ordinal, entry]:
                    continue
                expert = int(send_meta[src, dst_ordinal, entry, 1])
                entries_by_local_expert[expert] += 1
                entries_by_global_expert[dst * config.experts_per_rank + expert] += 1
                entries_by_source_destination_expert[src, dst, expert] += 1
    live_entries_total = int(np.sum(live))
    direct_self_entries_total = int(np.sum(direct_self_entries_per_rank))
    payload_send_entries_total = live_entries_total - direct_self_entries_total
    tail_entries_total = int(np.sum((valid_rows > 0) & (valid_rows < config.block_m)))
    n_tiles = config.intermediate_dim // config.block_n
    n_work_groups = n_tiles // config.n_group
    n_compute_jobs = (n_work_groups + config.n_groups_per_job - 1) // config.n_groups_per_job
    local_work_groups = config.inbox_slots * n_compute_jobs
    compute_jobs_per_entry = n_compute_jobs
    compute_wait_full_count = payload_send_entries_total * compute_jobs_per_entry
    capacity_entries_total = config.ep_size * config.traffic_fanout * config.entries_per_rank
    nonzero_expert_entries = entries_by_source_destination_expert[entries_by_source_destination_expert > 0]
    if nonzero_expert_entries.size:
        per_expert_pair_min_nonzero = int(np.min(nonzero_expert_entries))
        per_expert_pair_max = int(np.max(nonzero_expert_entries))
    else:
        per_expert_pair_min_nonzero = 0
        per_expert_pair_max = 0
    return {
        "routing": config.routing,
        "send_pipeline_depth": config.send_pipeline_depth,
        "n_groups_per_job": config.n_groups_per_job,
        "n_work_groups_per_entry": n_work_groups,
        "num_compute_jobs_per_entry": compute_jobs_per_entry,
        "local_work_groups_per_source": local_work_groups,
        "done_signals_per_entry": compute_jobs_per_entry,
        "send_entries_total": live_entries_total,
        "recv_entries_total": live_entries_total,
        "live_entries_total": live_entries_total,
        "logical_entries_total": live_entries_total,
        "direct_self_entries_total": direct_self_entries_total,
        "payload_send_entries_total": payload_send_entries_total,
        "remote_send_entries_total": payload_send_entries_total,
        "live_entries_per_rank_min": int(np.min(live_entries_per_rank)),
        "live_entries_per_rank_mean": float(np.mean(live_entries_per_rank)),
        "live_entries_per_rank_max": int(np.max(live_entries_per_rank)),
        "send_entries_per_rank_min": int(np.min(send_entries_per_rank)),
        "send_entries_per_rank_mean": float(np.mean(send_entries_per_rank)),
        "send_entries_per_rank_max": int(np.max(send_entries_per_rank)),
        "max_live_entries_per_pair": int(np.max(entries_per_pair)),
        "zero_source_destination_pairs": int(np.sum(entries_per_pair == 0)),
        "zero_source_destination_expert_pairs": int(np.sum(entries_by_source_destination_expert == 0)),
        "tail_entries": tail_entries_total,
        "tail_entries_total": tail_entries_total,
        "tail_fraction": float(tail_entries_total / live_entries_total) if live_entries_total else 0.0,
        "zero_entries_skipped": int(capacity_entries_total - live_entries_total),
        "masked_rows_total": int(np.sum(masked_rows)),
        "masked_rows_per_rank_min": int(np.min(masked_rows_per_rank)),
        "masked_rows_per_rank_mean": float(np.mean(masked_rows_per_rank)),
        "masked_rows_per_rank_max": int(np.max(masked_rows_per_rank)),
        "masked_row_fraction": (
            float(np.sum(masked_rows) / max(float(np.sum(rounded_rows_per_rank)), 1.0)) if live_entries_total else 0.0
        ),
        "valid_rows_per_rank_min": int(np.min(valid_rows_per_rank)),
        "valid_rows_per_rank_mean": float(np.mean(valid_rows_per_rank)),
        "valid_rows_per_rank_max": int(np.max(valid_rows_per_rank)),
        "rounded_rows_per_rank_min": int(np.min(rounded_rows_per_rank)),
        "rounded_rows_per_rank_mean": float(np.mean(rounded_rows_per_rank)),
        "rounded_rows_per_rank_max": int(np.max(rounded_rows_per_rank)),
        "send_masked_rows_per_rank_min": int(np.min(send_masked_rows_per_rank)),
        "send_masked_rows_per_rank_mean": float(np.mean(send_masked_rows_per_rank)),
        "send_masked_rows_per_rank_max": int(np.max(send_masked_rows_per_rank)),
        "send_masked_row_fraction": (
            float(np.sum(send_masked_rows_per_rank) / max(float(np.sum(send_rounded_rows_per_rank)), 1.0))
            if payload_send_entries_total
            else 0.0
        ),
        "send_rounded_rows_per_rank_min": int(np.min(send_rounded_rows_per_rank)),
        "send_rounded_rows_per_rank_mean": float(np.mean(send_rounded_rows_per_rank)),
        "send_rounded_rows_per_rank_max": int(np.max(send_rounded_rows_per_rank)),
        "capacity_entries_per_pair": config.entries_per_rank,
        "capacity_entries_per_rank": config.traffic_fanout * config.entries_per_rank,
        "capacity_entries_total": int(capacity_entries_total),
        "entries_by_src": [int(v) for v in live_entries_per_rank],
        "entries_by_dst": [int(v) for v in entries_by_dst],
        "entries_by_expert": [int(v) for v in entries_by_global_expert],
        "entries_by_local_expert": [int(v) for v in entries_by_local_expert],
        "entries_by_src_min": int(np.min(live_entries_per_rank)),
        "entries_by_src_max": int(np.max(live_entries_per_rank)),
        "entries_by_src_imbalance": float(np.max(live_entries_per_rank) / max(np.mean(live_entries_per_rank), 1.0)),
        "entries_by_dst_min": int(np.min(entries_by_dst)),
        "entries_by_dst_max": int(np.max(entries_by_dst)),
        "entries_by_dst_imbalance": float(np.max(entries_by_dst) / max(np.mean(entries_by_dst), 1.0)),
        "entries_by_expert_min_nonzero": per_expert_pair_min_nonzero,
        "entries_by_expert_max": per_expert_pair_max,
        "entries_by_expert_imbalance": (
            float(per_expert_pair_max / max(float(np.mean(nonzero_expert_entries)), 1.0))
            if nonzero_expert_entries.size
            else 0.0
        ),
        "slot_full_waits": compute_wait_full_count,
        "slot_empty_waits": payload_send_entries_total,
        "compute_wait_full_count": compute_wait_full_count,
        "send_wait_empty_count": payload_send_entries_total,
        "max_slot_residency_or_equivalent": int(max_slot_reuse_depth),
        "max_slot_reuse_depth": int(max_slot_reuse_depth),
    }


def _fill_block(x_host: np.ndarray, src: int, dst_ordinal: int, entry: int, valid_rows: int) -> None:
    value = 0.25 + 0.125 * src + 0.015 * dst_ordinal + 0.01 * entry
    x_host[src, dst_ordinal, entry, :, :] = 0.0
    x_host[src, dst_ordinal, entry, :valid_rows, :] = value


def _routing_counts(config: PushInboxConfig) -> np.ndarray:
    counts = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    if config.routing == "one_source_one_expert":
        counts[0, 0, 0] = config.block_m + 7
        return counts

    if config.routing == "many_sources_one_expert":
        for src in range(config.ep_size):
            counts[src, 0, 0] = config.block_m + 1 + src
        return counts

    if config.routing == "many_sources_many_experts":
        for src in range(config.ep_size):
            for dst in range(config.ep_size):
                for expert in range(config.experts_per_rank):
                    if (src + dst + expert) % 3 == 0:
                        counts[src, dst, expert] = config.block_m + 1 + ((src * 7 + dst * 3 + expert) % 17)
                    elif (src + 2 * dst + expert) % 5 == 0:
                        counts[src, dst, expert] = 1 + ((src + dst + expert) % (config.block_m - 1))
        return counts

    if config.routing == "tail_debug":
        for src in range(config.ep_size):
            for dst in range(config.ep_size):
                for expert in range(config.experts_per_rank):
                    counts[src, dst, expert] = (src + 3 * dst + 5 * expert) % (config.block_m + 1)
        return counts

    assignments = config.tokens_per_rank * config.topk
    global_experts = config.ep_size * config.experts_per_rank
    base = assignments // global_experts
    remainder = assignments % global_experts
    flat = np.full((global_experts,), base, dtype=np.int32)
    flat[:remainder] += 1
    if config.routing == "balanced":
        counts[:, :, :] = flat.reshape(config.ep_size, config.experts_per_rank)[None, :, :]
        return counts

    rng = np.random.default_rng(config.routing_seed)
    if config.routing == "roughly_balanced":
        max_count = int(np.max(flat))
        jitter_bound = min(config.block_m - 1, max(1, max_count // 8))
        lower = np.maximum(0, flat - jitter_bound)
        upper = flat + jitter_bound
        for src in range(config.ep_size):
            jitter = rng.integers(-jitter_bound, jitter_bound + 1, size=global_experts, dtype=np.int32)
            source_flat = np.clip(flat + jitter, lower, upper)
            diff = int(assignments - np.sum(source_flat))
            while diff != 0:
                if diff > 0:
                    eligible = np.flatnonzero(source_flat < upper)
                    if eligible.size == 0:
                        raise ValueError("roughly_balanced routing cannot add rows within configured bounds")
                    chosen = rng.permutation(eligible)[: min(diff, eligible.size)]
                    source_flat[chosen] += 1
                    diff -= int(chosen.size)
                else:
                    eligible = np.flatnonzero(source_flat > lower)
                    if eligible.size == 0:
                        raise ValueError("roughly_balanced routing cannot remove rows within configured bounds")
                    chosen = rng.permutation(eligible)[: min(-diff, eligible.size)]
                    source_flat[chosen] -= 1
                    diff += int(chosen.size)
            counts[src, :, :] = source_flat.reshape(config.ep_size, config.experts_per_rank)
        return counts

    probs = np.full((global_experts,), 1.0 / global_experts, dtype=np.float64)
    for src in range(config.ep_size):
        counts[src, :, :] = rng.multinomial(assignments, probs).reshape(config.ep_size, config.experts_per_rank)
    return counts


def _routing_row_bases(config: PushInboxConfig, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rounded_counts = ((counts + config.block_m - 1) // config.block_m) * config.block_m
    expert_base = np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    for dst in range(config.ep_size):
        row = 0
        for expert in range(config.experts_per_rank):
            expert_base[dst, expert] = row
            src_running = 0
            for src in range(config.ep_size):
                src_base[src, dst, expert] = src_running
                src_running += int(rounded_counts[src, dst, expert])
            row += src_running
    return rounded_counts, expert_base, src_base


def _src_base_by_expert_for_dst(src_base: np.ndarray) -> np.ndarray:
    return np.transpose(src_base, (1, 0, 2)).astype(np.int32)


def _source_padded_plan_send_meta(
    plan: SourcePushPlan,
    expert_base: np.ndarray,
    src_base_by_expert: np.ndarray,
) -> np.ndarray:
    send_meta = np.asarray(plan.send_meta, dtype=np.int32).copy()
    ep_size = send_meta.shape[0]
    for src in range(ep_size):
        for dst_ordinal in range(send_meta.shape[1]):
            dst = (src + dst_ordinal) % ep_size
            for entry in range(send_meta.shape[2]):
                valid_rows = int(send_meta[src, dst_ordinal, entry, 3])
                if valid_rows <= 0:
                    continue
                expert = int(send_meta[src, dst_ordinal, entry, 1])
                local_row_start = int(send_meta[src, dst_ordinal, entry, 2])
                send_meta[src, dst_ordinal, entry, 2] = (
                    int(expert_base[dst, expert]) + int(src_base_by_expert[dst, src, expert]) + local_row_start
                )
    return send_meta


def _make_routing_inputs(config: PushInboxConfig) -> HostInputs:
    counts = _routing_counts(config)
    _, expert_base, src_base = _routing_row_bases(config, counts)

    x_host = np.zeros(
        (config.ep_size, config.traffic_fanout, config.entries_per_rank, config.block_m, config.hidden_dim),
        dtype=np.float32,
    )
    send_meta = np.zeros((config.ep_size, config.traffic_fanout, config.entries_per_rank, META_FIELDS), dtype=np.int32)
    dropped_entries = 0
    dropped_rows = 0
    for src in range(config.ep_size):
        for dst in range(config.ep_size):
            dst_ordinal = _dst_ordinal(config, src, dst)
            entry = 0
            for expert in range(config.experts_per_rank):
                count = int(counts[src, dst, expert])
                block_count = (count + config.block_m - 1) // config.block_m
                for block in range(block_count):
                    valid_rows = min(config.block_m, count - block * config.block_m)
                    if entry >= config.entries_per_rank:
                        dropped_entries += 1
                        dropped_rows += valid_rows
                        continue
                    dst_row_start = int(expert_base[dst, expert] + src_base[src, dst, expert] + block * config.block_m)
                    send_meta[src, dst_ordinal, entry, :] = (src, expert, dst_row_start, valid_rows)
                    _fill_block(x_host, src, dst_ordinal, entry, valid_rows)
                    entry += 1

    recv_meta = _recv_meta_from_send_meta(config, send_meta)
    stats = _queue_stats(config, send_meta)
    stats["dropped_entries_total"] = dropped_entries
    stats["dropped_rows_total"] = dropped_rows
    stats["routing_assignments_per_source"] = config.tokens_per_rank * config.topk
    return HostInputs(
        x=x_host,
        send_meta=send_meta,
        recv_meta=recv_meta,
        expert_base=expert_base,
        src_base_by_expert=_src_base_by_expert_for_dst(src_base),
        queue_stats=stats,
    )


def _selected_experts_from_counts(config: PushInboxConfig, source_counts: np.ndarray, src: int) -> np.ndarray:
    assignments = config.tokens_per_rank * config.topk
    global_experts = config.ep_size * config.experts_per_rank
    flat_experts = np.repeat(np.arange(global_experts, dtype=np.int32), source_counts.reshape(-1))
    if flat_experts.size != assignments:
        raise ValueError(
            "compact routing source counts must match tokens_per_rank * topk; "
            f"got {flat_experts.size=} {assignments=} for {src=}"
        )
    rng = np.random.default_rng(config.routing_seed + 1009 * src)
    rng.shuffle(flat_experts)
    return flat_experts.reshape(config.tokens_per_rank, config.topk)


def _make_source_tokens(config: PushInboxConfig, src: int) -> np.ndarray:
    token_component = np.arange(config.tokens_per_rank, dtype=np.float32)[:, None] * 0.001
    hidden_component = np.arange(config.hidden_dim, dtype=np.float32)[None, :] * 0.00001
    return (0.125 + 0.03125 * src + token_component + hidden_component).astype(np.float32)


def _make_compact_routing_inputs(config: PushInboxConfig) -> HostInputs:
    """Build source-push queue inputs from the compact expert-sorted routing layout."""
    counts = _routing_counts(config)
    _, expert_base, src_base = _routing_row_bases(config, counts)
    global_experts = config.ep_size * config.experts_per_rank
    x_host = np.zeros(
        (config.ep_size, config.traffic_fanout, config.entries_per_rank, config.block_m, config.hidden_dim),
        dtype=np.float32,
    )
    send_meta = np.zeros((config.ep_size, config.traffic_fanout, config.entries_per_rank, META_FIELDS), dtype=np.int32)
    dropped_entries = 0
    dropped_rows = 0
    compact_pack_rows = 0
    for src in range(config.ep_size):
        selected_experts = _selected_experts_from_counts(config, counts[src], src)
        token_ids_sort, _, group_sizes, _ = _prepare_moe_dispatch_indices_with_assignment_ids(
            jnp.asarray(selected_experts, dtype=jnp.int32),
            num_experts=global_experts,
        )
        token_ids_sort_host = np.asarray(jax.device_get(token_ids_sort), dtype=np.int32)
        group_sizes_host = np.asarray(jax.device_get(group_sizes), dtype=np.int32)
        expected_group_sizes = counts[src].reshape(-1)
        if not np.array_equal(group_sizes_host, expected_group_sizes):
            raise ValueError("compact routing group sizes diverged from requested source counts")

        source_tokens = _make_source_tokens(config, src)
        packed_x = source_tokens[token_ids_sort_host]
        compact_pack_rows += int(packed_x.shape[0])
        group_offsets = np.cumsum(np.concatenate((np.array([0], dtype=np.int64), group_sizes_host[:-1])))
        for dst in range(config.ep_size):
            dst_ordinal = _dst_ordinal(config, src, dst)
            entry = 0
            for expert in range(config.experts_per_rank):
                global_expert = dst * config.experts_per_rank + expert
                count = int(group_sizes_host[global_expert])
                block_count = (count + config.block_m - 1) // config.block_m
                group_start = int(group_offsets[global_expert])
                for block in range(block_count):
                    valid_rows = min(config.block_m, count - block * config.block_m)
                    if entry >= config.entries_per_rank:
                        dropped_entries += 1
                        dropped_rows += valid_rows
                        continue
                    dst_row_start = int(expert_base[dst, expert] + src_base[src, dst, expert] + block * config.block_m)
                    send_meta[src, dst_ordinal, entry, :] = (src, expert, dst_row_start, valid_rows)
                    row_start = group_start + block * config.block_m
                    x_host[src, dst_ordinal, entry, :valid_rows, :] = packed_x[row_start : row_start + valid_rows]
                    entry += 1

    recv_meta = _recv_meta_from_send_meta(config, send_meta)
    stats = _queue_stats(config, send_meta)
    stats["input_mode"] = "compact_routing"
    stats["compact_pack_rows_total"] = compact_pack_rows
    stats["dropped_entries_total"] = dropped_entries
    stats["dropped_rows_total"] = dropped_rows
    stats["routing_assignments_per_source"] = config.tokens_per_rank * config.topk
    return HostInputs(
        x=x_host,
        send_meta=send_meta,
        recv_meta=recv_meta,
        expert_base=expert_base,
        src_base_by_expert=_src_base_by_expert_for_dst(src_base),
        queue_stats=stats,
    )


def _source_push_plan_and_packed_x(config: PushInboxConfig) -> tuple[SourcePushPlan, np.ndarray]:
    counts = _routing_counts(config)
    selected_experts = np.stack(
        [_selected_experts_from_counts(config, counts[src], src) for src in range(config.ep_size)],
        axis=0,
    )
    combine_weights = np.ones((config.ep_size, config.tokens_per_rank, config.topk), dtype=np.float32)
    plan = build_source_push_plan(
        jnp.asarray(selected_experts, dtype=jnp.int32),
        jnp.asarray(combine_weights, dtype=jnp.float32),
        ep_size=config.ep_size,
        experts_per_rank=config.experts_per_rank,
        block_m=config.block_m,
        capacity_factor=config.capacity_factor,
        entries_per_dst=config.entries_per_rank,
    )
    source_tokens = np.stack([_make_source_tokens(config, src) for src in range(config.ep_size)], axis=0)
    x_host = np.asarray(pack_source_push_tokens(jnp.asarray(source_tokens, dtype=jnp.float32), plan), dtype=np.float32)
    return plan, x_host


def _add_source_push_plan_queue_stats(
    config: PushInboxConfig,
    stats: dict[str, Any],
    plan: SourcePushPlan,
    *,
    row_start_mode: str,
    row_layout: str,
    layout_rows_total: int,
) -> None:
    plan_stats = source_push_plan_row_stats(plan)
    hidden_capacity_rows_total = config.ep_size * config.hidden_rows_per_rank
    stats.update(
        {
            "input_mode": DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN,
            "row_start_mode": row_start_mode,
            "row_layout": row_layout,
            "dropped_routes": plan_stats.dropped_routes,
            "dropped_entries_total": 0,
            "dropped_rows_total": plan_stats.dropped_routes,
            "routing_assignments_per_source": config.tokens_per_rank * config.topk,
            "compact_pack_rows_total": int(plan_stats.useful_rows),
            "plan_exact_rows_total": plan_stats.useful_rows,
            "plan_useful_rows_total": plan_stats.useful_rows,
            "plan_rounded_rows_total": plan_stats.rounded_rows,
            "plan_live_entries_total": plan_stats.live_entries,
            "plan_row_efficiency": plan_stats.row_efficiency,
            "plan_masked_row_fraction": plan_stats.masked_row_fraction,
            "plan_layout_rows_total": int(layout_rows_total),
            "plan_layout_rows_per_rank_mean": float(layout_rows_total / config.ep_size),
            "plan_layout_padding_rows_total": int(layout_rows_total - plan_stats.useful_rows),
            "plan_layout_padding_fraction": float(
                (layout_rows_total - plan_stats.useful_rows) / max(float(layout_rows_total), 1.0)
            ),
            "hidden_capacity_rows_total": int(hidden_capacity_rows_total),
            "hidden_capacity_rows_per_rank": int(config.hidden_rows_per_rank),
            "hidden_capacity_unused_rows_total": int(hidden_capacity_rows_total - layout_rows_total),
            "hidden_capacity_unused_fraction": float(
                (hidden_capacity_rows_total - layout_rows_total) / max(float(hidden_capacity_rows_total), 1.0)
            ),
        }
    )


def _make_source_push_plan_inputs(config: PushInboxConfig) -> HostInputs:
    """Build source-padded expert-major queue inputs from the invertible plan.

    The exact plan owns the inverse map and source-local row identity. Before
    the hot kernel launches, the host converts each block's local row start into
    a source-padded expert-major row start. That preserves expert-major order
    while giving each source full-block room so Lane/WGMMA stores never clobber
    a following source slice.
    """
    plan, x_host = _source_push_plan_and_packed_x(config)
    rounded_counts, expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, config.block_m)
    send_meta = _source_padded_plan_send_meta(plan, expert_base, src_base_by_expert)
    recv_meta = _recv_meta_from_send_meta(config, send_meta)
    stats = _queue_stats(config, send_meta)
    layout_rows_total = int(np.sum(rounded_counts))
    _add_source_push_plan_queue_stats(
        config,
        stats,
        plan,
        row_start_mode=ROW_START_MODE_SOURCE_PADDED,
        row_layout=ROW_LAYOUT_SOURCE_PADDED_EXPERT_MAJOR,
        layout_rows_total=layout_rows_total,
    )
    stats["plan_padded_rows_total"] = layout_rows_total
    stats["plan_padded_rows_per_rank_mean"] = float(layout_rows_total / config.ep_size)
    return HostInputs(
        x=x_host,
        send_meta=send_meta,
        recv_meta=recv_meta,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        queue_stats=stats,
    )


def _make_exact_source_push_plan_inputs(config: PushInboxConfig) -> HostInputs:
    """Build exact expert-major queue inputs when every live block is full.

    The current W13 kernel stores full `block_m` rows. Exact count-derived bases
    are therefore safe only when each accepted `(src, dst, expert)` run is
    block-aligned; tail blocks must keep using the source-padded layout.
    """
    plan, x_host = _source_push_plan_and_packed_x(config)
    send_meta = np.asarray(jax.device_get(plan.send_meta), dtype=np.int32)
    valid_rows = send_meta[..., 3]
    tail_blocks = (valid_rows > 0) & (valid_rows < config.block_m)
    if np.any(tail_blocks):
        raise ValueError(
            "exact source-push W13 layout requires block_m-aligned live blocks; "
            "use the source-padded layout for tail blocks"
        )

    expert_base = np.asarray(jax.device_get(plan.expert_base), dtype=np.int32)
    src_base_by_expert = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
    recv_meta = np.asarray(jax.device_get(plan.recv_meta), dtype=np.int32)
    stats = _queue_stats(config, send_meta)
    layout_rows_total = int(np.sum(jax.device_get(plan.rows_per_local_expert)))
    _add_source_push_plan_queue_stats(
        config,
        stats,
        plan,
        row_start_mode=ROW_START_MODE_EXACT_EXPERT_MAJOR,
        row_layout=ROW_LAYOUT_EXACT_EXPERT_MAJOR,
        layout_rows_total=layout_rows_total,
    )
    return HostInputs(
        x=x_host,
        send_meta=send_meta,
        recv_meta=recv_meta,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        queue_stats=stats,
        use_exact_expert_major=True,
    )


def _make_host_inputs(config: PushInboxConfig) -> HostInputs:
    host_inputs = _make_routing_inputs(config)
    host_inputs.queue_stats["input_mode"] = "synthetic_blocks"
    return host_inputs


def _device_inputs_from_host(config: PushInboxConfig, host_inputs: HostInputs) -> DeviceInputs:
    w_host = _make_weights(config)
    return DeviceInputs(
        x=jnp.asarray(host_inputs.x, dtype=jnp.bfloat16),
        send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
        expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        w=w_host,
        queue_stats=host_inputs.queue_stats,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def _make_inputs(config: PushInboxConfig) -> DeviceInputs:
    return _device_inputs_from_host(config, _make_host_inputs(config))


def _sharded_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_kernel(config, diagnostic_variant, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        x_local: Float[Array, "1 DST Q M D"],
        send_meta_local: Int[Array, "1 DST Q F"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_local: Float[Array, "1 E D twoI"],
    ):
        x_local = x_local[0]
        send_meta_local = send_meta_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_local = w_local[0]
        inbox, hidden = kernel(
            x_local,
            send_meta_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_local,
        )
        return inbox[None, ...], hidden[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None),
        ),
        check_vma=False,
    )


def _make_w13_h_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    return _make_kernel(
        config,
        DIAGNOSTIC_VARIANT_FULL,
        use_exact_expert_major=use_exact_expert_major,
        output_preactivation_h=True,
    )


def _make_raw_token_w13_h_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    return _make_kernel(
        config,
        DIAGNOSTIC_VARIANT_FULL,
        use_exact_expert_major=use_exact_expert_major,
        output_preactivation_h=True,
        source_input_mode=SOURCE_INPUT_RAW_TOKENS,
    )


def _sharded_w13_h_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_w13_h_kernel(config, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        x_local: Float[Array, "1 DST Q M D"],
        send_meta_local: Int[Array, "1 DST Q F"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_local: Float[Array, "1 E D twoI"],
    ):
        x_local = x_local[0]
        send_meta_local = send_meta_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_local = w_local[0]
        inbox, h = kernel(
            x_local,
            send_meta_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_local,
        )
        return inbox[None, ...], h[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None),
        ),
        check_vma=False,
    )


def _sharded_raw_token_w13_h_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_raw_token_w13_h_kernel(config, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        x_local: Float[Array, "1 T D"],
        token_ids_local: Int[Array, "1 DST Q M"],
        send_meta_local: Int[Array, "1 DST Q F"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_local: Float[Array, "1 E D twoI"],
    ):
        x_local = x_local[0]
        token_ids_local = token_ids_local[0]
        send_meta_local = send_meta_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_local = w_local[0]
        inbox, h = kernel(
            x_local,
            token_ids_local,
            send_meta_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_local,
        )
        return inbox[None, ...], h[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None),
        ),
        check_vma=False,
    )


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)


def _emit_progress(config: PushInboxConfig, progress_events: bool, event: str) -> None:
    if not progress_events:
        return
    print(
        json.dumps(
            {
                "config": asdict(config),
                "event": event,
                "time": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _time_jitted(
    fn,
    *args,
    warmup: int,
    steps: int,
    repeat_runs: int,
    separate_compile: bool,
    progress: Callable[[str], None] | None = None,
) -> TimingResult:
    call_args = tuple(args)

    lower_compile_time = None
    first_run_time = None
    if separate_compile:
        if progress is not None:
            progress("lower_start")
        lowered = fn.lower(*call_args)
        if progress is not None:
            progress("compile_start")
        start = time.perf_counter()
        compiled = lowered.compile()
        lower_compile_time = time.perf_counter() - start
        if progress is not None:
            progress("compile_done")

        if progress is not None:
            progress("first_run_start")
        start = time.perf_counter()
        out = compiled(*call_args)
        _block_until_ready(out)
        first_run_time = time.perf_counter() - start
        compile_time = lower_compile_time + first_run_time
        if progress is not None:
            progress("first_run_done")
    else:
        if progress is not None:
            progress("first_call_start")
        start = time.perf_counter()
        out = fn(*call_args)
        _block_until_ready(out)
        compile_time = time.perf_counter() - start
        if progress is not None:
            progress("first_call_done")

    if progress is not None:
        progress("warmup_start")
    for _ in range(warmup):
        out = fn(*call_args)
        _block_until_ready(out)

    steady_state_times = []
    for _ in range(repeat_runs):
        if progress is not None:
            progress("steady_state_start")
        start = time.perf_counter()
        for _ in range(steps):
            out = fn(*call_args)
            _block_until_ready(out)
        steady_state_times.append((time.perf_counter() - start) / steps)
        if progress is not None:
            progress("steady_state_done")
    return TimingResult(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        lower_compile_time=lower_compile_time,
        first_run_time=first_run_time,
    )


def _metadata_hidden_row_start(
    send_meta: np.ndarray,
    expert_base: np.ndarray,
    src_base_by_expert: np.ndarray,
    *,
    src: int,
    dst: int,
    dst_ordinal: int,
    entry: int,
    use_exact_expert_major: bool,
) -> int:
    if use_exact_expert_major:
        expert = int(send_meta[src, dst_ordinal, entry, 1])
        local_row_start = int(send_meta[src, dst_ordinal, entry, 2])
        return int(expert_base[dst, expert] + src_base_by_expert[dst, src, expert] + local_row_start)
    return int(send_meta[src, dst_ordinal, entry, 2])


def _reference_hidden(
    config: PushInboxConfig,
    x_host,
    send_meta_host,
    w_host,
    expert_base_host=None,
    src_base_by_expert_host=None,
    *,
    use_exact_expert_major: bool = False,
) -> np.ndarray:
    hidden = np.zeros(
        (config.ep_size, config.hidden_rows_per_rank, config.intermediate_dim),
        dtype=np.float32,
    )
    x_float = np.asarray(x_host, dtype=np.float32)
    send_meta = np.asarray(send_meta_host, dtype=np.int32)
    w_float = np.asarray(w_host, dtype=np.float32)
    expert_base = np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base_by_expert = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    if expert_base_host is not None:
        expert_base = np.asarray(expert_base_host, dtype=np.int32)
    if src_base_by_expert_host is not None:
        src_base_by_expert = np.asarray(src_base_by_expert_host, dtype=np.int32)
    for src in range(config.ep_size):
        for dst in range(config.ep_size):
            dst_ordinal = _dst_ordinal(config, src, dst)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                expert = send_meta[src, dst_ordinal, entry, 1]
                row = _metadata_hidden_row_start(
                    send_meta,
                    expert_base,
                    src_base_by_expert,
                    src=src,
                    dst=dst,
                    dst_ordinal=dst_ordinal,
                    entry=entry,
                    use_exact_expert_major=use_exact_expert_major,
                )
                gate = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, : config.intermediate_dim]
                up = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, config.intermediate_dim :]
                block_hidden = gate * (1.0 / (1.0 + np.exp(-gate))) * up
                rows_to_store = valid_rows if use_exact_expert_major else config.block_m
                hidden[dst, row : row + rows_to_store, :] = block_hidden[:rows_to_store]
    return hidden


def _reference_h_flat(
    config: PushInboxConfig,
    x_host,
    send_meta_host,
    w_host,
    expert_base_host=None,
    src_base_by_expert_host=None,
    *,
    use_exact_expert_major: bool = False,
) -> np.ndarray:
    h = np.zeros(
        (config.ep_size, config.hidden_rows_per_rank, 2 * config.intermediate_dim),
        dtype=np.float32,
    )
    x_float = np.asarray(x_host, dtype=np.float32)
    send_meta = np.asarray(send_meta_host, dtype=np.int32)
    w_float = np.asarray(w_host, dtype=np.float32)
    expert_base = np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base_by_expert = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    if expert_base_host is not None:
        expert_base = np.asarray(expert_base_host, dtype=np.int32)
    if src_base_by_expert_host is not None:
        src_base_by_expert = np.asarray(src_base_by_expert_host, dtype=np.int32)
    for src in range(config.ep_size):
        for dst in range(config.ep_size):
            dst_ordinal = _dst_ordinal(config, src, dst)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                expert = send_meta[src, dst_ordinal, entry, 1]
                row = _metadata_hidden_row_start(
                    send_meta,
                    expert_base,
                    src_base_by_expert,
                    src=src,
                    dst=dst,
                    dst_ordinal=dst_ordinal,
                    entry=entry,
                    use_exact_expert_major=use_exact_expert_major,
                )
                h_block = x_float[src, dst_ordinal, entry] @ w_float[dst, expert]
                rows_to_store = valid_rows if use_exact_expert_major else config.block_m
                h[dst, row : row + rows_to_store, :] = h_block[:rows_to_store]
    return h


def _hidden_live_row_mask(
    config: PushInboxConfig,
    send_meta_host,
    expert_base_host=None,
    src_base_by_expert_host=None,
    *,
    use_exact_expert_major: bool = False,
) -> np.ndarray:
    mask = np.zeros((config.ep_size, config.hidden_rows_per_rank), dtype=np.bool_)
    send_meta = np.asarray(send_meta_host, dtype=np.int32)
    expert_base = np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    src_base_by_expert = np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    if expert_base_host is not None:
        expert_base = np.asarray(expert_base_host, dtype=np.int32)
    if src_base_by_expert_host is not None:
        src_base_by_expert = np.asarray(src_base_by_expert_host, dtype=np.int32)
    for src in range(config.ep_size):
        for dst in range(config.ep_size):
            dst_ordinal = _dst_ordinal(config, src, dst)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                row = _metadata_hidden_row_start(
                    send_meta,
                    expert_base,
                    src_base_by_expert,
                    src=src,
                    dst=dst,
                    dst_ordinal=dst_ordinal,
                    entry=entry,
                    use_exact_expert_major=use_exact_expert_major,
                )
                rows_to_store = valid_rows if use_exact_expert_major else config.block_m
                mask[dst, row : row + rows_to_store] = True
    return mask


def _validate(
    config: PushInboxConfig,
    x_host,
    send_meta_host,
    expert_base_host,
    src_base_by_expert_host,
    w_host,
    inbox,
    hidden,
    *,
    use_exact_expert_major: bool = False,
) -> ValidationMetrics:
    inbox_host = np.asarray(inbox, dtype=np.float32)
    x_expected = np.asarray(x_host, dtype=np.float32)
    send_meta_expected = np.asarray(send_meta_host, dtype=np.int32)
    max_abs_diff = 0.0
    metadata_mismatches = 0
    for src in range(config.ep_size):
        for dst in _destination_ranks(config):
            dst_ordinal = _dst_ordinal(config, src, dst)
            live_entries = int(np.sum(send_meta_expected[src, dst_ordinal, :, 3] > 0))
            for slot in range(min(config.inbox_slots, live_entries)):
                entry = slot + ((live_entries - 1 - slot) // config.inbox_slots) * config.inbox_slots
                observed = inbox_host[dst, src, slot, :, :]
                expected = x_expected[src, dst_ordinal, entry, :, :]
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(observed - expected))))
    hidden_max_abs_diff = None
    hidden_mean_abs_diff = None
    hidden_all_max_abs_diff = None
    hidden_unwritten_max_abs = None
    hidden_expected = _reference_hidden(
        config,
        x_host,
        send_meta_host,
        w_host,
        expert_base_host,
        src_base_by_expert_host,
        use_exact_expert_major=use_exact_expert_major,
    )
    hidden_diff = np.abs(np.asarray(hidden, dtype=np.float32) - hidden_expected)
    hidden_all_max_abs_diff = float(np.max(hidden_diff))
    live_row_mask = _hidden_live_row_mask(
        config,
        send_meta_host,
        expert_base_host,
        src_base_by_expert_host,
        use_exact_expert_major=use_exact_expert_major,
    )
    if np.any(live_row_mask):
        hidden_live_diff = hidden_diff[live_row_mask, :]
        hidden_max_abs_diff = float(np.max(hidden_live_diff))
        hidden_mean_abs_diff = float(np.mean(hidden_live_diff))
    else:
        hidden_max_abs_diff = 0.0
        hidden_mean_abs_diff = 0.0
    if np.any(~live_row_mask):
        hidden_unwritten_max_abs = float(np.max(hidden_diff[~live_row_mask, :]))
    else:
        hidden_unwritten_max_abs = 0.0
    max_abs_diff = max(max_abs_diff, hidden_max_abs_diff)
    return ValidationMetrics(
        max_abs_diff=max_abs_diff,
        metadata_mismatches=metadata_mismatches,
        hidden_max_abs_diff=hidden_max_abs_diff,
        hidden_mean_abs_diff=hidden_mean_abs_diff,
        hidden_all_max_abs_diff=hidden_all_max_abs_diff,
        hidden_unwritten_max_abs=hidden_unwritten_max_abs,
    )


def _run_one(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    input_builder: Callable[[PushInboxConfig], HostInputs] = _make_host_inputs,
    *,
    diagnostic_variant: str = DIAGNOSTIC_VARIANT_FULL,
    row_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    row_metadata = row_metadata or {}
    try:
        _emit_progress(config, settings.progress_events, "validate_start")
        if settings.repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {settings.repeat_runs}")
        _validate_diagnostic_variant(diagnostic_variant)
        if settings.check and diagnostic_variant != DIAGNOSTIC_VARIANT_FULL:
            raise ValueError("diagnostic variants other than full do not produce validation-equivalent outputs")
        config.validate()
        _emit_progress(config, settings.progress_events, "mesh_start")
        mesh = _make_mesh(config.ep_size)
        _emit_progress(config, settings.progress_events, "make_inputs_start")
        host_inputs = input_builder(config)
        inputs = _device_inputs_from_host(config, host_inputs)
        _emit_progress(config, settings.progress_events, "device_put_start")
        x = jax.device_put(inputs.x, NamedSharding(mesh, P(AXIS, None, None, None, None)))
        send_meta = jax.device_put(inputs.send_meta, NamedSharding(mesh, P(AXIS, None, None, None)))
        recv_meta = jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None)))
        expert_base = jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None)))
        src_base_by_expert = jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None)))
        w = jax.device_put(inputs.w, NamedSharding(mesh, P(AXIS, None, None, None)))
        _emit_progress(config, settings.progress_events, "jit_start")
        fn = jax.jit(
            _sharded_kernel(
                mesh,
                config,
                diagnostic_variant,
                use_exact_expert_major=inputs.use_exact_expert_major,
            )
        )

        timing = _time_jitted(
            fn,
            x,
            send_meta,
            recv_meta,
            expert_base,
            src_base_by_expert,
            w,
            warmup=settings.warmup,
            steps=settings.steps,
            repeat_runs=settings.repeat_runs,
            separate_compile=settings.separate_compile,
            progress=lambda event: _emit_progress(config, settings.progress_events, event),
        )
        inbox, hidden = timing.output
        max_abs_diff = None
        metadata_mismatches = None
        hidden_max_abs_diff = None
        hidden_mean_abs_diff = None
        hidden_all_max_abs_diff = None
        hidden_unwritten_max_abs = None
        if settings.check:
            validation = _validate(
                config,
                inputs.x,
                inputs.send_meta,
                inputs.expert_base,
                inputs.src_base_by_expert,
                inputs.w,
                inbox,
                hidden,
                use_exact_expert_major=inputs.use_exact_expert_major,
            )
            max_abs_diff = validation.max_abs_diff
            metadata_mismatches = validation.metadata_mismatches
            hidden_max_abs_diff = validation.hidden_max_abs_diff
            hidden_mean_abs_diff = validation.hidden_mean_abs_diff
            hidden_all_max_abs_diff = validation.hidden_all_max_abs_diff
            hidden_unwritten_max_abs = validation.hidden_unwritten_max_abs
        queue_stats = inputs.queue_stats
        bytes_per_rank = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
        rounded_w13_flops_per_rank = (
            queue_stats["rounded_rows_per_rank_mean"] * config.hidden_dim * config.intermediate_dim * 4
        )
        useful_w13_flops_per_rank = (
            queue_stats["valid_rows_per_rank_mean"] * config.hidden_dim * config.intermediate_dim * 4
        )
        rows = []
        for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
            row = {
                "kernel": KERNEL_NAME,
                "implementation": KERNEL_NAME,
                "config": asdict(config),
                "queue_stats": queue_stats,
                **queue_stats,
                "compile_time": timing.compile_time,
                "lower_compile_time": timing.lower_compile_time,
                "first_run_time": timing.first_run_time,
                "repeat_run": repeat_run,
                "repeat_runs": settings.repeat_runs,
                "steady_state_time": steady_state_time,
                "bytes_per_rank": bytes_per_rank,
                "send_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
                "w13_tflops_per_rank": rounded_w13_flops_per_rank / steady_state_time / 1e12,
                "rounded_w13_tflops_per_rank": rounded_w13_flops_per_rank / steady_state_time / 1e12,
                "useful_w13_tflops_per_rank": useful_w13_flops_per_rank / steady_state_time / 1e12,
                "max_abs_diff": max_abs_diff,
                "metadata_mismatches": metadata_mismatches,
                "hidden_max_abs_diff": hidden_max_abs_diff,
                "hidden_mean_abs_diff": hidden_mean_abs_diff,
                "hidden_all_max_abs_diff": hidden_all_max_abs_diff,
                "hidden_unwritten_max_abs": hidden_unwritten_max_abs,
                "error": None,
                "error_type": None,
                "error_message": None,
            }
            row.update(row_metadata)
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 - repro rows should capture unsupported candidates.
        if settings.debug_exceptions:
            raise
        row = {
            "kernel": KERNEL_NAME,
            "implementation": KERNEL_NAME,
            "config": asdict(config),
            "compile_time": None,
            "lower_compile_time": None,
            "first_run_time": None,
            "repeat_run": None,
            "repeat_runs": settings.repeat_runs,
            "steady_state_time": None,
            "bytes_per_rank": None,
            "send_gbps_per_rank": None,
            "w13_tflops_per_rank": None,
            "rounded_w13_tflops_per_rank": None,
            "useful_w13_tflops_per_rank": None,
            "max_abs_diff": None,
            "metadata_mismatches": None,
            "hidden_max_abs_diff": None,
            "hidden_mean_abs_diff": None,
            "hidden_all_max_abs_diff": None,
            "hidden_unwritten_max_abs": None,
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        row.update(row_metadata)
        return [row]
    finally:
        jax.clear_caches()


def run_source_push_inbox(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
) -> list[dict[str, Any]]:
    """Run one package-private source-push inbox benchmark configuration."""
    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_one(config, settings)


def run_source_push_inbox_compact_routing(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
) -> list[dict[str, Any]]:
    """Run the source-push inbox kernel fed by compact expert-sorted routing inputs."""
    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_one(config, settings, input_builder=_make_compact_routing_inputs)


def run_source_push_inbox_source_plan(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
) -> list[dict[str, Any]]:
    """Run the source-push inbox kernel fed by exact invertible SourcePushPlan metadata."""
    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_one(config, settings, input_builder=_make_source_push_plan_inputs)


def run_source_push_inbox_diagnostic(
    config: PushInboxConfig,
    *,
    diagnostic_variant: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
    compact_routing: bool = False,
    input_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Run a non-production diagnostic variant of the source-push inbox kernel."""
    _validate_diagnostic_variant(diagnostic_variant)
    if input_mode is None:
        input_mode = (
            DIAGNOSTIC_INPUT_MODE_COMPACT_ROUTING if compact_routing else DIAGNOSTIC_INPUT_MODE_SYNTHETIC_BLOCKS
        )
    _validate_diagnostic_input_mode(input_mode)
    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=False,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    input_builder_by_mode = {
        DIAGNOSTIC_INPUT_MODE_SYNTHETIC_BLOCKS: _make_host_inputs,
        DIAGNOSTIC_INPUT_MODE_COMPACT_ROUTING: _make_compact_routing_inputs,
        DIAGNOSTIC_INPUT_MODE_SOURCE_PUSH_PLAN: _make_source_push_plan_inputs,
    }
    input_builder = input_builder_by_mode[input_mode]
    return _run_one(
        config,
        settings,
        input_builder=input_builder,
        diagnostic_variant=diagnostic_variant,
        row_metadata={
            "kernel": DIAGNOSTIC_KERNEL_NAME,
            "implementation": f"{DIAGNOSTIC_KERNEL_NAME}:{diagnostic_variant}",
            "diagnostic_variant": diagnostic_variant,
            "diagnostic_compact_routing": compact_routing,
            "diagnostic_input_mode": input_mode,
        },
    )
