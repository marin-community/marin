# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue-shaped source-push inbox prototype for MGPU MoE permute_up.

This is a compact phase-1 repro for the source-push design:

    source local GMEM -> source SMEM -> remote destination GMEM inbox

Each source rank sends a deterministic queue of token blocks to the configured destinations.
The destination owns bounded inbox slots, waits on remote-signaled full
semaphores, optionally computes W13 from the local inbox, and releases slots
back to the source.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref, lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int


AXIS = "expert"
LOWERING_SEMANTICS = {
    "warpgroup": mgpu.LoweringSemantics.Warpgroup,
    "lane": mgpu.LoweringSemantics.Lane,
}
META_FIELDS = 4
BYTES_PER_BF16 = 2
IMPLEMENTATIONS = ("send_only", "m_owner", "m_owner_slots", "m_n_slots")
TRAFFIC_PATTERNS = ("next_rank", "all_to_all")
PEER_LOOP_MODES = ("static", "switch", "dynamic", "grid", "grid_switch")
COMPUTE_PIPELINE_MODES = ("manual", "emit")
QUEUE_MODES = ("rectangular", "routing")
METADATA_MODES = ("remote_slot", "static_recv")
OUTPUT_MODES = ("debug", "perf")
ROUTING_MODES = (
    "balanced",
    "uniform",
    "tail_debug",
    "one_source_one_expert",
    "many_sources_one_expert",
    "many_sources_many_experts",
)
WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _wgmma_transforms(shape: tuple[int, int], dtype: Any, *, lowering_semantics: str):
    if lowering_semantics != "lane":
        return ()
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


def _wgmma_smem(shape: tuple[int, int], dtype: Any, *, lowering_semantics: str):
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=_wgmma_transforms(shape, dtype, lowering_semantics=lowering_semantics),
    )


@dataclass(frozen=True)
class PushInboxConfig:
    implementation: str = "send_only"
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
    num_send_sms: int = 4
    num_sms: int = 16
    lowering_semantics: str = "lane"
    traffic_pattern: str = "next_rank"
    peer_loop: str = "static"
    compute_pipeline: str = "manual"
    max_concurrent_steps: int = 4
    queue_mode: str = "rectangular"
    metadata_mode: str = "static_recv"
    output_mode: str = "debug"
    n_groups_per_job: int = 1
    routing: str = "balanced"
    tokens_per_rank: int = 32768
    topk: int = 4
    routing_seed: int = 0
    direct_self_compute: bool = False

    def validate(self) -> None:
        if self.implementation not in IMPLEMENTATIONS:
            raise ValueError(f"unknown implementation={self.implementation!r}; expected one of {IMPLEMENTATIONS}")
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
        if self.n_group != 1 and self.implementation != "m_n_slots":
            raise ValueError("n_group > 1 is currently implemented only for implementation=m_n_slots")
        if self.experts_per_rank <= 0:
            raise ValueError(f"experts_per_rank must be positive, got {self.experts_per_rank}")
        if self.num_send_sms <= 0:
            raise ValueError(f"num_send_sms must be positive, got {self.num_send_sms}")
        if self.num_sms <= self.num_send_sms:
            raise ValueError(
                "num_sms must leave at least one destination receiver program; "
                f"got {self.num_sms=} {self.num_send_sms=}"
            )
        if self.lowering_semantics not in LOWERING_SEMANTICS:
            raise ValueError(
                f"unknown lowering_semantics={self.lowering_semantics!r}; "
                f"expected one of {sorted(LOWERING_SEMANTICS)}"
            )
        if self.traffic_pattern not in TRAFFIC_PATTERNS:
            raise ValueError(f"unknown traffic_pattern={self.traffic_pattern!r}; expected one of {TRAFFIC_PATTERNS}")
        if self.peer_loop not in PEER_LOOP_MODES:
            raise ValueError(f"unknown peer_loop={self.peer_loop!r}; expected one of {PEER_LOOP_MODES}")
        if self.compute_pipeline not in COMPUTE_PIPELINE_MODES:
            raise ValueError(
                f"unknown compute_pipeline={self.compute_pipeline!r}; expected one of {COMPUTE_PIPELINE_MODES}"
            )
        if self.queue_mode not in QUEUE_MODES:
            raise ValueError(f"unknown queue_mode={self.queue_mode!r}; expected one of {QUEUE_MODES}")
        if self.metadata_mode not in METADATA_MODES:
            raise ValueError(f"unknown metadata_mode={self.metadata_mode!r}; expected one of {METADATA_MODES}")
        if self.metadata_mode == "static_recv" and self.implementation not in ("send_only", "m_n_slots"):
            raise ValueError("metadata_mode=static_recv is currently implemented only for send_only and m_n_slots")
        if self.metadata_mode == "static_recv" and self.peer_loop in ("dynamic", "grid"):
            raise ValueError(
                "metadata_mode=static_recv requires static peer offsets; use static, switch, or grid_switch"
            )
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(f"unknown output_mode={self.output_mode!r}; expected one of {OUTPUT_MODES}")
        if self.output_mode == "perf":
            if self.implementation != "m_n_slots":
                raise ValueError("output_mode=perf is currently implemented only for implementation=m_n_slots")
            if self.metadata_mode != "static_recv":
                raise ValueError("output_mode=perf requires metadata_mode=static_recv")
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
        if self.queue_mode == "routing" and self.traffic_pattern != "all_to_all":
            raise ValueError("queue_mode=routing currently supports traffic_pattern=all_to_all only")
        if self.tokens_per_rank <= 0:
            raise ValueError(f"tokens_per_rank must be positive, got {self.tokens_per_rank}")
        if self.topk <= 0:
            raise ValueError(f"topk must be positive, got {self.topk}")
        if self.max_concurrent_steps <= 0:
            raise ValueError(f"max_concurrent_steps must be positive, got {self.max_concurrent_steps}")
        if self.compute_pipeline == "emit" and self.n_group != 1:
            raise ValueError("compute_pipeline=emit is currently implemented only for n_group=1")
        if self.direct_self_compute:
            if self.implementation != "m_n_slots":
                raise ValueError("direct_self_compute is currently implemented only for implementation=m_n_slots")
            if self.traffic_pattern != "all_to_all":
                raise ValueError("direct_self_compute is currently implemented only for traffic_pattern=all_to_all")
            if self.n_group != 1:
                raise ValueError("direct_self_compute is currently implemented only for n_group=1")
            if self.compute_pipeline != "manual":
                raise ValueError("direct_self_compute is currently implemented only for compute_pipeline=manual")
            if self.peer_loop in ("dynamic", "grid"):
                raise ValueError(
                    "direct_self_compute requires static peer offsets; use static, switch, or grid_switch"
                )

    @property
    def traffic_fanout(self) -> int:
        if self.traffic_pattern == "all_to_all":
            return self.ep_size
        return 1

    @property
    def hidden_rows_per_rank(self) -> int:
        return self.traffic_fanout * self.entries_per_rank * self.block_m

    def host_dst_row_start(self, src: int, entry: int) -> int:
        if self.traffic_pattern == "all_to_all":
            return (src * self.entries_per_rank + entry) * self.block_m
        return entry * self.block_m

    @property
    def hidden_output_shape(self) -> tuple[int, int]:
        if self.implementation == "send_only":
            return (1, 1)
        return (self.hidden_rows_per_rank, self.intermediate_dim)


def _make_kernel(config: PushInboxConfig):
    k_tiles = config.hidden_dim // config.block_k
    n_tiles = config.intermediate_dim // config.block_n
    n_work_groups = n_tiles // config.n_group
    n_compute_jobs = (n_work_groups + config.n_groups_per_job - 1) // config.n_groups_per_job
    rounds_per_slot = (config.entries_per_rank + config.inbox_slots - 1) // config.inbox_slots
    num_compute_sms = config.num_sms - config.num_send_sms
    send_dst_offsets = tuple(range(config.ep_size)) if config.traffic_pattern == "all_to_all" else (1,)
    recv_src_offsets = (
        tuple(range(config.ep_size)) if config.traffic_pattern == "all_to_all" else (config.ep_size - 1,)
    )

    def body(
        x_ref: Float[Ref, "DST Q M D"],
        send_meta_ref: Int[Ref, "DST Q F"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        w_ref: Float[Ref, "E D twoI"],
        inbox_ref: Float[Ref, "SRC SLOTS M D"],
        meta_ref: Int[Ref, "SRC SLOTS F"],
        seen_payload_ref: Float[Ref, "SRC Q"],
        seen_meta_ref: Int[Ref, "SRC Q F"],
        hidden_ref: Float[Ref, "rows I"],
    ) -> None:
        empty_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        full_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        done_sem_ref = pl.get_global(mgpu.SemaphoreType.REGULAR((config.ep_size, config.inbox_slots)))
        rank = lax.axis_index(AXIS)
        if config.peer_loop in ("grid", "grid_switch"):
            peer_ordinal = pl.program_id(0)
            sm = pl.program_id(1)
        else:
            sm = pl.program_id(0)
            peer_ordinal = None

        @pl.when(sm < config.num_send_sms)
        def _send_worker() -> None:

            def _send_to_dynamic_dst(dst_ordinal) -> None:
                if config.traffic_pattern == "all_to_all":
                    dst_offset = dst_ordinal
                else:
                    dst_offset = 1
                dst = (rank + dst_offset) % config.ep_size

                @pl.loop(0, rounds_per_slot)
                def _round_loop(round_i) -> None:
                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot) -> None:
                        send_task = dst_ordinal * config.inbox_slots + slot
                        should_send_slot = (send_task % config.num_send_sms) == sm

                        @pl.when(should_send_slot)
                        def _send_slot_entry() -> None:
                            entry = slot + round_i * config.inbox_slots

                            @pl.when(entry < config.entries_per_rank)
                            def _maybe_send_entry() -> None:
                                valid_rows = send_meta_ref[dst_ordinal, entry, 3]

                                @pl.when(valid_rows > 0)
                                def _send_entry() -> None:
                                    if config.direct_self_compute and dst_offset == 0:
                                        return

                                    expert = send_meta_ref[dst_ordinal, entry, 1]
                                    dst_row_start = send_meta_ref[dst_ordinal, entry, 2]
                                    pl.semaphore_wait(empty_sem_ref.at[dst, slot])

                                    def _copy_scope(tile_smem) -> None:
                                        @pl.loop(0, k_tiles)
                                        def _k_loop(kk) -> None:
                                            k_start = kk * config.block_k
                                            tile_smem[:, :] = x_ref[
                                                dst_ordinal,
                                                entry,
                                                pl.ds(0, config.block_m),
                                                pl.ds(k_start, config.block_k),
                                            ]
                                            mgpu.commit_smem()

                                            @pl.when(dst_offset == 0)
                                            def _copy_local() -> None:
                                                mgpu.copy_smem_to_gmem(
                                                    tile_smem,
                                                    inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )

                                            @pl.when(dst_offset != 0)
                                            def _copy_remote() -> None:
                                                remote_inbox_ref = mgpu.remote_ref(
                                                    inbox_ref,
                                                    dst,
                                                    device_id_type=pl.DeviceIdType.LOGICAL,
                                                )
                                                mgpu.copy_smem_to_gmem(
                                                    tile_smem,
                                                    remote_inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )

                                            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                    pl.run_scoped(
                                        _copy_scope,
                                        tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                    )

                                    @pl.when(dst_offset == 0)
                                    def _write_local_meta() -> None:
                                        meta_ref[rank, slot, 0] = rank
                                        meta_ref[rank, slot, 1] = expert
                                        meta_ref[rank, slot, 2] = dst_row_start
                                        meta_ref[rank, slot, 3] = valid_rows
                                        pl.semaphore_signal(full_sem_ref.at[rank, slot])

                                    @pl.when(dst_offset != 0)
                                    def _write_remote_meta() -> None:
                                        remote_meta_ref = mgpu.remote_ref(
                                            meta_ref,
                                            dst,
                                            device_id_type=pl.DeviceIdType.LOGICAL,
                                        )
                                        remote_meta_ref[rank, slot, 0] = rank
                                        remote_meta_ref[rank, slot, 1] = expert
                                        remote_meta_ref[rank, slot, 2] = dst_row_start
                                        remote_meta_ref[rank, slot, 3] = valid_rows
                                        pl.semaphore_signal(
                                            full_sem_ref.at[rank, slot],
                                            device_id=dst,
                                            device_id_type=pl.DeviceIdType.LOGICAL,
                                        )

            def _send_to_dst(dst_ordinal: int, dst_offset: int) -> None:
                if config.direct_self_compute and dst_offset == 0:
                    return

                dst = (rank + dst_offset) % config.ep_size
                remote_inbox_ref = None
                remote_meta_ref = None
                write_slot_metadata = config.metadata_mode == "remote_slot" or config.implementation == "send_only"
                if dst_offset != 0:
                    remote_inbox_ref = mgpu.remote_ref(inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                    if write_slot_metadata:
                        remote_meta_ref = mgpu.remote_ref(meta_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                @pl.loop(0, rounds_per_slot)
                def _round_loop(round_i) -> None:
                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot) -> None:
                        send_task = dst_ordinal * config.inbox_slots + slot
                        should_send_slot = (send_task % config.num_send_sms) == sm

                        @pl.when(should_send_slot)
                        def _send_slot_entry() -> None:
                            entry = slot + round_i * config.inbox_slots

                            @pl.when(entry < config.entries_per_rank)
                            def _maybe_send_entry() -> None:
                                valid_rows = send_meta_ref[dst_ordinal, entry, 3]

                                @pl.when(valid_rows > 0)
                                def _send_entry() -> None:
                                    expert = send_meta_ref[dst_ordinal, entry, 1]
                                    dst_row_start = send_meta_ref[dst_ordinal, entry, 2]
                                    pl.semaphore_wait(empty_sem_ref.at[dst, slot])

                                    def _copy_scope(tile_smem) -> None:
                                        @pl.loop(0, k_tiles)
                                        def _k_loop(kk) -> None:
                                            k_start = kk * config.block_k
                                            tile_smem[:, :] = x_ref[
                                                dst_ordinal,
                                                entry,
                                                pl.ds(0, config.block_m),
                                                pl.ds(k_start, config.block_k),
                                            ]
                                            mgpu.commit_smem()
                                            if dst_offset == 0:
                                                mgpu.copy_smem_to_gmem(
                                                    tile_smem,
                                                    inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )
                                            else:
                                                mgpu.copy_smem_to_gmem(
                                                    tile_smem,
                                                    remote_inbox_ref.at[
                                                        rank,
                                                        slot,
                                                        pl.ds(0, config.block_m),
                                                        pl.ds(k_start, config.block_k),
                                                    ],
                                                )
                                            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                    pl.run_scoped(
                                        _copy_scope,
                                        tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                    )
                                    if dst_offset == 0:
                                        if write_slot_metadata:
                                            meta_ref[rank, slot, 0] = rank
                                            meta_ref[rank, slot, 1] = expert
                                            meta_ref[rank, slot, 2] = dst_row_start
                                            meta_ref[rank, slot, 3] = valid_rows
                                        pl.semaphore_signal(full_sem_ref.at[rank, slot])
                                    else:
                                        if write_slot_metadata:
                                            remote_meta_ref[rank, slot, 0] = rank
                                            remote_meta_ref[rank, slot, 1] = expert
                                            remote_meta_ref[rank, slot, 2] = dst_row_start
                                            remote_meta_ref[rank, slot, 3] = valid_rows
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

            if config.peer_loop == "grid":
                _send_to_dynamic_dst(peer_ordinal)

            elif config.peer_loop == "grid_switch":
                _switch_send_to_dst(peer_ordinal)

            elif config.peer_loop == "dynamic":

                @pl.loop(0, len(send_dst_offsets))
                def _dst_loop(dst_ordinal) -> None:
                    _send_to_dynamic_dst(dst_ordinal)

            elif config.peer_loop == "switch":

                @pl.loop(0, len(send_dst_offsets))
                def _dst_loop(dst_ordinal) -> None:
                    _switch_send_to_dst(dst_ordinal)

            else:
                for dst_ordinal, dst_offset in enumerate(send_dst_offsets):
                    _send_to_dst(dst_ordinal, dst_offset)

        def _init_empty_slots() -> None:
            def _init_empty_for_dynamic_src(src_ordinal) -> None:
                if config.traffic_pattern == "all_to_all":
                    src_offset = src_ordinal
                else:
                    src_offset = config.ep_size - 1
                src = (rank + src_offset) % config.ep_size

                @pl.loop(0, config.inbox_slots)
                def _init_empty_slot(slot) -> None:
                    @pl.when(src_offset == 0)
                    def _signal_local() -> None:
                        pl.semaphore_signal(empty_sem_ref.at[rank, slot])

                    @pl.when(src_offset != 0)
                    def _signal_remote() -> None:
                        pl.semaphore_signal(
                            empty_sem_ref.at[rank, slot],
                            device_id=src,
                            device_id_type=pl.DeviceIdType.LOGICAL,
                        )

            def _init_empty_for_src(src_offset: int) -> None:
                if config.direct_self_compute and src_offset == 0:
                    return

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

            if config.peer_loop == "grid":
                _init_empty_for_dynamic_src(peer_ordinal)

            elif config.peer_loop == "grid_switch":
                _switch_init_empty_for_src(peer_ordinal)

            elif config.peer_loop == "dynamic":

                @pl.loop(0, len(recv_src_offsets))
                def _src_loop(src_ordinal) -> None:
                    _init_empty_for_dynamic_src(src_ordinal)

            elif config.peer_loop == "switch":

                @pl.loop(0, len(recv_src_offsets))
                def _src_loop(src_ordinal) -> None:
                    _switch_init_empty_for_src(src_ordinal)

            else:
                for src_offset in recv_src_offsets:
                    _init_empty_for_src(src_offset)

        def _signal_empty_to_src(src, src_offset: int, slot) -> None:
            if config.peer_loop in ("dynamic", "grid"):

                @pl.when(src_offset == 0)
                def _signal_local() -> None:
                    pl.semaphore_signal(empty_sem_ref.at[rank, slot])

                @pl.when(src_offset != 0)
                def _signal_remote() -> None:
                    pl.semaphore_signal(
                        empty_sem_ref.at[rank, slot],
                        device_id=src,
                        device_id_type=pl.DeviceIdType.LOGICAL,
                    )

                return

            if src_offset == 0:
                pl.semaphore_signal(empty_sem_ref.at[rank, slot])
            else:
                pl.semaphore_signal(
                    empty_sem_ref.at[rank, slot],
                    device_id=src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )

        def _compute_hidden_n_tile(src, slot, expert, dst_row_start, n_tile) -> None:
            def acc_scope(gate_acc_ref, up_acc_ref) -> jax.Array:
                if config.compute_pipeline == "emit":

                    def wgmma_step(_, lhs_smem, gate_smem, up_smem) -> None:
                        mgpu.wgmma(gate_acc_ref, lhs_smem, gate_smem)
                        mgpu.wgmma(up_acc_ref, lhs_smem, up_smem)

                    mgpu.emit_pipeline(
                        wgmma_step,
                        grid=(k_tiles,),
                        in_specs=[
                            mgpu.BlockSpec(
                                (config.block_m, config.block_k),
                                lambda kk: (0, kk),
                                delay_release=1,
                                transforms=_wgmma_transforms(
                                    (config.block_m, config.block_k),
                                    inbox_ref.dtype,
                                    lowering_semantics=config.lowering_semantics,
                                ),
                            ),
                            mgpu.BlockSpec(
                                (config.block_k, config.block_n),
                                lambda kk: (kk, n_tile),
                                delay_release=1,
                                transforms=_wgmma_transforms(
                                    (config.block_k, config.block_n),
                                    w_ref.dtype,
                                    lowering_semantics=config.lowering_semantics,
                                ),
                            ),
                            mgpu.BlockSpec(
                                (config.block_k, config.block_n),
                                lambda kk: (kk, n_tile + n_tiles),
                                delay_release=1,
                                transforms=_wgmma_transforms(
                                    (config.block_k, config.block_n),
                                    w_ref.dtype,
                                    lowering_semantics=config.lowering_semantics,
                                ),
                            ),
                        ],
                        max_concurrent_steps=config.max_concurrent_steps,
                    )(
                        inbox_ref.at[src, slot],
                        w_ref.at[expert],
                        w_ref.at[expert],
                    )
                    return _silu(gate_acc_ref[...]) * up_acc_ref[...]

                def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier) -> None:
                    @pl.loop(0, k_tiles)
                    def _k_loop(kk) -> None:
                        k_start = kk * config.block_k
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
                    lhs_smem=_wgmma_smem(
                        (config.block_m, config.block_k),
                        inbox_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    gate_smem=_wgmma_smem(
                        (config.block_k, config.block_n),
                        w_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    up_smem=_wgmma_smem(
                        (config.block_k, config.block_n),
                        w_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    ready_barrier=mgpu.Barrier(num_arrivals=3),
                )
                return _silu(gate_acc_ref[...]) * up_acc_ref[...]

            hidden = pl.run_scoped(
                acc_scope,
                gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
            )
            hidden_ref[
                pl.ds(dst_row_start, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            ] = hidden.astype(hidden_ref.dtype)

        def _compute_hidden_n_tile_from_x(dst_ordinal, entry, expert, dst_row_start, n_tile) -> None:
            def acc_scope(gate_acc_ref, up_acc_ref) -> jax.Array:
                def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier) -> None:
                    @pl.loop(0, k_tiles)
                    def _k_loop(kk) -> None:
                        k_start = kk * config.block_k
                        mgpu.copy_gmem_to_smem(
                            x_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                            lhs_smem,
                            ready_barrier,
                        )
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
                    lhs_smem=_wgmma_smem(
                        (config.block_m, config.block_k),
                        x_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    gate_smem=_wgmma_smem(
                        (config.block_k, config.block_n),
                        w_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    up_smem=_wgmma_smem(
                        (config.block_k, config.block_n),
                        w_ref.dtype,
                        lowering_semantics=config.lowering_semantics,
                    ),
                    ready_barrier=mgpu.Barrier(num_arrivals=3),
                )
                return _silu(gate_acc_ref[...]) * up_acc_ref[...]

            hidden = pl.run_scoped(
                acc_scope,
                gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
            )
            hidden_ref[
                pl.ds(dst_row_start, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            ] = hidden.astype(hidden_ref.dtype)

        def _compute_hidden_n_group(src, slot, expert, dst_row_start, n_group_i) -> None:
            if config.n_group == 1:
                _compute_hidden_n_tile(src, slot, expert, dst_row_start, n_group_i)
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
                        lhs_smem=_wgmma_smem(
                            (config.block_m, config.block_k),
                            inbox_ref.dtype,
                            lowering_semantics=config.lowering_semantics,
                        ),
                        gate_n0_smem=_wgmma_smem(
                            (config.block_k, config.block_n),
                            w_ref.dtype,
                            lowering_semantics=config.lowering_semantics,
                        ),
                        up_n0_smem=_wgmma_smem(
                            (config.block_k, config.block_n),
                            w_ref.dtype,
                            lowering_semantics=config.lowering_semantics,
                        ),
                        gate_n1_smem=_wgmma_smem(
                            (config.block_k, config.block_n),
                            w_ref.dtype,
                            lowering_semantics=config.lowering_semantics,
                        ),
                        up_n1_smem=_wgmma_smem(
                            (config.block_k, config.block_n),
                            w_ref.dtype,
                            lowering_semantics=config.lowering_semantics,
                        ),
                        ready_barrier=mgpu.Barrier(num_arrivals=5),
                    )
                    hidden_n0 = _silu(gate_n0_acc[...]) * up_n0_acc[...]
                    hidden_n1 = _silu(gate_n1_acc[...]) * up_n1_acc[...]
                    hidden_ref[
                        pl.ds(dst_row_start, config.block_m),
                        pl.ds(n_tile * config.block_n, config.block_n),
                    ] = hidden_n0.astype(hidden_ref.dtype)
                    hidden_ref[
                        pl.ds(dst_row_start, config.block_m),
                        pl.ds((n_tile + 1) * config.block_n, config.block_n),
                    ] = hidden_n1.astype(hidden_ref.dtype)

                pl.run_scoped(
                    acc_scope,
                    gate_n0_acc=mgpu.ACC((config.block_m, config.block_n)),
                    up_n0_acc=mgpu.ACC((config.block_m, config.block_n)),
                    gate_n1_acc=mgpu.ACC((config.block_m, config.block_n)),
                    up_n1_acc=mgpu.ACC((config.block_m, config.block_n)),
                )

        def _consume_entry(src, src_offset: int, entry, slot) -> None:
            pl.semaphore_wait(full_sem_ref.at[src, slot])
            expert = meta_ref[src, slot, 1]
            dst_row_start = meta_ref[src, slot, 2]

            if config.implementation in ("m_owner", "m_owner_slots"):

                @pl.loop(0, n_tiles)
                def _n_tile_loop(n_tile) -> None:
                    _compute_hidden_n_tile(src, slot, expert, dst_row_start, n_tile)

            else:
                seen_payload_ref[src, entry] = inbox_ref[src, slot, 0, 0]
                seen_meta_ref[src, entry, 0] = meta_ref[src, slot, 0]
                seen_meta_ref[src, entry, 1] = meta_ref[src, slot, 1]
                seen_meta_ref[src, entry, 2] = meta_ref[src, slot, 2]
                seen_meta_ref[src, entry, 3] = meta_ref[src, slot, 3]
            _signal_empty_to_src(src, src_offset, slot)

        if config.implementation == "m_n_slots":

            @pl.when(sm == config.num_send_sms)
            def _init_worker() -> None:
                _init_empty_slots()

            @pl.when(sm >= config.num_send_sms)
            def _n_tile_recv_worker() -> None:
                compute_worker = sm - config.num_send_sms

                def _recv_n_tile_dynamic_src(src_ordinal) -> None:
                    if config.traffic_pattern == "all_to_all":
                        src_offset = src_ordinal
                    else:
                        src_offset = config.ep_size - 1
                    _recv_n_tile_src(src_ordinal, src_offset)

                def _recv_n_tile_src(src_ordinal: int, src_offset: int) -> None:
                    src = (rank + src_offset) % config.ep_size

                    if config.direct_self_compute and src_offset == 0:

                        @pl.loop(0, rounds_per_slot)
                        def _self_round_loop(round_i) -> None:
                            @pl.loop(0, config.inbox_slots)
                            def _self_slot_loop(slot) -> None:
                                entry = slot + round_i * config.inbox_slots

                                @pl.when(entry < config.entries_per_rank)
                                def _maybe_self_entry() -> None:
                                    valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                    @pl.when(valid_rows > 0)
                                    def _self_entry() -> None:
                                        @pl.loop(0, n_compute_jobs)
                                        def _self_job_loop(job_i) -> None:
                                            work_group = (
                                                src_ordinal * config.inbox_slots + slot
                                            ) * n_compute_jobs + job_i
                                            should_compute = (work_group % num_compute_sms) == compute_worker

                                            @pl.when(should_compute)
                                            def _self_job() -> None:
                                                expert = recv_meta_ref[src_ordinal, entry, 1]
                                                dst_row_start = recv_meta_ref[src_ordinal, entry, 2]

                                                @pl.loop(0, config.n_groups_per_job)
                                                def _self_job_n_group_loop(group_offset) -> None:
                                                    n_group_i = job_i * config.n_groups_per_job + group_offset

                                                    @pl.when(n_group_i < n_work_groups)
                                                    def _self_job_n_group() -> None:
                                                        _compute_hidden_n_tile_from_x(
                                                            src_ordinal,
                                                            entry,
                                                            expert,
                                                            dst_row_start,
                                                            n_group_i,
                                                        )

                        return

                    @pl.loop(0, rounds_per_slot)
                    def _round_loop(round_i) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _slot_loop(slot) -> None:
                            entry = slot + round_i * config.inbox_slots

                            @pl.when(entry < config.entries_per_rank)
                            def _maybe_recv_slot_entry() -> None:
                                valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                @pl.when(valid_rows > 0)
                                def _recv_slot_entry() -> None:
                                    @pl.loop(0, n_compute_jobs)
                                    def _job_loop(job_i) -> None:
                                        work_group = (src_ordinal * config.inbox_slots + slot) * n_compute_jobs + job_i
                                        should_compute = (work_group % num_compute_sms) == compute_worker

                                        @pl.when(should_compute)
                                        def _recv_job() -> None:
                                            pl.semaphore_wait(
                                                full_sem_ref.at[src, slot],
                                                value=round_i + 1,
                                                decrement=False,
                                            )
                                            if config.metadata_mode == "static_recv":
                                                expert = recv_meta_ref[src_ordinal, entry, 1]
                                                dst_row_start = recv_meta_ref[src_ordinal, entry, 2]
                                            else:
                                                expert = meta_ref[src, slot, 1]
                                                dst_row_start = meta_ref[src, slot, 2]

                                            @pl.loop(0, config.n_groups_per_job)
                                            def _job_n_group_loop(group_offset) -> None:
                                                n_group_i = job_i * config.n_groups_per_job + group_offset

                                                @pl.when(n_group_i < n_work_groups)
                                                def _job_n_group() -> None:
                                                    _compute_hidden_n_group(
                                                        src,
                                                        slot,
                                                        expert,
                                                        dst_row_start,
                                                        n_group_i,
                                                    )

                                            pl.semaphore_signal(done_sem_ref.at[src, slot])

                                    release_group = src_ordinal * config.inbox_slots + slot
                                    should_release = (release_group % num_compute_sms) == compute_worker

                                    @pl.when(should_release)
                                    def _release_slot() -> None:
                                        pl.semaphore_wait(
                                            done_sem_ref.at[src, slot],
                                            value=(round_i + 1) * n_compute_jobs,
                                            decrement=False,
                                        )
                                        _signal_empty_to_src(src, src_offset, slot)

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

                if config.peer_loop == "grid":
                    _recv_n_tile_dynamic_src(peer_ordinal)

                elif config.peer_loop == "grid_switch":
                    _switch_recv_n_tile_src(peer_ordinal)

                elif config.peer_loop == "dynamic":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _recv_n_tile_dynamic_src(src_ordinal)

                elif config.peer_loop == "switch":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _switch_recv_n_tile_src(src_ordinal)

                else:
                    for src_ordinal, src_offset in enumerate(recv_src_offsets):
                        _recv_n_tile_src(src_ordinal, src_offset)

        elif config.implementation == "m_owner_slots":

            @pl.when(sm == config.num_send_sms)
            def _init_worker() -> None:
                _init_empty_slots()

            @pl.when(sm >= config.num_send_sms)
            def _slot_recv_worker() -> None:
                compute_worker = sm - config.num_send_sms

                def _recv_slot_dynamic_src(src_ordinal) -> None:
                    if config.traffic_pattern == "all_to_all":
                        src_offset = src_ordinal
                    else:
                        src_offset = config.ep_size - 1
                    _recv_slot_src(src_ordinal, src_offset)

                def _recv_slot_src(src_ordinal: int, src_offset: int) -> None:
                    src = (rank + src_offset) % config.ep_size

                    @pl.loop(0, rounds_per_slot)
                    def _round_loop(round_i) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _slot_loop(slot) -> None:
                            entry = slot + round_i * config.inbox_slots
                            recv_group = src_ordinal * config.inbox_slots + slot
                            should_recv = (recv_group % num_compute_sms == compute_worker) & (
                                entry < config.entries_per_rank
                            )

                            @pl.when(should_recv)
                            def _maybe_recv_slot_entry() -> None:
                                valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                @pl.when(valid_rows > 0)
                                def _recv_slot_entry() -> None:
                                    _consume_entry(src, src_offset, entry, slot)

                def _switch_recv_slot_src(src_ordinal) -> None:
                    def _branch(static_src_ordinal: int, static_src_offset: int):
                        def _recv_branch(_) -> None:
                            _recv_slot_src(static_src_ordinal, static_src_offset)

                        return _recv_branch

                    branches = tuple(
                        _branch(static_src_ordinal, static_src_offset)
                        for static_src_ordinal, static_src_offset in enumerate(recv_src_offsets)
                    )
                    lax.switch(src_ordinal, branches, None)

                if config.peer_loop == "grid":
                    _recv_slot_dynamic_src(peer_ordinal)

                elif config.peer_loop == "grid_switch":
                    _switch_recv_slot_src(peer_ordinal)

                elif config.peer_loop == "dynamic":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _recv_slot_dynamic_src(src_ordinal)

                elif config.peer_loop == "switch":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _switch_recv_slot_src(src_ordinal)

                else:
                    for src_ordinal, src_offset in enumerate(recv_src_offsets):
                        _recv_slot_src(src_ordinal, src_offset)

        else:

            @pl.when(sm == config.num_send_sms)
            def _recv_worker() -> None:
                _init_empty_slots()

                def _recv_dynamic_src(src_ordinal) -> None:
                    if config.traffic_pattern == "all_to_all":
                        src_offset = src_ordinal
                    else:
                        src_offset = config.ep_size - 1
                    _recv_src(src_offset)

                def _recv_src(src_offset: int) -> None:
                    src = (rank + src_offset) % config.ep_size
                    if config.traffic_pattern == "all_to_all":
                        src_ordinal = src_offset
                    else:
                        src_ordinal = 0

                    @pl.loop(0, config.entries_per_rank)
                    def _recv_entry(entry) -> None:
                        valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                        @pl.when(valid_rows > 0)
                        def _recv_live_entry() -> None:
                            slot = entry % config.inbox_slots
                            _consume_entry(src, src_offset, entry, slot)

                def _switch_recv_src(src_ordinal) -> None:
                    def _branch(static_src_offset: int):
                        def _recv_branch(_) -> None:
                            _recv_src(static_src_offset)

                        return _recv_branch

                    branches = tuple(_branch(static_src_offset) for static_src_offset in recv_src_offsets)
                    lax.switch(src_ordinal, branches, None)

                if config.peer_loop == "grid":
                    _recv_dynamic_src(peer_ordinal)

                elif config.peer_loop == "grid_switch":
                    _switch_recv_src(peer_ordinal)

                elif config.peer_loop == "dynamic":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _recv_dynamic_src(src_ordinal)

                elif config.peer_loop == "switch":

                    @pl.loop(0, len(recv_src_offsets))
                    def _src_loop(src_ordinal) -> None:
                        _switch_recv_src(src_ordinal)

                else:
                    for src_offset in recv_src_offsets:
                        _recv_src(src_offset)

    if config.output_mode == "perf":
        meta_shape = (1, 1, META_FIELDS)
        seen_payload_shape = (1, 1)
        seen_meta_shape = (1, 1, META_FIELDS)
    else:
        meta_shape = (config.ep_size, config.inbox_slots, META_FIELDS)
        seen_payload_shape = (config.ep_size, config.entries_per_rank)
        seen_meta_shape = (config.ep_size, config.entries_per_rank, META_FIELDS)

    return mgpu.kernel(
        body,
        out_shape=[
            jax.ShapeDtypeStruct(
                (config.ep_size, config.inbox_slots, config.block_m, config.hidden_dim),
                jnp.bfloat16,
            ),
            jax.ShapeDtypeStruct(meta_shape, jnp.int32),
            jax.ShapeDtypeStruct(seen_payload_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(seen_meta_shape, jnp.int32),
            jax.ShapeDtypeStruct(config.hidden_output_shape, jnp.bfloat16),
        ],
        grid=(
            (len(send_dst_offsets), config.num_sms)
            if config.peer_loop in ("grid", "grid_switch")
            else (config.num_sms,)
        ),
        grid_names=("peer_phase", "sm") if config.peer_loop in ("grid", "grid_switch") else ("sm",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=LOWERING_SEMANTICS[config.lowering_semantics]),
    )


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,))


def _destination_ranks(config: PushInboxConfig, src: int):
    if config.traffic_pattern == "all_to_all":
        return range(config.ep_size)
    return ((src + 1) % config.ep_size,)


def _dst_ordinal(config: PushInboxConfig, src: int, dst: int) -> int:
    if config.traffic_pattern == "all_to_all":
        return (dst - src) % config.ep_size
    return 0


def _recv_src_ordinal(config: PushInboxConfig, dst: int, src: int) -> int:
    if config.traffic_pattern == "all_to_all":
        return (src - dst) % config.ep_size
    return 0


def _make_weights(config: PushInboxConfig):
    if config.implementation == "send_only":
        return jnp.zeros((config.ep_size, 1, 1, 1), dtype=jnp.bfloat16)

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
        for src in _destination_ranks(config, dst):
            send_dst_ordinal = _dst_ordinal(config, src, dst)
            recv_src_ordinal = _recv_src_ordinal(config, dst, src)
            recv_meta[dst, recv_src_ordinal, :, :] = send_meta[src, send_dst_ordinal, :, :]
    return recv_meta


def _queue_stats(config: PushInboxConfig, send_meta: np.ndarray) -> dict[str, Any]:
    valid_rows = send_meta[:, :, :, 3]
    live = valid_rows > 0
    live_entries_per_rank = np.sum(live, axis=(1, 2))
    direct_self_entries_per_rank = (
        np.sum(live[:, 0, :], axis=1)
        if config.direct_self_compute and config.traffic_pattern == "all_to_all"
        else np.zeros((config.ep_size,), dtype=np.int64)
    )
    send_entries_per_rank = live_entries_per_rank - direct_self_entries_per_rank
    valid_rows_per_rank = np.sum(valid_rows, axis=(1, 2))
    rounded_rows_per_rank = live_entries_per_rank * config.block_m
    send_rounded_rows_per_rank = send_entries_per_rank * config.block_m
    entries_per_pair = np.sum(live, axis=2)
    entries_by_dst = np.zeros((config.ep_size,), dtype=np.int64)
    entries_by_local_expert = np.zeros((config.experts_per_rank,), dtype=np.int64)
    entries_by_global_expert = np.zeros((config.ep_size * config.experts_per_rank,), dtype=np.int64)
    entries_by_source_destination_expert = np.zeros(
        (config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int64
    )
    max_slot_reuse_depth = 0
    for src in range(config.ep_size):
        for dst in _destination_ranks(config, src):
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
    if config.implementation == "m_n_slots":
        compute_wait_full_count = payload_send_entries_total * n_compute_jobs
    elif config.implementation in ("m_owner", "m_owner_slots"):
        compute_wait_full_count = payload_send_entries_total
    else:
        compute_wait_full_count = payload_send_entries_total
    capacity_entries_total = config.ep_size * config.traffic_fanout * config.entries_per_rank
    nonzero_expert_entries = entries_by_source_destination_expert[entries_by_source_destination_expert > 0]
    if nonzero_expert_entries.size:
        per_expert_pair_min_nonzero = int(np.min(nonzero_expert_entries))
        per_expert_pair_max = int(np.max(nonzero_expert_entries))
    else:
        per_expert_pair_min_nonzero = 0
        per_expert_pair_max = 0
    return {
        "queue_mode": config.queue_mode,
        "routing": config.routing,
        "metadata_mode": config.metadata_mode,
        "direct_self_compute": config.direct_self_compute,
        "output_mode": config.output_mode,
        "n_groups_per_job": config.n_groups_per_job,
        "n_work_groups_per_entry": n_work_groups,
        "num_compute_jobs_per_entry": n_compute_jobs,
        "done_signals_per_entry": n_compute_jobs,
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
        "valid_rows_per_rank_min": int(np.min(valid_rows_per_rank)),
        "valid_rows_per_rank_mean": float(np.mean(valid_rows_per_rank)),
        "valid_rows_per_rank_max": int(np.max(valid_rows_per_rank)),
        "rounded_rows_per_rank_min": int(np.min(rounded_rows_per_rank)),
        "rounded_rows_per_rank_mean": float(np.mean(rounded_rows_per_rank)),
        "rounded_rows_per_rank_max": int(np.max(rounded_rows_per_rank)),
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


def _make_rectangular_inputs(config: PushInboxConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x_host = np.zeros(
        (config.ep_size, config.traffic_fanout, config.entries_per_rank, config.block_m, config.hidden_dim),
        dtype=np.float32,
    )
    send_meta = np.zeros((config.ep_size, config.traffic_fanout, config.entries_per_rank, META_FIELDS), dtype=np.int32)
    for src in range(config.ep_size):
        for dst in _destination_ranks(config, src):
            dst_ordinal = _dst_ordinal(config, src, dst)
            for entry in range(config.entries_per_rank):
                expert = entry % config.experts_per_rank
                dst_row_start = config.host_dst_row_start(src, entry)
                send_meta[src, dst_ordinal, entry, :] = (src, expert, dst_row_start, config.block_m)
                _fill_block(x_host, src, dst_ordinal, entry, config.block_m)

    recv_meta = _recv_meta_from_send_meta(config, send_meta)
    stats = _queue_stats(config, send_meta)
    stats["dropped_entries_total"] = 0
    stats["dropped_rows_total"] = 0
    return x_host, send_meta, recv_meta, stats


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
    if config.routing == "balanced":
        base = assignments // global_experts
        remainder = assignments % global_experts
        flat = np.full((global_experts,), base, dtype=np.int32)
        flat[:remainder] += 1
        counts[:, :, :] = flat.reshape(config.ep_size, config.experts_per_rank)[None, :, :]
        return counts

    rng = np.random.default_rng(config.routing_seed)
    probs = np.full((global_experts,), 1.0 / global_experts, dtype=np.float64)
    for src in range(config.ep_size):
        counts[src, :, :] = rng.multinomial(assignments, probs).reshape(config.ep_size, config.experts_per_rank)
    return counts


def _make_routing_inputs(config: PushInboxConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    counts = _routing_counts(config)
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
    return x_host, send_meta, recv_meta, stats


def _make_inputs(config: PushInboxConfig):
    if config.queue_mode == "routing":
        x_host, send_meta, recv_meta, stats = _make_routing_inputs(config)
    else:
        x_host, send_meta, recv_meta, stats = _make_rectangular_inputs(config)
    w_host = _make_weights(config)
    return (
        jnp.asarray(x_host, dtype=jnp.bfloat16),
        jnp.asarray(send_meta, dtype=jnp.int32),
        jnp.asarray(recv_meta, dtype=jnp.int32),
        w_host,
        stats,
    )


def _sharded_kernel(mesh: Mesh, config: PushInboxConfig):
    kernel = _make_kernel(config)

    def local_fn(
        x_local: Float[Array, "1 DST Q M D"],
        send_meta_local: Int[Array, "1 DST Q F"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        w_local: Float[Array, "1 E D twoI"],
    ):
        x_local = x_local[0]
        send_meta_local = send_meta_local[0]
        recv_meta_local = recv_meta_local[0]
        w_local = w_local[0]
        inbox, meta, seen_payload, seen_meta, hidden = kernel(x_local, send_meta_local, recv_meta_local, w_local)
        return inbox[None, ...], meta[None, ...], seen_payload[None, ...], seen_meta[None, ...], hidden[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
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
    separate_compile: bool,
    progress: Callable[[str], None] | None = None,
):
    lower_compile_time = None
    first_run_time = None
    if separate_compile:
        if progress is not None:
            progress("lower_start")
        lowered = fn.lower(*args)
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
        out = compiled(*args)
        _block_until_ready(out)
        first_run_time = time.perf_counter() - start
        compile_time = lower_compile_time + first_run_time
        if progress is not None:
            progress("first_run_done")
    else:
        if progress is not None:
            progress("first_call_start")
        start = time.perf_counter()
        out = fn(*args)
        _block_until_ready(out)
        compile_time = time.perf_counter() - start
        if progress is not None:
            progress("first_call_done")

    if progress is not None:
        progress("warmup_start")
    for _ in range(warmup):
        out = fn(*args)
        _block_until_ready(out)
    if progress is not None:
        progress("steady_state_start")
    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        _block_until_ready(out)
    steady_state_time = (time.perf_counter() - start) / steps
    if progress is not None:
        progress("steady_state_done")
    return compile_time, steady_state_time, out, lower_compile_time, first_run_time


def _reference_hidden(config: PushInboxConfig, x_host, send_meta_host, w_host) -> np.ndarray:
    hidden = np.zeros(
        (config.ep_size, config.hidden_rows_per_rank, config.intermediate_dim),
        dtype=np.float32,
    )
    x_float = np.asarray(x_host, dtype=np.float32)
    send_meta = np.asarray(send_meta_host, dtype=np.int32)
    w_float = np.asarray(w_host, dtype=np.float32)
    for src in range(config.ep_size):
        dsts = range(config.ep_size) if config.traffic_pattern == "all_to_all" else ((src + 1) % config.ep_size,)
        for dst in dsts:
            dst_ordinal = _dst_ordinal(config, src, dst)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                expert = send_meta[src, dst_ordinal, entry, 1]
                row = send_meta[src, dst_ordinal, entry, 2]
                gate = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, : config.intermediate_dim]
                up = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, config.intermediate_dim :]
                hidden[dst, row : row + config.block_m, :] = gate * (1.0 / (1.0 + np.exp(-gate))) * up
    return hidden


def _validate(
    config: PushInboxConfig,
    x_host,
    send_meta_host,
    w_host,
    inbox,
    meta,
    seen_payload,
    seen_meta,
    hidden,
) -> dict[str, Any]:
    inbox_host = np.asarray(inbox, dtype=np.float32)
    meta_host = np.asarray(meta, dtype=np.int32)
    seen_payload_host = np.asarray(seen_payload, dtype=np.float32)
    seen_meta_host = np.asarray(seen_meta, dtype=np.int32)
    x_expected = np.asarray(x_host, dtype=np.float32)
    send_meta_expected = np.asarray(send_meta_host, dtype=np.int32)
    max_abs_diff = 0.0
    metadata_mismatches = 0
    for src in range(config.ep_size):
        for dst in _destination_ranks(config, src):
            dst_ordinal = _dst_ordinal(config, src, dst)
            live_entries = int(np.sum(send_meta_expected[src, dst_ordinal, :, 3] > 0))
            if config.implementation == "send_only":
                for entry in range(config.entries_per_rank):
                    expected_meta = send_meta_expected[src, dst_ordinal, entry, :]
                    if expected_meta[3] <= 0:
                        continue
                    expected_payload = x_expected[src, dst_ordinal, entry, 0, 0]
                    max_abs_diff = max(
                        max_abs_diff,
                        float(np.max(np.abs(seen_payload_host[dst, src, entry] - expected_payload))),
                    )
                    metadata_mismatches += int(np.sum(seen_meta_host[dst, src, entry, :] != expected_meta))
            if config.direct_self_compute and src == dst:
                continue
            for slot in range(min(config.inbox_slots, live_entries)):
                entry = slot + ((live_entries - 1 - slot) // config.inbox_slots) * config.inbox_slots
                observed = inbox_host[dst, src, slot, :, :]
                expected = x_expected[src, dst_ordinal, entry, :, :]
                max_abs_diff = max(max_abs_diff, float(np.max(np.abs(observed - expected))))
                if config.metadata_mode == "remote_slot":
                    expected_meta = send_meta_expected[src, dst_ordinal, entry, :]
                    metadata_mismatches += int(np.sum(meta_host[dst, src, slot, :] != expected_meta))
    hidden_max_abs_diff = None
    hidden_mean_abs_diff = None
    if config.implementation in ("m_owner", "m_owner_slots", "m_n_slots"):
        hidden_expected = _reference_hidden(config, x_host, send_meta_host, w_host)
        hidden_diff = np.abs(np.asarray(hidden, dtype=np.float32) - hidden_expected)
        hidden_max_abs_diff = float(np.max(hidden_diff))
        hidden_mean_abs_diff = float(np.mean(hidden_diff))
        max_abs_diff = max(max_abs_diff, hidden_max_abs_diff)
    return {
        "max_abs_diff": max_abs_diff,
        "metadata_mismatches": metadata_mismatches,
        "hidden_max_abs_diff": hidden_max_abs_diff,
        "hidden_mean_abs_diff": hidden_mean_abs_diff,
    }


def _run_one(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    check: bool,
    debug_exceptions: bool,
    separate_compile: bool,
    progress_events: bool,
) -> dict[str, Any]:
    try:
        _emit_progress(config, progress_events, "validate_start")
        config.validate()
        if check and config.output_mode == "perf":
            raise ValueError("output_mode=perf uses tiny debug outputs and requires --no-check")
        _emit_progress(config, progress_events, "mesh_start")
        mesh = _make_mesh(config.ep_size)
        _emit_progress(config, progress_events, "make_inputs_start")
        x_host, send_meta_host, recv_meta_host, w_host, queue_stats = _make_inputs(config)
        _emit_progress(config, progress_events, "device_put_start")
        x = jax.device_put(x_host, NamedSharding(mesh, P(AXIS, None, None, None, None)))
        send_meta = jax.device_put(send_meta_host, NamedSharding(mesh, P(AXIS, None, None, None)))
        recv_meta = jax.device_put(recv_meta_host, NamedSharding(mesh, P(AXIS, None, None, None)))
        w = jax.device_put(w_host, NamedSharding(mesh, P(AXIS, None, None, None)))
        _emit_progress(config, progress_events, "jit_start")
        fn = jax.jit(_sharded_kernel(mesh, config))

        (
            compile_time,
            steady_state_time,
            (inbox, meta, seen_payload, seen_meta, hidden),
            lower_compile_time,
            first_run_time,
        ) = _time_jitted(
            fn,
            x,
            send_meta,
            recv_meta,
            w,
            warmup=warmup,
            steps=steps,
            separate_compile=separate_compile,
            progress=lambda event: _emit_progress(config, progress_events, event),
        )
        max_abs_diff = None
        metadata_mismatches = None
        hidden_max_abs_diff = None
        hidden_mean_abs_diff = None
        if check:
            validation = _validate(
                config,
                x_host,
                send_meta_host,
                w_host,
                inbox,
                meta,
                seen_payload,
                seen_meta,
                hidden,
            )
            max_abs_diff = validation["max_abs_diff"]
            metadata_mismatches = validation["metadata_mismatches"]
            hidden_max_abs_diff = validation["hidden_max_abs_diff"]
            hidden_mean_abs_diff = validation["hidden_mean_abs_diff"]
        bytes_per_rank = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
        w13_tflops_per_rank = None
        if config.implementation != "send_only":
            flops_per_rank = (
                queue_stats["rounded_rows_per_rank_mean"] * config.hidden_dim * config.intermediate_dim * 4
            )
            w13_tflops_per_rank = flops_per_rank / steady_state_time / 1e12
        return {
            "config": asdict(config),
            "queue_stats": queue_stats,
            "compile_time": compile_time,
            "lower_compile_time": lower_compile_time,
            "first_run_time": first_run_time,
            "steady_state_time": steady_state_time,
            "bytes_per_rank": bytes_per_rank,
            "send_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
            "w13_tflops_per_rank": w13_tflops_per_rank,
            "max_abs_diff": max_abs_diff,
            "metadata_mismatches": metadata_mismatches,
            "hidden_max_abs_diff": hidden_max_abs_diff,
            "hidden_mean_abs_diff": hidden_mean_abs_diff,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - repro rows should capture unsupported candidates.
        if debug_exceptions:
            raise
        return {
            "config": asdict(config),
            "compile_time": None,
            "lower_compile_time": None,
            "first_run_time": None,
            "steady_state_time": None,
            "bytes_per_rank": None,
            "send_gbps_per_rank": None,
            "w13_tflops_per_rank": None,
            "max_abs_diff": None,
            "metadata_mismatches": None,
            "hidden_max_abs_diff": None,
            "hidden_mean_abs_diff": None,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        jax.clear_caches()


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default="send_only")
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--entries-per-rank", type=int, default=2)
    parser.add_argument("--sweep-entries-per-rank", type=_parse_int_csv, default=None)
    parser.add_argument("--inbox-slots", type=int, default=2)
    parser.add_argument("--sweep-inbox-slots", type=_parse_int_csv, default=None)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--sweep-block-n", type=_parse_int_csv, default=None)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--sweep-block-k", type=_parse_int_csv, default=None)
    parser.add_argument("--n-group", type=int, default=1)
    parser.add_argument("--sweep-n-groups", type=_parse_int_csv, default=None)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--num-send-sms", type=int, default=4)
    parser.add_argument("--sweep-num-send-sms", type=_parse_int_csv, default=None)
    parser.add_argument("--num-sms", type=int, default=16)
    parser.add_argument("--sweep-num-sms", type=_parse_int_csv, default=None)
    parser.add_argument("--lowering-semantics", choices=tuple(LOWERING_SEMANTICS), default="lane")
    parser.add_argument("--traffic-pattern", choices=TRAFFIC_PATTERNS, default="next_rank")
    parser.add_argument("--peer-loop", choices=PEER_LOOP_MODES, default="static")
    parser.add_argument("--compute-pipeline", choices=COMPUTE_PIPELINE_MODES, default="manual")
    parser.add_argument("--max-concurrent-steps", type=int, default=4)
    parser.add_argument("--sweep-max-concurrent-steps", type=_parse_int_csv, default=None)
    parser.add_argument("--queue-mode", choices=QUEUE_MODES, default="rectangular")
    parser.add_argument("--metadata-mode", choices=METADATA_MODES, default="static_recv")
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default="debug")
    parser.add_argument("--n-groups-per-job", type=int, default=1)
    parser.add_argument("--sweep-n-groups-per-job", type=_parse_int_csv, default=None)
    parser.add_argument("--routing", choices=ROUTING_MODES, default="balanced")
    parser.add_argument("--tokens-per-rank", type=int, default=32768)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--routing-seed", type=int, default=0)
    parser.add_argument("--direct-self-compute", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--separate-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--progress-events", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entries_per_rank_values = args.sweep_entries_per_rank or (args.entries_per_rank,)
    inbox_slots_values = args.sweep_inbox_slots or (args.inbox_slots,)
    block_n_values = args.sweep_block_n or (args.block_n,)
    block_k_values = args.sweep_block_k or (args.block_k,)
    num_send_sms_values = args.sweep_num_send_sms or (args.num_send_sms,)
    num_sms_values = args.sweep_num_sms or (args.num_sms,)
    n_group_values = args.sweep_n_groups or (args.n_group,)
    n_groups_per_job_values = args.sweep_n_groups_per_job or (args.n_groups_per_job,)
    max_concurrent_steps_values = args.sweep_max_concurrent_steps or (args.max_concurrent_steps,)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for entries_per_rank in entries_per_rank_values:
        for inbox_slots in inbox_slots_values:
            for block_n in block_n_values:
                for block_k in block_k_values:
                    for num_send_sms in num_send_sms_values:
                        for num_sms in num_sms_values:
                            for n_group in n_group_values:
                                for n_groups_per_job in n_groups_per_job_values:
                                    for max_concurrent_steps in max_concurrent_steps_values:
                                        config = PushInboxConfig(
                                            implementation=args.implementation,
                                            ep_size=args.ep_size,
                                            entries_per_rank=entries_per_rank,
                                            inbox_slots=inbox_slots,
                                            hidden_dim=args.hidden_dim,
                                            intermediate_dim=args.intermediate_dim,
                                            block_m=args.block_m,
                                            block_n=block_n,
                                            block_k=block_k,
                                            n_group=n_group,
                                            n_groups_per_job=n_groups_per_job,
                                            experts_per_rank=args.experts_per_rank,
                                            num_send_sms=num_send_sms,
                                            num_sms=num_sms,
                                            lowering_semantics=args.lowering_semantics,
                                            traffic_pattern=args.traffic_pattern,
                                            peer_loop=args.peer_loop,
                                            compute_pipeline=args.compute_pipeline,
                                            max_concurrent_steps=max_concurrent_steps,
                                            queue_mode=args.queue_mode,
                                            metadata_mode=args.metadata_mode,
                                            output_mode=args.output_mode,
                                            routing=args.routing,
                                            tokens_per_rank=args.tokens_per_rank,
                                            topk=args.topk,
                                            routing_seed=args.routing_seed,
                                            direct_self_compute=args.direct_self_compute,
                                        )
                                        row = _run_one(
                                            config,
                                            warmup=args.warmup,
                                            steps=args.steps,
                                            check=args.check,
                                            debug_exceptions=args.debug_exceptions,
                                            separate_compile=args.separate_compile,
                                            progress_events=args.progress_events,
                                        )
                                        line = json.dumps(row, sort_keys=True)
                                        print(line, flush=True)
                                        if args.jsonl:
                                            with open(args.jsonl, "a", encoding="utf-8") as f:
                                                print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
