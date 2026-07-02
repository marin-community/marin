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

import argparse
import json
import os
import time
import traceback
from collections.abc import Sequence
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
from levanter.grug._moe.source_push_inbox_profiles import SOURCE_PUSH_PROFILES, source_push_profile_defaults


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
HIDDEN_OUTPUT_MODES = ("full", "queue", "sink")
HIDDEN_COMPUTE_MODES = ("wgmma", "store_zero")
RECEIVER_SCHEDULES = ("fixed_wait", "owned_jobs", "owned_slots", "ordered_wait", "ready_scan", "slot_group")
SCAN_ORDERS = ("current", "rotated", "hash")
SLOT_ORDERS = ("current", "rotated", "hash")
SOURCE_PUSH_MODES = ("packed",)
INBOX_STORAGE_MODES = ("output", "alias", "scratch")
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
    send_pipeline_depth: int = 1
    lowering_semantics: str = "lane"
    traffic_pattern: str = "next_rank"
    peer_loop: str = "static"
    compute_pipeline: str = "manual"
    max_concurrent_steps: int = 4
    queue_mode: str = "rectangular"
    metadata_mode: str = "static_recv"
    output_mode: str = "debug"
    hidden_output_mode: str = "full"
    hidden_compute_mode: str = "wgmma"
    n_groups_per_job: int = 1
    receiver_schedule: str = "fixed_wait"
    scan_candidates: int = 8
    scan_order: str = "rotated"
    slot_order: str = "current"
    workers_per_slot: int = 1
    source_push_mode: str = "packed"
    inbox_storage: str = "output"
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
        if self.send_pipeline_depth not in (1, 2, 3, 4):
            raise ValueError(f"send_pipeline_depth must be in (1, 2, 3, 4), got {self.send_pipeline_depth}")
        if self.send_pipeline_depth > 1 and self.peer_loop in ("dynamic", "grid"):
            raise ValueError("send_pipeline_depth > 1 is implemented only for static, switch, and grid_switch")
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
        if self.hidden_output_mode not in HIDDEN_OUTPUT_MODES:
            raise ValueError(
                f"unknown hidden_output_mode={self.hidden_output_mode!r}; expected one of {HIDDEN_OUTPUT_MODES}"
            )
        if self.hidden_output_mode == "sink" and self.output_mode != "perf":
            raise ValueError("hidden_output_mode=sink requires output_mode=perf")
        if self.hidden_output_mode == "sink" and self.implementation != "m_n_slots":
            raise ValueError("hidden_output_mode=sink is currently implemented only for implementation=m_n_slots")
        if self.hidden_output_mode == "queue" and self.implementation != "m_n_slots":
            raise ValueError("hidden_output_mode=queue is currently implemented only for implementation=m_n_slots")
        if self.hidden_compute_mode not in HIDDEN_COMPUTE_MODES:
            raise ValueError(
                f"unknown hidden_compute_mode={self.hidden_compute_mode!r}; expected one of {HIDDEN_COMPUTE_MODES}"
            )
        if self.hidden_compute_mode == "store_zero":
            if self.implementation != "m_n_slots":
                raise ValueError("hidden_compute_mode=store_zero is currently implemented only for m_n_slots")
            if self.output_mode != "perf":
                raise ValueError("hidden_compute_mode=store_zero requires output_mode=perf")
        if self.output_mode == "perf":
            if self.implementation not in ("send_only", "m_n_slots"):
                raise ValueError("output_mode=perf is currently implemented only for send_only and m_n_slots")
            if self.metadata_mode != "static_recv":
                raise ValueError("output_mode=perf requires metadata_mode=static_recv")
        if self.inbox_storage not in INBOX_STORAGE_MODES:
            raise ValueError(f"unknown inbox_storage={self.inbox_storage!r}; expected one of {INBOX_STORAGE_MODES}")
        if self.inbox_storage == "scratch":
            raise ValueError(
                "inbox_storage=scratch is disabled: Mosaic GPU scratch_shapes with mgpu.GMEM fail with "
                "NotImplementedError: Unsupported memory space: gmem"
            )
        if self.inbox_storage == "alias":
            raise ValueError(
                "inbox_storage=alias is disabled: forcing direct pl.pallas_call specs to GMEM avoids the earlier "
                "whole-inbox SMEM blowup, but global semaphore lowering then fails with an out-of-bounds "
                "reserve_semaphores slice. This path needs an mgpu.kernel-level alias mechanism."
            )
            if self.implementation != "m_n_slots":
                raise ValueError("inbox_storage=alias is currently implemented only for implementation=m_n_slots")
            if self.output_mode != "perf":
                raise ValueError("inbox_storage=alias requires output_mode=perf")
        if self.n_groups_per_job <= 0:
            raise ValueError(f"n_groups_per_job must be positive, got {self.n_groups_per_job}")
        if self.n_groups_per_job > self.intermediate_dim // self.block_n // self.n_group:
            raise ValueError(
                "n_groups_per_job must not exceed the number of N work groups; "
                f"got {self.n_groups_per_job=} with "
                f"{self.intermediate_dim=} {self.block_n=} {self.n_group=}"
            )
        if self.receiver_schedule not in RECEIVER_SCHEDULES:
            raise ValueError(
                f"unknown receiver_schedule={self.receiver_schedule!r}; expected one of {RECEIVER_SCHEDULES}"
            )
        if self.receiver_schedule == "ready_scan":
            raise ValueError(
                "receiver_schedule=ready_scan is disabled: ready-scan variants compile but deadlock in checked "
                "H100 smokes, including the multi-candidate receiver scan"
            )
        if self.receiver_schedule == "slot_group":
            num_compute_sms = self.num_sms - self.num_send_sms
            if self.workers_per_slot > num_compute_sms:
                raise ValueError(
                    "receiver_schedule=slot_group requires workers_per_slot <= num_compute_sms; "
                    f"got {self.workers_per_slot=} {num_compute_sms=}"
                )
            if self.direct_self_compute:
                raise ValueError("receiver_schedule=slot_group does not support direct_self_compute")
        if self.receiver_schedule != "fixed_wait" and self.implementation != "m_n_slots":
            raise ValueError("receiver_schedule modes are currently implemented only for implementation=m_n_slots")
        if self.scan_candidates < 0:
            raise ValueError("scan_candidates must be non-negative; use 0 to mean all candidates")
        if self.scan_order not in SCAN_ORDERS:
            raise ValueError(f"unknown scan_order={self.scan_order!r}; expected one of {SCAN_ORDERS}")
        if self.slot_order not in SLOT_ORDERS:
            raise ValueError(f"unknown slot_order={self.slot_order!r}; expected one of {SLOT_ORDERS}")
        if self.workers_per_slot <= 0:
            raise ValueError(f"workers_per_slot must be positive, got {self.workers_per_slot}")
        if self.source_push_mode not in SOURCE_PUSH_MODES:
            raise ValueError(
                f"unknown source_push_mode={self.source_push_mode!r}; expected one of {SOURCE_PUSH_MODES}"
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
        if self.hidden_output_mode == "sink":
            return (self.ep_size * self.inbox_slots * self.block_m, self.intermediate_dim)
        return (self.hidden_rows_per_rank, self.intermediate_dim)


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


def _make_kernel(config: PushInboxConfig):
    k_tiles = config.hidden_dim // config.block_k
    n_tiles = config.intermediate_dim // config.block_n
    n_work_groups = n_tiles // config.n_group
    n_compute_jobs = (n_work_groups + config.n_groups_per_job - 1) // config.n_groups_per_job
    num_compute_sms = config.num_sms - config.num_send_sms
    local_work_groups = config.inbox_slots * n_compute_jobs
    owned_work_groups_per_worker = (local_work_groups + num_compute_sms - 1) // num_compute_sms
    owned_slot_jobs_per_worker = (n_compute_jobs + num_compute_sms - 1) // num_compute_sms
    ready_scan_jobs = config.inbox_slots * n_compute_jobs
    ready_scan_candidates = (
        ready_scan_jobs if config.scan_candidates == 0 else min(config.scan_candidates, ready_scan_jobs)
    )
    rounds_per_slot = (config.entries_per_rank + config.inbox_slots - 1) // config.inbox_slots
    num_slot_groups = num_compute_sms // config.workers_per_slot
    send_dst_offsets = tuple(range(config.ep_size)) if config.traffic_pattern == "all_to_all" else (1,)
    recv_src_offsets = (
        tuple(range(config.ep_size)) if config.traffic_pattern == "all_to_all" else (config.ep_size - 1,)
    )

    def body(
        x_ref: Float[Ref, "DST Q M D"],
        send_meta_ref: Int[Ref, "DST Q F"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        w_ref: Float[Ref, "E D twoI"],
        inbox_out_ref: Float[Ref, "SRC SLOTS M D"],
        meta_ref: Int[Ref, "SRC SLOTS F"],
        seen_payload_ref: Float[Ref, "SRC Q"],
        seen_meta_ref: Int[Ref, "SRC Q F"],
        hidden_ref: Float[Ref, "rows I"],
        inbox_scratch_ref: Float[Ref, "SRC SLOTS M D"] | None = None,
    ) -> None:
        inbox_ref = inbox_scratch_ref if config.inbox_storage == "scratch" else inbox_out_ref
        if inbox_ref is None:
            raise AssertionError("inbox_ref must be available for enabled inbox storage modes")
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

        def _slot_for_position(peer_index, slot_pos, round_i):
            if config.slot_order == "current":
                return slot_pos
            if config.slot_order == "rotated":
                return (slot_pos + peer_index + 3 * round_i) % config.inbox_slots
            return (5 * slot_pos + 3 * peer_index + 7 * round_i) % config.inbox_slots

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
                    def _slot_loop(slot_pos) -> None:
                        slot = _slot_for_position(dst_ordinal, slot_pos, round_i)
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
                                        if config.output_mode != "perf":
                                            meta_ref[rank, slot, 0] = rank
                                            meta_ref[rank, slot, 1] = expert
                                            meta_ref[rank, slot, 2] = dst_row_start
                                            meta_ref[rank, slot, 3] = valid_rows
                                        pl.semaphore_signal(full_sem_ref.at[rank, slot])

                                    @pl.when(dst_offset != 0)
                                    def _write_remote_meta() -> None:
                                        if config.output_mode != "perf":
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
                write_slot_metadata = config.metadata_mode == "remote_slot" or (
                    config.implementation == "send_only" and config.output_mode != "perf"
                )
                if dst_offset != 0:
                    remote_inbox_ref = mgpu.remote_ref(inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)
                    if write_slot_metadata:
                        remote_meta_ref = mgpu.remote_ref(meta_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                @pl.loop(0, rounds_per_slot)
                def _round_loop(round_i) -> None:
                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot_pos) -> None:
                        slot = _slot_for_position(dst_ordinal, slot_pos, round_i)
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

                                    def _copy_buffer(buffer_smem, k_start) -> None:
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

                                    if config.send_pipeline_depth == 1:

                                        def _copy_scope(tile_smem) -> None:
                                            @pl.loop(0, k_tiles)
                                            def _k_loop(kk) -> None:
                                                k_start = kk * config.block_k
                                                _copy_buffer(tile_smem, k_start)
                                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                        pl.run_scoped(
                                            _copy_scope,
                                            tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
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
                                            tile_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_next=mgpu.SMEM(
                                                (config.block_m, config.block_k), dtype=x_ref.dtype
                                            ),
                                        )
                                    elif config.send_pipeline_depth == 3:

                                        def _copy_scope(tile_smem_0, tile_smem_1, tile_smem_2) -> None:
                                            @pl.loop(0, k_tiles)
                                            def _k_loop(kk) -> None:
                                                k_start = kk * config.block_k

                                                @pl.when(kk >= 3)
                                                def _wait_for_reusable_buffer() -> None:
                                                    mgpu.wait_smem_to_gmem(2, wait_read_only=False)

                                                @pl.when((kk % 3) == 0)
                                                def _copy_buffer_0() -> None:
                                                    _copy_buffer(tile_smem_0, k_start)

                                                @pl.when((kk % 3) == 1)
                                                def _copy_buffer_1() -> None:
                                                    _copy_buffer(tile_smem_1, k_start)

                                                @pl.when((kk % 3) == 2)
                                                def _copy_buffer_2() -> None:
                                                    _copy_buffer(tile_smem_2, k_start)

                                            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                        pl.run_scoped(
                                            _copy_scope,
                                            tile_smem_0=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_1=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_2=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                        )
                                    else:

                                        def _copy_scope(tile_smem_0, tile_smem_1, tile_smem_2, tile_smem_3) -> None:
                                            @pl.loop(0, k_tiles)
                                            def _k_loop(kk) -> None:
                                                k_start = kk * config.block_k

                                                @pl.when(kk >= 4)
                                                def _wait_for_reusable_buffer() -> None:
                                                    mgpu.wait_smem_to_gmem(3, wait_read_only=False)

                                                @pl.when((kk % 4) == 0)
                                                def _copy_buffer_0() -> None:
                                                    _copy_buffer(tile_smem_0, k_start)

                                                @pl.when((kk % 4) == 1)
                                                def _copy_buffer_1() -> None:
                                                    _copy_buffer(tile_smem_1, k_start)

                                                @pl.when((kk % 4) == 2)
                                                def _copy_buffer_2() -> None:
                                                    _copy_buffer(tile_smem_2, k_start)

                                                @pl.when((kk % 4) == 3)
                                                def _copy_buffer_3() -> None:
                                                    _copy_buffer(tile_smem_3, k_start)

                                            mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                                        pl.run_scoped(
                                            _copy_scope,
                                            tile_smem_0=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_1=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_2=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
                                            tile_smem_3=mgpu.SMEM((config.block_m, config.block_k), dtype=x_ref.dtype),
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
                                            assert remote_meta_ref is not None
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

        def _sink_hidden(src_ordinal, slot, n_tile, hidden) -> None:
            sink_row = (src_ordinal * config.inbox_slots + slot) * config.block_m
            hidden_ref[
                pl.ds(sink_row, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            ] = hidden.astype(hidden_ref.dtype)

        def _store_hidden(src_ordinal, entry, slot, dst_row_start, n_tile, hidden) -> None:
            if config.hidden_output_mode == "sink":
                _sink_hidden(src_ordinal, slot, n_tile, hidden)
            else:
                if config.hidden_output_mode == "queue":
                    dst_row_start = (src_ordinal * config.entries_per_rank + entry) * config.block_m
                hidden_ref[
                    pl.ds(dst_row_start, config.block_m),
                    pl.ds(n_tile * config.block_n, config.block_n),
                ] = hidden.astype(hidden_ref.dtype)

        def _store_zero_hidden(src_ordinal, entry, slot, dst_row_start, n_tile) -> None:
            zero_hidden = jnp.zeros((config.block_m, config.block_n), dtype=jnp.float32)
            _store_hidden(src_ordinal, entry, slot, dst_row_start, n_tile, zero_hidden)

        def _compute_hidden_n_tile(src, src_ordinal, entry, slot, expert, dst_row_start, n_tile) -> None:
            if config.hidden_compute_mode == "store_zero":
                _store_zero_hidden(src_ordinal, entry, slot, dst_row_start, n_tile)
                return

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
            _store_hidden(src_ordinal, entry, slot, dst_row_start, n_tile, hidden)

        def _compute_hidden_n_tile_from_x(dst_ordinal, entry, expert, dst_row_start, n_tile) -> None:
            if config.hidden_compute_mode == "store_zero":
                slot = entry % config.inbox_slots
                _store_zero_hidden(dst_ordinal, entry, slot, dst_row_start, n_tile)
                return

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
            slot = entry % config.inbox_slots
            _store_hidden(dst_ordinal, entry, slot, dst_row_start, n_tile, hidden)

        def _compute_hidden_n_group(src, src_ordinal, entry, slot, expert, dst_row_start, n_group_i) -> None:
            if config.n_group == 1:
                _compute_hidden_n_tile(src, src_ordinal, entry, slot, expert, dst_row_start, n_group_i)
            else:
                n_tile = n_group_i * 2
                if config.hidden_compute_mode == "store_zero":
                    _store_zero_hidden(src_ordinal, entry, slot, dst_row_start, n_tile)
                    _store_zero_hidden(src_ordinal, entry, slot, dst_row_start, n_tile + 1)
                    return

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
                    _store_hidden(src_ordinal, entry, slot, dst_row_start, n_tile, hidden_n0)
                    _store_hidden(src_ordinal, entry, slot, dst_row_start, n_tile + 1, hidden_n1)

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
                    _compute_hidden_n_tile(src, src, entry, slot, expert, dst_row_start, n_tile)

            else:
                if config.output_mode != "perf":
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

                    def _job_owner(slot, job_i):
                        work_group = (src_ordinal * config.inbox_slots + slot) * n_compute_jobs + job_i
                        return (work_group % num_compute_sms) == compute_worker

                    def _compute_recv_job(entry, slot, job_i) -> None:
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
                                    src_ordinal,
                                    entry,
                                    slot,
                                    expert,
                                    dst_row_start,
                                    n_group_i,
                                )

                    def _slot_group_id(slot):
                        return (src_ordinal * config.inbox_slots + slot) % num_slot_groups

                    def _slot_group_owner(slot):
                        return (compute_worker // config.workers_per_slot) == _slot_group_id(slot)

                    def _slot_group_lane():
                        return compute_worker % config.workers_per_slot

                    def _compute_slot_group_entry(entry, slot) -> None:
                        if config.metadata_mode == "static_recv":
                            expert = recv_meta_ref[src_ordinal, entry, 1]
                            dst_row_start = recv_meta_ref[src_ordinal, entry, 2]
                        else:
                            expert = meta_ref[src, slot, 1]
                            dst_row_start = meta_ref[src, slot, 2]

                        lane = _slot_group_lane()

                        @pl.loop(0, n_work_groups)
                        def _n_group_loop(n_group_i) -> None:
                            should_compute_group = (n_group_i % config.workers_per_slot) == lane

                            @pl.when(should_compute_group)
                            def _n_group() -> None:
                                _compute_hidden_n_group(
                                    src,
                                    src_ordinal,
                                    entry,
                                    slot,
                                    expert,
                                    dst_row_start,
                                    n_group_i,
                                )

                    def _signal_recv_job_done(slot) -> None:
                        pl.semaphore_signal(done_sem_ref.at[src, slot])

                    def _scan_candidate_id(work_i, scan_attempt, round_i):
                        if config.scan_order == "current":
                            start = work_i
                        elif config.scan_order == "rotated":
                            start = compute_worker + 5 * round_i + work_i
                        else:
                            start = compute_worker * 17 + round_i * 31 + work_i * 7
                        return (start + scan_attempt) % ready_scan_jobs

                    def _ordered_wait_slot(slot_pos, round_i):
                        if config.receiver_schedule == "fixed_wait" or config.scan_order == "current":
                            return slot_pos
                        if config.scan_order == "rotated":
                            return (slot_pos + compute_worker + 5 * round_i) % config.inbox_slots
                        return (compute_worker * 3 + round_i * 5 + slot_pos * 3) % config.inbox_slots

                    def _maybe_select_ready_candidate(round_i, candidate_id, selected_ref) -> None:
                        slot = candidate_id // n_compute_jobs
                        job_i = candidate_id % n_compute_jobs
                        entry = slot + round_i * config.inbox_slots

                        @pl.when(entry < config.entries_per_rank)
                        def _maybe_live_entry() -> None:
                            valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                            @pl.when(valid_rows > 0)
                            def _live_entry() -> None:
                                should_compute = _job_owner(slot, job_i)

                                @pl.when(should_compute)
                                def _owned_job() -> None:
                                    full_value = pl.semaphore_read(full_sem_ref.at[src, slot])
                                    ready = full_value >= (round_i + 1)
                                    not_selected = selected_ref[0] < 0

                                    @pl.when(ready & not_selected)
                                    def _select_ready_job() -> None:
                                        selected_ref[0] = candidate_id

                    def _maybe_compute_ready_candidate(round_i, candidate_id, computed_ref) -> None:
                        slot = candidate_id // n_compute_jobs
                        job_i = candidate_id % n_compute_jobs
                        entry = slot + round_i * config.inbox_slots

                        @pl.when(entry < config.entries_per_rank)
                        def _maybe_live_entry() -> None:
                            valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                            @pl.when(valid_rows > 0)
                            def _live_entry() -> None:
                                should_compute = _job_owner(slot, job_i)

                                @pl.when(should_compute)
                                def _owned_job() -> None:
                                    not_computed = computed_ref[candidate_id] == 0

                                    @pl.when(not_computed)
                                    def _maybe_ready_job() -> None:
                                        full_value = pl.semaphore_read(full_sem_ref.at[src, slot])
                                        ready = full_value >= (round_i + 1)

                                        @pl.when(ready)
                                        def _compute_ready_job() -> None:
                                            _compute_recv_job(entry, slot, job_i)
                                            pl.semaphore_signal(done_sem_ref.at[src, slot])
                                            computed_ref[candidate_id] = jnp.int32(1)

                    def _wait_and_compute_fixed_candidate(round_i, candidate_id) -> None:
                        slot = candidate_id // n_compute_jobs
                        job_i = candidate_id % n_compute_jobs
                        entry = slot + round_i * config.inbox_slots

                        @pl.when(entry < config.entries_per_rank)
                        def _maybe_live_entry() -> None:
                            valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                            @pl.when(valid_rows > 0)
                            def _live_entry() -> None:
                                should_compute = _job_owner(slot, job_i)

                                @pl.when(should_compute)
                                def _owned_job() -> None:
                                    pl.semaphore_wait(
                                        full_sem_ref.at[src, slot],
                                        value=round_i + 1,
                                        decrement=False,
                                    )
                                    _compute_recv_job(entry, slot, job_i)
                                    pl.semaphore_signal(done_sem_ref.at[src, slot])

                    def _release_completed_slot(round_i, slot) -> None:
                        entry = slot + round_i * config.inbox_slots

                        @pl.when(entry < config.entries_per_rank)
                        def _maybe_release_entry() -> None:
                            valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                            @pl.when(valid_rows > 0)
                            def _release_live_entry() -> None:
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

                    def _release_completed_slots(round_i) -> None:
                        @pl.loop(0, config.inbox_slots)
                        def _release_slot_loop(slot) -> None:
                            _release_completed_slot(round_i, slot)

                    def _owned_local_work_group(owned_i):
                        source_base_work_group = src_ordinal * local_work_groups
                        first_local_work_group = (
                            compute_worker - (source_base_work_group % num_compute_sms)
                        ) % num_compute_sms
                        return first_local_work_group + owned_i * num_compute_sms

                    def _owned_slot_job(slot, owned_i):
                        slot_base_work_group = (src_ordinal * config.inbox_slots + slot) * n_compute_jobs
                        first_job = (compute_worker - (slot_base_work_group % num_compute_sms)) % num_compute_sms
                        return first_job + owned_i * num_compute_sms

                    if config.receiver_schedule == "fixed_wait":

                        @pl.loop(0, rounds_per_slot)
                        def _round_loop(round_i) -> None:
                            @pl.loop(0, config.inbox_slots)
                            def _slot_loop(slot_pos) -> None:
                                slot = _slot_for_position(src_ordinal, slot_pos, round_i)
                                entry = slot + round_i * config.inbox_slots

                                @pl.when(entry < config.entries_per_rank)
                                def _maybe_recv_slot_entry() -> None:
                                    valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                    @pl.when(valid_rows > 0)
                                    def _recv_slot_entry() -> None:
                                        @pl.loop(0, n_compute_jobs)
                                        def _job_loop(job_i) -> None:
                                            work_group = (
                                                src_ordinal * config.inbox_slots + slot
                                            ) * n_compute_jobs + job_i
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
                                                            src_ordinal,
                                                            entry,
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

                    elif config.receiver_schedule == "owned_jobs":

                        @pl.loop(0, rounds_per_slot)
                        def _round_loop(round_i) -> None:
                            @pl.loop(0, owned_work_groups_per_worker)
                            def _owned_work_loop(owned_i) -> None:
                                local_work_group = _owned_local_work_group(owned_i)

                                @pl.when(local_work_group < local_work_groups)
                                def _owned_live_work_group() -> None:
                                    slot = local_work_group // n_compute_jobs
                                    job_i = local_work_group - slot * n_compute_jobs
                                    entry = slot + round_i * config.inbox_slots

                                    @pl.when(entry < config.entries_per_rank)
                                    def _maybe_live_entry() -> None:
                                        valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                        @pl.when(valid_rows > 0)
                                        def _live_entry() -> None:
                                            pl.semaphore_wait(
                                                full_sem_ref.at[src, slot],
                                                value=round_i + 1,
                                                decrement=False,
                                            )
                                            _compute_recv_job(entry, slot, job_i)
                                            pl.semaphore_signal(done_sem_ref.at[src, slot])

                            _release_completed_slots(round_i)

                    elif config.receiver_schedule == "owned_slots":

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
                                        @pl.loop(0, owned_slot_jobs_per_worker)
                                        def _owned_slot_job_loop(owned_i) -> None:
                                            job_i = _owned_slot_job(slot, owned_i)

                                            @pl.when(job_i < n_compute_jobs)
                                            def _owned_live_job() -> None:
                                                pl.semaphore_wait(
                                                    full_sem_ref.at[src, slot],
                                                    value=round_i + 1,
                                                    decrement=False,
                                                )
                                                _compute_recv_job(entry, slot, job_i)
                                                pl.semaphore_signal(done_sem_ref.at[src, slot])

                                        _release_completed_slot(round_i, slot)

                    elif config.receiver_schedule == "ordered_wait":

                        @pl.loop(0, rounds_per_slot)
                        def _round_loop(round_i) -> None:
                            @pl.loop(0, config.inbox_slots)
                            def _slot_loop(slot_pos) -> None:
                                slot = _ordered_wait_slot(slot_pos, round_i)
                                entry = slot + round_i * config.inbox_slots

                                @pl.when(entry < config.entries_per_rank)
                                def _maybe_recv_slot_entry() -> None:
                                    valid_rows = recv_meta_ref[src_ordinal, entry, 3]

                                    @pl.when(valid_rows > 0)
                                    def _recv_slot_entry() -> None:
                                        @pl.loop(0, n_compute_jobs)
                                        def _job_loop(job_i) -> None:
                                            should_compute = _job_owner(slot, job_i)

                                            @pl.when(should_compute)
                                            def _recv_job() -> None:
                                                pl.semaphore_wait(
                                                    full_sem_ref.at[src, slot],
                                                    value=round_i + 1,
                                                    decrement=False,
                                                )
                                                _compute_recv_job(entry, slot, job_i)
                                                pl.semaphore_signal(done_sem_ref.at[src, slot])

                                        _release_completed_slot(round_i, slot)

                    elif config.receiver_schedule == "slot_group":

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
                                        should_compute = _slot_group_owner(slot)

                                        @pl.when(should_compute)
                                        def _recv_slot_group() -> None:
                                            pl.semaphore_wait(
                                                full_sem_ref.at[src, slot],
                                                value=round_i + 1,
                                                decrement=False,
                                            )
                                            _compute_slot_group_entry(entry, slot)
                                            pl.semaphore_signal(done_sem_ref.at[src, slot])

                                            should_release = _slot_group_lane() == 0

                                            @pl.when(should_release)
                                            def _release_slot() -> None:
                                                pl.semaphore_wait(
                                                    done_sem_ref.at[src, slot],
                                                    value=(round_i + 1) * config.workers_per_slot,
                                                    decrement=False,
                                                )
                                                _signal_empty_to_src(src, src_offset, slot)

                    else:

                        @pl.loop(0, rounds_per_slot)
                        def _round_loop(round_i) -> None:
                            def _ready_scan_scope(computed_ref):
                                @pl.loop(0, ready_scan_jobs)
                                def _init_computed_loop(candidate_id) -> None:
                                    computed_ref[candidate_id] = jnp.int32(0)

                                @pl.loop(0, ready_scan_candidates)
                                def _scan_attempt_loop(scan_attempt) -> None:
                                    candidate_id = _scan_candidate_id(0, scan_attempt, round_i)
                                    _maybe_compute_ready_candidate(round_i, candidate_id, computed_ref)

                                @pl.loop(0, ready_scan_jobs)
                                def _fallback_loop(candidate_id) -> None:
                                    @pl.when(computed_ref[candidate_id] == 0)
                                    def _fallback_candidate() -> None:
                                        _wait_and_compute_fixed_candidate(round_i, candidate_id)

                            pl.run_scoped(
                                _ready_scan_scope,
                                computed_ref=mgpu.SMEM((ready_scan_jobs,), dtype=jnp.int32),
                            )

                            _release_completed_slots(round_i)

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

    inbox_shape = (config.ep_size, config.inbox_slots, config.block_m, config.hidden_dim)
    if config.output_mode == "perf":
        returned_inbox_shape = (1, 1, 1, 1) if config.inbox_storage == "scratch" else inbox_shape
        meta_shape = (1, 1, META_FIELDS)
        seen_payload_shape = (1, 1)
        seen_meta_shape = (1, 1, META_FIELDS)
    else:
        returned_inbox_shape = inbox_shape
        meta_shape = (config.ep_size, config.inbox_slots, META_FIELDS)
        seen_payload_shape = (config.ep_size, config.entries_per_rank)
        seen_meta_shape = (config.ep_size, config.entries_per_rank, META_FIELDS)
    scratch_shapes = ()
    if config.inbox_storage == "scratch":
        scratch_shapes = (mgpu.GMEM(inbox_shape, jnp.bfloat16),)
    grid = (
        (len(send_dst_offsets), config.num_sms) if config.peer_loop in ("grid", "grid_switch") else (config.num_sms,)
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=LOWERING_SEMANTICS[config.lowering_semantics])
    out_shape = [
        jax.ShapeDtypeStruct(returned_inbox_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(meta_shape, jnp.int32),
        jax.ShapeDtypeStruct(seen_payload_shape, jnp.bfloat16),
        jax.ShapeDtypeStruct(seen_meta_shape, jnp.int32),
        jax.ShapeDtypeStruct(config.hidden_output_shape, jnp.bfloat16),
    ]

    if config.inbox_storage == "alias":
        x_shape = (config.traffic_fanout, config.entries_per_rank, config.block_m, config.hidden_dim)
        send_meta_shape = (config.traffic_fanout, config.entries_per_rank, META_FIELDS)
        recv_meta_shape = (config.traffic_fanout, config.entries_per_rank, META_FIELDS)
        w_shape = (config.experts_per_rank, config.hidden_dim, 2 * config.intermediate_dim)

        def gmem_spec(shape):
            return pl.BlockSpec(
                shape,
                lambda *indices: (0,) * len(shape),
                memory_space=mgpu.GMEM,
            )

        in_specs = (
            gmem_spec(x_shape),
            gmem_spec(send_meta_shape),
            gmem_spec(recv_meta_shape),
            gmem_spec(w_shape),
            gmem_spec(inbox_shape),
        )
        out_specs = tuple(gmem_spec(shape_dtype.shape) for shape_dtype in out_shape)

        def alias_body(
            x_ref,
            send_meta_ref,
            recv_meta_ref,
            w_ref,
            inbox_seed_ref,
            inbox_ref,
            meta_ref,
            seen_payload_ref,
            seen_meta_ref,
            hidden_ref,
        ) -> None:
            del inbox_seed_ref
            body(
                x_ref,
                send_meta_ref,
                recv_meta_ref,
                w_ref,
                inbox_ref,
                meta_ref,
                seen_payload_ref,
                seen_meta_ref,
                hidden_ref,
            )

        return pl.pallas_call(
            alias_body,
            out_shape=out_shape,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            input_output_aliases={4: 0},
            compiler_params=compiler_params,
        )

    return mgpu.kernel(
        body,
        out_shape=out_shape,
        scratch_shapes=scratch_shapes,
        grid=grid,
        grid_names=("peer_phase", "sm") if config.peer_loop in ("grid", "grid_switch") else ("sm",),
        compiler_params=compiler_params,
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
    masked_rows = live.astype(np.int64) * config.block_m - valid_rows
    masked_rows_per_rank = np.sum(masked_rows, axis=(1, 2))
    direct_self_masked_rows_per_rank = (
        np.sum(masked_rows[:, 0, :], axis=1)
        if config.direct_self_compute and config.traffic_pattern == "all_to_all"
        else np.zeros((config.ep_size,), dtype=np.int64)
    )
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
    num_compute_sms = config.num_sms - config.num_send_sms
    local_work_groups = config.inbox_slots * n_compute_jobs
    owned_work_groups_per_worker = (local_work_groups + num_compute_sms - 1) // num_compute_sms
    owned_slot_jobs_per_worker = (n_compute_jobs + num_compute_sms - 1) // num_compute_sms
    compute_jobs_per_entry = config.workers_per_slot if config.receiver_schedule == "slot_group" else n_compute_jobs
    ready_scan_jobs = config.inbox_slots * n_compute_jobs
    ready_scan_candidates = (
        ready_scan_jobs if config.scan_candidates == 0 else min(config.scan_candidates, ready_scan_jobs)
    )
    if config.implementation == "m_n_slots":
        compute_wait_full_count = payload_send_entries_total * compute_jobs_per_entry
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
        "hidden_output_mode": config.hidden_output_mode,
        "hidden_compute_mode": config.hidden_compute_mode,
        "receiver_schedule": config.receiver_schedule,
        "scan_candidates": config.scan_candidates,
        "scan_candidates_effective": ready_scan_candidates,
        "scan_order": config.scan_order,
        "slot_order": config.slot_order,
        "workers_per_slot": config.workers_per_slot,
        "source_push_mode": config.source_push_mode,
        "send_pipeline_depth": config.send_pipeline_depth,
        "n_groups_per_job": config.n_groups_per_job,
        "n_work_groups_per_entry": n_work_groups,
        "num_compute_jobs_per_entry": compute_jobs_per_entry,
        "local_work_groups_per_source": local_work_groups,
        "owned_work_groups_per_worker_round": owned_work_groups_per_worker,
        "owned_slot_jobs_per_worker": owned_slot_jobs_per_worker,
        "done_signals_per_entry": compute_jobs_per_entry,
        "ready_scan_jobs_per_round": ready_scan_jobs,
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

    if config.inbox_storage == "alias":

        def local_alias_fn(
            x_local: Float[Array, "1 DST Q M D"],
            send_meta_local: Int[Array, "1 DST Q F"],
            recv_meta_local: Int[Array, "1 SRC Q F"],
            w_local: Float[Array, "1 E D twoI"],
            inbox_seed_local: Float[Array, "1 SRC SLOTS M D"],
        ):
            x_local = x_local[0]
            send_meta_local = send_meta_local[0]
            recv_meta_local = recv_meta_local[0]
            w_local = w_local[0]
            inbox_seed_local = inbox_seed_local[0]
            inbox, meta, seen_payload, seen_meta, hidden = kernel(
                x_local,
                send_meta_local,
                recv_meta_local,
                w_local,
                inbox_seed_local,
            )
            return inbox[None, ...], meta[None, ...], seen_payload[None, ...], seen_meta[None, ...], hidden[None, ...]

        return shard_map(
            local_alias_fn,
            mesh=mesh,
            in_specs=(
                P(AXIS, None, None, None, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None, None),
                P(AXIS, None, None, None, None),
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
    repeat_runs: int,
    separate_compile: bool,
    threaded_alias: tuple[int, int] | None = None,
    progress: Callable[[str], None] | None = None,
):
    call_args = tuple(args)

    def _thread_alias(current_args, out):
        if threaded_alias is None:
            return current_args
        next_args = list(current_args)
        input_index, output_index = threaded_alias
        next_args[input_index] = out[output_index]
        return tuple(next_args)

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
        call_args = _thread_alias(call_args, out)
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
        call_args = _thread_alias(call_args, out)
        compile_time = time.perf_counter() - start
        if progress is not None:
            progress("first_call_done")

    if progress is not None:
        progress("warmup_start")
    for _ in range(warmup):
        out = fn(*call_args)
        _block_until_ready(out)
        call_args = _thread_alias(call_args, out)

    steady_state_times = []
    for _ in range(repeat_runs):
        if progress is not None:
            progress("steady_state_start")
        start = time.perf_counter()
        for _ in range(steps):
            out = fn(*call_args)
            _block_until_ready(out)
            call_args = _thread_alias(call_args, out)
        steady_state_times.append((time.perf_counter() - start) / steps)
        if progress is not None:
            progress("steady_state_done")
    return compile_time, steady_state_times, out, lower_compile_time, first_run_time


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
            recv_src_ordinal = _recv_src_ordinal(config, dst, src)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                expert = send_meta[src, dst_ordinal, entry, 1]
                if config.hidden_output_mode == "queue":
                    row = (recv_src_ordinal * config.entries_per_rank + entry) * config.block_m
                else:
                    row = send_meta[src, dst_ordinal, entry, 2]
                gate = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, : config.intermediate_dim]
                up = x_float[src, dst_ordinal, entry] @ w_float[dst, expert, :, config.intermediate_dim :]
                hidden[dst, row : row + config.block_m, :] = gate * (1.0 / (1.0 + np.exp(-gate))) * up
    return hidden


def _hidden_live_row_mask(config: PushInboxConfig, send_meta_host) -> np.ndarray:
    mask = np.zeros((config.ep_size, config.hidden_rows_per_rank), dtype=np.bool_)
    send_meta = np.asarray(send_meta_host, dtype=np.int32)
    for src in range(config.ep_size):
        dsts = range(config.ep_size) if config.traffic_pattern == "all_to_all" else ((src + 1) % config.ep_size,)
        for dst in dsts:
            dst_ordinal = _dst_ordinal(config, src, dst)
            recv_src_ordinal = _recv_src_ordinal(config, dst, src)
            for entry in range(config.entries_per_rank):
                valid_rows = send_meta[src, dst_ordinal, entry, 3]
                if valid_rows <= 0:
                    continue
                if config.hidden_output_mode == "queue":
                    row = (recv_src_ordinal * config.entries_per_rank + entry) * config.block_m
                else:
                    row = send_meta[src, dst_ordinal, entry, 2]
                mask[dst, row : row + config.block_m] = True
    return mask


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
    hidden_all_max_abs_diff = None
    hidden_unwritten_max_abs = None
    if config.implementation in ("m_owner", "m_owner_slots", "m_n_slots"):
        hidden_expected = _reference_hidden(config, x_host, send_meta_host, w_host)
        hidden_diff = np.abs(np.asarray(hidden, dtype=np.float32) - hidden_expected)
        hidden_all_max_abs_diff = float(np.max(hidden_diff))
        live_row_mask = _hidden_live_row_mask(config, send_meta_host)
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
    return {
        "max_abs_diff": max_abs_diff,
        "metadata_mismatches": metadata_mismatches,
        "hidden_max_abs_diff": hidden_max_abs_diff,
        "hidden_mean_abs_diff": hidden_mean_abs_diff,
        "hidden_all_max_abs_diff": hidden_all_max_abs_diff,
        "hidden_unwritten_max_abs": hidden_unwritten_max_abs,
    }


def _run_one(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool,
    separate_compile: bool,
    progress_events: bool,
) -> list[dict[str, Any]]:
    try:
        _emit_progress(config, progress_events, "validate_start")
        if repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {repeat_runs}")
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
        kernel_args = (x, send_meta, recv_meta, w)
        threaded_alias = None
        if config.inbox_storage == "alias":
            inbox_seed_host = jnp.zeros(
                (config.ep_size, config.ep_size, config.inbox_slots, config.block_m, config.hidden_dim),
                dtype=jnp.bfloat16,
            )
            inbox_seed = jax.device_put(inbox_seed_host, NamedSharding(mesh, P(AXIS, None, None, None, None)))
            kernel_args = (x, send_meta, recv_meta, w, inbox_seed)
            threaded_alias = (4, 0)
        _emit_progress(config, progress_events, "jit_start")
        fn = jax.jit(_sharded_kernel(mesh, config))

        (
            compile_time,
            steady_state_times,
            (inbox, meta, seen_payload, seen_meta, hidden),
            lower_compile_time,
            first_run_time,
        ) = _time_jitted(
            fn,
            *kernel_args,
            warmup=warmup,
            steps=steps,
            repeat_runs=repeat_runs,
            separate_compile=separate_compile,
            threaded_alias=threaded_alias,
            progress=lambda event: _emit_progress(config, progress_events, event),
        )
        max_abs_diff = None
        metadata_mismatches = None
        hidden_max_abs_diff = None
        hidden_mean_abs_diff = None
        hidden_all_max_abs_diff = None
        hidden_unwritten_max_abs = None
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
            hidden_all_max_abs_diff = validation["hidden_all_max_abs_diff"]
            hidden_unwritten_max_abs = validation["hidden_unwritten_max_abs"]
        bytes_per_rank = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
        flops_per_rank = None
        if config.implementation != "send_only":
            flops_per_rank = (
                queue_stats["rounded_rows_per_rank_mean"] * config.hidden_dim * config.intermediate_dim * 4
            )
        rows = []
        for repeat_run, steady_state_time in enumerate(steady_state_times):
            rows.append(
                {
                    "kernel": "source_push_inbox",
                    "implementation": config.implementation,
                    "config": asdict(config),
                    "queue_stats": queue_stats,
                    **queue_stats,
                    "compile_time": compile_time,
                    "lower_compile_time": lower_compile_time,
                    "first_run_time": first_run_time,
                    "repeat_run": repeat_run,
                    "repeat_runs": repeat_runs,
                    "steady_state_time": steady_state_time,
                    "bytes_per_rank": bytes_per_rank,
                    "send_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
                    "w13_tflops_per_rank": (
                        None if flops_per_rank is None else flops_per_rank / steady_state_time / 1e12
                    ),
                    "max_abs_diff": max_abs_diff,
                    "metadata_mismatches": metadata_mismatches,
                    "hidden_max_abs_diff": hidden_max_abs_diff,
                    "hidden_mean_abs_diff": hidden_mean_abs_diff,
                    "hidden_all_max_abs_diff": hidden_all_max_abs_diff,
                    "hidden_unwritten_max_abs": hidden_unwritten_max_abs,
                    "error": None,
                }
            )
        return rows
    except Exception as exc:  # noqa: BLE001 - repro rows should capture unsupported candidates.
        if debug_exceptions:
            raise
        return [
            {
                "kernel": "source_push_inbox",
                "implementation": config.implementation,
                "config": asdict(config),
                "compile_time": None,
                "lower_compile_time": None,
                "first_run_time": None,
                "repeat_run": None,
                "repeat_runs": repeat_runs,
                "steady_state_time": None,
                "bytes_per_rank": None,
                "send_gbps_per_rank": None,
                "w13_tflops_per_rank": None,
                "max_abs_diff": None,
                "metadata_mismatches": None,
                "hidden_max_abs_diff": None,
                "hidden_mean_abs_diff": None,
                "hidden_all_max_abs_diff": None,
                "hidden_unwritten_max_abs": None,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        ]
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
    return _run_one(
        config,
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )


def _parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


def _parse_scan_candidates(value: str) -> int:
    if value == "all":
        return 0
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("scan candidates must be non-negative or 'all'")
    return parsed


def _parse_scan_candidates_csv(value: str) -> tuple[int, ...]:
    values = tuple(_parse_scan_candidates(part) for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers or 'all'")
    return values


def _parse_choice_csv(value: str, choices: tuple[str, ...], name: str) -> tuple[str, ...]:
    values = tuple(part for part in value.split(",") if part)
    if not values:
        raise argparse.ArgumentTypeError(f"expected a comma-separated list of {name} values")
    unknown = tuple(part for part in values if part not in choices)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {name} values {unknown}; expected one of {choices}")
    return values


def _parse_receiver_schedule_csv(value: str) -> tuple[str, ...]:
    return _parse_choice_csv(value, RECEIVER_SCHEDULES, "receiver_schedule")


def _parse_scan_order_csv(value: str) -> tuple[str, ...]:
    return _parse_choice_csv(value, SCAN_ORDERS, "scan_order")


def _parse_slot_order_csv(value: str) -> tuple[str, ...]:
    return _parse_choice_csv(value, SLOT_ORDERS, "slot_order")


def _parse_hidden_output_mode_csv(value: str) -> tuple[str, ...]:
    return _parse_choice_csv(value, HIDDEN_OUTPUT_MODES, "hidden_output_mode")


def _parse_hidden_compute_mode_csv(value: str) -> tuple[str, ...]:
    return _parse_choice_csv(value, HIDDEN_COMPUTE_MODES, "hidden_compute_mode")


def _source_push_profile_defaults(argv: Sequence[str] | None = None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    args, _ = pre_parser.parse_known_args(argv)
    return source_push_profile_defaults(args.source_push_profile)


def parse_source_push_inbox_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse source-push inbox benchmark arguments with profile defaults applied."""
    profile_defaults = _source_push_profile_defaults(argv)

    def default(name: str, fallback: Any) -> Any:
        return profile_defaults.get(name, fallback)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-push-profile", choices=SOURCE_PUSH_PROFILES, default="none")
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, default=default("implementation", "send_only"))
    parser.add_argument("--ep-size", type=int, default=default("ep_size", 8))
    parser.add_argument("--entries-per-rank", type=int, default=default("entries_per_rank", 2))
    parser.add_argument("--sweep-entries-per-rank", type=_parse_int_csv, default=None)
    parser.add_argument("--inbox-slots", type=int, default=default("inbox_slots", 2))
    parser.add_argument("--sweep-inbox-slots", type=_parse_int_csv, default=None)
    parser.add_argument("--hidden-dim", type=int, default=default("hidden_dim", 2560))
    parser.add_argument("--intermediate-dim", type=int, default=default("intermediate_dim", 1280))
    parser.add_argument("--block-m", type=int, default=default("block_m", 64))
    parser.add_argument("--sweep-block-m", type=_parse_int_csv, default=None)
    parser.add_argument("--block-n", type=int, default=default("block_n", 128))
    parser.add_argument("--sweep-block-n", type=_parse_int_csv, default=None)
    parser.add_argument("--block-k", type=int, default=default("block_k", 128))
    parser.add_argument("--sweep-block-k", type=_parse_int_csv, default=None)
    parser.add_argument("--n-group", type=int, default=default("n_group", 1))
    parser.add_argument("--sweep-n-groups", type=_parse_int_csv, default=None)
    parser.add_argument("--experts-per-rank", type=int, default=default("experts_per_rank", 32))
    parser.add_argument("--num-send-sms", type=int, default=default("num_send_sms", 4))
    parser.add_argument("--sweep-num-send-sms", type=_parse_int_csv, default=None)
    parser.add_argument("--num-sms", type=int, default=default("num_sms", 16))
    parser.add_argument("--sweep-num-sms", type=_parse_int_csv, default=None)
    parser.add_argument("--send-pipeline-depth", type=int, default=default("send_pipeline_depth", 1))
    parser.add_argument("--sweep-send-pipeline-depth", type=_parse_int_csv, default=None)
    parser.add_argument(
        "--lowering-semantics", choices=tuple(LOWERING_SEMANTICS), default=default("lowering_semantics", "lane")
    )
    parser.add_argument("--traffic-pattern", choices=TRAFFIC_PATTERNS, default=default("traffic_pattern", "next_rank"))
    parser.add_argument("--peer-loop", choices=PEER_LOOP_MODES, default=default("peer_loop", "static"))
    parser.add_argument(
        "--compute-pipeline", choices=COMPUTE_PIPELINE_MODES, default=default("compute_pipeline", "manual")
    )
    parser.add_argument("--max-concurrent-steps", type=int, default=default("max_concurrent_steps", 4))
    parser.add_argument("--sweep-max-concurrent-steps", type=_parse_int_csv, default=None)
    parser.add_argument("--queue-mode", choices=QUEUE_MODES, default=default("queue_mode", "rectangular"))
    parser.add_argument("--metadata-mode", choices=METADATA_MODES, default=default("metadata_mode", "static_recv"))
    parser.add_argument("--output-mode", choices=OUTPUT_MODES, default=default("output_mode", "debug"))
    parser.add_argument(
        "--hidden-output-mode", choices=HIDDEN_OUTPUT_MODES, default=default("hidden_output_mode", "full")
    )
    parser.add_argument("--sweep-hidden-output-mode", type=_parse_hidden_output_mode_csv, default=None)
    parser.add_argument(
        "--hidden-compute-mode", choices=HIDDEN_COMPUTE_MODES, default=default("hidden_compute_mode", "wgmma")
    )
    parser.add_argument("--sweep-hidden-compute-mode", type=_parse_hidden_compute_mode_csv, default=None)
    parser.add_argument("--n-groups-per-job", type=int, default=default("n_groups_per_job", 1))
    parser.add_argument("--sweep-n-groups-per-job", type=_parse_int_csv, default=None)
    parser.add_argument(
        "--receiver-schedule", choices=RECEIVER_SCHEDULES, default=default("receiver_schedule", "fixed_wait")
    )
    parser.add_argument("--sweep-receiver-schedule", type=_parse_receiver_schedule_csv, default=None)
    parser.add_argument("--scan-candidates", type=_parse_scan_candidates, default=default("scan_candidates", 8))
    parser.add_argument("--sweep-scan-candidates", type=_parse_scan_candidates_csv, default=None)
    parser.add_argument("--scan-order", choices=SCAN_ORDERS, default=default("scan_order", "rotated"))
    parser.add_argument("--sweep-scan-order", type=_parse_scan_order_csv, default=None)
    parser.add_argument("--slot-order", choices=SLOT_ORDERS, default=default("slot_order", "current"))
    parser.add_argument("--sweep-slot-order", type=_parse_slot_order_csv, default=None)
    parser.add_argument("--workers-per-slot", type=int, default=default("workers_per_slot", 1))
    parser.add_argument("--sweep-workers-per-slot", type=_parse_int_csv, default=None)
    parser.add_argument("--source-push-mode", choices=SOURCE_PUSH_MODES, default=default("source_push_mode", "packed"))
    parser.add_argument("--inbox-storage", choices=INBOX_STORAGE_MODES, default=default("inbox_storage", "output"))
    parser.add_argument("--routing", choices=ROUTING_MODES, default=default("routing", "balanced"))
    parser.add_argument("--tokens-per-rank", type=int, default=default("tokens_per_rank", 32768))
    parser.add_argument("--topk", type=int, default=default("topk", 4))
    parser.add_argument("--routing-seed", type=int, default=default("routing_seed", 0))
    parser.add_argument(
        "--direct-self-compute", action=argparse.BooleanOptionalAction, default=default("direct_self_compute", False)
    )
    parser.add_argument("--warmup", type=int, default=default("warmup", 1))
    parser.add_argument("--steps", type=int, default=default("steps", 5))
    parser.add_argument("--repeat-runs", type=int, default=default("repeat_runs", 1))
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=default("check", True))
    parser.add_argument(
        "--debug-exceptions", action=argparse.BooleanOptionalAction, default=default("debug_exceptions", False)
    )
    parser.add_argument(
        "--separate-compile", action=argparse.BooleanOptionalAction, default=default("separate_compile", False)
    )
    parser.add_argument(
        "--progress-events", action=argparse.BooleanOptionalAction, default=default("progress_events", False)
    )
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def _parse_args() -> argparse.Namespace:
    return parse_source_push_inbox_args()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_source_push_inbox_args(argv)
    entries_per_rank_values = args.sweep_entries_per_rank or (args.entries_per_rank,)
    inbox_slots_values = args.sweep_inbox_slots or (args.inbox_slots,)
    block_m_values = args.sweep_block_m or (args.block_m,)
    block_n_values = args.sweep_block_n or (args.block_n,)
    block_k_values = args.sweep_block_k or (args.block_k,)
    num_send_sms_values = args.sweep_num_send_sms or (args.num_send_sms,)
    num_sms_values = args.sweep_num_sms or (args.num_sms,)
    send_pipeline_depth_values = args.sweep_send_pipeline_depth or (args.send_pipeline_depth,)
    n_group_values = args.sweep_n_groups or (args.n_group,)
    n_groups_per_job_values = args.sweep_n_groups_per_job or (args.n_groups_per_job,)
    max_concurrent_steps_values = args.sweep_max_concurrent_steps or (args.max_concurrent_steps,)
    receiver_schedule_values = args.sweep_receiver_schedule or (args.receiver_schedule,)
    scan_candidates_values = args.sweep_scan_candidates or (args.scan_candidates,)
    scan_order_values = args.sweep_scan_order or (args.scan_order,)
    slot_order_values = args.sweep_slot_order or (args.slot_order,)
    hidden_output_mode_values = args.sweep_hidden_output_mode or (args.hidden_output_mode,)
    hidden_compute_mode_values = args.sweep_hidden_compute_mode or (args.hidden_compute_mode,)
    workers_per_slot_values = args.sweep_workers_per_slot or (args.workers_per_slot,)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for entries_per_rank in entries_per_rank_values:
        for inbox_slots in inbox_slots_values:
            for block_m in block_m_values:
                for block_n in block_n_values:
                    for block_k in block_k_values:
                        for num_send_sms in num_send_sms_values:
                            for num_sms in num_sms_values:
                                for send_pipeline_depth in send_pipeline_depth_values:
                                    for n_group in n_group_values:
                                        for n_groups_per_job in n_groups_per_job_values:
                                            for max_concurrent_steps in max_concurrent_steps_values:
                                                for receiver_schedule in receiver_schedule_values:
                                                    for scan_candidates in scan_candidates_values:
                                                        for scan_order in scan_order_values:
                                                            for slot_order in slot_order_values:
                                                                for hidden_output_mode in hidden_output_mode_values:
                                                                    for (
                                                                        hidden_compute_mode
                                                                    ) in hidden_compute_mode_values:
                                                                        for (
                                                                            workers_per_slot
                                                                        ) in workers_per_slot_values:
                                                                            config = PushInboxConfig(
                                                                                implementation=args.implementation,
                                                                                ep_size=args.ep_size,
                                                                                entries_per_rank=entries_per_rank,
                                                                                inbox_slots=inbox_slots,
                                                                                hidden_dim=args.hidden_dim,
                                                                                intermediate_dim=args.intermediate_dim,
                                                                                block_m=block_m,
                                                                                block_n=block_n,
                                                                                block_k=block_k,
                                                                                n_group=n_group,
                                                                                n_groups_per_job=n_groups_per_job,
                                                                                receiver_schedule=receiver_schedule,
                                                                                scan_candidates=scan_candidates,
                                                                                scan_order=scan_order,
                                                                                slot_order=slot_order,
                                                                                workers_per_slot=workers_per_slot,
                                                                                source_push_mode=args.source_push_mode,
                                                                                inbox_storage=args.inbox_storage,
                                                                                experts_per_rank=args.experts_per_rank,
                                                                                num_send_sms=num_send_sms,
                                                                                num_sms=num_sms,
                                                                                send_pipeline_depth=send_pipeline_depth,
                                                                                lowering_semantics=args.lowering_semantics,
                                                                                traffic_pattern=args.traffic_pattern,
                                                                                peer_loop=args.peer_loop,
                                                                                compute_pipeline=args.compute_pipeline,
                                                                                max_concurrent_steps=max_concurrent_steps,
                                                                                queue_mode=args.queue_mode,
                                                                                metadata_mode=args.metadata_mode,
                                                                                output_mode=args.output_mode,
                                                                                hidden_output_mode=hidden_output_mode,
                                                                                hidden_compute_mode=hidden_compute_mode,
                                                                                routing=args.routing,
                                                                                tokens_per_rank=args.tokens_per_rank,
                                                                                topk=args.topk,
                                                                                routing_seed=args.routing_seed,
                                                                                direct_self_compute=args.direct_self_compute,
                                                                            )
                                                                        rows = run_source_push_inbox(
                                                                            config,
                                                                            warmup=args.warmup,
                                                                            steps=args.steps,
                                                                            repeat_runs=args.repeat_runs,
                                                                            check=args.check,
                                                                            debug_exceptions=args.debug_exceptions,
                                                                            separate_compile=args.separate_compile,
                                                                            progress_events=args.progress_events,
                                                                        )
                                                                        for row in rows:
                                                                            line = json.dumps(row, sort_keys=True)
                                                                            print(line, flush=True)
                                                                            if args.jsonl:
                                                                                with open(
                                                                                    args.jsonl, "a", encoding="utf-8"
                                                                                ) as f:
                                                                                    print(line, file=f, flush=True)


if __name__ == "__main__":
    main()
