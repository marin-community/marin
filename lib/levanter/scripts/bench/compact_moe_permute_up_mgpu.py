# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact Mosaic GPU MoE permute_up kernel and tuner.

This file is intentionally self-contained. It is a small playground for the
Hopper MGPU MoE "up" half:

    destination-pull token access -> W_gate/W_up matmul -> SiLU(gate) * up

It avoids Marin/Levanter/Haliax dependencies so the kernel body is easy to edit.
The fast pull path takes per-token expert assignments, source-packs each rank's
tokens into a fixed-capacity rank buffer, then lets the destination rank pull
the matching bucket rows directly into WGMMA.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Ref, lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Float, Int


EXPERT_AXIS = "expert"
RING_IMPLEMENTATIONS = {"ring", "ring_copy", "ring_compute", "ring_queue"}
IMPLEMENTATIONS = {"pull", *RING_IMPLEMENTATIONS}


def _is_ring_implementation(implementation: str) -> bool:
    return implementation in RING_IMPLEMENTATIONS


def _ring_stage_for_implementation(implementation: str) -> str:
    if implementation == "ring":
        return "full"
    if implementation == "ring_copy":
        return "copy_only"
    if implementation == "ring_compute":
        return "compute_only"
    if implementation == "ring_queue":
        return "queue_only"
    raise ValueError(f"{implementation=} is not a ring implementation")


@dataclass(frozen=True)
class CompactConfig:
    block_m: int = 64
    block_n: int = 128
    block_k: int = 128
    max_concurrent_steps: int = 4
    grid_block_n: int = 4
    expert_group_size: int = 32
    n_group: int = 2
    copy_tile: int = 512
    copy_rows: int = 1
    capacity_factor: float = 1.25
    max_rows_per_rank: int | None = None
    skip_padded_blocks: bool = False
    ring_block_tokens: int = 64
    ring_max_rows_per_pair: int | None = None
    ring_num_prefetch: int = 2
    ring_max_active_srcs: int = 8
    ring_queue_order: str = "eg_block_src"
    ring_stage: str = "full"
    num_sms: int | None = None

    def validate(self, shape: "BenchShape", *, implementation: str) -> None:
        assignments = shape.assignments_per_rank
        global_experts = shape.ep_size * shape.experts_per_rank
        if assignments % global_experts != 0:
            raise ValueError(
                "compact benchmark requires balanced routing: "
                f"T*K={assignments} must be divisible by EP*E_local={global_experts}"
            )
        if implementation == "static" and shape.routing != "balanced":
            raise ValueError("compact static workqueue currently supports only --routing balanced")
        if self.ring_block_tokens <= 0:
            raise ValueError(f"ring_block_tokens must be positive, got {self.ring_block_tokens}")
        if _is_ring_implementation(implementation) and (
            self.ring_block_tokens < self.block_m or self.ring_block_tokens % self.block_m != 0
        ):
            raise ValueError(
                "compact ring prototype requires ring_block_tokens >= block_m and a multiple of block_m; "
                f"got ring_block_tokens={self.ring_block_tokens}, block_m={self.block_m}"
            )
        max_rows = self.inferred_max_rows_per_rank(shape)
        checks = [
            ("block_m", max_rows, self.block_m),
            ("block_k", shape.hidden_dim, self.block_k),
            ("block_n", shape.intermediate_dim, self.block_n),
            ("expert_group_size", shape.experts_per_rank, self.expert_group_size),
        ]
        for name, value, divisor in checks:
            if divisor <= 0:
                raise ValueError(f"{name} must be positive, got {divisor}")
            if value % divisor != 0:
                raise ValueError(f"{value=} must be divisible by {name}={divisor}")
        if self.max_concurrent_steps <= 0:
            raise ValueError(f"max_concurrent_steps must be positive, got {self.max_concurrent_steps}")
        if self.grid_block_n <= 0:
            raise ValueError(f"grid_block_n must be positive, got {self.grid_block_n}")
        if self.n_group not in (1, 2):
            raise ValueError(f"n_group must be 1 or 2 for the compact pull kernel, got {self.n_group}")
        if (shape.intermediate_dim // self.block_n) % self.n_group != 0:
            raise ValueError(
                "intermediate_dim / block_n must be divisible by n_group; "
                f"got I={shape.intermediate_dim}, block_n={self.block_n}, n_group={self.n_group}"
            )
        if self.num_sms is not None and self.num_sms <= 0:
            raise ValueError(f"num_sms must be positive when set, got {self.num_sms}")
        if self.capacity_factor <= 0:
            raise ValueError(f"capacity_factor must be positive, got {self.capacity_factor}")
        if self.max_rows_per_rank is not None and self.max_rows_per_rank <= 0:
            raise ValueError(f"max_rows_per_rank must be positive when set, got {self.max_rows_per_rank}")
        if self.ring_max_rows_per_pair is not None and self.ring_max_rows_per_pair <= 0:
            raise ValueError(f"ring_max_rows_per_pair must be positive when set, got {self.ring_max_rows_per_pair}")
        if self.ring_max_rows_per_pair is not None and self.ring_max_rows_per_pair % self.ring_block_tokens != 0:
            raise ValueError(
                "ring_max_rows_per_pair must be divisible by ring_block_tokens; "
                f"got ring_max_rows_per_pair={self.ring_max_rows_per_pair}, "
                f"ring_block_tokens={self.ring_block_tokens}"
            )
        if self.ring_num_prefetch <= 0:
            raise ValueError(f"ring_num_prefetch must be positive, got {self.ring_num_prefetch}")
        if self.ring_max_active_srcs <= 0:
            raise ValueError(f"ring_max_active_srcs must be positive, got {self.ring_max_active_srcs}")
        if self.ring_queue_order not in ("eg_block_src", "src_eg_block"):
            raise ValueError(
                "ring_queue_order must be 'eg_block_src' or 'src_eg_block', " f"got {self.ring_queue_order!r}"
            )
        if self.ring_stage not in ("full", "copy_only", "compute_only", "queue_only"):
            raise ValueError(
                "ring_stage must be 'full', 'copy_only', 'compute_only', or 'queue_only', " f"got {self.ring_stage!r}"
            )
        if _is_ring_implementation(implementation) and shape.hidden_dim % self.copy_tile != 0:
            raise ValueError(
                f"ring prototype requires hidden_dim={shape.hidden_dim} divisible by copy_tile={self.copy_tile}"
            )

    def inferred_max_rows_per_rank(self, shape: "BenchShape") -> int:
        if self.max_rows_per_rank is not None:
            return self.max_rows_per_rank
        global_experts = shape.ep_size * shape.experts_per_rank
        requested = int(math.ceil(shape.capacity_per_rank * self.capacity_factor))
        requested = max(requested, shape.capacity_per_rank + global_experts * (self.block_m - 1))
        return int(math.ceil(requested / self.block_m) * self.block_m)

    def inferred_ring_max_rows_per_pair(self, shape: "BenchShape") -> int:
        if self.ring_max_rows_per_pair is not None:
            return self.ring_max_rows_per_pair
        requested = int(math.ceil(shape.rows_per_source_expert * self.capacity_factor))
        return int(math.ceil(requested / self.ring_block_tokens) * self.ring_block_tokens)


@dataclass(frozen=True)
class BenchShape:
    ep_size: int = 8
    tokens_per_rank: int = 32768
    hidden_dim: int = 2560
    intermediate_dim: int = 1280
    experts_per_rank: int = 32
    topk: int = 4
    routing: str = "uniform"

    @property
    def assignments_per_rank(self) -> int:
        return self.tokens_per_rank * self.topk

    @property
    def rows_per_source_expert(self) -> int:
        return self.assignments_per_rank // (self.ep_size * self.experts_per_rank)

    @property
    def capacity_per_rank(self) -> int:
        return self.assignments_per_rank

    def label(self) -> str:
        return (
            f"EP={self.ep_size},T={self.tokens_per_rank},D={self.hidden_dim},"
            f"I={self.intermediate_dim},E_local={self.experts_per_rank},K={self.topk},routing={self.routing}"
        )


def _device_core_count() -> int:
    device = jax.devices()[0]
    return int(getattr(device, "core_count", 1) or 1)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _silu(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def _round_up_to_block(value, block_size: int):
    return ((value + block_size - 1) // block_size) * block_size


def _phase_indices(phase: jax.Array, rank: jax.Array, *, ep_size: int, expert_groups: int):
    del rank, expert_groups
    expert_group = phase // ep_size
    peer_phase = phase - expert_group * ep_size
    return expert_group, peer_phase


def _rank_limited_starts_and_counts(counts, *, max_rows_per_rank: int, block_m: int):
    rounded_counts = _round_up_to_block(counts, block_m)
    starts = jnp.cumsum(rounded_counts, axis=-1, dtype=jnp.int32) - rounded_counts
    available = jnp.maximum(jnp.asarray(max_rows_per_rank, dtype=jnp.int32) - starts, 0)
    accepted_span = jnp.minimum(rounded_counts, available)
    accepted_counts = jnp.minimum(counts, accepted_span)
    return starts, accepted_counts


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePack:
    packed_x: Float[Array, "packed D"]
    # True unclipped assignment counts for each destination rank/local expert.
    send_counts: Int[Array, "EP E"]
    # Original flattened assignment id for each packed row; -1 marks padding.
    packed_assignment: Int[Array, "packed"]


def _source_pack_by_assignment(
    x_local: Float[Array, "T D"],
    selected_experts_local: Int[Array, "T K"],
    *,
    ep_size: int,
    local_experts: int,
    max_rows_per_rank: int,
    block_m: int,
) -> SourcePack:
    """Pack one source rank into block-rounded destination/expert buckets."""

    tokens, hidden_dim = x_local.shape
    _tokens_for_routing, topk = selected_experts_local.shape
    assignments = tokens * topk
    global_experts = ep_size * local_experts
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    token_ids = assignment_ids // topk
    global_expert_ids = selected_experts_local.reshape((assignments,)).astype(jnp.int32)
    sort_idx = jnp.argsort(global_expert_ids)
    sorted_assignment = assignment_ids[sort_idx]
    sorted_token_ids = token_ids[sort_idx]
    sorted_global_expert = global_expert_ids[sort_idx]
    send_counts = jnp.bincount(global_expert_ids, length=global_experts).astype(jnp.int32)
    flat_offsets = jnp.cumsum(send_counts, dtype=jnp.int32) - send_counts
    sorted_position = jnp.arange(assignments, dtype=jnp.int32)
    sorted_local_pos = sorted_position - flat_offsets[sorted_global_expert]

    packed_starts, accepted_counts = _rank_limited_starts_and_counts(
        send_counts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=block_m,
    )
    packed_row = packed_starts[sorted_global_expert] + sorted_local_pos
    packed_row = jnp.where(sorted_local_pos < accepted_counts[sorted_global_expert], packed_row, max_rows_per_rank)
    packed_x = jnp.zeros((max_rows_per_rank, hidden_dim), dtype=x_local.dtype)
    packed_x = packed_x.at[packed_row].set(x_local[sorted_token_ids], mode="drop")
    packed_assignment = jnp.full((max_rows_per_rank,), -1, dtype=jnp.int32)
    packed_assignment = packed_assignment.at[packed_row].set(sorted_assignment, mode="drop")
    return SourcePack(
        packed_x=packed_x,
        send_counts=send_counts.reshape((ep_size, local_experts)),
        packed_assignment=packed_assignment,
    )


def _source_starts_and_counts(
    counts_by_src: Int[Array, "SRC DST E"],
    *,
    ep_size: int,
    local_experts: int,
    max_rows_per_rank: int,
    block_m: int,
):
    flat_counts = counts_by_src.reshape((ep_size, ep_size * local_experts))
    starts, accepted_counts = _rank_limited_starts_and_counts(
        flat_counts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=block_m,
    )
    return (
        starts.reshape((ep_size, ep_size, local_experts)),
        accepted_counts.reshape((ep_size, ep_size, local_experts)),
    )


def _destination_starts_and_counts(
    source_accepted_counts: Int[Array, "SRC DST E"],
    *,
    rank: jax.Array,
    ep_size: int,
    local_experts: int,
    max_rows_per_rank: int,
    block_m: int,
):
    pair_counts = source_accepted_counts[:, rank, :]
    expert_major_counts = jnp.swapaxes(pair_counts, 0, 1).reshape((local_experts * ep_size,))
    starts, accepted_counts = _rank_limited_starts_and_counts(
        expert_major_counts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=block_m,
    )
    return (
        jnp.swapaxes(starts.reshape((local_experts, ep_size)), 0, 1),
        jnp.swapaxes(accepted_counts.reshape((local_experts, ep_size)), 0, 1),
    )


def _dense_metadata_from_source_pack(
    packed_assignment: Int[Array, "packed"],
    counts_by_src: Int[Array, "EP EP E"],
    *,
    ep_size: int,
    local_experts: int,
    max_rows_per_rank: int,
    block_m: int,
) -> tuple[Int[Array, "C"], Int[Array, "C"]]:
    """Build destination dense metadata from packed assignments and counts."""

    rank = lax.axis_index(EXPERT_AXIS)
    capacity = max_rows_per_rank
    assignment_by_src = lax.all_gather(packed_assignment, EXPERT_AXIS)
    source_starts, source_accepted_counts = _source_starts_and_counts(
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=block_m,
    )
    dst_starts, dst_accepted_counts = _destination_starts_and_counts(
        source_accepted_counts,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=block_m,
    )
    local_positions = jnp.arange(max_rows_per_rank, dtype=jnp.int32)
    recv_src_rank = jnp.full((capacity,), -1, dtype=jnp.int32)
    recv_assignment = jnp.full((capacity,), -1, dtype=jnp.int32)

    for src in range(ep_size):
        for expert in range(local_experts):
            count = dst_accepted_counts[src, expert]
            source_rows = source_starts[src, rank, expert] + local_positions
            source_rows = jnp.minimum(source_rows, capacity - 1)
            rows = dst_starts[src, expert] + local_positions
            rows = jnp.where(local_positions < count, rows, capacity)
            recv_src_rank = recv_src_rank.at[rows].set(jnp.int32(src), mode="drop")
            recv_assignment = recv_assignment.at[rows].set(assignment_by_src[src, source_rows], mode="drop")

    return recv_src_rank, recv_assignment


# pseudocode we want:
#
# def body(
#    sorted_x_ref: Float[Ref, "packed D"],
#    expert_counts_ref: Int[Ref, "SRC DST localE"],
#    src_start_ref: Int[Ref, "SRC DST localE"],  # cumsum
#    dst_start_ref: Int[Ref, "SRC localE"], # cumsum, imputed from expert_counts_ref[:, rank, :]
#    w_ref: Float[Ref, "E D twoI"],
#    hidden_ref: Float[Ref, "C I"]
# ):
#   d = this_device_id()
#   # eg_size # number of experts we process per iter
#   src_group_starts = src_start_ref[:, d, ::eg_size]
#   dst_group_starts = dst_start_ref[:, d, ::eg_size]
#   # queue: (block_start, eg, src)  # block_id ragged
#   # probably do this outside kernel
#   queue_info = _compute_queue_info(src_group_starts, dst_group_starts, expert_counts_ref)
#   # per grid tile:
#   prefetch_ring_token = [NUM_PREFETCH, BSIZE, D]GMEM?
#   remote_refs = [mgpu.remote_ref(sorted_x_ref, src, XXX) for src in range(ep_size)]
#
#   # fill prefetch
#   for i in range(NUM_PREFETCH):
#       block_start_i, eg_i, src_i, _ = _compute_queue_indices(queue_info, i)
#       prefetch_ring_token[i] = _load_block(remote_refs, src_i, block_start_i)  # annoying dynamic indexing / just unroll
#
#   loop i using ndloop
#     # invariant: we've prefetched the next NUM_PREFETCH blocks into prefetch_ring
#     # peel off
#     prefetch_idx = i % NUM_PREFETCH
#     block_start_i, eg_i, src_i, dest_i = _compute_queue_indices(queue_info, next_i)
#     # do wgmma loop over experts/token
#     # whatever this is
#
#     next_i = i + NUM_PREFETCH
#     @pl.when(next_i < queue_info.num_blocks)
#     def prefetch_next():
#       block_start_i, eg_i, src_i, _ = _compute_queue_indices(queue_info, next_i)
#       prefetch_ring_token[next_i] = _load_block(remote_refs, src_i, block_start_i)  # annoying dynamic indexing / just unroll
#
#
# Notes:
#   number of expert groups is probably <= 32 or so. Probably closer to 8
#   ideally each SM would mostly do the same EG. this queue discipline doesn't really
#   help with that but that's ok for now
#
#
#


def compact_permute_up_mgpu(
    x_local: Float[Array, "T D"],
    selected_experts_local: Int[Array, "T K"],
    w_gate_up_local: Float[Array, "E D twoI"],
    *,
    config: CompactConfig,
    ep_size: int,
) -> tuple[Float[Array, "C I"], Int[Array, "C"], Int[Array, "C"]]:
    """Destination-pull MGPU permute_up kernel for assignment-driven routing.

    Each source rank first builds a source-packed `x` layout ordered by
    destination global expert and local position. Buckets are padded to a fixed
    block-rounded capacity. The destination rank then
    pulls the matching source rows directly into W13 WGMMA, avoiding both the
    full `recv_x` materialization and the source-push scratch buffer.

    Output rows are expert-major within each destination rank:

        row = sum(round_up(accepted_count[source, rank, expert], block_m)
                  for (expert, source) < (local_expert, src_rank))
              + local_pos
    """

    tokens, hidden_dim = x_local.shape
    routing_tokens, _topk = selected_experts_local.shape
    if routing_tokens != tokens:
        raise ValueError(f"selected_experts T={routing_tokens} must equal x T={tokens}")
    local_experts, weight_hidden_dim, intermediate2 = w_gate_up_local.shape
    if weight_hidden_dim != hidden_dim:
        raise ValueError(f"weight D={weight_hidden_dim} must equal x D={hidden_dim}")
    intermediate = intermediate2 // 2
    if intermediate2 != 2 * intermediate:
        raise ValueError(f"w_gate_up last dim must be 2*I, got {intermediate2}")

    shape = BenchShape(
        ep_size=ep_size,
        tokens_per_rank=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate,
        experts_per_rank=local_experts,
        topk=_topk,
    )
    max_rows_per_rank = config.inferred_max_rows_per_rank(shape)
    capacity = max_rows_per_rank
    expert_group_size = config.expert_group_size
    expert_groups = local_experts // expert_group_size
    total_phases = expert_groups * ep_size
    n_tiles = intermediate // config.block_n
    k_tiles = hidden_dim // config.block_k
    n_group = config.n_group
    n_groups = n_tiles // n_group
    blocks_per_rank = max_rows_per_rank // config.block_m
    chunk_grid_m = expert_group_size * blocks_per_rank
    num_sms = config.num_sms or _device_core_count()

    def body(
        sorted_x_ref: Float[Ref, "packed D"],
        source_starts_ref: Int[Ref, "SRC DST E"],
        dst_starts_ref: Int[Ref, "SRC E"],
        dst_accepted_counts_ref: Int[Ref, "SRC E"],
        w_ref: Float[Ref, "E D twoI"],
        hidden_ref: Float[Ref, "C I"],
    ):
        rank = lax.axis_index(EXPERT_AXIS)

        def _source_row(src_rank, dst, expert, local_pos):
            return source_starts_ref[src_rank, dst, expert] + local_pos

        def _remote_row(src_rank, expert, local_pos):
            return dst_starts_ref[src_rank, expert] + local_pos

        def _phase(phase):
            if config.ring_queue_order == "src_eg_block":
                peer_phase = phase // expert_groups
                expert_group = phase - peer_phase * expert_groups
                return expert_group, peer_phase
            return _phase_indices(
                phase,
                rank,
                ep_size=ep_size,
                expert_groups=expert_groups,
            )

        def _compute_phase(phase):
            expert_group, peer_phase = _phase(phase)
            expert_group_start = expert_group * expert_group_size
            src = (rank + ep_size - peer_phase) % ep_size
            remote_sorted_x_ref = mgpu.remote_ref(sorted_x_ref, src, device_id_type=pl.DeviceIdType.LOGICAL)

            @mgpu.nd_loop((chunk_grid_m * n_groups,), collective_axes="sm")
            def _compute_chunk(loop_info):
                mi, n_group_i = mgpu.planar_snake(loop_info.index[0], (chunk_grid_m, n_groups), 1, config.grid_block_n)
                expert_in_group = mi // blocks_per_rank
                block_in_expert = mi - expert_in_group * blocks_per_rank
                expert = expert_group_start + expert_in_group
                local_pos_start = block_in_expert * config.block_m
                token_offset = _source_row(src, rank, expert, local_pos_start)
                row = _remote_row(src, expert, local_pos_start)
                source_count = dst_accepted_counts_ref[src, expert]
                rounded_count = ((source_count + config.block_m - 1) // config.block_m) * config.block_m
                should_compute = local_pos_start < rounded_count

                @pl.when(should_compute)
                def _compute_live_block():
                    if n_group == 2:

                        def acc_scope(gate_n0_acc_ref, up_n0_acc_ref, gate_n1_acc_ref, up_n1_acc_ref):
                            def smem_scope(
                                lhs_smem, gate_n0_smem, up_n0_smem, gate_n1_smem, up_n1_smem, ready_barrier
                            ):
                                @pl.loop(0, k_tiles)
                                def _wgmma_k(kk):
                                    k_start = kk * config.block_k
                                    n0 = n_group_i * 2
                                    n1 = n0 + 1
                                    lhs_smem[:, :] = mgpu.load(
                                        remote_sorted_x_ref,
                                        (
                                            pl.ds(token_offset, config.block_m),
                                            pl.ds(k_start, config.block_k),
                                        ),
                                        layout=mgpu.Layout.WGMMA,
                                        optimized=False,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds(n0 * config.block_n, config.block_n),
                                        ],
                                        gate_n0_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds((n0 + n_tiles) * config.block_n, config.block_n),
                                        ],
                                        up_n0_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds(n1 * config.block_n, config.block_n),
                                        ],
                                        gate_n1_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds((n1 + n_tiles) * config.block_n, config.block_n),
                                        ],
                                        up_n1_smem,
                                        ready_barrier,
                                    )
                                    mgpu.barrier_wait(ready_barrier)
                                    mgpu.commit_smem()
                                    mgpu.wgmma(gate_n0_acc_ref, lhs_smem, gate_n0_smem)
                                    mgpu.wgmma(up_n0_acc_ref, lhs_smem, up_n0_smem)
                                    mgpu.wgmma(gate_n1_acc_ref, lhs_smem, gate_n1_smem)
                                    mgpu.wgmma(up_n1_acc_ref, lhs_smem, up_n1_smem)
                                    mgpu.wgmma_wait(0)

                            pl.run_scoped(
                                smem_scope,
                                lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=sorted_x_ref.dtype),
                                gate_n0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_n0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                gate_n1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_n1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                ready_barrier=mgpu.Barrier(num_arrivals=4),
                            )
                            return (
                                _silu(gate_n0_acc_ref[...]) * up_n0_acc_ref[...],
                                _silu(gate_n1_acc_ref[...]) * up_n1_acc_ref[...],
                            )

                        hidden_n0, hidden_n1 = pl.run_scoped(
                            acc_scope,
                            gate_n0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_n0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            gate_n1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_n1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )
                        hidden_ref[
                            pl.ds(row, config.block_m), pl.ds(n_group_i * 2 * config.block_n, config.block_n)
                        ] = hidden_n0.astype(hidden_ref.dtype)
                        hidden_ref[
                            pl.ds(row, config.block_m),
                            pl.ds((n_group_i * 2 + 1) * config.block_n, config.block_n),
                        ] = hidden_n1.astype(hidden_ref.dtype)
                    else:

                        def acc_scope(gate_acc_ref, up_acc_ref):
                            def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier):
                                @pl.loop(0, k_tiles)
                                def _wgmma_k(kk):
                                    k_start = kk * config.block_k
                                    n_tile = n_group_i
                                    lhs_smem[:, :] = mgpu.load(
                                        remote_sorted_x_ref,
                                        (
                                            pl.ds(token_offset, config.block_m),
                                            pl.ds(k_start, config.block_k),
                                        ),
                                        layout=mgpu.Layout.WGMMA,
                                        optimized=False,
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
                                lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=sorted_x_ref.dtype),
                                gate_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                ready_barrier=mgpu.Barrier(num_arrivals=2),
                            )
                            return _silu(gate_acc_ref[...]) * up_acc_ref[...]

                        hidden = pl.run_scoped(
                            acc_scope,
                            gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )
                        hidden_ref[pl.ds(row, config.block_m), pl.ds(n_group_i * config.block_n, config.block_n)] = (
                            hidden.astype(hidden_ref.dtype)
                        )

        @pl.loop(0, total_phases)
        def _phase_loop(phase):
            _compute_phase(phase)

    kernel = mgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Warpgroup),
    )

    source_pack = _source_pack_by_assignment(
        x_local,
        selected_experts_local,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    counts_by_src = lax.all_gather(source_pack.send_counts, EXPERT_AXIS)
    source_starts, source_accepted_counts = _source_starts_and_counts(
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    rank = lax.axis_index(EXPERT_AXIS)
    dst_starts, dst_accepted_counts = _destination_starts_and_counts(
        source_accepted_counts,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    hidden = kernel(source_pack.packed_x, source_starts, dst_starts, dst_accepted_counts, w_gate_up_local)
    recv_src_rank, recv_assignment = _dense_metadata_from_source_pack(
        source_pack.packed_assignment,
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    return hidden, recv_src_rank, recv_assignment


def compact_permute_up_mgpu_ring(
    x_local: Float[Array, "T D"],
    selected_experts_local: Int[Array, "T K"],
    w_gate_up_local: Float[Array, "E D twoI"],
    scratch_seed_local: Float[Array, "R rows D"] | None = None,
    *,
    config: CompactConfig,
    ep_size: int,
    return_scratch: bool = False,
) -> tuple[Float[Array, "C I"], Int[Array, "C"], Int[Array, "C"]]:
    """Destination-pull MGPU permute_up with a blocking local-GMEM ring.

    Queue entries are expert-group/source/token-block units. Each entry first
    copies `expert_group_size * ring_block_tokens` remote source-packed token
    rows into a local GMEM ring slot, then WGMMA reads the local slot into SMEM.
    This intentionally avoids peer GMEM refs in the WGMMA/SMEM fill path.
    """

    tokens, hidden_dim = x_local.shape
    routing_tokens, topk = selected_experts_local.shape
    if routing_tokens != tokens:
        raise ValueError(f"selected_experts T={routing_tokens} must equal x T={tokens}")
    local_experts, weight_hidden_dim, intermediate2 = w_gate_up_local.shape
    if weight_hidden_dim != hidden_dim:
        raise ValueError(f"weight D={weight_hidden_dim} must equal x D={hidden_dim}")
    intermediate = intermediate2 // 2
    if intermediate2 != 2 * intermediate:
        raise ValueError(f"w_gate_up last dim must be 2*I, got {intermediate2}")

    shape = BenchShape(
        ep_size=ep_size,
        tokens_per_rank=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate,
        experts_per_rank=local_experts,
        topk=topk,
    )
    max_rows_per_rank = config.inferred_max_rows_per_rank(shape)
    ring_max_rows_per_pair = config.inferred_ring_max_rows_per_pair(shape)
    capacity = max_rows_per_rank
    expert_group_size = config.expert_group_size
    expert_groups = local_experts // expert_group_size
    n_tiles = intermediate // config.block_n
    k_tiles = hidden_dim // config.block_k
    n_group = config.n_group
    n_groups = n_tiles // n_group
    blocks_per_ring_chunk = config.ring_block_tokens // config.block_m
    pair_chunks = ring_max_rows_per_pair // config.ring_block_tokens
    active_srcs = min(config.ring_max_active_srcs, ep_size)
    source_windows = int(math.ceil(ep_size / active_srcs))
    queue_entries = expert_groups * pair_chunks * source_windows * active_srcs
    prologue_entries = min(config.ring_num_prefetch, queue_entries)
    scratch_rows = expert_group_size * config.ring_block_tokens
    copy_tile = config.copy_tile
    d_tiles = hidden_dim // copy_tile
    copy_tiles_per_entry = expert_group_size * blocks_per_ring_chunk * d_tiles
    num_sms = config.num_sms or _device_core_count()
    workers = num_sms
    copy_steps = int(math.ceil(copy_tiles_per_entry / workers))
    stage = config.ring_stage
    do_copy = stage in ("full", "copy_only")
    do_compute = stage in ("full", "compute_only")
    do_ring_semaphores = stage in ("full", "copy_only", "queue_only")
    if stage == "compute_only" and scratch_seed_local is None:
        raise ValueError("ring_compute requires a prefilled scratch_seed_local input")

    def _run_body(
        sorted_x_ref: Float[Ref, "packed D"],
        source_starts_ref: Int[Ref, "SRC DST E"],
        dst_starts_ref: Int[Ref, "SRC E"],
        dst_accepted_counts_ref: Int[Ref, "SRC E"],
        w_ref: Float[Ref, "E D twoI"],
        hidden_ref: Float[Ref, "C I"],
        scratch_ref: Float[Ref, "R rows D"],
    ):
        prefetch_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        compute_sem = pl.get_global(mgpu.SemaphoreType.REGULAR)
        rank = lax.axis_index(EXPERT_AXIS)
        worker_id = lax.axis_index("sm")

        def _decode_queue_entry(queue_index):
            if config.ring_queue_order == "eg_block_src":
                src_slot = queue_index % active_srcs
                q = queue_index // active_srcs
                source_window = q % source_windows
                q = q // source_windows
                chunk_round = q % pair_chunks
                expert_group = q // pair_chunks
            else:
                chunk_round = queue_index % pair_chunks
                q = queue_index // pair_chunks
                expert_group = q % expert_groups
                q = q // expert_groups
                src_slot = q % active_srcs
                source_window = q // active_srcs
            src = source_window * active_srcs + src_slot
            return expert_group, chunk_round, src

        def _source_row(src_rank, dst, expert, local_pos):
            return source_starts_ref[src_rank, dst, expert] + local_pos

        def _remote_row(src_rank, expert, local_pos):
            return dst_starts_ref[src_rank, expert] + local_pos

        def _prefetch_queue_entry(queue_index, slot):
            expert_group, chunk_round, src = _decode_queue_entry(queue_index)
            expert_group_start = expert_group * expert_group_size
            src_safe = jnp.minimum(src, ep_size - 1)
            remote_sorted_x_ref = mgpu.remote_ref(sorted_x_ref, src_safe, device_id_type=pl.DeviceIdType.LOGICAL)
            chunk_pos_start = chunk_round * config.ring_block_tokens

            if do_copy:

                @pl.loop(0, copy_steps)
                def _copy_step(step):
                    linear_tile = step * workers + worker_id

                    @pl.when(linear_tile < copy_tiles_per_entry)
                    def _copy_tile():
                        expert_in_group = linear_tile // (blocks_per_ring_chunk * d_tiles)
                        q = linear_tile - expert_in_group * blocks_per_ring_chunk * d_tiles
                        block_in_chunk = q // d_tiles
                        d_tile = q - block_in_chunk * d_tiles
                        expert = expert_group_start + expert_in_group
                        d_start = d_tile * copy_tile
                        source_count = dst_accepted_counts_ref[src_safe, expert]
                        rounded_count = _round_up_to_block(source_count, config.block_m)
                        local_pos_start = chunk_pos_start + block_in_chunk * config.block_m
                        should_copy = (src < ep_size) & (local_pos_start < rounded_count)
                        source_row = _source_row(src_safe, rank, expert, local_pos_start)
                        scratch_row = expert_in_group * config.ring_block_tokens + block_in_chunk * config.block_m

                        @pl.when(should_copy)
                        def _copy_live_tile():
                            scratch_ref[
                                slot,
                                pl.ds(scratch_row, config.block_m),
                                pl.ds(d_start, copy_tile),
                            ] = remote_sorted_x_ref[
                                pl.ds(source_row, config.block_m),
                                pl.ds(d_start, copy_tile),
                            ]

            pl.semaphore_signal(prefetch_sem, device_id=rank, device_id_type=pl.DeviceIdType.LOGICAL)

        def _compute_queue_entry(queue_index, slot):
            expert_group, chunk_round, src = _decode_queue_entry(queue_index)
            expert_group_start = expert_group * expert_group_size
            src_safe = jnp.minimum(src, ep_size - 1)
            chunk_pos_start = chunk_round * config.ring_block_tokens

            @mgpu.nd_loop((expert_group_size * blocks_per_ring_chunk * n_groups,), collective_axes="sm")
            def _compute_chunk(loop_info):
                expert_in_group, n_group_i = mgpu.planar_snake(
                    loop_info.index[0],
                    (expert_group_size * blocks_per_ring_chunk, n_groups),
                    1,
                    config.grid_block_n,
                )
                block_in_chunk = expert_in_group % blocks_per_ring_chunk
                expert_in_group = expert_in_group // blocks_per_ring_chunk
                expert = expert_group_start + expert_in_group
                source_count = dst_accepted_counts_ref[src_safe, expert]
                rounded_count = _round_up_to_block(source_count, config.block_m)
                local_pos_start = chunk_pos_start + block_in_chunk * config.block_m
                should_compute = (src < ep_size) & (local_pos_start < rounded_count)
                row = _remote_row(src_safe, expert, local_pos_start)
                scratch_row = expert_in_group * config.ring_block_tokens + block_in_chunk * config.block_m

                @pl.when(should_compute)
                def _compute_live_block():
                    if n_group == 2:

                        def acc_scope(gate_n0_acc_ref, up_n0_acc_ref, gate_n1_acc_ref, up_n1_acc_ref):
                            def smem_scope(
                                lhs_smem, gate_n0_smem, up_n0_smem, gate_n1_smem, up_n1_smem, ready_barrier
                            ):
                                @pl.loop(0, k_tiles)
                                def _wgmma_k(kk):
                                    k_start = kk * config.block_k
                                    n0 = n_group_i * 2
                                    n1 = n0 + 1
                                    lhs_smem[:, :] = mgpu.load(
                                        scratch_ref,
                                        (
                                            slot,
                                            pl.ds(scratch_row, config.block_m),
                                            pl.ds(k_start, config.block_k),
                                        ),
                                        layout=mgpu.Layout.WGMMA,
                                        optimized=False,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds(n0 * config.block_n, config.block_n),
                                        ],
                                        gate_n0_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds((n0 + n_tiles) * config.block_n, config.block_n),
                                        ],
                                        up_n0_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds(n1 * config.block_n, config.block_n),
                                        ],
                                        gate_n1_smem,
                                        ready_barrier,
                                    )
                                    mgpu.copy_gmem_to_smem(
                                        w_ref.at[
                                            expert,
                                            pl.ds(k_start, config.block_k),
                                            pl.ds((n1 + n_tiles) * config.block_n, config.block_n),
                                        ],
                                        up_n1_smem,
                                        ready_barrier,
                                    )
                                    mgpu.barrier_wait(ready_barrier)
                                    mgpu.commit_smem()
                                    mgpu.wgmma(gate_n0_acc_ref, lhs_smem, gate_n0_smem)
                                    mgpu.wgmma(up_n0_acc_ref, lhs_smem, up_n0_smem)
                                    mgpu.wgmma(gate_n1_acc_ref, lhs_smem, gate_n1_smem)
                                    mgpu.wgmma(up_n1_acc_ref, lhs_smem, up_n1_smem)
                                    mgpu.wgmma_wait(0)

                            pl.run_scoped(
                                smem_scope,
                                lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=scratch_ref.dtype),
                                gate_n0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_n0_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                gate_n1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_n1_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                ready_barrier=mgpu.Barrier(num_arrivals=4),
                            )
                            return (
                                _silu(gate_n0_acc_ref[...]) * up_n0_acc_ref[...],
                                _silu(gate_n1_acc_ref[...]) * up_n1_acc_ref[...],
                            )

                        hidden_n0, hidden_n1 = pl.run_scoped(
                            acc_scope,
                            gate_n0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_n0_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            gate_n1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_n1_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )
                        hidden_ref[
                            pl.ds(row, config.block_m), pl.ds(n_group_i * 2 * config.block_n, config.block_n)
                        ] = hidden_n0.astype(hidden_ref.dtype)
                        hidden_ref[
                            pl.ds(row, config.block_m),
                            pl.ds((n_group_i * 2 + 1) * config.block_n, config.block_n),
                        ] = hidden_n1.astype(hidden_ref.dtype)
                    else:

                        def acc_scope(gate_acc_ref, up_acc_ref):
                            def smem_scope(lhs_smem, gate_smem, up_smem, ready_barrier):
                                @pl.loop(0, k_tiles)
                                def _wgmma_k(kk):
                                    k_start = kk * config.block_k
                                    n_tile = n_group_i
                                    lhs_smem[:, :] = mgpu.load(
                                        scratch_ref,
                                        (
                                            slot,
                                            pl.ds(scratch_row, config.block_m),
                                            pl.ds(k_start, config.block_k),
                                        ),
                                        layout=mgpu.Layout.WGMMA,
                                        optimized=False,
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
                                lhs_smem=mgpu.SMEM((config.block_m, config.block_k), dtype=scratch_ref.dtype),
                                gate_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                up_smem=mgpu.SMEM((config.block_k, config.block_n), dtype=w_ref.dtype),
                                ready_barrier=mgpu.Barrier(num_arrivals=2),
                            )
                            return _silu(gate_acc_ref[...]) * up_acc_ref[...]

                        hidden = pl.run_scoped(
                            acc_scope,
                            gate_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                            up_acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                        )
                        hidden_ref[pl.ds(row, config.block_m), pl.ds(n_group_i * config.block_n, config.block_n)] = (
                            hidden.astype(hidden_ref.dtype)
                        )

        if do_ring_semaphores:

            @pl.loop(0, prologue_entries)
            def _prefetch_prologue(queue_index):
                _prefetch_queue_entry(queue_index, queue_index)

        @pl.loop(0, queue_entries)
        def _queue_loop(queue_index):
            slot = queue_index % config.ring_num_prefetch
            if do_ring_semaphores:
                pl.semaphore_wait(prefetch_sem, value=(queue_index + 1) * workers, decrement=False)
            if do_compute:
                _compute_queue_entry(queue_index, slot)
            if stage == "queue_only":

                @pl.when(worker_id == 0)
                def _touch_output():
                    hidden_ref[0, 0] = jnp.asarray(queue_index, dtype=hidden_ref.dtype)

            if do_ring_semaphores:
                pl.semaphore_signal(compute_sem, device_id=rank, device_id_type=pl.DeviceIdType.LOGICAL)
                pl.semaphore_wait(compute_sem, value=(queue_index + 1) * workers, decrement=False)
                next_index = queue_index + config.ring_num_prefetch

                @pl.when(next_index < queue_entries)
                def _prefetch_next():
                    _prefetch_queue_entry(next_index, slot)

    if stage == "compute_only":

        def body(
            sorted_x_ref: Float[Ref, "packed D"],
            source_starts_ref: Int[Ref, "SRC DST E"],
            dst_starts_ref: Int[Ref, "SRC E"],
            dst_accepted_counts_ref: Int[Ref, "SRC E"],
            w_ref: Float[Ref, "E D twoI"],
            scratch_ref: Float[Ref, "R rows D"],
            hidden_ref: Float[Ref, "C I"],
        ):
            _run_body(
                sorted_x_ref,
                source_starts_ref,
                dst_starts_ref,
                dst_accepted_counts_ref,
                w_ref,
                hidden_ref,
                scratch_ref,
            )

        kernel = mgpu.kernel(
            body,
            out_shape=jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
            grid=(num_sms,),
            grid_names=("sm",),
            compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Warpgroup),
        )
    else:

        def body(
            sorted_x_ref: Float[Ref, "packed D"],
            source_starts_ref: Int[Ref, "SRC DST E"],
            dst_starts_ref: Int[Ref, "SRC E"],
            dst_accepted_counts_ref: Int[Ref, "SRC E"],
            w_ref: Float[Ref, "E D twoI"],
            hidden_ref: Float[Ref, "C I"],
            scratch_ref: Float[Ref, "R rows D"],
        ):
            _run_body(
                sorted_x_ref,
                source_starts_ref,
                dst_starts_ref,
                dst_accepted_counts_ref,
                w_ref,
                hidden_ref,
                scratch_ref,
            )

        kernel = mgpu.kernel(
            body,
            out_shape=[
                jax.ShapeDtypeStruct((capacity, intermediate), x_local.dtype),
                jax.ShapeDtypeStruct((config.ring_num_prefetch, scratch_rows, hidden_dim), x_local.dtype),
            ],
            grid=(num_sms,),
            grid_names=("sm",),
            compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Warpgroup),
        )

    source_pack = _source_pack_by_assignment(
        x_local,
        selected_experts_local,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    counts_by_src = lax.all_gather(source_pack.send_counts, EXPERT_AXIS)
    source_starts, source_accepted_counts = _source_starts_and_counts(
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    rank = lax.axis_index(EXPERT_AXIS)
    dst_starts, dst_accepted_counts = _destination_starts_and_counts(
        source_accepted_counts,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    if stage == "compute_only":
        hidden = kernel(
            source_pack.packed_x,
            source_starts,
            dst_starts,
            dst_accepted_counts,
            w_gate_up_local,
            scratch_seed_local,
        )
        scratch = None
    else:
        hidden, scratch = kernel(
            source_pack.packed_x,
            source_starts,
            dst_starts,
            dst_accepted_counts,
            w_gate_up_local,
        )
    recv_src_rank, recv_assignment = _dense_metadata_from_source_pack(
        source_pack.packed_assignment,
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    if return_scratch:
        return hidden, recv_src_rank, recv_assignment, scratch
    return hidden, recv_src_rank, recv_assignment


def compact_reference(
    x_local: Float[Array, "T D"],
    selected_experts_local: Int[Array, "T K"],
    w_gate_up_local: Float[Array, "E D twoI"],
    *,
    config: CompactConfig,
    ep_size: int,
) -> tuple[Float[Array, "C I"], Int[Array, "C"], Int[Array, "C"]]:
    """Slow assignment-driven JAX reference for value checks."""

    tokens, hidden_dim = x_local.shape
    _routing_tokens, topk = selected_experts_local.shape
    local_experts, _weight_hidden_dim, intermediate2 = w_gate_up_local.shape
    intermediate = intermediate2 // 2
    assignments = tokens * topk
    shape = BenchShape(
        ep_size=ep_size,
        tokens_per_rank=tokens,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate,
        experts_per_rank=local_experts,
        topk=topk,
    )
    max_rows_per_rank = config.inferred_max_rows_per_rank(shape)
    capacity = max_rows_per_rank
    rank = lax.axis_index(EXPERT_AXIS)
    x_by_src = lax.all_gather(x_local, EXPERT_AXIS)
    selected_experts_by_src = lax.all_gather(selected_experts_local, EXPERT_AXIS)
    selected_experts_flat = selected_experts_by_src.reshape((ep_size, assignments))
    counts_by_src_global_expert = jnp.stack(
        [
            jnp.bincount(selected_experts_flat[src], length=ep_size * local_experts).astype(jnp.int32)
            for src in range(ep_size)
        ]
    )
    counts_by_src = counts_by_src_global_expert.reshape((ep_size, ep_size, local_experts))
    _source_starts, source_accepted_counts = _source_starts_and_counts(
        counts_by_src,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    dst_starts, dst_accepted_counts = _destination_starts_and_counts(
        source_accepted_counts,
        rank=rank,
        ep_size=ep_size,
        local_experts=local_experts,
        max_rows_per_rank=max_rows_per_rank,
        block_m=config.block_m,
    )
    hidden = jnp.zeros((capacity, intermediate), dtype=x_local.dtype)
    recv_src_rank = jnp.full((capacity,), -1, dtype=jnp.int32)
    recv_assignment = jnp.full((capacity,), -1, dtype=jnp.int32)
    assignment_ids = jnp.arange(assignments, dtype=jnp.int32)
    sentinel = jnp.full((assignments,), assignments, dtype=jnp.int32)
    local_positions = jnp.arange(max_rows_per_rank, dtype=jnp.int32)

    for expert in range(local_experts):
        gate_w = w_gate_up_local[expert, :, :intermediate]
        up_w = w_gate_up_local[expert, :, intermediate:]
        global_expert = rank * local_experts + expert
        for src in range(ep_size):
            source_assignments = selected_experts_flat[src]
            kept_assignment_ids = jnp.where(source_assignments == global_expert, assignment_ids, sentinel)
            kept_assignment_ids = jnp.sort(kept_assignment_ids)
            kept_assignment_ids = jnp.pad(
                kept_assignment_ids,
                (0, max_rows_per_rank - assignments),
                constant_values=assignments,
            )[:max_rows_per_rank]
            valid = kept_assignment_ids < assignments
            token_ids = kept_assignment_ids // topk
            token_ids = jnp.minimum(token_ids, tokens - 1)
            x_rows = x_by_src[src, token_ids]
            x_rows = jnp.where(valid[:, None], x_rows, jnp.zeros((max_rows_per_rank, hidden_dim), x_local.dtype))
            gate = x_rows @ gate_w
            up = x_rows @ up_w
            out = _silu(gate) * up
            row_count = dst_accepted_counts[src, expert]
            rounded_count = _round_up_to_block(row_count, config.block_m)
            row_start = dst_starts[src, expert]
            rows = row_start + local_positions
            hidden_rows = jnp.where(local_positions < rounded_count, rows, capacity)
            hidden = hidden.at[hidden_rows].set(out, mode="drop")
            metadata_rows = jnp.where(local_positions < row_count, rows, capacity)
            recv_src_rank = recv_src_rank.at[metadata_rows].set(jnp.int32(src), mode="drop")
            recv_assignment = recv_assignment.at[metadata_rows].set(kept_assignment_ids, mode="drop")

    return hidden, recv_src_rank, recv_assignment


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size != ep_size:
        raise ValueError(f"requested EP={ep_size}, but only {devices.size} devices are visible")
    return Mesh(devices, (EXPERT_AXIS,))


def _make_inputs(shape: BenchShape, *, dtype: jnp.dtype, seed: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    key_x, key_w, key_routing = jax.random.split(jax.random.PRNGKey(seed), 3)
    x = jax.random.normal(key_x, (shape.ep_size, shape.tokens_per_rank, shape.hidden_dim), dtype=dtype) * 0.02
    w = (
        jax.random.normal(
            key_w,
            (shape.ep_size, shape.experts_per_rank, shape.hidden_dim, 2 * shape.intermediate_dim),
            dtype=dtype,
        )
        * 0.02
    )
    global_experts = shape.ep_size * shape.experts_per_rank
    if shape.routing == "balanced":
        routing = jnp.arange(shape.assignments_per_rank, dtype=jnp.int32) // shape.rows_per_source_expert
        routing = jnp.broadcast_to(routing.reshape((shape.tokens_per_rank, shape.topk)), (shape.ep_size, -1, -1))
    elif shape.routing == "uniform":
        routing = jax.random.randint(
            key_routing,
            (shape.ep_size, shape.tokens_per_rank, shape.topk),
            minval=0,
            maxval=global_experts,
            dtype=jnp.int32,
        )
    else:
        raise ValueError(f"unknown routing={shape.routing!r}; expected uniform or balanced")
    return x, routing, w


def _shard_inputs(
    mesh: Mesh, x: jax.Array, routing: jax.Array, w: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    x = jax.device_put(x, NamedSharding(mesh, P(EXPERT_AXIS, None, None)))
    routing = jax.device_put(routing, NamedSharding(mesh, P(EXPERT_AXIS, None, None)))
    w = jax.device_put(w, NamedSharding(mesh, P(EXPERT_AXIS, None, None, None)))
    return x, routing, w


def _ring_scratch_shape(config: CompactConfig, shape: BenchShape) -> tuple[int, int, int]:
    scratch_rows = config.expert_group_size * config.ring_block_tokens
    return config.ring_num_prefetch, scratch_rows, shape.hidden_dim


def _make_ring_scratch_seed(
    mesh: Mesh,
    shape: BenchShape,
    config: CompactConfig,
    *,
    dtype: jnp.dtype,
    seed: int,
) -> jax.Array:
    scratch_shape = (shape.ep_size, *_ring_scratch_shape(config, shape))
    scratch = jax.random.normal(jax.random.PRNGKey(seed), scratch_shape, dtype=dtype) * 0.02
    return jax.device_put(scratch, NamedSharding(mesh, P(EXPERT_AXIS, None, None, None)))


def _sharded_kernel(mesh: Mesh, config: CompactConfig, ep_size: int, topk: int, implementation: str):
    del topk
    if implementation == "ring_compute":

        def local_fn(x_local, routing_local, w_local, scratch_local):
            x_local = x_local[0]
            routing_local = routing_local[0]
            w_local = w_local[0]
            scratch_local = scratch_local[0]
            hidden, src_rank, assignment = compact_permute_up_mgpu_ring(
                x_local,
                routing_local,
                w_local,
                scratch_local,
                config=config,
                ep_size=ep_size,
            )
            return hidden[None, ...], src_rank[None, ...], assignment[None, ...]

        return shard_map(
            local_fn,
            mesh=mesh,
            in_specs=(
                P(EXPERT_AXIS, None, None),
                P(EXPERT_AXIS, None, None),
                P(EXPERT_AXIS, None, None, None),
                P(EXPERT_AXIS, None, None, None),
            ),
            out_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None), P(EXPERT_AXIS, None)),
            check_vma=False,
        )

    if implementation == "ring_copy":

        def local_fn(x_local, routing_local, w_local):
            x_local = x_local[0]
            routing_local = routing_local[0]
            w_local = w_local[0]
            hidden, src_rank, assignment, scratch = compact_permute_up_mgpu_ring(
                x_local,
                routing_local,
                w_local,
                config=config,
                ep_size=ep_size,
                return_scratch=True,
            )
            return hidden[None, ...], src_rank[None, ...], assignment[None, ...], scratch[None, ...]

        return shard_map(
            local_fn,
            mesh=mesh,
            in_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None, None)),
            out_specs=(
                P(EXPERT_AXIS, None, None),
                P(EXPERT_AXIS, None),
                P(EXPERT_AXIS, None),
                P(EXPERT_AXIS, None, None, None),
            ),
            check_vma=False,
        )

    def local_fn(x_local, routing_local, w_local):
        x_local = x_local[0]
        routing_local = routing_local[0]
        w_local = w_local[0]
        if implementation == "pull":
            hidden, src_rank, assignment = compact_permute_up_mgpu(
                x_local,
                routing_local,
                w_local,
                config=config,
                ep_size=ep_size,
            )
            return hidden[None, ...], src_rank[None, ...], assignment[None, ...]
        if implementation in ("ring", "ring_queue"):
            hidden, src_rank, assignment = compact_permute_up_mgpu_ring(
                x_local,
                routing_local,
                w_local,
                config=config,
                ep_size=ep_size,
            )
            return hidden[None, ...], src_rank[None, ...], assignment[None, ...]
        raise ValueError(f"unknown implementation={implementation!r}")

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None, None)),
        out_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None), P(EXPERT_AXIS, None)),
        check_vma=False,
    )


def _sharded_reference(mesh: Mesh, config: CompactConfig, ep_size: int, topk: int):
    def local_fn(x_local, routing_local, w_local):
        x_local = x_local[0]
        routing_local = routing_local[0]
        w_local = w_local[0]
        hidden, src_rank, assignment = compact_reference(
            x_local,
            routing_local,
            w_local,
            config=config,
            ep_size=ep_size,
        )
        return hidden[None, ...], src_rank[None, ...], assignment[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None, None, None)),
        out_specs=(P(EXPERT_AXIS, None, None), P(EXPERT_AXIS, None), P(EXPERT_AXIS, None)),
        check_vma=False,
    )


def _block_until_ready(value: Any) -> Any:
    return jax.tree.map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)


def _time_jitted(fn, *args, warmup: int, steps: int) -> tuple[float, float, Any]:
    start = time.perf_counter()
    out = fn(*args)
    _block_until_ready(out)
    compile_time = time.perf_counter() - start
    for _ in range(warmup):
        out = fn(*args)
        _block_until_ready(out)
    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        _block_until_ready(out)
    steady_state = (time.perf_counter() - start) / steps
    return compile_time, steady_state, out


def _estimate_flops(shape: BenchShape) -> float:
    return float(4 * shape.assignments_per_rank * shape.hidden_dim * shape.intermediate_dim)


def _estimate_dispatch_bytes(shape: BenchShape, dtype_bytes: int) -> float:
    return float(shape.assignments_per_rank * shape.hidden_dim * dtype_bytes)


def _rank_limited_starts_and_counts_np(
    counts: np.ndarray, *, max_rows_per_rank: int, block_m: int
) -> tuple[np.ndarray, np.ndarray]:
    rounded_counts = ((counts + block_m - 1) // block_m) * block_m
    starts = np.cumsum(rounded_counts, axis=-1, dtype=np.int64) - rounded_counts
    available = np.maximum(max_rows_per_rank - starts, 0)
    accepted_span = np.minimum(rounded_counts, available)
    accepted_counts = np.minimum(counts, accepted_span)
    return starts.astype(np.int64), accepted_counts.astype(np.int64)


def _source_accepted_counts_np(routing: np.ndarray, shape: BenchShape, config: CompactConfig) -> np.ndarray:
    global_experts = shape.ep_size * shape.experts_per_rank
    counts_by_src = np.stack(
        [
            np.bincount(routing[src].reshape((-1,)), minlength=global_experts)
            .astype(np.int64)
            .reshape((shape.ep_size, shape.experts_per_rank))
            for src in range(shape.ep_size)
        ],
        axis=0,
    )
    flat_counts = counts_by_src.reshape((shape.ep_size, global_experts))
    _starts, accepted_counts = _rank_limited_starts_and_counts_np(
        flat_counts,
        max_rows_per_rank=config.inferred_max_rows_per_rank(shape),
        block_m=config.block_m,
    )
    return accepted_counts.reshape((shape.ep_size, shape.ep_size, shape.experts_per_rank))


def _decode_ring_queue_entry_np(
    queue_index: int,
    *,
    config: CompactConfig,
    active_srcs: int,
    source_windows: int,
    expert_groups: int,
    pair_chunks: int,
) -> tuple[int, int, int]:
    if config.ring_queue_order == "eg_block_src":
        src_slot = queue_index % active_srcs
        q = queue_index // active_srcs
        source_window = q % source_windows
        q = q // source_windows
        chunk_round = q % pair_chunks
        expert_group = q // pair_chunks
    else:
        chunk_round = queue_index % pair_chunks
        q = queue_index // pair_chunks
        expert_group = q % expert_groups
        q = q // expert_groups
        src_slot = q % active_srcs
        source_window = q // active_srcs
    return expert_group, chunk_round, source_window * active_srcs + src_slot


def _ring_queue_stats(shape: BenchShape, config: CompactConfig, routing: jax.Array) -> dict[str, Any] | None:
    if config.ring_block_tokens < config.block_m or config.ring_block_tokens % config.block_m != 0:
        return {
            "supported": False,
            "unsupported_reason": "ring_block_tokens must be >= block_m and divisible by block_m",
        }
    if shape.hidden_dim % config.copy_tile != 0:
        return {
            "supported": False,
            "unsupported_reason": "hidden_dim must be divisible by copy_tile",
        }
    expert_group_size = config.expert_group_size
    expert_groups = shape.experts_per_rank // expert_group_size
    ring_max_rows_per_pair = config.inferred_ring_max_rows_per_pair(shape)
    blocks_per_ring_chunk = config.ring_block_tokens // config.block_m
    pair_chunks = ring_max_rows_per_pair // config.ring_block_tokens
    active_srcs = min(config.ring_max_active_srcs, shape.ep_size)
    source_windows = int(math.ceil(shape.ep_size / active_srcs))
    queue_entries = expert_groups * pair_chunks * source_windows * active_srcs
    d_tiles = shape.hidden_dim // config.copy_tile
    n_tiles = shape.intermediate_dim // config.block_n
    n_groups = n_tiles // config.n_group
    source_accepted_counts = _source_accepted_counts_np(np.asarray(routing), shape, config)

    live_entries_by_rank: list[int] = []
    no_live_expert_entries_by_rank: list[int] = []
    no_live_token_block_entries_by_rank: list[int] = []
    live_copy_tiles_by_rank: list[int] = []
    live_compute_blocks_by_rank: list[int] = []
    total_copy_tiles = queue_entries * expert_group_size * blocks_per_ring_chunk * d_tiles
    total_compute_blocks = queue_entries * expert_group_size * blocks_per_ring_chunk * n_groups

    block_offsets = np.arange(blocks_per_ring_chunk, dtype=np.int64) * config.block_m
    for dst in range(shape.ep_size):
        dst_counts = source_accepted_counts[:, dst, :]
        live_entries = 0
        no_live_expert_entries = 0
        no_live_token_block_entries = 0
        live_copy_tiles = 0
        live_compute_blocks = 0
        for queue_index in range(queue_entries):
            expert_group, chunk_round, src = _decode_ring_queue_entry_np(
                queue_index,
                config=config,
                active_srcs=active_srcs,
                source_windows=source_windows,
                expert_groups=expert_groups,
                pair_chunks=pair_chunks,
            )
            if src >= shape.ep_size:
                no_live_expert_entries += 1
                no_live_token_block_entries += 1
                continue
            expert_start = expert_group * expert_group_size
            counts = dst_counts[src, expert_start : expert_start + expert_group_size]
            rounded_counts = ((counts + config.block_m - 1) // config.block_m) * config.block_m
            chunk_start = chunk_round * config.ring_block_tokens
            live_block_mask = chunk_start + block_offsets[None, :] < rounded_counts[:, None]
            live_token_blocks = int(np.sum(live_block_mask))
            has_live_expert = bool(np.any(counts > 0))
            if live_token_blocks > 0:
                live_entries += 1
            if not has_live_expert:
                no_live_expert_entries += 1
            if live_token_blocks == 0:
                no_live_token_block_entries += 1
            live_copy_tiles += live_token_blocks * d_tiles
            live_compute_blocks += live_token_blocks * n_groups
        live_entries_by_rank.append(live_entries)
        no_live_expert_entries_by_rank.append(no_live_expert_entries)
        no_live_token_block_entries_by_rank.append(no_live_token_block_entries)
        live_copy_tiles_by_rank.append(live_copy_tiles)
        live_compute_blocks_by_rank.append(live_compute_blocks)

    live_entries_arr = np.asarray(live_entries_by_rank, dtype=np.int64)
    no_live_expert_arr = np.asarray(no_live_expert_entries_by_rank, dtype=np.int64)
    no_live_token_block_arr = np.asarray(no_live_token_block_entries_by_rank, dtype=np.int64)
    live_copy_tiles_arr = np.asarray(live_copy_tiles_by_rank, dtype=np.int64)
    live_compute_blocks_arr = np.asarray(live_compute_blocks_by_rank, dtype=np.int64)
    all_rank_entries = queue_entries * shape.ep_size
    all_rank_copy_tiles = total_copy_tiles * shape.ep_size
    all_rank_compute_blocks = total_compute_blocks * shape.ep_size
    masked_copy_tiles = all_rank_copy_tiles - int(np.sum(live_copy_tiles_arr))
    masked_compute_blocks = all_rank_compute_blocks - int(np.sum(live_compute_blocks_arr))
    return {
        "supported": True,
        "total_queue_entries_per_rank": int(queue_entries),
        "total_queue_entries_all_ranks": int(all_rank_entries),
        "live_queue_entries_all_ranks": int(np.sum(live_entries_arr)),
        "live_queue_entries_per_rank_min": int(np.min(live_entries_arr)),
        "live_queue_entries_per_rank_mean": float(np.mean(live_entries_arr)),
        "live_queue_entries_per_rank_max": int(np.max(live_entries_arr)),
        "live_queue_entry_fraction": float(np.sum(live_entries_arr) / all_rank_entries),
        "entries_with_no_live_experts_all_ranks": int(np.sum(no_live_expert_arr)),
        "entries_with_no_live_token_blocks_all_ranks": int(np.sum(no_live_token_block_arr)),
        "no_live_token_block_entry_fraction": float(np.sum(no_live_token_block_arr) / all_rank_entries),
        "total_copy_tiles_all_ranks": int(all_rank_copy_tiles),
        "live_copy_tiles_all_ranks": int(np.sum(live_copy_tiles_arr)),
        "masked_copy_tiles_all_ranks": int(masked_copy_tiles),
        "masked_copy_tile_fraction": float(masked_copy_tiles / all_rank_copy_tiles) if all_rank_copy_tiles else 0.0,
        "total_compute_blocks_all_ranks": int(all_rank_compute_blocks),
        "live_compute_blocks_all_ranks": int(np.sum(live_compute_blocks_arr)),
        "masked_compute_blocks_all_ranks": int(masked_compute_blocks),
        "masked_compute_block_fraction": (
            float(masked_compute_blocks / all_rank_compute_blocks) if all_rank_compute_blocks else 0.0
        ),
    }


def _row(
    *,
    shape: BenchShape,
    config: CompactConfig,
    implementation: str,
    compile_time: float | None,
    steady_state_time: float | None,
    error: str | None,
    max_abs_diff: float | None,
    mean_abs_diff: float | None,
    metadata_mismatches: int | None,
    queue_stats: dict[str, Any] | None,
) -> dict[str, Any]:
    device = jax.devices()[0]
    dtype = "bfloat16"
    tflops = None
    if steady_state_time and not error and implementation not in ("ring_copy", "ring_queue"):
        tflops = _estimate_flops(shape) / steady_state_time / 1e12
    return {
        "kernel": "compact_moe_permute_up_mgpu",
        "implementation": f"pallas_mgpu_compact_{implementation}",
        "shape": shape.label(),
        "dtype": dtype,
        "backend": jax.default_backend(),
        "device_type": getattr(device, "device_kind", str(device)),
        "device_count": len(jax.devices()),
        "block_sizes": json.dumps(asdict(config), sort_keys=True),
        "compile_time": compile_time,
        "steady_state_time": steady_state_time,
        "effective_tflops_per_rank": tflops,
        "estimated_flops_per_rank": _estimate_flops(shape),
        "estimated_dispatch_bytes_per_rank": _estimate_dispatch_bytes(shape, 2),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "metadata_mismatches": metadata_mismatches,
        "queue_stats": queue_stats,
        "error": error,
        "git_sha": _git_sha(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": json.dumps({k: os.environ[k] for k in sorted(os.environ) if k.startswith("JAX_")}),
    }


def _run_one(
    *,
    shape: BenchShape,
    config: CompactConfig,
    implementation: str,
    warmup: int,
    steps: int,
    seed: int,
    check: bool,
) -> dict[str, Any]:
    if _is_ring_implementation(implementation):
        config = replace(config, ring_stage=_ring_stage_for_implementation(implementation))
    queue_stats = None
    try:
        config.validate(shape, implementation=implementation)
        mesh = _make_mesh(shape.ep_size)
        x_host, routing_host, w_host = _make_inputs(shape, dtype=jnp.bfloat16, seed=seed)
        if _is_ring_implementation(implementation):
            queue_stats = _ring_queue_stats(shape, config, routing_host)
        x, routing, w = _shard_inputs(mesh, x_host, routing_host, w_host)
        kernel_fn = jax.jit(_sharded_kernel(mesh, config, shape.ep_size, shape.topk, implementation))
        if implementation == "ring_compute":
            scratch = _make_ring_scratch_seed(mesh, shape, config, dtype=jnp.bfloat16, seed=seed + 17)
            compile_time, steady_state_time, candidate = _time_jitted(
                kernel_fn,
                x,
                routing,
                w,
                scratch,
                warmup=warmup,
                steps=steps,
            )
        else:
            compile_time, steady_state_time, candidate = _time_jitted(
                kernel_fn,
                x,
                routing,
                w,
                warmup=warmup,
                steps=steps,
            )
        max_abs_diff = None
        mean_abs_diff = None
        metadata_mismatches = None
        if check and implementation in ("pull", "ring"):
            reference_fn = jax.jit(_sharded_reference(mesh, config, shape.ep_size, shape.topk))
            reference = reference_fn(x, routing, w)
            _block_until_ready(reference)
            hidden, src_rank, assignment = candidate
            ref_hidden, ref_src_rank, ref_assignment = reference
            valid_rows = np.asarray(ref_src_rank) >= 0
            diff = np.abs(
                np.asarray(hidden, dtype=np.float32)[valid_rows] - np.asarray(ref_hidden, dtype=np.float32)[valid_rows]
            )
            max_abs_diff = float(np.max(diff))
            mean_abs_diff = float(np.mean(diff))
            metadata_mismatches = int(
                np.sum(np.asarray(src_rank) != np.asarray(ref_src_rank))
                + np.sum(np.asarray(assignment) != np.asarray(ref_assignment))
            )
        return _row(
            shape=shape,
            config=config,
            implementation=implementation,
            compile_time=compile_time,
            steady_state_time=steady_state_time,
            error=None,
            max_abs_diff=max_abs_diff,
            mean_abs_diff=mean_abs_diff,
            metadata_mismatches=metadata_mismatches,
            queue_stats=queue_stats,
        )
    except Exception as exc:  # noqa: BLE001 - tuning rows should capture unsupported candidates.
        return _row(
            shape=shape,
            config=config,
            implementation=implementation,
            compile_time=None,
            steady_state_time=None,
            error=f"{type(exc).__name__}: {exc}",
            max_abs_diff=None,
            mean_abs_diff=None,
            metadata_mismatches=None,
            queue_stats=queue_stats,
        )


def _int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def _str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _optional_int_list(values: str | None, fallback: int) -> list[int]:
    return _int_list(values) if values else [fallback]


def _optional_str_list(values: str | None, fallback: str) -> list[str]:
    return _str_list(values) if values else [fallback]


def _config_sweep(args: argparse.Namespace) -> list[CompactConfig]:
    configs = []
    for (
        expert_group_size,
        n_group,
        copy_tile,
        copy_rows,
        max_concurrent_steps,
        grid_block_n,
        ring_block_tokens,
        ring_num_prefetch,
        ring_max_active_srcs,
        ring_queue_order,
    ) in itertools.product(
        _int_list(args.expert_group_sizes),
        _int_list(args.n_groups),
        _int_list(args.copy_tiles),
        _int_list(args.copy_rows),
        _int_list(args.max_concurrent_steps_values),
        _int_list(args.grid_block_n_values),
        _optional_int_list(args.ring_block_tokens_values, args.ring_block_tokens),
        _optional_int_list(args.ring_num_prefetch_values, args.ring_num_prefetch),
        _optional_int_list(args.ring_max_active_srcs_values, args.ring_max_active_srcs),
        _optional_str_list(args.ring_queue_orders, args.ring_queue_order),
    ):
        configs.append(
            CompactConfig(
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                max_concurrent_steps=max_concurrent_steps,
                grid_block_n=grid_block_n,
                expert_group_size=expert_group_size,
                n_group=n_group,
                copy_tile=copy_tile,
                copy_rows=copy_rows,
                capacity_factor=args.capacity_factor,
                max_rows_per_rank=args.max_rows_per_rank,
                skip_padded_blocks=args.skip_padded_blocks,
                ring_block_tokens=ring_block_tokens,
                ring_max_rows_per_pair=args.ring_max_rows_per_pair,
                ring_num_prefetch=ring_num_prefetch,
                ring_max_active_srcs=ring_max_active_srcs,
                ring_queue_order=ring_queue_order,
                num_sms=args.num_sms,
            )
        )
    return configs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=32768)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--routing",
        type=str,
        choices=("uniform", "balanced"),
        default="uniform",
        help="Assignment generator. uniform is roughly balanced in expectation; balanced is exact synthetic routing.",
    )
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument(
        "--implementations",
        type=str,
        default="pull",
        help="Comma-separated compact implementations: pull,ring,ring_copy,ring_compute,ring_queue.",
    )
    parser.add_argument("--expert-group-sizes", type=str, default="32")
    parser.add_argument("--n-groups", type=str, default="2")
    parser.add_argument("--copy-tiles", type=str, default="512")
    parser.add_argument("--copy-rows", type=str, default="1")
    parser.add_argument("--max-concurrent-steps-values", type=str, default="4")
    parser.add_argument("--grid-block-n-values", type=str, default="2")
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--max-rows-per-rank", type=int, default=None)
    parser.add_argument("--skip-padded-blocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ring-block-tokens", type=int, default=64)
    parser.add_argument("--ring-block-tokens-values", type=str, default=None)
    parser.add_argument("--ring-max-rows-per-pair", type=int, default=None)
    parser.add_argument("--ring-num-prefetch", type=int, default=2)
    parser.add_argument("--ring-num-prefetch-values", type=str, default=None)
    parser.add_argument("--ring-max-active-srcs", type=int, default=8)
    parser.add_argument("--ring-max-active-srcs-values", type=str, default=None)
    parser.add_argument(
        "--ring-queue-order",
        type=str,
        choices=("eg_block_src", "src_eg_block"),
        default="eg_block_src",
    )
    parser.add_argument("--ring-queue-orders", type=str, default=None)
    parser.add_argument("--num-sms", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    implementations = _str_list(args.implementations)
    unknown = sorted(set(implementations) - IMPLEMENTATIONS)
    if unknown:
        raise ValueError(f"unknown compact implementations: {unknown}; expected one of {sorted(IMPLEMENTATIONS)}")
    shape = BenchShape(
        ep_size=args.ep_size,
        tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        experts_per_rank=args.experts_per_rank,
        topk=args.topk,
        routing=args.routing,
    )
    configs = _config_sweep(args)
    best: dict[str, Any] | None = None
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)
    jsonl = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    try:
        for implementation in implementations:
            for config in configs:
                row = _run_one(
                    shape=shape,
                    config=config,
                    implementation=implementation,
                    warmup=args.warmup,
                    steps=args.steps,
                    seed=args.seed,
                    check=args.check,
                )
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                if jsonl is not None:
                    print(line, file=jsonl, flush=True)
                if row["error"] is None and row["steady_state_time"] is not None:
                    if best is None or row["steady_state_time"] < best["steady_state_time"]:
                        best = row
        if best is not None:
            print(json.dumps({"best": best}, sort_keys=True), flush=True)
    finally:
        if jsonl is not None:
            jsonl.close()


if __name__ == "__main__":
    main()
