# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistent semantic source-push permutation fused with expert W13."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import AbstractMesh, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    SourcePushSemanticPlan,
    source_push_semantic_queue_metadata_jax,
)
from levanter.grug._moe.source_push_semantic_inbox_pallas import source_push_semantic_inbox_metadata_jax


WGMMA_SWIZZLE_BYTES = 128
WGMMA_TILE_M = 8
RAW_GATHER_ROWS = 8
TOKEN_TRANSFER_ROWS = 128


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13Config:
    """Physical Hopper tile and persistent-CTA dimensions."""

    compute_m: int = 64
    send_m: int = 256
    block_n: int = 128
    block_k: int = 128
    send_k: int = 256
    inbox_slots: int = 12
    producer_programs_per_peer: int = 12
    consumer_programs_per_peer: int = 20
    n_groups_per_job: int = 2

    def validate(self) -> None:
        for name in (
            "compute_m",
            "send_m",
            "block_n",
            "block_k",
            "send_k",
            "inbox_slots",
            "producer_programs_per_peer",
            "consumer_programs_per_peer",
            "n_groups_per_job",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.compute_m != 64:
            raise ValueError(f"the initial Hopper lowering requires compute_m=64, got {self.compute_m}")
        if self.send_m != 256:
            raise ValueError(f"the initial Hopper lowering requires send_m=256, got {self.send_m}")
        if self.block_n != 128 or self.block_k != 128:
            raise ValueError(
                f"the initial Hopper lowering requires block_n=block_k=128, got {self.block_n=} {self.block_k=}"
            )
        if self.send_k != 256:
            raise ValueError(f"the initial Hopper lowering requires send_k=256, got {self.send_k}")
        if self.inbox_slots != 12:
            raise ValueError(f"the initial Hopper lowering requires inbox_slots=12, got {self.inbox_slots}")
        if self.producer_programs_per_peer != 12:
            raise ValueError(
                "the Hopper lowering requires twelve producer programs per peer, "
                f"got {self.producer_programs_per_peer}"
            )
        if self.consumer_programs_per_peer != 20:
            raise ValueError(f"consumer_programs_per_peer must be 20, got {self.consumer_programs_per_peer}")
        if self.n_groups_per_job != 2:
            raise ValueError(f"the initial Hopper lowering requires n_groups_per_job=2, got {self.n_groups_per_job}")

    @property
    def compute_blocks_per_send(self) -> int:
        return self.send_m // self.compute_m

    @property
    def worker_programs_per_peer(self) -> int:
        return self.producer_programs_per_peer + self.consumer_programs_per_peer


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourcePushSemanticFusedW13Metadata:
    """Source-owned send chunks and destination-owned receive descriptors."""

    token_ids: Int[Array, "S DstOrd C B M"]
    send_expert: Int[Array, "S DstOrd C B"]
    send_row_start: Int[Array, "S DstOrd C B"]
    send_valid_rows: Int[Array, "S DstOrd C B"]
    recv_expert: Int[Array, "Dst SrcOrd C B"]
    recv_row_start: Int[Array, "Dst SrcOrd C B"]
    recv_valid_rows: Int[Array, "Dst SrcOrd C B"]
    valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]
    rows_per_expert_capacity: int = field(metadata={"static": True})
    send_chunks_per_dst: int = field(metadata={"static": True})


class SourcePushSemanticFusedW13Result(NamedTuple):
    """Source-padded expert-major gate/up preactivation and validity."""

    z: Float[Array, "Dst E C twoI"]
    valid: Bool[Array, "Dst E C"]
    queue_overflow_routes: Int[Array, ""]
    layout_overflow_rows: Int[Array, ""]


@dataclass(frozen=True, slots=True)
class SourcePushSemanticFusedW13GenerationAccounting:
    """Cumulative semaphore targets for one physical send generation."""

    slot: int
    generation: int
    owner: int
    producer_programs: int
    consumer_programs: int
    copy_tiles: int
    copy_tiles_per_compute_block: int
    empty_generation: int
    block_ready_generation: int
    consumer_done_generation: int
    released_generation: int


def source_push_semantic_fused_w13_generation_accounting(
    chunk: int,
    *,
    hidden_dim: int,
    intermediate_dim: int,
    config: SourcePushSemanticFusedW13Config = SourcePushSemanticFusedW13Config(),
) -> SourcePushSemanticFusedW13GenerationAccounting:
    """Return cumulative semaphore targets for ``chunk``."""

    config.validate()
    _validate_kernel_dimensions(hidden_dim, intermediate_dim, config)
    if chunk < 0:
        raise ValueError(f"chunk must be nonnegative, got {chunk}")
    generation = chunk // config.inbox_slots + 1
    copy_tiles_per_compute_block = hidden_dim // config.send_k
    copy_tiles = config.compute_blocks_per_send * copy_tiles_per_compute_block
    n_jobs = math.ceil((intermediate_dim // config.block_n) / config.n_groups_per_job)
    consumer_jobs = config.compute_blocks_per_send * n_jobs
    return SourcePushSemanticFusedW13GenerationAccounting(
        slot=chunk % config.inbox_slots,
        generation=generation,
        owner=chunk % config.producer_programs_per_peer,
        producer_programs=config.producer_programs_per_peer,
        consumer_programs=config.consumer_programs_per_peer,
        copy_tiles=copy_tiles,
        copy_tiles_per_compute_block=copy_tiles_per_compute_block,
        empty_generation=generation,
        block_ready_generation=generation,
        consumer_done_generation=generation * consumer_jobs,
        released_generation=generation + 1,
    )


def source_push_semantic_fused_w13_metadata_jax(
    x: Float[Array, "S T H"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW13Config = SourcePushSemanticFusedW13Config(),
) -> SourcePushSemanticFusedW13Metadata:
    """Lower semantic expert rows into four-B64 physical send chunks."""

    config.validate()
    if send_chunks_per_dst <= 0:
        raise ValueError(f"send_chunks_per_dst must be positive, got {send_chunks_per_dst}")
    entries_per_dst = send_chunks_per_dst * config.compute_blocks_per_send
    queue = source_push_semantic_queue_metadata_jax(
        plan,
        return_row_block=config.compute_m,
        entries_per_dst=entries_per_dst,
    )
    inbox_metadata = source_push_semantic_inbox_metadata_jax(
        x,
        plan,
        queue,
        rows_per_expert_capacity=rows_per_expert_capacity,
    )
    source_count, destination_count, _ = queue.local_expert.shape
    chunk_shape = (
        source_count,
        destination_count,
        send_chunks_per_dst,
        config.compute_blocks_per_send,
    )
    send_meta = inbox_metadata.send_meta.reshape(*chunk_shape, -1)
    token_ids = inbox_metadata.token_ids.reshape(*chunk_shape, config.compute_m)

    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None]
    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = (source + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(send_meta[..., 1], 0)
    expert_base = inbox_metadata.layout.expert_base.at[destination, safe_expert].get()
    send_row_start = jnp.where(send_meta[..., 3] > 0, send_meta[..., 2] - expert_base, 0)

    destination_index = jnp.arange(destination_count, dtype=jnp.int32)[:, None]
    source_ordinal = jnp.arange(source_count, dtype=jnp.int32)[None, :]
    recv_source = (destination_index + source_ordinal) % source_count
    recv_dst_ordinal = (-source_ordinal) % destination_count

    def _to_recv(value: Array) -> Array:
        return value.at[recv_source, recv_dst_ordinal].get()

    send_expert = send_meta[..., 1].astype(jnp.int32)
    send_valid_rows = send_meta[..., 3].astype(jnp.int32)
    return SourcePushSemanticFusedW13Metadata(
        token_ids=token_ids,
        send_expert=send_expert,
        send_row_start=send_row_start.astype(jnp.int32),
        send_valid_rows=send_valid_rows,
        recv_expert=_to_recv(send_expert),
        recv_row_start=_to_recv(send_row_start).astype(jnp.int32),
        recv_valid_rows=_to_recv(send_valid_rows),
        valid=inbox_metadata.layout.valid,
        queue_overflow_routes=queue.overflow_routes,
        layout_overflow_rows=inbox_metadata.layout.overflow_rows,
        rows_per_expert_capacity=rows_per_expert_capacity,
        send_chunks_per_dst=send_chunks_per_dst,
    )


def source_push_semantic_fused_w13_reference_jax(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    metadata: SourcePushSemanticFusedW13Metadata,
) -> Float[Array, "Dst E C twoI"]:
    """Obvious gather, matmul, and expert-major scatter reference."""

    source_count, destination_count, chunks, blocks, compute_m = metadata.token_ids.shape
    source = jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None, None]
    gathered_x = x.at[source, metadata.token_ids].get()
    row = jnp.arange(compute_m, dtype=jnp.int32)[None, None, None, None, :]
    row_valid = row < metadata.send_valid_rows[..., None]
    gathered_x = jnp.where(row_valid[..., None], gathered_x, jnp.zeros((), dtype=x.dtype))

    dst_ordinal = jnp.arange(destination_count, dtype=jnp.int32)[None, :, None, None]
    destination = (jnp.arange(source_count, dtype=jnp.int32)[:, None, None, None] + dst_ordinal) % destination_count
    safe_expert = jnp.maximum(metadata.send_expert, 0)
    weights = w_gate_up.at[destination, safe_expert].get()
    queue_z = jnp.einsum(
        "sdcbmh,sdcbho->sdcbmo",
        gathered_x.astype(jnp.float32),
        weights.astype(jnp.float32),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
    queue_z = jnp.where(row_valid[..., None], queue_z, jnp.zeros((), dtype=queue_z.dtype))

    output = jnp.zeros(
        (
            destination_count,
            w_gate_up.shape[1],
            metadata.rows_per_expert_capacity + 1,
            w_gate_up.shape[-1],
        ),
        dtype=queue_z.dtype,
    )
    scatter_destination = jnp.broadcast_to(destination[..., None], row_valid.shape)
    scatter_expert = jnp.broadcast_to(safe_expert[..., None], row_valid.shape)
    scatter_row = metadata.send_row_start[..., None] + row
    scatter_row = jnp.where(row_valid, scatter_row, metadata.rows_per_expert_capacity)
    output = output.at[scatter_destination, scatter_expert, scatter_row].set(queue_z)
    output = output[..., : metadata.rows_per_expert_capacity, :]
    return jnp.where(metadata.valid[..., None], output, jnp.zeros((), dtype=output.dtype))


def source_push_semantic_fused_w13(
    x: Float[Array, "S T H"],
    w_gate_up: Float[Array, "Dst E H twoI"],
    plan: SourcePushSemanticPlan,
    *,
    send_chunks_per_dst: int,
    rows_per_expert_capacity: int,
    config: SourcePushSemanticFusedW13Config = SourcePushSemanticFusedW13Config(),
    mesh: Mesh | AbstractMesh | None = None,
    interpret: bool = False,
) -> SourcePushSemanticFusedW13Result:
    """Run fused raw-token permutation and W13, or its JAX interpret path."""

    _validate_request(x, w_gate_up, plan, config)
    metadata = source_push_semantic_fused_w13_metadata_jax(
        x,
        plan,
        send_chunks_per_dst=send_chunks_per_dst,
        rows_per_expert_capacity=rows_per_expert_capacity,
        config=config,
    )
    if interpret:
        z = source_push_semantic_fused_w13_reference_jax(x, w_gate_up, metadata)
    else:
        if jax.default_backend() != "gpu":
            raise NotImplementedError("persistent semantic fused W13 requires a GPU backend")
        if mesh is None:
            mesh = jax.sharding.get_abstract_mesh()
            if mesh.empty:
                raise ValueError("mesh is required for persistent semantic fused W13")
        z = _source_push_semantic_fused_w13_sharded(x, w_gate_up, metadata, config=config, mesh=mesh)
    z = jnp.where(metadata.valid[..., None], z, jnp.zeros((), dtype=z.dtype))
    return SourcePushSemanticFusedW13Result(
        z=z,
        valid=metadata.valid,
        queue_overflow_routes=metadata.queue_overflow_routes,
        layout_overflow_rows=metadata.layout_overflow_rows,
    )


def _source_push_semantic_fused_w13_sharded(
    x: Array,
    w_gate_up: Array,
    metadata: SourcePushSemanticFusedW13Metadata,
    *,
    config: SourcePushSemanticFusedW13Config,
    mesh: Mesh | AbstractMesh,
) -> Array:
    if mesh.shape[SOURCE_PUSH_MESH_AXIS] != x.shape[0]:
        raise ValueError(
            f"mesh {SOURCE_PUSH_MESH_AXIS!r} size must match source count {x.shape[0]}, "
            f"got {mesh.shape[SOURCE_PUSH_MESH_AXIS]}"
        )
    kernel = _make_source_push_semantic_fused_w13_kernel(
        ep_size=x.shape[0],
        hidden_dim=x.shape[-1],
        intermediate_dim=w_gate_up.shape[-1] // 2,
        experts_per_rank=w_gate_up.shape[1],
        rows_per_expert_capacity=metadata.rows_per_expert_capacity,
        send_chunks_per_dst=metadata.send_chunks_per_dst,
        dtype=x.dtype,
        config=config,
    )

    def local_fn(
        x_local,
        token_ids_local,
        send_valid_local,
        recv_expert_local,
        recv_row_local,
        recv_valid_local,
        weights_local,
    ):
        token_ids_flat = token_ids_local[0].reshape(
            token_ids_local.shape[1],
            metadata.send_chunks_per_dst,
            config.send_m,
        )
        token_ids_padded = jnp.pad(token_ids_flat, ((0, 0), (0, 0), (0, config.compute_m)))
        _inbox, z_flat_local = kernel(
            x_local[0],
            token_ids_padded,
            send_valid_local[0],
            recv_expert_local[0],
            recv_row_local[0],
            recv_valid_local[0],
            weights_local[0],
        )
        z_local = z_flat_local.reshape(
            w_gate_up.shape[1],
            metadata.rows_per_expert_capacity,
            w_gate_up.shape[-1],
        )
        return z_local[None, ...]

    source_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))
    source_token_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None, None))
    source_metadata_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    destination_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    weights_sharding = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(
        jax.sharding.reshard(x, source_3d),
        jax.sharding.reshard(metadata.token_ids, source_token_sharding),
        jax.sharding.reshard(metadata.send_valid_rows, source_metadata_sharding),
        jax.sharding.reshard(metadata.recv_expert, destination_4d),
        jax.sharding.reshard(metadata.recv_row_start, destination_4d),
        jax.sharding.reshard(metadata.recv_valid_rows, destination_4d),
        jax.sharding.reshard(w_gate_up, weights_sharding),
    )


def _make_source_push_semantic_fused_w13_kernel(
    *,
    ep_size: int,
    hidden_dim: int,
    intermediate_dim: int,
    experts_per_rank: int,
    rows_per_expert_capacity: int,
    send_chunks_per_dst: int,
    dtype: jnp.dtype,
    config: SourcePushSemanticFusedW13Config,
):
    config.validate()
    _validate_kernel_dimensions(hidden_dim, intermediate_dim, config)
    blocks_per_send = config.compute_blocks_per_send
    n_tiles = intermediate_dim // config.block_n
    n_jobs = math.ceil(n_tiles / config.n_groups_per_job)
    consumer_jobs = blocks_per_send * n_jobs
    consumer_start = config.producer_programs_per_peer
    worker_programs = config.worker_programs_per_peer

    def body(
        x_ref,
        token_ids_ref,
        send_valid_ref,
        recv_expert_ref,
        recv_row_ref,
        recv_valid_ref,
        w_ref,
        inbox_ref,
        z_ref,
    ) -> None:
        empty_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        block_ready_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots, blocks_per_send)))
        consumer_done_sem = pl.get_global(mgpu.SemaphoreType.REGULAR((ep_size, config.inbox_slots)))
        rank = lax.axis_index(SOURCE_PUSH_MESH_AXIS)
        peer_ordinal = pl.program_id(0)
        worker = pl.program_id(1)

        def _signal_remote(sem, peer, local_index, slot) -> None:
            pl.semaphore_signal(
                sem.at[local_index, slot],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        def _signal_remote_block(sem, peer, local_index, slot, block) -> None:
            pl.semaphore_signal(
                sem.at[local_index, slot, block],
                device_id=peer,
                device_id_type=pl.DeviceIdType.LOGICAL,
            )

        @pl.when(worker < config.producer_programs_per_peer)
        def _producer() -> None:
            producer = worker

            def _run_peer(static_peer_ordinal: int) -> None:
                dst = (rank + static_peer_ordinal) % ep_size
                src = (rank + static_peer_ordinal) % ep_size
                remote_inbox = None
                if static_peer_ordinal != 0:
                    remote_inbox = mgpu.remote_ref(inbox_ref, dst, device_id_type=pl.DeviceIdType.LOGICAL)

                @pl.when(producer == 0)
                def _init_empty_slots() -> None:
                    @pl.loop(0, config.inbox_slots)
                    def _slot_loop(slot) -> None:
                        if static_peer_ordinal == 0:
                            pl.semaphore_signal(empty_sem.at[rank, slot])
                        else:
                            _signal_remote(empty_sem, src, rank, slot)

                def _send_block(chunk, slot, block) -> None:
                    valid_rows = send_valid_ref[static_peer_ordinal, chunk, block]

                    @pl.when(valid_rows > 0)
                    def _send_live_block() -> None:
                        def _copy_scope(tile_smem, token_smem, token_barrier) -> None:
                            mgpu.copy_gmem_to_smem(
                                token_ids_ref.at[
                                    static_peer_ordinal,
                                    chunk,
                                    pl.ds(block * config.compute_m, TOKEN_TRANSFER_ROWS),
                                ],
                                token_smem,
                                token_barrier,
                            )
                            mgpu.barrier_wait(token_barrier)

                            def _fill_and_send(tile, k_start) -> None:
                                for row_group in range(config.compute_m // RAW_GATHER_ROWS):
                                    row_start = row_group * RAW_GATHER_ROWS
                                    for row in range(RAW_GATHER_ROWS):
                                        output_row = row_start + row
                                        row_valid = output_row < valid_rows
                                        token = jnp.where(row_valid, token_smem[output_row], 0)
                                        x_row = x_ref[token, pl.ds(k_start, config.send_k)]
                                        tile[output_row, :] = jnp.where(
                                            row_valid,
                                            x_row,
                                            jnp.zeros((config.send_k,), dtype=dtype),
                                        )
                                mgpu.commit_smem()
                                destination_ref = inbox_ref if static_peer_ordinal == 0 else remote_inbox
                                mgpu.copy_smem_to_gmem(
                                    tile,
                                    destination_ref.at[
                                        rank,
                                        slot,
                                        pl.ds(block * config.compute_m, config.compute_m),
                                        pl.ds(k_start, config.send_k),
                                    ],
                                )

                            @pl.loop(0, hidden_dim // config.send_k)
                            def _k_loop(k_tile) -> None:
                                k_start = k_tile * config.send_k
                                _fill_and_send(tile_smem, k_start)
                                mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                        pl.run_scoped(
                            _copy_scope,
                            tile_smem=mgpu.SMEM((config.compute_m, config.send_k), dtype=dtype),
                            token_smem=mgpu.SMEM((TOKEN_TRANSFER_ROWS,), dtype=jnp.int32),
                            token_barrier=mgpu.Barrier(num_arrivals=1),
                        )

                    if static_peer_ordinal == 0:
                        pl.semaphore_signal(block_ready_sem.at[rank, slot, block])
                    else:
                        _signal_remote_block(block_ready_sem, dst, rank, slot, block)

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1

                    @pl.when((chunk % config.producer_programs_per_peer) == producer)
                    def _send_chunk() -> None:
                        pl.semaphore_wait(empty_sem.at[dst, slot], value=generation, decrement=False)

                        @pl.loop(0, blocks_per_send)
                        def _block_loop(block) -> None:
                            _send_block(chunk, slot, block)

            branches = tuple((lambda ordinal: lambda _: _run_peer(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

        @pl.when(worker >= consumer_start)
        def _consumer() -> None:
            consumer = worker - consumer_start

            def _consume_peer(static_peer_ordinal: int) -> None:
                peer = (rank + static_peer_ordinal) % ep_size

                def _compute(block, chunk, slot, expert, n_tile) -> None:
                    def _acc_scope(gate_acc, up_acc) -> None:
                        def _smem_scope(lhs_smem, gate_smem, up_smem, barrier) -> None:
                            @pl.loop(0, hidden_dim // config.block_k)
                            def _k_loop(k_tile) -> None:
                                k_start = k_tile * config.block_k
                                mgpu.copy_gmem_to_smem(
                                    inbox_ref.at[
                                        peer,
                                        slot,
                                        pl.ds(block * config.compute_m, config.compute_m),
                                        pl.ds(k_start, config.block_k),
                                    ],
                                    lhs_smem,
                                    barrier,
                                )
                                mgpu.copy_gmem_to_smem(
                                    w_ref.at[
                                        expert,
                                        pl.ds(k_start, config.block_k),
                                        pl.ds(n_tile * config.block_n, config.block_n),
                                    ],
                                    gate_smem,
                                    barrier,
                                )
                                mgpu.copy_gmem_to_smem(
                                    w_ref.at[
                                        expert,
                                        pl.ds(k_start, config.block_k),
                                        pl.ds((n_tile + n_tiles) * config.block_n, config.block_n),
                                    ],
                                    up_smem,
                                    barrier,
                                )
                                mgpu.barrier_wait(barrier)
                                mgpu.commit_smem()
                                mgpu.wgmma(gate_acc, lhs_smem, gate_smem)
                                mgpu.wgmma(up_acc, lhs_smem, up_smem)
                                mgpu.wgmma_wait(0)

                        pl.run_scoped(
                            _smem_scope,
                            lhs_smem=_wgmma_smem((config.compute_m, config.block_k), dtype),
                            gate_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                            up_smem=_wgmma_smem((config.block_k, config.block_n), dtype),
                            barrier=mgpu.Barrier(num_arrivals=3),
                        )
                        row_start = recv_row_ref[static_peer_ordinal, chunk, block]
                        flat_row_start = expert * rows_per_expert_capacity + row_start
                        z_ref[
                            pl.ds(flat_row_start, config.compute_m),
                            pl.ds(n_tile * config.block_n, config.block_n),
                        ] = gate_acc[...].astype(dtype)
                        z_ref[
                            pl.ds(flat_row_start, config.compute_m),
                            pl.ds((n_tile + n_tiles) * config.block_n, config.block_n),
                        ] = up_acc[...].astype(dtype)

                    pl.run_scoped(
                        _acc_scope,
                        gate_acc=mgpu.ACC((config.compute_m, config.block_n)),
                        up_acc=mgpu.ACC((config.compute_m, config.block_n)),
                    )

                @pl.loop(0, send_chunks_per_dst)
                def _chunk_loop(chunk) -> None:
                    slot = chunk % config.inbox_slots
                    generation = chunk // config.inbox_slots + 1

                    @pl.loop(0, consumer_jobs)
                    def _job_loop(job) -> None:
                        work_group = (static_peer_ordinal * config.inbox_slots + slot) * consumer_jobs + job

                        @pl.when((work_group % config.consumer_programs_per_peer) == consumer)
                        def _consume_job() -> None:
                            block = job // n_jobs
                            n_job = job % n_jobs
                            pl.semaphore_wait(
                                block_ready_sem.at[peer, slot, block],
                                value=generation,
                                decrement=False,
                            )
                            valid_rows = recv_valid_ref[static_peer_ordinal, chunk, block]

                            @pl.when(valid_rows > 0)
                            def _compute_live_block() -> None:
                                expert = recv_expert_ref[static_peer_ordinal, chunk, block]

                                @pl.loop(0, config.n_groups_per_job)
                                def _n_group_loop(n_offset) -> None:
                                    n_tile = n_job * config.n_groups_per_job + n_offset

                                    @pl.when(n_tile < n_tiles)
                                    def _compute_n_tile() -> None:
                                        _compute(block, chunk, slot, expert, n_tile)

                            pl.semaphore_signal(consumer_done_sem.at[peer, slot])

                    release_group = static_peer_ordinal * config.inbox_slots + slot

                    @pl.when((release_group % config.consumer_programs_per_peer) == consumer)
                    def _release_slot() -> None:
                        # Fixed job accounting includes masked subblocks, so release cannot race a live reader.
                        pl.semaphore_wait(
                            consumer_done_sem.at[peer, slot],
                            value=generation * consumer_jobs,
                            decrement=False,
                        )
                        if static_peer_ordinal == 0:
                            pl.semaphore_signal(empty_sem.at[rank, slot])
                        else:
                            _signal_remote(empty_sem, peer, rank, slot)

            branches = tuple((lambda ordinal: lambda _: _consume_peer(ordinal))(i) for i in range(ep_size))
            lax.switch(peer_ordinal, branches, None)

    inbox_shape = (ep_size, config.inbox_slots, config.send_m, hidden_dim)
    output_shape = (experts_per_rank * rows_per_expert_capacity, 2 * intermediate_dim)
    return mgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct(inbox_shape, dtype),
            jax.ShapeDtypeStruct(output_shape, dtype),
        ),
        grid=(ep_size, worker_programs),
        grid_names=("peer_phase", "worker_program"),
        compiler_params=mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane),
    )


def _wgmma_smem(shape: tuple[int, int], dtype: jnp.dtype):
    swizzle_elements = WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % WGMMA_TILE_M or shape[-1] % swizzle_elements:
        raise ValueError(f"WGMMA SMEM shape {shape} is incompatible with 128-byte swizzling")
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((WGMMA_TILE_M, swizzle_elements)),
            mgpu.SwizzleTransform(WGMMA_SWIZZLE_BYTES),
        ),
    )


def _validate_kernel_dimensions(
    hidden_dim: int,
    intermediate_dim: int,
    config: SourcePushSemanticFusedW13Config,
) -> None:
    if hidden_dim % config.send_k or hidden_dim % config.block_k:
        raise ValueError(
            f"hidden_dim must be divisible by send_k and block_k, got {hidden_dim=} "
            f"{config.send_k=} {config.block_k=}"
        )
    if intermediate_dim % config.block_n:
        raise ValueError(f"intermediate_dim must be divisible by block_n, got {intermediate_dim=} {config.block_n=}")


def _validate_request(
    x: Array, w_gate_up: Array, plan: SourcePushSemanticPlan, config: SourcePushSemanticFusedW13Config
) -> None:
    config.validate()
    if x.ndim != 3:
        raise ValueError(f"x must have shape [source, token, hidden], got {x.shape}")
    if w_gate_up.ndim != 4:
        raise ValueError(f"w_gate_up must have shape [destination, expert, hidden, twoI], got {w_gate_up.shape}")
    source_count, destination_count, experts_per_rank = plan.xcounts.shape
    if source_count != destination_count or x.shape[:2] != (source_count, plan.tokens_per_source):
        raise ValueError(f"x shape {x.shape} is incompatible with semantic plan shape {plan.xcounts.shape}")
    if w_gate_up.shape[:3] != (destination_count, experts_per_rank, x.shape[-1]):
        raise ValueError(
            f"w_gate_up leading shape {w_gate_up.shape[:3]} must be "
            f"{(destination_count, experts_per_rank, x.shape[-1])}"
        )
    if w_gate_up.shape[-1] % 2:
        raise ValueError(f"w_gate_up output dimension must be even, got {w_gate_up.shape[-1]}")
    _validate_kernel_dimensions(x.shape[-1], w_gate_up.shape[-1] // 2, config)
