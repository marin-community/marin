# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# pyrefly: ignore-errors

"""Package-private W2 return kernel for the source-push MGPU MoE prototype."""

from __future__ import annotations

import json
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

from levanter.grug._moe.source_push_inbox import (
    AXIS,
    BYTES_PER_BF16,
    KERNEL_NAME as W13_KERNEL_NAME,
    PushInboxConfig,
    SourcePushInboxRunSettings,
    _make_exact_source_push_plan_inputs,
    _make_mesh,
    _make_source_push_plan_inputs,
    _make_weights_host,
    _reference_hidden,
    _silu,
    _time_jitted,
    _wgmma_smem,
)
from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_META_FIELDS,
    SOURCE_PUSH_META_LOCAL_EXPERT,
    SOURCE_PUSH_META_LOCAL_ROW_START,
    SOURCE_PUSH_META_SRC_RANK,
    SOURCE_PUSH_META_VALID_ROWS,
)


KERNEL_NAME = "source_push_w2_return"
RETURN_COPY_KERNEL_NAME = "source_push_w2_return_copy"
DIRECT_RETURN_KERNEL_NAME = "source_push_w2_return_direct"
W2_RETURN_MODE_DESTINATION_LOCAL = "destination_local"
W2_RETURN_MODE_SEPARATE_COPY = "separate_copy"
W2_RETURN_MODE_DIRECT_REMOTE = "direct_remote"
W2_RETURN_MODES = (
    W2_RETURN_MODE_DESTINATION_LOCAL,
    W2_RETURN_MODE_SEPARATE_COPY,
    W2_RETURN_MODE_DIRECT_REMOTE,
)
W2_HIDDEN_INPUT_W13_REFERENCE = "w13_reference"
W2_HIDDEN_INPUT_SYNTHETIC = "synthetic"
W2_HIDDEN_INPUT_MODES = (W2_HIDDEN_INPUT_W13_REFERENCE, W2_HIDDEN_INPUT_SYNTHETIC)


@dataclass(frozen=True)
class W2ReturnHostInputs:
    """Host-side inputs for the destination-local W2 return kernel."""

    hidden: np.ndarray
    recv_meta: np.ndarray
    expert_base: np.ndarray
    src_base_by_expert: np.ndarray
    w_down: np.ndarray
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool = False


@dataclass(frozen=True)
class W2ReturnDeviceInputs:
    """Device inputs for the destination-local W2 return kernel."""

    hidden: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    w_down: jax.Array
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool


@dataclass(frozen=True)
class W2ReturnValidationMetrics:
    """Validation summary for destination-local W2 return rows."""

    max_abs_diff: float
    mean_abs_diff: float
    all_max_abs_diff: float
    source_queue_max_abs_diff: float


def validate_w2_return_config(config: PushInboxConfig) -> None:
    """Validate the extra tile constraints required by the W2 return kernel."""

    config.validate()
    if config.intermediate_dim % config.block_k:
        raise ValueError(
            "intermediate_dim must be divisible by block_k for W2; "
            f"got {config.intermediate_dim=} {config.block_k=}"
        )
    if config.hidden_dim % config.block_n:
        raise ValueError(f"hidden_dim must be divisible by block_n for W2, got {config.hidden_dim=} {config.block_n=}")


def make_w_down(config: PushInboxConfig) -> np.ndarray:
    """Create deterministic W2 weights for source-push W2 return benchmarks."""

    expert_scale = (((np.arange(config.experts_per_rank, dtype=np.float32) % 5) + 1.0) / config.intermediate_dim)[
        None, :, None, None
    ]
    dst_scale = (((np.arange(config.ep_size, dtype=np.float32) % 7) + 1.0) / (2.0 * config.intermediate_dim))[
        :, None, None, None
    ]
    k_component = ((np.arange(config.intermediate_dim, dtype=np.float32) % 11) + 1.0)[None, None, :, None]
    n_component = ((np.arange(config.hidden_dim, dtype=np.float32) % 13) + 1.0)[None, None, None, :]
    w_down = (
        expert_scale + dst_scale + (k_component * n_component) / (32.0 * config.intermediate_dim * config.hidden_dim)
    )
    return w_down.astype(np.float32)


def make_synthetic_hidden(config: PushInboxConfig) -> np.ndarray:
    """Create deterministic expert-major hidden rows for W2-only timing."""

    hidden = np.empty((config.ep_size, config.hidden_rows_per_rank, config.intermediate_dim), dtype=np.float32)
    k_component = (np.arange(config.intermediate_dim, dtype=np.float32) % 17)[None, :] * 0.0001
    for dst in range(config.ep_size):
        row_component = (np.arange(config.hidden_rows_per_rank, dtype=np.float32)[:, None] % 2048) * 0.00001
        hidden[dst] = 0.125 + 0.03125 * dst + row_component + k_component
    return hidden


def source_queue_from_destination_return(
    config: PushInboxConfig,
    return_by_destination: Float[Array, "Dst Src Q M D"],
) -> Float[Array, "S Dst Q M D"]:
    """Reorder destination-local return blocks into the source-visible queue layout.

    The Pallas W2 kernel in this module writes destination-local rows indexed by
    `(dst_rank, recv_src_ordinal, entry)`. This host-side adapter is a
    validation bridge to the final source-owned layout `(src_rank, dst_ordinal,
    entry)`. A later return kernel should perform the same mapping with remote
    writes instead of materializing this transpose on the host.
    """

    return_host = np.asarray(jax.device_get(return_by_destination))
    expected_shape = (
        config.ep_size,
        config.ep_size,
        config.entries_per_rank,
        config.block_m,
        config.hidden_dim,
    )
    if return_host.shape != expected_shape:
        raise ValueError(f"return_by_destination shape {return_host.shape} must be {expected_shape}")

    source_queue = np.zeros_like(return_host)
    for dst in range(config.ep_size):
        for src_ordinal in range(config.ep_size):
            src = (dst + src_ordinal) % config.ep_size
            dst_ordinal = (dst - src) % config.ep_size
            source_queue[src, dst_ordinal] = return_host[dst, src_ordinal]
    return jnp.asarray(source_queue)


def reference_w2_return_by_destination(
    config: PushInboxConfig,
    hidden: Float[Array, "Dst rows I"],
    recv_meta: Int[Array, "Dst Src Q F"],
    w_down: Float[Array, "Dst E I D"],
    expert_base: Int[Array, "Dst E"] | np.ndarray | None = None,
    src_base_by_expert: Int[Array, "Dst Src E"] | np.ndarray | None = None,
    *,
    use_exact_expert_major: bool = False,
) -> Float[Array, "Dst Src Q M D"]:
    """Reference W2 over expert-major hidden rows in destination-local return order."""

    hidden_host = np.asarray(jax.device_get(hidden), dtype=np.float32)
    recv_meta_host = np.asarray(jax.device_get(recv_meta), dtype=np.int32)
    w_down_host = np.asarray(jax.device_get(w_down), dtype=np.float32)
    expert_base_host = _expert_base_or_zeros(config, expert_base)
    src_base_host = _src_base_by_expert_or_zeros(config, src_base_by_expert)
    _validate_reference_shapes(config, hidden_host, recv_meta_host, w_down_host, expert_base_host, src_base_host)

    return_by_destination = np.zeros(
        (
            config.ep_size,
            config.ep_size,
            config.entries_per_rank,
            config.block_m,
            config.hidden_dim,
        ),
        dtype=np.float32,
    )
    for dst in range(config.ep_size):
        for src_ordinal in range(config.ep_size):
            for entry in range(config.entries_per_rank):
                valid_rows = int(recv_meta_host[dst, src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS])
                if valid_rows <= 0:
                    continue
                expert = int(recv_meta_host[dst, src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT])
                row_start = _w2_hidden_row_start(
                    recv_meta_host,
                    expert_base_host,
                    src_base_host,
                    dst=dst,
                    src_ordinal=src_ordinal,
                    entry=entry,
                    use_exact_expert_major=use_exact_expert_major,
                )
                hidden_rows = hidden_host[dst, row_start : row_start + config.block_m]
                return_by_destination[dst, src_ordinal, entry] = hidden_rows @ w_down_host[dst, expert]
    return jnp.asarray(return_by_destination)


def make_w2_return_source_plan_inputs(
    config: PushInboxConfig,
    *,
    hidden_input_mode: str = W2_HIDDEN_INPUT_SYNTHETIC,
) -> W2ReturnHostInputs:
    """Build W2 inputs from the current source-padded SourcePushPlan W13 layout."""

    validate_w2_return_config(config)
    if hidden_input_mode not in W2_HIDDEN_INPUT_MODES:
        raise ValueError(f"unknown hidden_input_mode={hidden_input_mode!r}; expected one of {W2_HIDDEN_INPUT_MODES}")
    host_inputs = _make_source_push_plan_inputs(config)
    if hidden_input_mode == W2_HIDDEN_INPUT_W13_REFERENCE:
        w_gate_up = _make_weights_host(config)
        hidden = _reference_hidden(
            config,
            host_inputs.x,
            host_inputs.send_meta,
            w_gate_up,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
    else:
        hidden = make_synthetic_hidden(config)
    return W2ReturnHostInputs(
        hidden=hidden.astype(np.float32),
        recv_meta=host_inputs.recv_meta,
        expert_base=host_inputs.expert_base,
        src_base_by_expert=host_inputs.src_base_by_expert,
        w_down=make_w_down(config),
        queue_stats={
            **host_inputs.queue_stats,
            "w2_input_mode": "source_push_plan",
            "w2_hidden_input_mode": hidden_input_mode,
        },
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def make_w2_return_exact_source_plan_inputs(
    config: PushInboxConfig,
    *,
    hidden_input_mode: str = W2_HIDDEN_INPUT_SYNTHETIC,
) -> W2ReturnHostInputs:
    """Build W2 inputs from exact count-derived SourcePushPlan metadata."""

    validate_w2_return_config(config)
    if hidden_input_mode not in W2_HIDDEN_INPUT_MODES:
        raise ValueError(f"unknown hidden_input_mode={hidden_input_mode!r}; expected one of {W2_HIDDEN_INPUT_MODES}")
    host_inputs = _make_exact_source_push_plan_inputs(config)
    if hidden_input_mode == W2_HIDDEN_INPUT_W13_REFERENCE:
        w_gate_up = _make_weights_host(config)
        hidden = _reference_hidden(
            config,
            host_inputs.x,
            host_inputs.send_meta,
            w_gate_up,
            host_inputs.expert_base,
            host_inputs.src_base_by_expert,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        )
    else:
        hidden = make_synthetic_hidden(config)
    return W2ReturnHostInputs(
        hidden=hidden.astype(np.float32),
        recv_meta=host_inputs.recv_meta,
        expert_base=host_inputs.expert_base,
        src_base_by_expert=host_inputs.src_base_by_expert,
        w_down=make_w_down(config),
        queue_stats={
            **host_inputs.queue_stats,
            "w2_input_mode": "exact_source_push_plan",
            "w2_hidden_input_mode": hidden_input_mode,
        },
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def device_w2_return_inputs_from_host(host_inputs: W2ReturnHostInputs) -> W2ReturnDeviceInputs:
    """Move host W2-return inputs into their benchmark dtypes."""

    return W2ReturnDeviceInputs(
        hidden=jnp.asarray(host_inputs.hidden, dtype=jnp.bfloat16),
        recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
        expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        w_down=jnp.asarray(host_inputs.w_down, dtype=jnp.bfloat16),
        queue_stats=host_inputs.queue_stats,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def _make_w2_return_kernel(config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    validate_w2_return_config(config)
    k_tiles = config.intermediate_dim // config.block_k
    n_tiles = config.hidden_dim // config.block_n

    def body(
        hidden_ref: Float[Ref, "rows I"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        w_down_ref: Float[Ref, "E I D"],
        return_ref: Float[Ref, "SRC Q M D"],
    ) -> None:
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)
        valid_rows = recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

        def _hidden_row_start() -> jax.Array:
            if use_exact_expert_major:
                src_rank = recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK]
                expert = recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                local_row_start = recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                return expert_base_ref[expert] + src_base_by_expert_ref[src_rank, expert] + local_row_start
            return recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]

        @pl.when(valid_rows > 0)
        def _compute_return_block() -> None:
            expert = recv_meta_ref[src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
            row_start = _hidden_row_start()

            def acc_scope(acc_ref) -> jax.Array:
                def smem_scope(hidden_smem, w_down_smem, ready_barrier) -> None:
                    @pl.loop(0, k_tiles)
                    def _k_loop(kk) -> None:
                        k_start = kk * config.block_k
                        mgpu.copy_gmem_to_smem(
                            hidden_ref.at[
                                pl.ds(row_start, config.block_m),
                                pl.ds(k_start, config.block_k),
                            ],
                            hidden_smem,
                            ready_barrier,
                        )
                        mgpu.copy_gmem_to_smem(
                            w_down_ref.at[
                                expert,
                                pl.ds(k_start, config.block_k),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                            w_down_smem,
                            ready_barrier,
                        )
                        mgpu.barrier_wait(ready_barrier)
                        mgpu.commit_smem()
                        mgpu.wgmma(acc_ref, hidden_smem, w_down_smem)
                        mgpu.wgmma_wait(0)

                pl.run_scoped(
                    smem_scope,
                    hidden_smem=_wgmma_smem((config.block_m, config.block_k), hidden_ref.dtype),
                    w_down_smem=_wgmma_smem((config.block_k, config.block_n), w_down_ref.dtype),
                    ready_barrier=mgpu.Barrier(num_arrivals=2),
                )
                return acc_ref[...].astype(return_ref.dtype)

            output = pl.run_scoped(
                acc_scope,
                acc_ref=mgpu.ACC((config.block_m, config.block_n)),
            )
            return_ref[
                src_ordinal,
                entry,
                pl.ds(0, config.block_m),
                pl.ds(n_tile * config.block_n, config.block_n),
            ] = output

    out_shape = jax.ShapeDtypeStruct(
        (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
        jnp.bfloat16,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_w2_return_kernel(mesh: Mesh, config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    kernel = _make_w2_return_kernel(config, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        hidden_local: Float[Array, "1 rows I"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_down_local: Float[Array, "1 E I D"],
    ):
        hidden_local = hidden_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_down_local = w_down_local[0]
        return_by_src_ordinal = kernel(
            hidden_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_down_local,
        )
        return return_by_src_ordinal[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _make_w2_return_direct_to_source_kernel(config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    validate_w2_return_config(config)
    k_tiles = config.intermediate_dim // config.block_k
    n_tiles = config.hidden_dim // config.block_n
    src_offsets = tuple(range(config.ep_size))

    def body(
        hidden_ref: Float[Ref, "rows I"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        w_down_ref: Float[Ref, "E I D"],
        source_return_ref: Float[Ref, "DST Q M D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)

        def _compute_to_src(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % config.ep_size
            dst_ordinal = (-static_src_ordinal) % config.ep_size
            remote_source_return_ref = None
            if static_src_ordinal != 0:
                remote_source_return_ref = mgpu.remote_ref(
                    source_return_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            def _hidden_row_start() -> jax.Array:
                if use_exact_expert_major:
                    src_rank = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK]
                    expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                    local_row_start = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                    return expert_base_ref[expert] + src_base_by_expert_ref[src_rank, expert] + local_row_start
                return recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]

            @pl.when(valid_rows > 0)
            def _compute_return_block() -> None:
                expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                row_start = _hidden_row_start()

                def acc_scope(acc_ref) -> jax.Array:
                    def smem_scope(hidden_smem, w_down_smem, ready_barrier) -> None:
                        @pl.loop(0, k_tiles)
                        def _k_loop(kk) -> None:
                            k_start = kk * config.block_k
                            mgpu.copy_gmem_to_smem(
                                hidden_ref.at[
                                    pl.ds(row_start, config.block_m),
                                    pl.ds(k_start, config.block_k),
                                ],
                                hidden_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_down_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds(n_tile * config.block_n, config.block_n),
                                ],
                                w_down_smem,
                                ready_barrier,
                            )
                            mgpu.barrier_wait(ready_barrier)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_ref, hidden_smem, w_down_smem)
                            mgpu.wgmma_wait(0)

                    pl.run_scoped(
                        smem_scope,
                        hidden_smem=_wgmma_smem((config.block_m, config.block_k), hidden_ref.dtype),
                        w_down_smem=_wgmma_smem((config.block_k, config.block_n), w_down_ref.dtype),
                        ready_barrier=mgpu.Barrier(num_arrivals=2),
                    )
                    return acc_ref[...].astype(source_return_ref.dtype)

                output = pl.run_scoped(
                    acc_scope,
                    acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                )

                def store_scope(output_smem) -> None:
                    output_smem[:, :] = output
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            remote_source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    store_scope,
                    output_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=source_return_ref.dtype),
                )

        def _switch_compute_to_src(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _compute_branch(_) -> None:
                    _compute_to_src(static_src_ordinal)

                return _compute_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_compute_to_src(src_ordinal)

    out_shape = jax.ShapeDtypeStruct(
        (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
        jnp.bfloat16,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_w2_return_direct_to_source_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_w2_return_direct_to_source_kernel(config, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        hidden_local: Float[Array, "1 rows I"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_down_local: Float[Array, "1 E I D"],
    ):
        hidden_local = hidden_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_down_local = w_down_local[0]
        source_return = kernel(
            hidden_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_down_local,
        )
        return source_return[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _make_w2_from_h_return_direct_to_source_kernel(config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    validate_w2_return_config(config)
    k_tiles = config.intermediate_dim // config.block_k
    n_tiles = config.hidden_dim // config.block_n
    src_offsets = tuple(range(config.ep_size))

    def body(
        h_ref: Float[Ref, "rows twoI"],
        h_route_weights_ref: Float[Ref, "rows"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        w_down_ref: Float[Ref, "E I D"],
        source_return_ref: Float[Ref, "DST Q M D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)

        def _compute_to_src(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % config.ep_size
            dst_ordinal = (-static_src_ordinal) % config.ep_size
            remote_source_return_ref = None
            if static_src_ordinal != 0:
                remote_source_return_ref = mgpu.remote_ref(
                    source_return_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            def _h_row_start() -> jax.Array:
                if use_exact_expert_major:
                    src_rank = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK]
                    expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                    local_row_start = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                    return expert_base_ref[expert] + src_base_by_expert_ref[src_rank, expert] + local_row_start
                return recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]

            @pl.when(valid_rows > 0)
            def _compute_return_block() -> None:
                expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                row_start = _h_row_start()

                def acc_scope(acc_ref) -> jax.Array:
                    def smem_scope(
                        gate_smem,
                        up_smem,
                        route_weight_smem,
                        activation_smem,
                        w_down_smem,
                        ready_barrier,
                        route_ready_barrier,
                    ) -> jax.Array:
                        mgpu.copy_gmem_to_smem(
                            h_route_weights_ref.at[pl.ds(row_start, config.block_m)],
                            route_weight_smem,
                            route_ready_barrier,
                        )

                        @pl.loop(0, k_tiles)
                        def _k_loop(kk) -> None:
                            k_start = kk * config.block_k
                            mgpu.copy_gmem_to_smem(
                                h_ref.at[
                                    pl.ds(row_start, config.block_m),
                                    pl.ds(k_start, config.block_k),
                                ],
                                gate_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                h_ref.at[
                                    pl.ds(row_start, config.block_m),
                                    pl.ds(config.intermediate_dim + k_start, config.block_k),
                                ],
                                up_smem,
                                ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_down_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds(n_tile * config.block_n, config.block_n),
                                ],
                                w_down_smem,
                                ready_barrier,
                            )
                            mgpu.barrier_wait(ready_barrier)
                            activation_smem[:, :] = (
                                _silu(gate_smem[:, :].astype(jnp.float32)) * up_smem[:, :].astype(jnp.float32)
                            ).astype(activation_smem.dtype)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_ref, activation_smem, w_down_smem)
                            mgpu.wgmma_wait(0)

                        # Route scaling moves to the fp32 accumulator: the weight is per-row, so
                        # (diag(w) @ A) @ W2 == diag(w) @ (A @ W2), and the k-loop drops both the
                        # k-invariant weight-tile reload and one multiply per element.
                        mgpu.barrier_wait(route_ready_barrier)
                        route_weight_vec = mgpu.load(
                            route_weight_smem,
                            (pl.ds(0, config.block_m),),
                            layout=mgpu.Layout.WGMMA.reduce(1),
                        ).astype(jnp.float32)
                        route_weight = lax.broadcast_in_dim(
                            route_weight_vec,
                            (config.block_m, config.block_n),
                            (0,),
                        )
                        return (acc_ref[...] * route_weight).astype(source_return_ref.dtype)

                    return pl.run_scoped(
                        smem_scope,
                        gate_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        up_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        route_weight_smem=mgpu.SMEM((config.block_m,), dtype=h_route_weights_ref.dtype),
                        activation_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        w_down_smem=_wgmma_smem((config.block_k, config.block_n), w_down_ref.dtype),
                        ready_barrier=mgpu.Barrier(num_arrivals=3),
                        route_ready_barrier=mgpu.Barrier(num_arrivals=1),
                    )

                output = pl.run_scoped(
                    acc_scope,
                    acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                )

                def store_scope(output_smem) -> None:
                    output_smem[:, :] = output
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            remote_source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    store_scope,
                    output_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=source_return_ref.dtype),
                )

        def _switch_compute_to_src(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _compute_branch(_) -> None:
                    _compute_to_src(static_src_ordinal)

                return _compute_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_compute_to_src(src_ordinal)

    out_shape = jax.ShapeDtypeStruct(
        (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
        jnp.bfloat16,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_w2_from_h_return_direct_to_source_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_w2_from_h_return_direct_to_source_kernel(config, use_exact_expert_major=use_exact_expert_major)

    def local_fn(
        h_local: Float[Array, "1 rows twoI"],
        h_route_weights_local: Float[Array, "1 rows"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_down_local: Float[Array, "1 E I D"],
    ):
        h_local = h_local[0]
        h_route_weights_local = h_route_weights_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_down_local = w_down_local[0]
        source_return = kernel(
            h_local,
            h_route_weights_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_down_local,
        )
        return source_return[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _make_w2_from_compact_h_return_direct_to_source_kernel(
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    validate_w2_return_config(config)
    k_tiles = config.intermediate_dim // config.block_k
    n_tiles = config.hidden_dim // config.block_n
    src_offsets = tuple(range(config.ep_size))

    def body(
        h_ref: Float[Ref, "E C twoI"],
        h_route_weights_ref: Float[Ref, "E C"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        expert_base_ref: Int[Ref, "E"],
        src_base_by_expert_ref: Int[Ref, "SRC E"],
        w_down_ref: Float[Ref, "E I D"],
        source_return_ref: Float[Ref, "DST Q M D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)

        def _compute_to_src(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % config.ep_size
            dst_ordinal = (-static_src_ordinal) % config.ep_size
            remote_source_return_ref = None
            if static_src_ordinal != 0:
                remote_source_return_ref = mgpu.remote_ref(
                    source_return_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            def _h_expert_row_start(expert) -> jax.Array:
                local_row_start = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START]
                if use_exact_expert_major:
                    src_rank = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK]
                    return src_base_by_expert_ref[src_rank, expert] + local_row_start
                return local_row_start - expert_base_ref[expert]

            @pl.when(valid_rows > 0)
            def _compute_return_block() -> None:
                expert = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT]
                row_start = _h_expert_row_start(expert)

                def acc_scope(acc_ref) -> jax.Array:
                    def smem_scope(
                        gate_smem,
                        up_smem,
                        route_weight_smem,
                        activation_smem,
                        w_down_smem,
                        weight_ready_barrier,
                        tile_ready_barrier,
                    ) -> jax.Array:
                        mgpu.copy_gmem_to_smem(
                            h_route_weights_ref.at[expert, pl.ds(row_start, config.block_m)],
                            route_weight_smem,
                            weight_ready_barrier,
                        )

                        @pl.loop(0, k_tiles)
                        def _k_loop(kk) -> None:
                            k_start = kk * config.block_k
                            mgpu.copy_gmem_to_smem(
                                h_ref.at[
                                    expert,
                                    pl.ds(row_start, config.block_m),
                                    pl.ds(k_start, config.block_k),
                                ],
                                gate_smem,
                                tile_ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                h_ref.at[
                                    expert,
                                    pl.ds(row_start, config.block_m),
                                    pl.ds(config.intermediate_dim + k_start, config.block_k),
                                ],
                                up_smem,
                                tile_ready_barrier,
                            )
                            mgpu.copy_gmem_to_smem(
                                w_down_ref.at[
                                    expert,
                                    pl.ds(k_start, config.block_k),
                                    pl.ds(n_tile * config.block_n, config.block_n),
                                ],
                                w_down_smem,
                                tile_ready_barrier,
                            )
                            mgpu.barrier_wait(tile_ready_barrier)
                            activation_smem[:, :] = (
                                _silu(gate_smem[:, :].astype(jnp.float32)) * up_smem[:, :].astype(jnp.float32)
                            ).astype(activation_smem.dtype)
                            mgpu.commit_smem()
                            mgpu.wgmma(acc_ref, activation_smem, w_down_smem)
                            mgpu.wgmma_wait(0)

                        # Route scaling moves to the fp32 accumulator: the weight is per-row, so
                        # (diag(w) @ A) @ W2 == diag(w) @ (A @ W2), and the k-loop drops one
                        # multiply per element.
                        mgpu.barrier_wait(weight_ready_barrier)
                        route_weight_vec = mgpu.load(
                            route_weight_smem,
                            (pl.ds(0, config.block_m),),
                            layout=mgpu.Layout.WGMMA.reduce(1),
                        ).astype(jnp.float32)
                        route_weight = lax.broadcast_in_dim(
                            route_weight_vec,
                            (config.block_m, config.block_n),
                            (0,),
                        )
                        return (acc_ref[...] * route_weight).astype(source_return_ref.dtype)

                    return pl.run_scoped(
                        smem_scope,
                        gate_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        up_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        route_weight_smem=mgpu.SMEM((config.block_m,), dtype=h_route_weights_ref.dtype),
                        activation_smem=_wgmma_smem((config.block_m, config.block_k), h_ref.dtype),
                        w_down_smem=_wgmma_smem((config.block_k, config.block_n), w_down_ref.dtype),
                        weight_ready_barrier=mgpu.Barrier(num_arrivals=1),
                        tile_ready_barrier=mgpu.Barrier(num_arrivals=3),
                    )

                output = pl.run_scoped(
                    acc_scope,
                    acc_ref=mgpu.ACC((config.block_m, config.block_n)),
                )

                def store_scope(output_smem) -> None:
                    output_smem[:, :] = output
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            output_smem,
                            remote_source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    store_scope,
                    output_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=source_return_ref.dtype),
                )

        def _switch_compute_to_src(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _compute_branch(_) -> None:
                    _compute_to_src(static_src_ordinal)

                return _compute_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_compute_to_src(src_ordinal)

    out_shape = jax.ShapeDtypeStruct(
        (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
        jnp.bfloat16,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_w2_from_compact_h_return_direct_to_source_kernel(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool = False,
):
    kernel = _make_w2_from_compact_h_return_direct_to_source_kernel(
        config,
        use_exact_expert_major=use_exact_expert_major,
    )

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        h_route_weights_local: Float[Array, "1 E C"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        expert_base_local: Int[Array, "1 E"],
        src_base_by_expert_local: Int[Array, "1 SRC E"],
        w_down_local: Float[Array, "1 E I D"],
    ):
        h_local = h_local[0]
        h_route_weights_local = h_route_weights_local[0]
        recv_meta_local = recv_meta_local[0]
        expert_base_local = expert_base_local[0]
        src_base_by_expert_local = src_base_by_expert_local[0]
        w_down_local = w_down_local[0]
        source_return = kernel(
            h_local,
            h_route_weights_local,
            recv_meta_local,
            expert_base_local,
            src_base_by_expert_local,
            w_down_local,
        )
        return source_return[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None),
            P(AXIS, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _make_return_copy_kernel(config: PushInboxConfig):
    validate_w2_return_config(config)
    n_tiles = config.hidden_dim // config.block_n
    src_offsets = tuple(range(config.ep_size))

    def body(
        return_by_src_ref: Float[Ref, "SRC Q M D"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        source_return_ref: Float[Ref, "DST Q M D"],
    ) -> None:
        rank = lax.axis_index(AXIS)
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)

        def _copy_to_src(static_src_ordinal: int) -> None:
            src = (rank + static_src_ordinal) % config.ep_size
            dst_ordinal = (-static_src_ordinal) % config.ep_size
            remote_source_return_ref = None
            if static_src_ordinal != 0:
                remote_source_return_ref = mgpu.remote_ref(
                    source_return_ref,
                    src,
                    device_id_type=pl.DeviceIdType.LOGICAL,
                )
            valid_rows = recv_meta_ref[static_src_ordinal, entry, SOURCE_PUSH_META_VALID_ROWS]

            @pl.when(valid_rows > 0)
            def _copy_entry_tile() -> None:
                def _copy_scope(tile_smem) -> None:
                    tile_smem[:, :] = return_by_src_ref[
                        static_src_ordinal,
                        entry,
                        pl.ds(0, config.block_m),
                        pl.ds(n_tile * config.block_n, config.block_n),
                    ]
                    mgpu.commit_smem()
                    if static_src_ordinal == 0:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    else:
                        mgpu.copy_smem_to_gmem(
                            tile_smem,
                            remote_source_return_ref.at[
                                dst_ordinal,
                                entry,
                                pl.ds(0, config.block_m),
                                pl.ds(n_tile * config.block_n, config.block_n),
                            ],
                        )
                    mgpu.wait_smem_to_gmem(0, wait_read_only=False)

                pl.run_scoped(
                    _copy_scope,
                    tile_smem=mgpu.SMEM((config.block_m, config.block_n), dtype=return_by_src_ref.dtype),
                )

        def _switch_copy_to_src(dynamic_src_ordinal) -> None:
            def _branch(static_src_ordinal: int):
                def _copy_branch(_) -> None:
                    _copy_to_src(static_src_ordinal)

                return _copy_branch

            branches = tuple(_branch(static_src_ordinal) for static_src_ordinal in src_offsets)
            lax.switch(dynamic_src_ordinal, branches, None)

        _switch_copy_to_src(src_ordinal)

    out_shape = jax.ShapeDtypeStruct(
        (config.ep_size, config.entries_per_rank, config.block_m, config.hidden_dim),
        jnp.bfloat16,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(config.ep_size, config.entries_per_rank, n_tiles),
        grid_names=("src_ordinal", "entry", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_return_copy_kernel(mesh: Mesh, config: PushInboxConfig):
    kernel = _make_return_copy_kernel(config)

    def local_fn(
        return_by_destination_local: Float[Array, "1 SRC Q M D"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
    ):
        return_by_destination_local = return_by_destination_local[0]
        recv_meta_local = recv_meta_local[0]
        source_return = kernel(return_by_destination_local, recv_meta_local)
        return source_return[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _sharded_w2_return_to_source_kernel(mesh: Mesh, config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    w2_kernel = _sharded_w2_return_kernel(mesh, config, use_exact_expert_major=use_exact_expert_major)
    return_copy_kernel = _sharded_return_copy_kernel(mesh, config)

    def fn(
        hidden: Float[Array, "Dst rows I"],
        recv_meta: Int[Array, "Dst SRC Q F"],
        expert_base: Int[Array, "Dst E"],
        src_base_by_expert: Int[Array, "Dst SRC E"],
        w_down: Float[Array, "Dst E I D"],
    ):
        return_by_destination = w2_kernel(hidden, recv_meta, expert_base, src_base_by_expert, w_down)
        source_return = return_copy_kernel(return_by_destination, recv_meta)
        return return_by_destination, source_return

    return fn


def _validate_w2_return(
    config: PushInboxConfig,
    hidden_host: np.ndarray,
    recv_meta_host: np.ndarray,
    expert_base_host: np.ndarray,
    src_base_by_expert_host: np.ndarray,
    w_down_host: np.ndarray,
    return_by_destination: jax.Array,
    source_return: jax.Array | None = None,
    *,
    use_exact_expert_major: bool = False,
) -> W2ReturnValidationMetrics:
    expected = reference_w2_return_by_destination(
        config,
        hidden_host,
        recv_meta_host,
        w_down_host,
        expert_base_host,
        src_base_by_expert_host,
        use_exact_expert_major=use_exact_expert_major,
    )
    observed = np.asarray(jax.device_get(return_by_destination), dtype=np.float32)
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    diff = np.abs(observed - expected_host)
    live_entries = recv_meta_host[..., SOURCE_PUSH_META_VALID_ROWS] > 0
    if np.any(live_entries):
        live_diff = diff[live_entries]
        max_abs_diff = float(np.max(live_diff))
        mean_abs_diff = float(np.mean(live_diff))
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    if source_return is None:
        source_observed = source_queue_from_destination_return(config, observed)
    else:
        source_observed = source_return
    source_expected = source_queue_from_destination_return(config, expected_host)
    source_queue_diff = np.abs(
        np.asarray(jax.device_get(source_observed), dtype=np.float32)
        - np.asarray(jax.device_get(source_expected), dtype=np.float32)
    )
    return W2ReturnValidationMetrics(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        all_max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        source_queue_max_abs_diff=float(np.max(source_queue_diff)) if source_queue_diff.size else 0.0,
    )


def _validate_w2_return_source_queue(
    config: PushInboxConfig,
    hidden_host: np.ndarray,
    recv_meta_host: np.ndarray,
    expert_base_host: np.ndarray,
    src_base_by_expert_host: np.ndarray,
    w_down_host: np.ndarray,
    source_return: jax.Array,
    *,
    use_exact_expert_major: bool = False,
) -> W2ReturnValidationMetrics:
    expected = reference_w2_return_by_destination(
        config,
        hidden_host,
        recv_meta_host,
        w_down_host,
        expert_base_host,
        src_base_by_expert_host,
        use_exact_expert_major=use_exact_expert_major,
    )
    source_expected = source_queue_from_destination_return(config, expected)
    observed_host = np.asarray(jax.device_get(source_return), dtype=np.float32)
    expected_host = np.asarray(jax.device_get(source_expected), dtype=np.float32)
    diff = np.abs(observed_host - expected_host)
    return W2ReturnValidationMetrics(
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
        all_max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        source_queue_max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
    )


def _run_w2_return_one(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    input_builder: Callable[[PushInboxConfig], W2ReturnHostInputs],
    *,
    row_metadata: dict[str, Any] | None = None,
    return_mode: str = W2_RETURN_MODE_DESTINATION_LOCAL,
) -> list[dict[str, Any]]:
    row_metadata = row_metadata or {}
    try:
        _validate_w2_return_mode(return_mode)
        if settings.repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {settings.repeat_runs}")
        validate_w2_return_config(config)
        _emit_progress(config, settings.progress_events, "mesh_start")
        mesh = _make_mesh(config.ep_size)
        _emit_progress(config, settings.progress_events, "make_inputs_start")
        host_inputs = input_builder(config)
        inputs = device_w2_return_inputs_from_host(host_inputs)
        _emit_progress(config, settings.progress_events, "device_put_start")
        hidden = jax.device_put(inputs.hidden, NamedSharding(mesh, P(AXIS, None, None)))
        recv_meta = jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None)))
        expert_base = jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None)))
        src_base_by_expert = jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None)))
        w_down = jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None)))
        _emit_progress(config, settings.progress_events, "jit_start")
        if return_mode == W2_RETURN_MODE_DIRECT_REMOTE:
            fn = jax.jit(
                _sharded_w2_return_direct_to_source_kernel(
                    mesh,
                    config,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )
            )
        elif return_mode == W2_RETURN_MODE_SEPARATE_COPY:
            fn = jax.jit(
                _sharded_w2_return_to_source_kernel(
                    mesh,
                    config,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )
            )
        else:
            fn = jax.jit(
                _sharded_w2_return_kernel(
                    mesh,
                    config,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )
            )

        timing = _time_jitted(
            fn,
            hidden,
            recv_meta,
            expert_base,
            src_base_by_expert,
            w_down,
            warmup=settings.warmup,
            steps=settings.steps,
            repeat_runs=settings.repeat_runs,
            separate_compile=settings.separate_compile,
            progress=lambda event: _emit_progress(config, settings.progress_events, event),
        )
        validation = None
        if settings.check:
            if return_mode == W2_RETURN_MODE_DIRECT_REMOTE:
                validation = _validate_w2_return_source_queue(
                    config,
                    host_inputs.hidden,
                    host_inputs.recv_meta,
                    host_inputs.expert_base,
                    host_inputs.src_base_by_expert,
                    host_inputs.w_down,
                    timing.output,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )
            else:
                return_by_destination = (
                    timing.output[0] if return_mode == W2_RETURN_MODE_SEPARATE_COPY else timing.output
                )
                source_return = timing.output[1] if return_mode == W2_RETURN_MODE_SEPARATE_COPY else None
                validation = _validate_w2_return(
                    config,
                    host_inputs.hidden,
                    host_inputs.recv_meta,
                    host_inputs.expert_base,
                    host_inputs.src_base_by_expert,
                    host_inputs.w_down,
                    return_by_destination,
                    source_return,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )

        queue_stats = inputs.queue_stats
        flops_per_rank = queue_stats["rounded_rows_per_rank_mean"] * config.intermediate_dim * config.hidden_dim * 2
        bytes_per_rank = (
            queue_stats["rounded_rows_per_rank_mean"] * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16
        )
        rows = []
        for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
            row = {
                "kernel": KERNEL_NAME,
                "implementation": _w2_return_implementation_name(return_mode),
                "copy_to_source": return_mode == W2_RETURN_MODE_SEPARATE_COPY,
                "direct_to_source": return_mode == W2_RETURN_MODE_DIRECT_REMOTE,
                "return_mode": return_mode,
                "depends_on_kernel": W13_KERNEL_NAME,
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
                "return_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
                "w2_tflops_per_rank": flops_per_rank / steady_state_time / 1e12,
                "max_abs_diff": None if validation is None else validation.max_abs_diff,
                "mean_abs_diff": None if validation is None else validation.mean_abs_diff,
                "all_max_abs_diff": None if validation is None else validation.all_max_abs_diff,
                "source_queue_max_abs_diff": None if validation is None else validation.source_queue_max_abs_diff,
                "error": None,
                "error_type": None,
                "error_message": None,
            }
            row.update(row_metadata)
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        if settings.debug_exceptions:
            raise
        row = {
            "kernel": KERNEL_NAME,
            "implementation": _w2_return_implementation_name(return_mode),
            "copy_to_source": return_mode == W2_RETURN_MODE_SEPARATE_COPY,
            "direct_to_source": return_mode == W2_RETURN_MODE_DIRECT_REMOTE,
            "return_mode": return_mode,
            "config": asdict(config),
            "compile_time": None,
            "lower_compile_time": None,
            "first_run_time": None,
            "repeat_run": None,
            "repeat_runs": settings.repeat_runs,
            "steady_state_time": None,
            "bytes_per_rank": None,
            "return_gbps_per_rank": None,
            "w2_tflops_per_rank": None,
            "max_abs_diff": None,
            "mean_abs_diff": None,
            "all_max_abs_diff": None,
            "source_queue_max_abs_diff": None,
            "error": f"{type(exc).__name__}: {exc}",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        row.update(row_metadata)
        return [row]
    finally:
        jax.clear_caches()


def _validate_w2_return_mode(return_mode: str) -> None:
    if return_mode not in W2_RETURN_MODES:
        raise ValueError(f"unknown return_mode={return_mode!r}; expected one of {W2_RETURN_MODES}")


def _w2_return_implementation_name(return_mode: str) -> str:
    if return_mode == W2_RETURN_MODE_DIRECT_REMOTE:
        return DIRECT_RETURN_KERNEL_NAME
    if return_mode == W2_RETURN_MODE_SEPARATE_COPY:
        return RETURN_COPY_KERNEL_NAME
    return KERNEL_NAME


def run_source_push_w2_return_source_plan(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
    hidden_input_mode: str = W2_HIDDEN_INPUT_SYNTHETIC,
    return_mode: str = W2_RETURN_MODE_DESTINATION_LOCAL,
) -> list[dict[str, Any]]:
    """Run W2 over source-padded SourcePushPlan hidden rows."""

    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_w2_return_one(
        config,
        settings,
        input_builder=lambda run_config: make_w2_return_source_plan_inputs(
            run_config,
            hidden_input_mode=hidden_input_mode,
        ),
        return_mode=return_mode,
    )


def _emit_progress(config: PushInboxConfig, progress_events: bool, event: str) -> None:
    if not progress_events:
        return
    print(
        json.dumps(
            {
                "config": asdict(config),
                "event": event,
                "kernel": KERNEL_NAME,
                "time": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _expert_base_or_zeros(config: PushInboxConfig, expert_base: Any) -> np.ndarray:
    if expert_base is None:
        return np.zeros((config.ep_size, config.experts_per_rank), dtype=np.int32)
    return np.asarray(jax.device_get(expert_base), dtype=np.int32)


def _src_base_by_expert_or_zeros(config: PushInboxConfig, src_base_by_expert: Any) -> np.ndarray:
    if src_base_by_expert is None:
        return np.zeros((config.ep_size, config.ep_size, config.experts_per_rank), dtype=np.int32)
    return np.asarray(jax.device_get(src_base_by_expert), dtype=np.int32)


def _w2_hidden_row_start(
    recv_meta: np.ndarray,
    expert_base: np.ndarray,
    src_base_by_expert: np.ndarray,
    *,
    dst: int,
    src_ordinal: int,
    entry: int,
    use_exact_expert_major: bool,
) -> int:
    if use_exact_expert_major:
        src_rank = int(recv_meta[dst, src_ordinal, entry, SOURCE_PUSH_META_SRC_RANK])
        expert = int(recv_meta[dst, src_ordinal, entry, SOURCE_PUSH_META_LOCAL_EXPERT])
        local_row_start = int(recv_meta[dst, src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START])
        return int(expert_base[dst, expert] + src_base_by_expert[dst, src_rank, expert] + local_row_start)
    return int(recv_meta[dst, src_ordinal, entry, SOURCE_PUSH_META_LOCAL_ROW_START])


def _validate_reference_shapes(
    config: PushInboxConfig,
    hidden: np.ndarray,
    recv_meta: np.ndarray,
    w_down: np.ndarray,
    expert_base: np.ndarray | None = None,
    src_base_by_expert: np.ndarray | None = None,
) -> None:
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [dst, rows, I], got {hidden.shape}")
    expected_recv_meta = (config.ep_size, config.ep_size, config.entries_per_rank, SOURCE_PUSH_META_FIELDS)
    if recv_meta.shape != expected_recv_meta:
        raise ValueError("recv_meta shape must be " f"{expected_recv_meta}, got {recv_meta.shape}")
    if w_down.shape != (config.ep_size, config.experts_per_rank, config.intermediate_dim, config.hidden_dim):
        raise ValueError(
            "w_down shape must be "
            f"{(config.ep_size, config.experts_per_rank, config.intermediate_dim, config.hidden_dim)}, "
            f"got {w_down.shape}"
        )
    if hidden.shape[0] != config.ep_size or hidden.shape[-1] != config.intermediate_dim:
        raise ValueError(
            f"hidden shape {hidden.shape} must have leading dim {config.ep_size} and trailing dim "
            f"{config.intermediate_dim}"
        )
    if expert_base is not None and expert_base.shape != (config.ep_size, config.experts_per_rank):
        raise ValueError(
            f"expert_base shape must be {(config.ep_size, config.experts_per_rank)}, got {expert_base.shape}"
        )
    expected_src_base = (config.ep_size, config.ep_size, config.experts_per_rank)
    if src_base_by_expert is not None and src_base_by_expert.shape != expected_src_base:
        raise ValueError(f"src_base_by_expert shape must be {expected_src_base}, got {src_base_by_expert.shape}")
