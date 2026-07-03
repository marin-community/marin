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
from jax import Ref, shard_map
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
    _make_mesh,
    _make_source_push_plan_inputs,
    _make_weights,
    _reference_hidden,
    _time_jitted,
    _wgmma_smem,
)


KERNEL_NAME = "source_push_w2_return"


@dataclass(frozen=True)
class W2ReturnHostInputs:
    """Host-side inputs for the destination-local W2 return kernel."""

    hidden: np.ndarray
    recv_meta: np.ndarray
    w_down: np.ndarray
    queue_stats: dict[str, Any]


@dataclass(frozen=True)
class W2ReturnDeviceInputs:
    """Device inputs for the destination-local W2 return kernel."""

    hidden: jax.Array
    recv_meta: jax.Array
    w_down: jax.Array
    queue_stats: dict[str, Any]


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


def make_w_down(config: PushInboxConfig) -> jax.Array:
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
    return jnp.asarray(w_down, dtype=jnp.bfloat16)


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
) -> Float[Array, "Dst Src Q M D"]:
    """Reference W2 over expert-major hidden rows in destination-local return order."""

    hidden_host = np.asarray(jax.device_get(hidden), dtype=np.float32)
    recv_meta_host = np.asarray(jax.device_get(recv_meta), dtype=np.int32)
    w_down_host = np.asarray(jax.device_get(w_down), dtype=np.float32)
    _validate_reference_shapes(config, hidden_host, recv_meta_host, w_down_host)

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
                valid_rows = int(recv_meta_host[dst, src_ordinal, entry, 3])
                if valid_rows <= 0:
                    continue
                expert = int(recv_meta_host[dst, src_ordinal, entry, 1])
                row_start = int(recv_meta_host[dst, src_ordinal, entry, 2])
                hidden_rows = hidden_host[dst, row_start : row_start + config.block_m]
                return_by_destination[dst, src_ordinal, entry] = hidden_rows @ w_down_host[dst, expert]
    return jnp.asarray(return_by_destination)


def make_w2_return_source_plan_inputs(config: PushInboxConfig) -> W2ReturnHostInputs:
    """Build W2 inputs from the current source-padded SourcePushPlan W13 layout."""

    validate_w2_return_config(config)
    host_inputs = _make_source_push_plan_inputs(config)
    w_gate_up = np.asarray(jax.device_get(_make_weights(config)), dtype=np.float32)
    hidden = _reference_hidden(
        config,
        host_inputs.x,
        host_inputs.send_meta,
        w_gate_up,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    return W2ReturnHostInputs(
        hidden=hidden.astype(np.float32),
        recv_meta=host_inputs.recv_meta,
        w_down=np.asarray(jax.device_get(make_w_down(config)), dtype=np.float32),
        queue_stats={**host_inputs.queue_stats, "w2_input_mode": "source_push_plan"},
    )


def device_w2_return_inputs_from_host(host_inputs: W2ReturnHostInputs) -> W2ReturnDeviceInputs:
    """Move host W2-return inputs into their benchmark dtypes."""

    return W2ReturnDeviceInputs(
        hidden=jnp.asarray(host_inputs.hidden, dtype=jnp.bfloat16),
        recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
        w_down=jnp.asarray(host_inputs.w_down, dtype=jnp.bfloat16),
        queue_stats=host_inputs.queue_stats,
    )


def _make_w2_return_kernel(config: PushInboxConfig):
    validate_w2_return_config(config)
    k_tiles = config.intermediate_dim // config.block_k
    n_tiles = config.hidden_dim // config.block_n

    def body(
        hidden_ref: Float[Ref, "rows I"],
        recv_meta_ref: Int[Ref, "SRC Q F"],
        w_down_ref: Float[Ref, "E I D"],
        return_ref: Float[Ref, "SRC Q M D"],
    ) -> None:
        src_ordinal = pl.program_id(0)
        entry = pl.program_id(1)
        n_tile = pl.program_id(2)
        valid_rows = recv_meta_ref[src_ordinal, entry, 3]

        @pl.when(valid_rows > 0)
        def _compute_return_block() -> None:
            expert = recv_meta_ref[src_ordinal, entry, 1]
            row_start = recv_meta_ref[src_ordinal, entry, 2]

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


def _sharded_w2_return_kernel(mesh: Mesh, config: PushInboxConfig):
    kernel = _make_w2_return_kernel(config)

    def local_fn(
        hidden_local: Float[Array, "1 rows I"],
        recv_meta_local: Int[Array, "1 SRC Q F"],
        w_down_local: Float[Array, "1 E I D"],
    ):
        hidden_local = hidden_local[0]
        recv_meta_local = recv_meta_local[0]
        w_down_local = w_down_local[0]
        return_by_src_ordinal = kernel(hidden_local, recv_meta_local, w_down_local)
        return return_by_src_ordinal[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None),
            P(AXIS, None, None, None),
            P(AXIS, None, None, None),
        ),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _validate_w2_return(
    config: PushInboxConfig,
    hidden_host: np.ndarray,
    recv_meta_host: np.ndarray,
    w_down_host: np.ndarray,
    return_by_destination: jax.Array,
) -> W2ReturnValidationMetrics:
    expected = reference_w2_return_by_destination(config, hidden_host, recv_meta_host, w_down_host)
    observed = np.asarray(jax.device_get(return_by_destination), dtype=np.float32)
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    diff = np.abs(observed - expected_host)
    live_entries = recv_meta_host[..., 3] > 0
    if np.any(live_entries):
        live_diff = diff[live_entries]
        max_abs_diff = float(np.max(live_diff))
        mean_abs_diff = float(np.mean(live_diff))
    else:
        max_abs_diff = 0.0
        mean_abs_diff = 0.0
    source_observed = source_queue_from_destination_return(config, observed)
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


def _run_w2_return_one(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    input_builder: Callable[[PushInboxConfig], W2ReturnHostInputs] = make_w2_return_source_plan_inputs,
    *,
    row_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    row_metadata = row_metadata or {}
    try:
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
        w_down = jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None)))
        _emit_progress(config, settings.progress_events, "jit_start")
        fn = jax.jit(_sharded_w2_return_kernel(mesh, config))

        timing = _time_jitted(
            fn,
            hidden,
            recv_meta,
            w_down,
            warmup=settings.warmup,
            steps=settings.steps,
            repeat_runs=settings.repeat_runs,
            separate_compile=settings.separate_compile,
            progress=lambda event: _emit_progress(config, settings.progress_events, event),
        )
        validation = None
        if settings.check:
            validation = _validate_w2_return(
                config,
                host_inputs.hidden,
                host_inputs.recv_meta,
                host_inputs.w_down,
                timing.output,
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
                "implementation": KERNEL_NAME,
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
            "implementation": KERNEL_NAME,
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
) -> list[dict[str, Any]]:
    """Run destination-local W2 over source-padded SourcePushPlan hidden rows."""

    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_w2_return_one(config, settings)


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


def _validate_reference_shapes(
    config: PushInboxConfig,
    hidden: np.ndarray,
    recv_meta: np.ndarray,
    w_down: np.ndarray,
) -> None:
    if hidden.ndim != 3:
        raise ValueError(f"hidden must have shape [dst, rows, I], got {hidden.shape}")
    if recv_meta.shape != (config.ep_size, config.ep_size, config.entries_per_rank, 4):
        raise ValueError(
            "recv_meta shape must be "
            f"{(config.ep_size, config.ep_size, config.entries_per_rank, 4)}, got {recv_meta.shape}"
        )
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
