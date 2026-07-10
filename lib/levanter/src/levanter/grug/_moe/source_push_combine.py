# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# pyrefly: ignore-errors

"""Package-private source combine kernel for the source-push MGPU MoE prototype."""

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
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_inbox import (
    AXIS,
    BYTES_PER_BF16,
    PushInboxConfig,
    SourcePushInboxRunSettings,
    _make_mesh,
    _routing_counts,
    _selected_experts_from_counts,
    _time_jitted,
)
from levanter.grug._moe.source_push_plan import SourcePushPlan, build_source_push_plan, source_push_combine


KERNEL_NAME = "source_push_combine"
SOURCE_COMBINE_MODE_DIRECT_GATHER_SUM = "direct_gather_sum"


@dataclass(frozen=True)
class SourceCombineHostInputs:
    """Host-side inputs for source-local deterministic combine."""

    return_y: np.ndarray
    queue_dst_ord: np.ndarray
    queue_entry: np.ndarray
    queue_row: np.ndarray
    route_combine_weights: np.ndarray
    route_valid_mask: np.ndarray
    plan: SourcePushPlan
    queue_stats: dict[str, Any]


@dataclass(frozen=True)
class SourceCombineDeviceInputs:
    """Device inputs for source-local deterministic combine."""

    return_y: jax.Array
    queue_dst_ord: jax.Array
    queue_entry: jax.Array
    queue_row: jax.Array
    route_combine_weights: jax.Array
    route_valid_mask: jax.Array
    queue_stats: dict[str, Any]


@dataclass(frozen=True)
class SourceCombineValidationMetrics:
    """Validation summary for source-local deterministic combine."""

    max_abs_diff: float
    mean_abs_diff: float


def validate_source_combine_config(config: PushInboxConfig) -> None:
    """Validate tile constraints required by the source combine kernel."""

    config.validate()
    if config.hidden_dim % config.block_n:
        raise ValueError(
            f"hidden_dim must be divisible by block_n for source combine, got {config.hidden_dim=} {config.block_n=}"
        )


def make_source_push_combine_source_plan_inputs(config: PushInboxConfig) -> SourceCombineHostInputs:
    """Build source-visible return queue and inverse metadata for combine timing."""

    validate_source_combine_config(config)
    counts = _routing_counts(config)
    selected_experts = np.stack(
        [_selected_experts_from_counts(config, counts[src], src) for src in range(config.ep_size)],
        axis=0,
    )
    combine_weights = _make_combine_weights(config)
    plan = build_source_push_plan(
        jnp.asarray(selected_experts, dtype=jnp.int32),
        jnp.asarray(combine_weights, dtype=jnp.float32),
        ep_size=config.ep_size,
        experts_per_rank=config.experts_per_rank,
        block_m=config.block_m,
        capacity_factor=config.capacity_factor,
        entries_per_dst=config.entries_per_rank,
    )
    return_y = _make_source_return(config, plan)
    route_inverse = _make_route_inverse(config, plan)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    queue_stats = _combine_queue_stats(config, valid_mask, int(jax.device_get(plan.dropped_routes)))
    return SourceCombineHostInputs(
        return_y=return_y,
        queue_dst_ord=route_inverse["queue_dst_ord"],
        queue_entry=route_inverse["queue_entry"],
        queue_row=route_inverse["queue_row"],
        route_combine_weights=route_inverse["route_combine_weights"],
        route_valid_mask=route_inverse["route_valid_mask"],
        plan=plan,
        queue_stats=queue_stats,
    )


def device_source_combine_inputs_from_host(host_inputs: SourceCombineHostInputs) -> SourceCombineDeviceInputs:
    """Move host combine inputs into benchmark dtypes."""

    return SourceCombineDeviceInputs(
        return_y=jnp.asarray(host_inputs.return_y, dtype=jnp.bfloat16),
        queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
        queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
        queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
        route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
        route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
        queue_stats=host_inputs.queue_stats,
    )


def reference_source_push_combine(
    return_y: Float[Array, "S Dst Q M D"],
    plan: SourcePushPlan,
) -> Float[Array, "S T D"]:
    """Reference source combine using the source-owned inverse plan."""

    return source_push_combine(return_y, plan)


def _make_direct_gather_sum_kernel(config: PushInboxConfig):
    validate_source_combine_config(config)
    n_tiles = config.hidden_dim // config.block_n
    token_blocks = (config.tokens_per_rank + config.block_m - 1) // config.block_m

    def body(
        return_y_ref: Float[Ref, "DST Q M D"],
        queue_dst_ord_ref: Int[Ref, "T K"],
        queue_entry_ref: Int[Ref, "T K"],
        queue_row_ref: Int[Ref, "T K"],
        route_combine_weights_ref: Float[Ref, "T K"],
        route_valid_mask_ref: Bool[Ref, "T K"],
        out_ref: Float[Ref, "T D"],
    ) -> None:
        token_block = pl.program_id(0)
        n_tile = pl.program_id(1)

        @pl.loop(0, config.block_m)
        def _row_loop(row) -> None:
            token = token_block * config.block_m + row

            @pl.when(token < config.tokens_per_rank)
            def _combine_token_row() -> None:
                acc = jnp.zeros((config.block_n,), dtype=jnp.float32)
                for route_slot in range(config.topk):
                    dst_ordinal = queue_dst_ord_ref[token, route_slot]
                    entry = queue_entry_ref[token, route_slot]
                    queue_row = queue_row_ref[token, route_slot]
                    weight = route_combine_weights_ref[token, route_slot]
                    valid = route_valid_mask_ref[token, route_slot]
                    value = return_y_ref[
                        dst_ordinal,
                        entry,
                        queue_row,
                        pl.ds(n_tile * config.block_n, config.block_n),
                    ]
                    weighted = (value * weight).astype(out_ref.dtype).astype(jnp.float32)
                    zero = jnp.zeros((config.block_n,), dtype=jnp.float32)
                    acc += jnp.where(valid, weighted, zero)
                out_ref[token, pl.ds(n_tile * config.block_n, config.block_n)] = acc.astype(out_ref.dtype)

    out_shape = jax.ShapeDtypeStruct((config.tokens_per_rank, config.hidden_dim), jnp.bfloat16)
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(token_blocks, n_tiles),
        grid_names=("token_block", "n_tile"),
        compiler_params=compiler_params,
    )


def _sharded_source_combine_kernel(mesh: Mesh, config: PushInboxConfig):
    gather_sum_kernel = _make_direct_gather_sum_kernel(config)

    def local_fn(
        return_y_local: Float[Array, "1 DST Q M D"],
        queue_dst_ord_local: Int[Array, "1 T K"],
        queue_entry_local: Int[Array, "1 T K"],
        queue_row_local: Int[Array, "1 T K"],
        route_combine_weights_local: Float[Array, "1 T K"],
        route_valid_mask_local: Bool[Array, "1 T K"],
    ):
        return_y_local = return_y_local[0]
        queue_dst_ord_local = queue_dst_ord_local[0]
        queue_entry_local = queue_entry_local[0]
        queue_row_local = queue_row_local[0]
        route_combine_weights_local = route_combine_weights_local[0]
        route_valid_mask_local = route_valid_mask_local[0]
        out = gather_sum_kernel(
            return_y_local,
            queue_dst_ord_local,
            queue_entry_local,
            queue_row_local,
            route_combine_weights_local,
            route_valid_mask_local,
        )
        return out[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(AXIS, None, None, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
            P(AXIS, None, None),
        ),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def _validate_source_combine(
    host_inputs: SourceCombineHostInputs,
    observed: jax.Array,
) -> SourceCombineValidationMetrics:
    expected = reference_source_push_combine(
        jnp.asarray(host_inputs.return_y, dtype=jnp.bfloat16),
        host_inputs.plan,
    )
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    observed_host = np.asarray(jax.device_get(observed), dtype=np.float32)
    diff = np.abs(observed_host - expected_host)
    return SourceCombineValidationMetrics(
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
    )


def _run_source_combine_one(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    input_builder: Callable[[PushInboxConfig], SourceCombineHostInputs] = make_source_push_combine_source_plan_inputs,
) -> list[dict[str, Any]]:
    try:
        if settings.repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {settings.repeat_runs}")
        validate_source_combine_config(config)
        _emit_progress(config, settings.progress_events, "mesh_start")
        mesh = _make_mesh(config.ep_size)
        _emit_progress(config, settings.progress_events, "make_inputs_start")
        host_inputs = input_builder(config)
        inputs = device_source_combine_inputs_from_host(host_inputs)
        _emit_progress(config, settings.progress_events, "device_put_start")
        return_y = jax.device_put(inputs.return_y, NamedSharding(mesh, P(AXIS, None, None, None, None)))
        queue_dst_ord = jax.device_put(inputs.queue_dst_ord, NamedSharding(mesh, P(AXIS, None, None)))
        queue_entry = jax.device_put(inputs.queue_entry, NamedSharding(mesh, P(AXIS, None, None)))
        queue_row = jax.device_put(inputs.queue_row, NamedSharding(mesh, P(AXIS, None, None)))
        route_combine_weights = jax.device_put(
            inputs.route_combine_weights,
            NamedSharding(mesh, P(AXIS, None, None)),
        )
        route_valid_mask = jax.device_put(inputs.route_valid_mask, NamedSharding(mesh, P(AXIS, None, None)))
        _emit_progress(config, settings.progress_events, "jit_start")
        fn = jax.jit(_sharded_source_combine_kernel(mesh, config))
        timing = _time_jitted(
            fn,
            return_y,
            queue_dst_ord,
            queue_entry,
            queue_row,
            route_combine_weights,
            route_valid_mask,
            warmup=settings.warmup,
            steps=settings.steps,
            repeat_runs=settings.repeat_runs,
            separate_compile=settings.separate_compile,
            progress=lambda event: _emit_progress(config, settings.progress_events, event),
        )
        validation = _validate_source_combine(host_inputs, timing.output) if settings.check else None
        queue_stats = inputs.queue_stats
        rows = []
        for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
            bytes_per_rank = _combine_bytes_per_rank(config, queue_stats)
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
                "combine_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
                "max_abs_diff": None if validation is None else validation.max_abs_diff,
                "mean_abs_diff": None if validation is None else validation.mean_abs_diff,
                "error": None,
                "error_type": None,
                "error_message": None,
            }
            rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        if settings.debug_exceptions:
            raise
        return [
            {
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
                "combine_gbps_per_rank": None,
                "max_abs_diff": None,
                "mean_abs_diff": None,
                "error": f"{type(exc).__name__}: {exc}",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        ]
    finally:
        jax.clear_caches()


def run_source_push_combine_source_plan(
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
    """Run source-local deterministic combine from source-visible return queues."""

    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_source_combine_one(config, settings)


def _make_combine_weights(config: PushInboxConfig) -> np.ndarray:
    assignment_ids = np.arange(config.tokens_per_rank * config.topk, dtype=np.float32).reshape(
        config.tokens_per_rank,
        config.topk,
    )
    weights = 0.125 + ((assignment_ids % 11.0) / 32.0)
    source_scale = 1.0 + np.arange(config.ep_size, dtype=np.float32)[:, None, None] * 0.001
    return (weights[None, :, :] * source_scale).astype(np.float32)


def _make_source_return(config: PushInboxConfig, plan: SourcePushPlan) -> np.ndarray:
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    assignment_ids = np.asarray(jax.device_get(plan.assignment_ids), dtype=np.int32)
    return_y = np.zeros(
        (
            config.ep_size,
            config.ep_size,
            config.entries_per_rank,
            config.block_m,
            config.hidden_dim,
        ),
        dtype=np.float32,
    )
    d_component = (np.arange(config.hidden_dim, dtype=np.float32) % 17)[None, :] * 0.0001
    for src in range(config.ep_size):
        source_valid = valid_mask[src]
        values = 0.25 + (np.maximum(assignment_ids[src], 0).astype(np.float32) % 8192) * 0.00001
        source_return = return_y[src]
        source_return[source_valid] = values[source_valid][:, None] + d_component
        return_y[src] = source_return
    return return_y


def _make_route_inverse(config: PushInboxConfig, plan: SourcePushPlan) -> dict[str, np.ndarray]:
    token_ids = np.asarray(jax.device_get(plan.token_ids), dtype=np.int32)
    route_slots = np.asarray(jax.device_get(plan.route_slots), dtype=np.int32)
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    combine_weights = np.asarray(jax.device_get(plan.combine_weights), dtype=np.float32)
    queue_dst_ord = np.zeros((config.ep_size, config.tokens_per_rank, config.topk), dtype=np.int32)
    queue_entry = np.zeros_like(queue_dst_ord)
    queue_row = np.zeros_like(queue_dst_ord)
    route_valid_mask = np.zeros((config.ep_size, config.tokens_per_rank, config.topk), dtype=np.bool_)
    route_combine_weights = np.zeros((config.ep_size, config.tokens_per_rank, config.topk), dtype=np.float32)

    dst_ord_indices = np.broadcast_to(
        np.arange(config.ep_size, dtype=np.int32)[:, None, None],
        valid_mask.shape[1:],
    )
    entry_indices = np.broadcast_to(
        np.arange(config.entries_per_rank, dtype=np.int32)[None, :, None],
        valid_mask.shape[1:],
    )
    row_indices = np.broadcast_to(
        np.arange(config.block_m, dtype=np.int32)[None, None, :],
        valid_mask.shape[1:],
    )
    for src in range(config.ep_size):
        source_valid = valid_mask[src]
        tokens = token_ids[src][source_valid]
        slots = route_slots[src][source_valid]
        if tokens.size == 0:
            continue
        assignment_ids = tokens.astype(np.int64) * config.topk + slots.astype(np.int64)
        if np.unique(assignment_ids).size != assignment_ids.size:
            raise ValueError("source-push route inverse has duplicate token/route-slot assignments")
        queue_dst_ord[src, tokens, slots] = dst_ord_indices[source_valid]
        queue_entry[src, tokens, slots] = entry_indices[source_valid]
        queue_row[src, tokens, slots] = row_indices[source_valid]
        route_combine_weights[src, tokens, slots] = combine_weights[src][source_valid]
        route_valid_mask[src, tokens, slots] = True

    return {
        "queue_dst_ord": queue_dst_ord,
        "queue_entry": queue_entry,
        "queue_row": queue_row,
        "route_combine_weights": route_combine_weights,
        "route_valid_mask": route_valid_mask,
    }


def _combine_queue_stats(config: PushInboxConfig, valid_mask: np.ndarray, dropped_routes: int) -> dict[str, Any]:
    live_entries_per_rank = np.sum(np.any(valid_mask, axis=-1), axis=(1, 2))
    useful_rows_per_rank = np.sum(valid_mask, axis=(1, 2, 3))
    rounded_rows_per_rank = live_entries_per_rank * config.block_m
    useful_rows_total = int(np.sum(useful_rows_per_rank))
    rounded_rows_total = int(np.sum(rounded_rows_per_rank))
    output_elements_per_rank = config.tokens_per_rank * config.hidden_dim
    return {
        "input_mode": "source_push_plan",
        "combine_mode": SOURCE_COMBINE_MODE_DIRECT_GATHER_SUM,
        "dropped_routes": dropped_routes,
        "live_entries_total": int(np.sum(live_entries_per_rank)),
        "live_entries_per_rank_min": int(np.min(live_entries_per_rank)),
        "live_entries_per_rank_mean": float(np.mean(live_entries_per_rank)),
        "live_entries_per_rank_max": int(np.max(live_entries_per_rank)),
        "valid_rows_per_rank_min": int(np.min(useful_rows_per_rank)),
        "valid_rows_per_rank_mean": float(np.mean(useful_rows_per_rank)),
        "valid_rows_per_rank_max": int(np.max(useful_rows_per_rank)),
        "rounded_rows_per_rank_min": int(np.min(rounded_rows_per_rank)),
        "rounded_rows_per_rank_mean": float(np.mean(rounded_rows_per_rank)),
        "rounded_rows_per_rank_max": int(np.max(rounded_rows_per_rank)),
        "useful_rows_total": useful_rows_total,
        "rounded_rows_total": rounded_rows_total,
        "row_efficiency": useful_rows_total / rounded_rows_total if rounded_rows_total else 1.0,
        "masked_row_fraction": 1.0 - useful_rows_total / rounded_rows_total if rounded_rows_total else 0.0,
        "route_buffer_elements_per_rank": 0,
        "output_elements_per_rank": output_elements_per_rank,
    }


def _combine_bytes_per_rank(config: PushInboxConfig, queue_stats: dict[str, Any]) -> float:
    valid_rows = queue_stats["valid_rows_per_rank_mean"]
    output_elements = queue_stats["output_elements_per_rank"]
    route_gather_read = valid_rows * config.hidden_dim * BYTES_PER_BF16
    output_write = output_elements * BYTES_PER_BF16
    return float(route_gather_read + output_write)


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
