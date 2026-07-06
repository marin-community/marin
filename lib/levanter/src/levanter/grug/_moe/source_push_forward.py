# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# pyrefly: ignore-errors

"""Package-private full forward harness for the source-push MGPU MoE prototype."""

from __future__ import annotations

import json
import math
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_combine import (
    _combine_bytes_per_rank,
    _combine_queue_stats,
    _make_combine_weights,
    _make_route_inverse,
    _sharded_source_combine_kernel,
    validate_source_combine_config,
)
from levanter.grug._moe.source_push_inbox import (
    AXIS,
    BYTES_PER_BF16,
    HostInputs,
    PushInboxConfig,
    SLOW_USEFUL_W13_TFLOPS_PER_RANK,
    SourcePushInboxRunSettings,
    TimingResult,
    ROW_LAYOUT_EXACT_EXPERT_MAJOR,
    ROW_LAYOUT_SOURCE_PADDED_EXPERT_MAJOR,
    ROW_START_MODE_EXACT_EXPERT_MAJOR,
    ROW_START_MODE_SOURCE_PADDED,
    _add_source_push_plan_queue_stats,
    _block_until_ready,
    _device_inputs_from_host,
    _make_mesh,
    _make_source_tokens,
    _make_weights,
    _queue_stats,
    _recv_meta_from_send_meta,
    _reference_hidden,
    _reference_h_flat,
    _routing_counts,
    _selected_experts_from_counts,
    _sharded_kernel,
    _sharded_w13_h_kernel,
    _source_padded_plan_send_meta,
    _time_jitted,
)
from levanter.grug._moe.source_push_inbox_blackwell import (
    sharded_flat_y_return_transport,
    sharded_blackwell_local_w13_h,
    sharded_blackwell_local_w2_y,
    sharded_destination_local_x_transport,
    sharded_raw_destination_local_x_transport,
)
from levanter.grug._moe.source_push_plan import (
    SourcePushPlan,
    build_source_push_plan,
    pack_source_push_tokens,
    pack_source_push_tokens_jax,
    source_push_combine,
    source_push_combine_preweighted,
    source_push_h_row_route_weights_jax,
    source_push_w13_h,
    source_push_w2_from_h_return,
    source_push_source_padded_row_bases,
    source_push_w2_return,
)
from levanter.grug._moe.source_push_w2_return import (
    _sharded_w2_from_h_return_direct_to_source_kernel,
    _sharded_w2_return_direct_to_source_kernel,
    make_w_down,
    validate_w2_return_config,
)


KERNEL_NAME = "source_push_forward"
FORWARD_EXECUTION_SINGLE_JIT = "single_jit"
FORWARD_EXECUTION_STAGED_HOST_SYNC = "staged_host_sync"
FORWARD_EXECUTION_STAGED_DEVICE_SYNC = "staged_device_sync"
FORWARD_EXECUTION_MODES = (
    FORWARD_EXECUTION_SINGLE_JIT,
    FORWARD_EXECUTION_STAGED_HOST_SYNC,
    FORWARD_EXECUTION_STAGED_DEVICE_SYNC,
)
FORWARD_STAGE_TOTAL = "total"
FORWARD_STAGE_W13 = "w13"
FORWARD_STAGE_W2_RETURN = "w2_return"
FORWARD_STAGE_COMBINE = "combine"
FORWARD_STAGES = (FORWARD_STAGE_W13, FORWARD_STAGE_W2_RETURN, FORWARD_STAGE_COMBINE)
SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED = "blackwell_staged"
_SOURCE_PUSH_FORWARD_IMPLEMENTATIONS = (
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
)
SourcePushForwardImplementation: TypeAlias = Literal["reference", "pallas_mgpu", "blackwell_staged"]


def _set_mesh_if_needed(mesh: Mesh):
    if jax.sharding.get_abstract_mesh().empty:
        return jax.set_mesh(mesh)
    return nullcontext()


FORWARD_SUMMARY_METRICS = (
    "steady_state_time",
    "bytes_per_rank",
    "forward_gbps_per_rank",
    "rounded_forward_tflops_per_rank",
    "useful_forward_tflops_per_rank",
    "w13_tflops_per_rank",
    "rounded_w13_tflops_per_rank",
    "useful_w13_tflops_per_rank",
    "w2_tflops_per_rank",
    "combine_gbps_per_rank",
    "compile_time",
    "lower_compile_time",
    "first_run_time",
    "max_abs_diff",
    "mean_abs_diff",
)


@dataclass(frozen=True)
class SourcePushForwardTiming:
    """Timing result for the chained forward harness, optionally with per-stage rows."""

    compile_time: float
    steady_state_times: list[float]
    output: Any
    lower_compile_time: float | None
    first_run_time: float | None
    stage_steady_state_times: dict[str, list[float]] | None = None
    stage_compile_times: dict[str, float] | None = None


@dataclass(frozen=True)
class _StagedSourcePushForwardCallables:
    """Jitted stage callables for a fixed mesh/config/layout."""

    destination_local_x_fn: Callable[..., Any] | None
    raw_destination_local_x_fn: Callable[..., Any] | None
    destination_x_barrier_fn: Callable[..., Any] | None
    w13_h_fn: Callable[..., Any]
    w2_y_fn: Callable[..., Any] | None
    return_fn: Callable[..., Any] | None
    w2_from_h_return_fn: Callable[..., Any] | None
    source_return_barrier_fn: Callable[..., Any] | None
    combine_fn: Callable[..., Any]


_STAGED_FORWARD_CALLABLE_CACHE: dict[
    tuple[int, PushInboxConfig, bool, SourcePushForwardImplementation, str],
    _StagedSourcePushForwardCallables,
] = {}


@dataclass(frozen=True)
class SourcePushForwardHostInputs:
    """Host-side inputs shared by source-push W13, W2 return, and source combine."""

    x: np.ndarray
    send_meta: np.ndarray
    recv_meta: np.ndarray
    expert_base: np.ndarray
    src_base_by_expert: np.ndarray
    h_group_sizes: np.ndarray
    w_gate_up: np.ndarray
    w_down: np.ndarray
    queue_dst_ord: np.ndarray
    queue_entry: np.ndarray
    queue_row: np.ndarray
    h_route_weights: np.ndarray
    route_combine_weights: np.ndarray
    route_valid_mask: np.ndarray
    plan: SourcePushPlan
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool = False


@dataclass(frozen=True)
class SourcePushForwardRawInputs:
    """Real source-major arrays used to build a source-push full-forward plan."""

    x: np.ndarray
    selected_experts: np.ndarray
    combine_weights: np.ndarray
    w_gate_up: np.ndarray
    w_down: np.ndarray


@dataclass(frozen=True)
class SourcePushForwardDeviceInputs:
    """Device inputs for the chained source-push forward harness."""

    x: jax.Array | None
    source_x: jax.Array | None
    token_ids: jax.Array | None
    send_meta: jax.Array
    recv_meta: jax.Array
    expert_base: jax.Array
    src_base_by_expert: jax.Array
    h_group_sizes: jax.Array
    w_gate_up: jax.Array
    w_down: jax.Array
    queue_dst_ord: jax.Array
    queue_entry: jax.Array
    queue_row: jax.Array
    h_route_weights: jax.Array
    route_combine_weights: jax.Array
    route_valid_mask: jax.Array
    queue_stats: dict[str, Any]
    use_exact_expert_major: bool


@dataclass(frozen=True)
class PreparedSourcePushForwardDeviceInputs:
    """Sharded device inputs prepared for repeated source-push forward calls."""

    mesh: Mesh
    inputs: SourcePushForwardDeviceInputs


@dataclass(frozen=True)
class SourcePushForwardValidationMetrics:
    """Validation summary for the full source-push forward output."""

    max_abs_diff: float
    mean_abs_diff: float


def validate_source_push_forward_config(config: PushInboxConfig) -> None:
    """Validate all tile constraints used by the chained forward harness."""

    config.validate()
    validate_w2_return_config(config)
    validate_source_combine_config(config)


def make_source_push_forward_inputs(
    config: PushInboxConfig,
    x: Float[Array, "S T D"],
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    input_mode: str = "real_arrays",
    use_exact_expert_major: bool = False,
) -> SourcePushForwardHostInputs:
    """Build full-forward source-push inputs from real source-local MoE arrays."""

    validate_source_push_forward_config(config)
    x_host = np.asarray(jax.device_get(x), dtype=np.float32)
    selected_host = np.asarray(jax.device_get(selected_experts), dtype=np.int32)
    combine_host = np.asarray(jax.device_get(combine_weights), dtype=np.float32)
    w_gate_up_host = np.asarray(jax.device_get(w_gate_up), dtype=np.float32)
    w_down_host = np.asarray(jax.device_get(w_down), dtype=np.float32)
    _validate_source_push_forward_array_shapes(
        config,
        x_host,
        selected_host,
        combine_host,
        w_gate_up_host,
        w_down_host,
    )

    plan = build_source_push_plan(
        jnp.asarray(selected_host, dtype=jnp.int32),
        jnp.asarray(combine_host, dtype=jnp.float32),
        ep_size=config.ep_size,
        experts_per_rank=config.experts_per_rank,
        block_m=config.block_m,
        capacity_factor=config.capacity_factor,
        entries_per_dst=config.entries_per_rank,
    )
    packed_x = np.asarray(pack_source_push_tokens(jnp.asarray(x_host, dtype=jnp.float32), plan), dtype=np.float32)
    return _make_source_push_forward_inputs_from_plan(
        config,
        plan,
        packed_x,
        combine_host,
        w_gate_up_host,
        w_down_host,
        input_mode=input_mode,
        use_exact_expert_major=use_exact_expert_major,
    )


def make_source_push_forward_plan_inputs(
    config: PushInboxConfig,
    selected_experts: Int[Array, "S T K"],
    *,
    input_mode: str = "source_push_plan",
    use_exact_expert_major: bool = False,
) -> SourcePushForwardHostInputs:
    """Build source-push host metadata from nondifferentiable routing only."""

    validate_source_push_forward_config(config)
    selected_host = np.asarray(jax.device_get(selected_experts), dtype=np.int32)
    expected_routes = (config.ep_size, config.tokens_per_rank, config.topk)
    if selected_host.shape != expected_routes:
        raise ValueError(f"selected_experts shape {selected_host.shape} must match {expected_routes}")

    placeholder_route_weights = np.zeros(expected_routes, dtype=np.float32)
    plan = build_source_push_plan(
        jnp.asarray(selected_host, dtype=jnp.int32),
        jnp.asarray(placeholder_route_weights, dtype=jnp.float32),
        ep_size=config.ep_size,
        experts_per_rank=config.experts_per_rank,
        block_m=config.block_m,
        capacity_factor=config.capacity_factor,
        entries_per_dst=config.entries_per_rank,
    )
    packed_x = np.zeros(
        (
            config.ep_size,
            config.ep_size,
            config.entries_per_rank,
            config.block_m,
            config.hidden_dim,
        ),
        dtype=np.float32,
    )
    w_gate_up = np.zeros(
        (
            config.ep_size,
            config.experts_per_rank,
            config.hidden_dim,
            2 * config.intermediate_dim,
        ),
        dtype=np.float32,
    )
    w_down = np.zeros(
        (
            config.ep_size,
            config.experts_per_rank,
            config.intermediate_dim,
            config.hidden_dim,
        ),
        dtype=np.float32,
    )
    return _make_source_push_forward_inputs_from_plan(
        config,
        plan,
        packed_x,
        placeholder_route_weights,
        w_gate_up,
        w_down,
        input_mode=input_mode,
        use_exact_expert_major=use_exact_expert_major,
    )


def make_source_push_forward_source_plan_inputs(config: PushInboxConfig) -> SourcePushForwardHostInputs:
    """Build shared source-push plan inputs for full forward timing."""

    raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
    return make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="source_push_plan",
    )


def make_source_push_forward_exact_source_plan_inputs(config: PushInboxConfig) -> SourcePushForwardHostInputs:
    """Build exact expert-major full-forward inputs for block-aligned plans."""

    raw_inputs = make_source_push_forward_source_plan_raw_inputs(config)
    return make_source_push_forward_inputs(
        config,
        raw_inputs.x,
        raw_inputs.selected_experts,
        raw_inputs.combine_weights,
        raw_inputs.w_gate_up,
        raw_inputs.w_down,
        input_mode="exact_source_push_plan",
        use_exact_expert_major=True,
    )


def make_source_push_forward_source_plan_raw_inputs(config: PushInboxConfig) -> SourcePushForwardRawInputs:
    """Build synthetic-but-real source-major arrays for source-push integration smokes."""

    validate_source_push_forward_config(config)
    counts = _routing_counts(config)
    selected_experts = np.stack(
        [_selected_experts_from_counts(config, counts[src], src) for src in range(config.ep_size)],
        axis=0,
    )
    combine_weights = _make_combine_weights(config)
    source_tokens = np.stack([_make_source_tokens(config, src) for src in range(config.ep_size)], axis=0)
    return SourcePushForwardRawInputs(
        x=source_tokens,
        selected_experts=selected_experts,
        combine_weights=combine_weights,
        w_gate_up=np.asarray(jax.device_get(_make_weights(config)), dtype=np.float32),
        w_down=make_w_down(config),
    )


def _make_source_push_forward_inputs_from_plan(
    config: PushInboxConfig,
    plan: SourcePushPlan,
    packed_x: np.ndarray,
    route_weights: np.ndarray,
    w_gate_up: np.ndarray,
    w_down: np.ndarray,
    *,
    input_mode: str,
    use_exact_expert_major: bool = False,
) -> SourcePushForwardHostInputs:
    if use_exact_expert_major:
        send_meta = np.asarray(jax.device_get(plan.send_meta), dtype=np.int32)
        valid_rows = send_meta[..., 3]
        tail_blocks = (valid_rows > 0) & (valid_rows < config.block_m)
        if np.any(tail_blocks):
            raise ValueError(
                "exact source-push full forward layout requires block_m-aligned live blocks; "
                "use the source-padded layout for tail blocks"
            )
        expert_base = np.asarray(jax.device_get(plan.expert_base), dtype=np.int32)
        src_base_by_expert = np.asarray(jax.device_get(plan.src_base_by_expert), dtype=np.int32)
        recv_meta = np.asarray(jax.device_get(plan.recv_meta), dtype=np.int32)
        row_start_mode = ROW_START_MODE_EXACT_EXPERT_MAJOR
        row_layout = ROW_LAYOUT_EXACT_EXPERT_MAJOR
        h_group_sizes = np.asarray(jax.device_get(plan.rows_per_local_expert), dtype=np.int32)
        layout_rows_total = int(np.sum(h_group_sizes))
    else:
        rounded_counts, expert_base, src_base_by_expert = source_push_source_padded_row_bases(plan, config.block_m)
        h_group_sizes = np.sum(rounded_counts, axis=0).astype(np.int32)
        send_meta = _source_padded_plan_send_meta(plan, expert_base, src_base_by_expert)
        recv_meta = _recv_meta_from_send_meta(config, send_meta)
        row_start_mode = ROW_START_MODE_SOURCE_PADDED
        row_layout = ROW_LAYOUT_SOURCE_PADDED_EXPERT_MAJOR
        layout_rows_total = int(np.sum(rounded_counts))
    route_inverse = _make_route_inverse(config, plan)
    h_route_weights = np.asarray(
        source_push_h_row_route_weights_jax(
            jnp.asarray(route_weights, dtype=jnp.float32),
            plan,
            send_meta,
            expert_base,
            src_base_by_expert,
            hidden_rows_per_rank=config.hidden_rows_per_rank,
            use_exact_expert_major=use_exact_expert_major,
        ),
        dtype=np.float32,
    )
    valid_mask = np.asarray(jax.device_get(plan.valid_mask), dtype=np.bool_)
    queue_stats = _queue_stats(config, send_meta)
    combine_stats = _combine_queue_stats(config, valid_mask, int(jax.device_get(plan.dropped_routes)))
    _add_source_push_plan_queue_stats(
        config,
        queue_stats,
        plan,
        row_start_mode=row_start_mode,
        row_layout=row_layout,
        layout_rows_total=layout_rows_total,
    )
    queue_stats.update(
        {
            "input_mode": input_mode,
            "forward_mode": "w13_w2_direct_return_combine",
            "combine_mode": combine_stats["combine_mode"],
            "plan_padded_rows_total": layout_rows_total,
            "plan_padded_rows_per_rank_mean": float(layout_rows_total / config.ep_size),
            "route_buffer_elements_per_rank": combine_stats["route_buffer_elements_per_rank"],
            "output_elements_per_rank": combine_stats["output_elements_per_rank"],
        }
    )
    return SourcePushForwardHostInputs(
        x=packed_x,
        send_meta=send_meta,
        recv_meta=recv_meta,
        expert_base=expert_base,
        src_base_by_expert=src_base_by_expert,
        h_group_sizes=h_group_sizes,
        w_gate_up=w_gate_up,
        w_down=w_down,
        queue_dst_ord=route_inverse["queue_dst_ord"],
        queue_entry=route_inverse["queue_entry"],
        queue_row=route_inverse["queue_row"],
        h_route_weights=h_route_weights,
        route_combine_weights=route_inverse["route_combine_weights"],
        route_valid_mask=route_inverse["route_valid_mask"],
        plan=plan,
        queue_stats=queue_stats,
        use_exact_expert_major=use_exact_expert_major,
    )


def _validate_source_push_forward_array_shapes(
    config: PushInboxConfig,
    x: np.ndarray,
    selected_experts: np.ndarray,
    combine_weights: np.ndarray,
    w_gate_up: np.ndarray,
    w_down: np.ndarray,
) -> None:
    expected_x = (config.ep_size, config.tokens_per_rank, config.hidden_dim)
    expected_routes = (config.ep_size, config.tokens_per_rank, config.topk)
    expected_w_gate_up = (
        config.ep_size,
        config.experts_per_rank,
        config.hidden_dim,
        2 * config.intermediate_dim,
    )
    expected_w_down = (
        config.ep_size,
        config.experts_per_rank,
        config.intermediate_dim,
        config.hidden_dim,
    )
    if x.shape != expected_x:
        raise ValueError(f"x shape {x.shape} must match {expected_x}")
    if selected_experts.shape != expected_routes:
        raise ValueError(f"selected_experts shape {selected_experts.shape} must match {expected_routes}")
    if combine_weights.shape != expected_routes:
        raise ValueError(f"combine_weights shape {combine_weights.shape} must match {expected_routes}")
    if w_gate_up.shape != expected_w_gate_up:
        raise ValueError(f"w_gate_up shape {w_gate_up.shape} must match {expected_w_gate_up}")
    if w_down.shape != expected_w_down:
        raise ValueError(f"w_down shape {w_down.shape} must match {expected_w_down}")


def device_source_push_forward_inputs_from_host(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> SourcePushForwardDeviceInputs:
    """Move full forward inputs into benchmark dtypes."""

    device_w13_inputs = _device_inputs_from_host(
        config,
        HostInputs(
            x=host_inputs.x,
            send_meta=host_inputs.send_meta,
            recv_meta=host_inputs.recv_meta,
            expert_base=host_inputs.expert_base,
            src_base_by_expert=host_inputs.src_base_by_expert,
            queue_stats=host_inputs.queue_stats,
            use_exact_expert_major=host_inputs.use_exact_expert_major,
        ),
    )
    return SourcePushForwardDeviceInputs(
        x=device_w13_inputs.x,
        source_x=None,
        token_ids=None,
        send_meta=device_w13_inputs.send_meta,
        recv_meta=device_w13_inputs.recv_meta,
        expert_base=device_w13_inputs.expert_base,
        src_base_by_expert=device_w13_inputs.src_base_by_expert,
        h_group_sizes=jnp.asarray(host_inputs.h_group_sizes, dtype=jnp.int32),
        w_gate_up=jnp.asarray(host_inputs.w_gate_up, dtype=jnp.bfloat16),
        w_down=jnp.asarray(host_inputs.w_down, dtype=jnp.bfloat16),
        queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
        queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
        queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
        h_route_weights=jnp.asarray(host_inputs.h_route_weights, dtype=jnp.bfloat16),
        route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
        route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
        queue_stats=host_inputs.queue_stats,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def device_source_push_forward_inputs_from_plan(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    pack_source_tokens: bool = True,
) -> SourcePushForwardDeviceInputs:
    """Build device inputs from static plan metadata and dynamic differentiable arrays."""

    h_route_weights = source_push_h_row_route_weights_jax(
        route_weights,
        host_inputs.plan,
        host_inputs.send_meta,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        hidden_rows_per_rank=config.hidden_rows_per_rank,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    packed_x = pack_source_push_tokens_jax(x, host_inputs.plan).astype(jnp.bfloat16) if pack_source_tokens else None
    return SourcePushForwardDeviceInputs(
        x=packed_x,
        source_x=jnp.asarray(x, dtype=jnp.bfloat16) if not pack_source_tokens else None,
        token_ids=jnp.asarray(host_inputs.plan.token_ids, dtype=jnp.int32) if not pack_source_tokens else None,
        send_meta=jnp.asarray(host_inputs.send_meta, dtype=jnp.int32),
        recv_meta=jnp.asarray(host_inputs.recv_meta, dtype=jnp.int32),
        expert_base=jnp.asarray(host_inputs.expert_base, dtype=jnp.int32),
        src_base_by_expert=jnp.asarray(host_inputs.src_base_by_expert, dtype=jnp.int32),
        h_group_sizes=jnp.asarray(host_inputs.h_group_sizes, dtype=jnp.int32),
        w_gate_up=jnp.asarray(w_gate_up, dtype=jnp.bfloat16),
        w_down=jnp.asarray(w_down, dtype=jnp.bfloat16),
        queue_dst_ord=jnp.asarray(host_inputs.queue_dst_ord, dtype=jnp.int32),
        queue_entry=jnp.asarray(host_inputs.queue_entry, dtype=jnp.int32),
        queue_row=jnp.asarray(host_inputs.queue_row, dtype=jnp.int32),
        h_route_weights=h_route_weights.astype(jnp.bfloat16),
        route_combine_weights=jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.bfloat16),
        route_valid_mask=jnp.asarray(host_inputs.route_valid_mask, dtype=jnp.bool_),
        queue_stats=host_inputs.queue_stats,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )


def reference_source_push_forward(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> Float[Array, "S T D"]:
    """Reference W13 -> W2 return -> source combine using the shared plan."""

    hidden = _reference_hidden(
        config,
        host_inputs.x,
        host_inputs.send_meta,
        host_inputs.w_gate_up,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    source_return = source_push_w2_return(
        jnp.asarray(hidden, dtype=jnp.bfloat16),
        jnp.asarray(host_inputs.w_down, dtype=jnp.bfloat16),
        host_inputs.plan,
        expert_base=host_inputs.expert_base,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    return source_push_combine(jnp.asarray(source_return, dtype=jnp.bfloat16), host_inputs.plan)


def reference_source_push_forward_h(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> Float[Array, "S T D"]:
    """Reference forward using H as the W13/W2 checkpoint boundary."""

    h = source_push_w13_h(
        jnp.asarray(host_inputs.x, dtype=jnp.bfloat16),
        jnp.asarray(host_inputs.w_gate_up, dtype=jnp.bfloat16),
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    source_return = source_push_w2_from_h_return(
        jnp.asarray(h, dtype=jnp.bfloat16),
        jnp.asarray(host_inputs.route_combine_weights, dtype=jnp.float32),
        jnp.asarray(host_inputs.w_down, dtype=jnp.bfloat16),
        host_inputs.plan,
        src_base_by_expert=host_inputs.src_base_by_expert,
    )
    return source_push_combine_preweighted(jnp.asarray(source_return, dtype=jnp.bfloat16), host_inputs.plan)


def reference_source_push_forward_with_h(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    """Reference forward plus the flat source-padded W13 preactivation checkpoint."""

    h = _reference_h_flat(
        config,
        host_inputs.x,
        host_inputs.send_meta,
        host_inputs.w_gate_up,
        host_inputs.expert_base,
        host_inputs.src_base_by_expert,
        use_exact_expert_major=host_inputs.use_exact_expert_major,
    )
    out = reference_source_push_forward_h(config, host_inputs)
    return out, jnp.asarray(h, dtype=jnp.bfloat16)


def source_push_forward(
    config: PushInboxConfig,
    x: Float[Array, "S T D"],
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Int[Array, ""]]:
    """Run source-push full forward from explicit source-major EP inputs.

    This is package-private while the planner remains host-side. It is the
    integration-facing boundary for the staged source-push prototype: callers
    provide real source-local tokens, routing, combine weights, and local expert
    weights, and receive source-local token outputs plus the clipped-route count.
    """

    if implementation not in (
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    ):
        raise ValueError(
            "source-push forward implementation must be one of "
            f"{_SOURCE_PUSH_FORWARD_IMPLEMENTATIONS}, "
            f"got {implementation!r}"
        )
    host_inputs = make_source_push_forward_inputs(
        config,
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
    )
    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE:
        return reference_source_push_forward(config, host_inputs), host_inputs.plan.dropped_routes

    out = _execute_source_push_forward_host_inputs(
        config,
        host_inputs,
        mesh=mesh,
        execution_mode=execution_mode,
        implementation=implementation,
    )
    return out, host_inputs.plan.dropped_routes


def source_push_forward_with_h(
    config: PushInboxConfig,
    x: Float[Array, "S T D"],
    selected_experts: Int[Array, "S T K"],
    combine_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"], Int[Array, ""]]:
    """Run source-push forward and return the W13 preactivation checkpoint.

    The returned H uses the production W13 output layout:
    ``[destination_rank, source_padded_expert_major_row, 2 * intermediate_dim]``.
    It is before SwiGLU and before route-weight scaling, so it is the intended
    MLP-level custom-VJP residual rather than post-activation ``A``.
    """

    if implementation not in (
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    ):
        raise ValueError(
            "source-push forward implementation must be one of "
            f"{_SOURCE_PUSH_FORWARD_IMPLEMENTATIONS}, "
            f"got {implementation!r}"
        )
    host_inputs = make_source_push_forward_inputs(
        config,
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
    )
    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_REFERENCE:
        out, h = reference_source_push_forward_with_h(config, host_inputs)
        return out, h, host_inputs.plan.dropped_routes

    out, h = _execute_source_push_forward_host_inputs_with_h(
        config,
        host_inputs,
        mesh=mesh,
        execution_mode=execution_mode,
        implementation=implementation,
    )
    return out, h, host_inputs.plan.dropped_routes


def source_push_forward_with_h_from_plan(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"], Int[Array, ""]]:
    """Run staged source-push forward from fixed plan metadata and dynamic arrays.

    ``host_inputs`` supplies the nondifferentiable routing metadata. The
    differentiable arrays are packed/gathered with JAX before entering the
    Pallas stages, so callers do not have to rebuild ``packed_x`` or receive
    route weights on the host.
    """

    if implementation not in (
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    ):
        raise ValueError(
            "source_push_forward_with_h_from_plan supports staged Pallas implementations "
            f"{(SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU, SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED)}, "
            f"got {implementation!r}"
        )
    if execution_mode not in FORWARD_EXECUTION_MODES:
        raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")
    if mesh is None:
        mesh = _make_mesh(config.ep_size)
    with _set_mesh_if_needed(mesh):
        inputs = _shard_source_push_forward_inputs(
            mesh,
            device_source_push_forward_inputs_from_plan(
                config,
                host_inputs,
                x,
                route_weights,
                w_gate_up,
                w_down,
                pack_source_tokens=implementation != SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
            ),
        )
        out, h = _call_source_push_forward_device_inputs_with_h(
            mesh,
            config,
            inputs,
            implementation=implementation,
            execution_mode=execution_mode,
        )
    _block_until_ready((out, h))
    return out, h, host_inputs.plan.dropped_routes


def prepare_source_push_forward_device_inputs_from_plan(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    x: Float[Array, "S T D"],
    route_weights: Float[Array, "S T K"],
    w_gate_up: Float[Array, "S E D twoI"],
    w_down: Float[Array, "S E I D"],
    *,
    mesh: Mesh | None = None,
) -> PreparedSourcePushForwardDeviceInputs:
    """Prepare and shard dynamic arrays for repeated source-push forward calls."""

    if mesh is None:
        mesh = _make_mesh(config.ep_size)
    with _set_mesh_if_needed(mesh):
        return PreparedSourcePushForwardDeviceInputs(
            mesh=mesh,
            inputs=_shard_source_push_forward_inputs(
                mesh,
                device_source_push_forward_inputs_from_plan(
                    config,
                    host_inputs,
                    x,
                    route_weights,
                    w_gate_up,
                    w_down,
                ),
            ),
        )


def source_push_forward_with_h_from_prepared_device_inputs(
    config: PushInboxConfig,
    prepared: PreparedSourcePushForwardDeviceInputs,
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    """Run source-push forward from already prepared and sharded device inputs."""

    if implementation not in (
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
        SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED,
    ):
        raise ValueError(
            "source_push_forward_with_h_from_prepared_device_inputs supports staged Pallas implementations "
            f"{(SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU, SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED)}, "
            f"got {implementation!r}"
        )
    if execution_mode not in FORWARD_EXECUTION_MODES:
        raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")
    out, h = _call_source_push_forward_device_inputs_with_h(
        prepared.mesh,
        config,
        prepared.inputs,
        implementation=implementation,
        execution_mode=execution_mode,
    )
    _block_until_ready((out, h))
    return out, h


def _sharded_source_push_forward_kernel(mesh: Mesh, config: PushInboxConfig, *, use_exact_expert_major: bool = False):
    w13_kernel = _sharded_kernel(mesh, config, use_exact_expert_major=use_exact_expert_major)
    w2_return_kernel = _sharded_w2_return_direct_to_source_kernel(
        mesh,
        config,
        use_exact_expert_major=use_exact_expert_major,
    )
    remote_write_barrier = _sharded_remote_write_completion_barrier(mesh)
    combine_kernel = _sharded_source_combine_kernel(mesh, config)

    def fn(
        x: Float[Array, "S DST Q M D"],
        send_meta: Int[Array, "S DST Q F"],
        recv_meta: Int[Array, "Dst SRC Q F"],
        expert_base: Int[Array, "Dst E"],
        src_base_by_expert: Int[Array, "Dst S E"],
        w_gate_up: Float[Array, "Dst E D twoI"],
        w_down: Float[Array, "Dst E I D"],
        queue_dst_ord: Int[Array, "S T K"],
        queue_entry: Int[Array, "S T K"],
        queue_row: Int[Array, "S T K"],
        route_combine_weights: Float[Array, "S T K"],
        route_valid_mask: Bool[Array, "S T K"],
    ):
        _, hidden = w13_kernel(
            x,
            send_meta,
            recv_meta,
            expert_base,
            src_base_by_expert,
            w_gate_up,
        )
        source_return = w2_return_kernel(hidden, recv_meta, expert_base, src_base_by_expert, w_down)
        source_return = remote_write_barrier(source_return)
        return combine_kernel(
            source_return,
            queue_dst_ord,
            queue_entry,
            queue_row,
            route_combine_weights,
            route_valid_mask,
        )

    return fn


def _sharded_source_push_forward_h_kernel(
    mesh: Mesh, config: PushInboxConfig, *, use_exact_expert_major: bool = False
):
    w13_h_kernel = _sharded_w13_h_kernel(mesh, config, use_exact_expert_major=use_exact_expert_major)
    w2_from_h_kernel = _sharded_w2_from_h_return_direct_to_source_kernel(
        mesh,
        config,
        use_exact_expert_major=use_exact_expert_major,
    )
    remote_write_barrier = _sharded_remote_write_completion_barrier(mesh)
    combine_kernel = _sharded_source_combine_kernel(mesh, config)

    def fn(
        x: Float[Array, "S DST Q M D"],
        send_meta: Int[Array, "S DST Q F"],
        recv_meta: Int[Array, "Dst SRC Q F"],
        expert_base: Int[Array, "Dst E"],
        src_base_by_expert: Int[Array, "Dst S E"],
        w_gate_up: Float[Array, "Dst E D twoI"],
        h_route_weights: Float[Array, "Dst rows"],
        w_down: Float[Array, "Dst E I D"],
        queue_dst_ord: Int[Array, "S T K"],
        queue_entry: Int[Array, "S T K"],
        queue_row: Int[Array, "S T K"],
        route_combine_weights: Float[Array, "S T K"],
        route_valid_mask: Bool[Array, "S T K"],
    ):
        _, h = w13_h_kernel(
            x,
            send_meta,
            recv_meta,
            expert_base,
            src_base_by_expert,
            w_gate_up,
        )
        source_return = w2_from_h_kernel(
            h,
            h_route_weights,
            recv_meta,
            expert_base,
            src_base_by_expert,
            w_down,
        )
        source_return = remote_write_barrier(source_return)
        return combine_kernel(
            source_return,
            queue_dst_ord,
            queue_entry,
            queue_row,
            jnp.ones_like(route_combine_weights),
            route_valid_mask,
        )

    return fn


def _sharded_remote_write_completion_barrier(mesh: Mesh):
    """Synchronize ranks after return-copy remote writes before source-local reads."""

    def local_fn(source_return_local: Float[Array, "1 DST Q M D"]):
        source_return_local = source_return_local[0]
        marker = source_return_local[0, 0, 0, 0].astype(jnp.float32)
        barrier = lax.psum(marker, AXIS)
        zero = (barrier - lax.optimization_barrier(barrier)).astype(source_return_local.dtype)
        source_return_local = source_return_local.at[0, 0, 0, 0].add(zero)
        return source_return_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(AXIS, None, None, None, None),
        out_specs=P(AXIS, None, None, None, None),
        check_vma=False,
    )


def _sharded_destination_x_completion_barrier(mesh: Mesh):
    """Synchronize ranks after destination-local peer writes before local W13 reads."""

    def local_fn(destination_x_local: Float[Array, "1 rows D"]):
        destination_x_local = destination_x_local[0]
        marker = destination_x_local.reshape((-1,))[0].astype(jnp.float32)
        barrier = lax.psum(marker, AXIS)
        zero = (barrier - lax.optimization_barrier(barrier)).astype(destination_x_local.dtype)
        destination_x_local = destination_x_local.at[0, 0].add(zero)
        return destination_x_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=P(AXIS, None, None),
        out_specs=P(AXIS, None, None),
        check_vma=False,
    )


def _staged_source_push_forward_callables(
    mesh: Mesh,
    config: PushInboxConfig,
    *,
    use_exact_expert_major: bool,
    implementation: SourcePushForwardImplementation,
    execution_mode: str,
) -> _StagedSourcePushForwardCallables:
    key = (id(mesh), config, use_exact_expert_major, implementation, execution_mode)
    callables = _STAGED_FORWARD_CALLABLE_CACHE.get(key)
    if callables is not None:
        return callables

    combine_fn = jax.jit(_sharded_source_combine_kernel(mesh, config))
    source_return_barrier_fn = None
    if execution_mode == FORWARD_EXECUTION_STAGED_DEVICE_SYNC:
        source_return_barrier_fn = jax.jit(_sharded_remote_write_completion_barrier(mesh))

    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
        destination_x_barrier_fn = None
        if execution_mode == FORWARD_EXECUTION_STAGED_DEVICE_SYNC:
            destination_x_barrier_fn = jax.jit(_sharded_destination_x_completion_barrier(mesh))
        callables = _StagedSourcePushForwardCallables(
            destination_local_x_fn=jax.jit(
                sharded_destination_local_x_transport(
                    mesh,
                    config,
                    use_exact_expert_major=use_exact_expert_major,
                )
            ),
            raw_destination_local_x_fn=jax.jit(
                sharded_raw_destination_local_x_transport(
                    mesh,
                    config,
                    use_exact_expert_major=use_exact_expert_major,
                )
            ),
            destination_x_barrier_fn=destination_x_barrier_fn,
            w13_h_fn=jax.jit(sharded_blackwell_local_w13_h(mesh)),
            w2_y_fn=jax.jit(sharded_blackwell_local_w2_y(mesh)),
            return_fn=jax.jit(
                sharded_flat_y_return_transport(
                    mesh,
                    config,
                    use_exact_expert_major=use_exact_expert_major,
                )
            ),
            w2_from_h_return_fn=None,
            source_return_barrier_fn=source_return_barrier_fn,
            combine_fn=combine_fn,
        )
    else:
        callables = _StagedSourcePushForwardCallables(
            destination_local_x_fn=None,
            raw_destination_local_x_fn=None,
            destination_x_barrier_fn=None,
            w13_h_fn=jax.jit(_sharded_w13_h_kernel(mesh, config, use_exact_expert_major=use_exact_expert_major)),
            w2_y_fn=None,
            return_fn=None,
            w2_from_h_return_fn=jax.jit(
                _sharded_w2_from_h_return_direct_to_source_kernel(
                    mesh,
                    config,
                    use_exact_expert_major=use_exact_expert_major,
                )
            ),
            source_return_barrier_fn=source_return_barrier_fn,
            combine_fn=combine_fn,
        )

    _STAGED_FORWARD_CALLABLE_CACHE[key] = callables
    return callables


def _execute_source_push_forward_host_inputs(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    *,
    mesh: Mesh | None,
    execution_mode: str,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
) -> Float[Array, "S T D"]:
    if execution_mode not in FORWARD_EXECUTION_MODES:
        raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")
    if mesh is None:
        mesh = _make_mesh(config.ep_size)
    inputs = _shard_source_push_forward_inputs(
        mesh,
        device_source_push_forward_inputs_from_host(config, host_inputs),
    )
    out = _call_source_push_forward_device_inputs(
        mesh,
        config,
        inputs,
        implementation=implementation,
        execution_mode=execution_mode,
    )
    _block_until_ready(out)
    return out


def _execute_source_push_forward_host_inputs_with_h(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    *,
    mesh: Mesh | None,
    execution_mode: str,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    if execution_mode not in FORWARD_EXECUTION_MODES:
        raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")
    if mesh is None:
        mesh = _make_mesh(config.ep_size)
    inputs = _shard_source_push_forward_inputs(
        mesh,
        device_source_push_forward_inputs_from_host(config, host_inputs),
    )
    out, h = _call_source_push_forward_device_inputs_with_h(
        mesh,
        config,
        inputs,
        implementation=implementation,
        execution_mode=execution_mode,
    )
    _block_until_ready((out, h))
    return out, h


def _shard_source_push_forward_inputs(
    mesh: Mesh,
    inputs: SourcePushForwardDeviceInputs,
) -> SourcePushForwardDeviceInputs:
    return SourcePushForwardDeviceInputs(
        x=(
            None
            if inputs.x is None
            else jax.device_put(inputs.x, NamedSharding(mesh, P(AXIS, None, None, None, None)))
        ),
        source_x=(
            None
            if inputs.source_x is None
            else jax.device_put(inputs.source_x, NamedSharding(mesh, P(AXIS, None, None)))
        ),
        token_ids=(
            None
            if inputs.token_ids is None
            else jax.device_put(inputs.token_ids, NamedSharding(mesh, P(AXIS, None, None, None)))
        ),
        send_meta=jax.device_put(inputs.send_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        recv_meta=jax.device_put(inputs.recv_meta, NamedSharding(mesh, P(AXIS, None, None, None))),
        expert_base=jax.device_put(inputs.expert_base, NamedSharding(mesh, P(AXIS, None))),
        src_base_by_expert=jax.device_put(inputs.src_base_by_expert, NamedSharding(mesh, P(AXIS, None, None))),
        h_group_sizes=jax.device_put(inputs.h_group_sizes, NamedSharding(mesh, P(AXIS, None))),
        w_gate_up=jax.device_put(inputs.w_gate_up, NamedSharding(mesh, P(AXIS, None, None, None))),
        w_down=jax.device_put(inputs.w_down, NamedSharding(mesh, P(AXIS, None, None, None))),
        queue_dst_ord=jax.device_put(inputs.queue_dst_ord, NamedSharding(mesh, P(AXIS, None, None))),
        queue_entry=jax.device_put(inputs.queue_entry, NamedSharding(mesh, P(AXIS, None, None))),
        queue_row=jax.device_put(inputs.queue_row, NamedSharding(mesh, P(AXIS, None, None))),
        h_route_weights=jax.device_put(inputs.h_route_weights, NamedSharding(mesh, P(AXIS, None))),
        route_combine_weights=jax.device_put(
            inputs.route_combine_weights,
            NamedSharding(mesh, P(AXIS, None, None)),
        ),
        route_valid_mask=jax.device_put(inputs.route_valid_mask, NamedSharding(mesh, P(AXIS, None, None))),
        queue_stats=inputs.queue_stats,
        use_exact_expert_major=inputs.use_exact_expert_major,
    )


def _call_source_push_forward_device_inputs(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: SourcePushForwardDeviceInputs,
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str,
) -> Float[Array, "S T D"]:
    if execution_mode in (FORWARD_EXECUTION_STAGED_HOST_SYNC, FORWARD_EXECUTION_STAGED_DEVICE_SYNC):
        out, _ = _call_staged_source_push_forward_device_inputs(
            mesh,
            config,
            inputs,
            implementation=implementation,
            execution_mode=execution_mode,
        )
        return out
    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
        raise ValueError(
            f"{SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED!r} requires "
            f"execution_mode={FORWARD_EXECUTION_STAGED_HOST_SYNC!r}"
        )

    fn = jax.jit(
        _sharded_source_push_forward_h_kernel(mesh, config, use_exact_expert_major=inputs.use_exact_expert_major)
    )
    if inputs.x is None:
        raise ValueError(f"{implementation!r} requires packed source-push x inputs")
    return fn(
        inputs.x,
        inputs.send_meta,
        inputs.recv_meta,
        inputs.expert_base,
        inputs.src_base_by_expert,
        inputs.w_gate_up,
        inputs.h_route_weights,
        inputs.w_down,
        inputs.queue_dst_ord,
        inputs.queue_entry,
        inputs.queue_row,
        inputs.route_combine_weights,
        inputs.route_valid_mask,
    )


def _call_source_push_forward_device_inputs_with_h(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: SourcePushForwardDeviceInputs,
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    if execution_mode not in (FORWARD_EXECUTION_STAGED_HOST_SYNC, FORWARD_EXECUTION_STAGED_DEVICE_SYNC):
        raise ValueError(
            f"source_push_forward_with_h currently requires a staged execution mode; got {execution_mode!r}"
        )
    return _call_staged_source_push_forward_device_inputs(
        mesh,
        config,
        inputs,
        implementation=implementation,
        execution_mode=execution_mode,
    )


def _call_staged_source_push_forward_device_inputs(
    mesh: Mesh,
    config: PushInboxConfig,
    inputs: SourcePushForwardDeviceInputs,
    *,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
) -> tuple[Float[Array, "S T D"], Float[Array, "Dst rows twoI"]]:
    if execution_mode not in (FORWARD_EXECUTION_STAGED_HOST_SYNC, FORWARD_EXECUTION_STAGED_DEVICE_SYNC):
        raise ValueError(
            "staged source-push forward requires "
            f"{(FORWARD_EXECUTION_STAGED_HOST_SYNC, FORWARD_EXECUTION_STAGED_DEVICE_SYNC)}, got {execution_mode!r}"
        )
    stage_fns = _staged_source_push_forward_callables(
        mesh,
        config,
        use_exact_expert_major=inputs.use_exact_expert_major,
        implementation=implementation,
        execution_mode=execution_mode,
    )
    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
        if stage_fns.destination_local_x_fn is None or stage_fns.w2_y_fn is None or stage_fns.return_fn is None:
            raise AssertionError("Blackwell staged callables were not initialized")
        destination_transport_expert_base = jax.device_put(inputs.expert_base, NamedSharding(mesh, P(None, None)))
        destination_transport_src_base_by_expert = jax.device_put(
            inputs.src_base_by_expert,
            NamedSharding(mesh, P(None, None, None)),
        )
        if inputs.source_x is not None and inputs.token_ids is not None:
            if stage_fns.raw_destination_local_x_fn is None:
                raise AssertionError("Blackwell raw-token destination callable was not initialized")
            destination_x = stage_fns.raw_destination_local_x_fn(
                inputs.source_x,
                inputs.token_ids,
                inputs.send_meta,
                destination_transport_expert_base,
                destination_transport_src_base_by_expert,
            )
        else:
            if inputs.x is None:
                raise ValueError("Blackwell staged transport requires either raw source tokens or packed x inputs")
            destination_x = stage_fns.destination_local_x_fn(
                inputs.x,
                inputs.send_meta,
                destination_transport_expert_base,
                destination_transport_src_base_by_expert,
            )
        if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
            _block_until_ready(destination_x)
        elif stage_fns.destination_x_barrier_fn is not None:
            destination_x = stage_fns.destination_x_barrier_fn(destination_x)
        h = stage_fns.w13_h_fn(destination_x, inputs.w_gate_up, inputs.h_group_sizes)
    else:
        if inputs.x is None:
            raise ValueError(f"{implementation!r} requires packed source-push x inputs")
        _, h = stage_fns.w13_h_fn(
            inputs.x,
            inputs.send_meta,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_gate_up,
        )
    if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
        _block_until_ready(h)
    if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
        y = stage_fns.w2_y_fn(
            h,
            inputs.h_route_weights,
            inputs.w_down,
            inputs.h_group_sizes,
        )
        if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
            _block_until_ready(y)
        source_return = stage_fns.return_fn(
            y,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
        )
    else:
        if stage_fns.w2_from_h_return_fn is None:
            raise AssertionError("source-push W2-return callable was not initialized")
        source_return = stage_fns.w2_from_h_return_fn(
            h,
            inputs.h_route_weights,
            inputs.recv_meta,
            inputs.expert_base,
            inputs.src_base_by_expert,
            inputs.w_down,
        )
    if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
        _block_until_ready(source_return)
    elif stage_fns.source_return_barrier_fn is not None:
        source_return = stage_fns.source_return_barrier_fn(source_return)
    out = stage_fns.combine_fn(
        source_return,
        inputs.queue_dst_ord,
        inputs.queue_entry,
        inputs.queue_row,
        jnp.ones_like(inputs.route_combine_weights),
        inputs.route_valid_mask,
    )
    return out, h


def _source_push_w2_from_flat_h_return_jax(
    config: PushInboxConfig,
    h: Float[Array, "Dst rows twoI"],
    h_route_weights: Float[Array, "Dst rows"],
    recv_meta: Int[Array, "Dst Src Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    w_down: Float[Array, "Dst E I D"],
    *,
    use_exact_expert_major: bool,
) -> Float[Array, "S Dst Q M D"]:
    source_return = jnp.zeros(
        (
            config.ep_size,
            config.ep_size,
            config.entries_per_rank,
            config.block_m,
            config.hidden_dim,
        ),
        dtype=w_down.dtype,
    )
    rows = jnp.arange(config.block_m, dtype=jnp.int32)
    for dst in range(config.ep_size):
        for src_ordinal in range(config.ep_size):
            src = (dst + src_ordinal) % config.ep_size
            dst_ordinal = (-src_ordinal) % config.ep_size
            for entry in range(config.entries_per_rank):
                meta = recv_meta[dst, src_ordinal, entry]
                valid_rows = meta[3]
                expert = jnp.maximum(meta[1], 0)
                local_row_start = meta[2]
                if use_exact_expert_major:
                    src_rank = meta[0]
                    row_start = expert_base[dst, expert] + src_base_by_expert[dst, src_rank, expert] + local_row_start
                else:
                    row_start = local_row_start
                row_indices = jnp.maximum(row_start, 0) + rows
                h_rows = h[dst, row_indices].astype(jnp.float32)
                gate = h_rows[:, : config.intermediate_dim]
                up = h_rows[:, config.intermediate_dim :]
                route_weights = h_route_weights[dst, row_indices].astype(jnp.float32)
                activation = jax.nn.silu(gate) * up * route_weights[:, None]
                output = activation @ w_down[dst, expert].astype(jnp.float32)
                output = jnp.where(rows[:, None] < valid_rows, output, jnp.zeros((), dtype=output.dtype))
                source_return = source_return.at[src, dst_ordinal, entry].set(output.astype(w_down.dtype))
    return source_return


def _source_push_flat_y_to_source_return_jax(
    config: PushInboxConfig,
    y: Float[Array, "Dst rows D"],
    recv_meta: Int[Array, "Dst Src Q F"],
    expert_base: Int[Array, "Dst E"],
    src_base_by_expert: Int[Array, "Dst S E"],
    *,
    use_exact_expert_major: bool,
) -> Float[Array, "S Dst Q M D"]:
    source_return = jnp.zeros(
        (
            config.ep_size,
            config.ep_size,
            config.entries_per_rank,
            config.block_m,
            config.hidden_dim,
        ),
        dtype=y.dtype,
    )
    rows = jnp.arange(config.block_m, dtype=jnp.int32)
    for dst in range(config.ep_size):
        for src_ordinal in range(config.ep_size):
            src = (dst + src_ordinal) % config.ep_size
            dst_ordinal = (-src_ordinal) % config.ep_size
            for entry in range(config.entries_per_rank):
                meta = recv_meta[dst, src_ordinal, entry]
                valid_rows = meta[3]
                expert = jnp.maximum(meta[1], 0)
                local_row_start = meta[2]
                if use_exact_expert_major:
                    src_rank = meta[0]
                    row_start = expert_base[dst, expert] + src_base_by_expert[dst, src_rank, expert] + local_row_start
                else:
                    row_start = local_row_start
                row_indices = jnp.maximum(row_start, 0) + rows
                output = y[dst, row_indices]
                output = jnp.where(rows[:, None] < valid_rows, output, jnp.zeros((), dtype=output.dtype))
                source_return = source_return.at[src, dst_ordinal, entry].set(output)
    return source_return


def _time_staged_source_push_forward(
    mesh: Mesh,
    config: PushInboxConfig,
    x: jax.Array,
    send_meta: jax.Array,
    recv_meta: jax.Array,
    expert_base: jax.Array,
    src_base_by_expert: jax.Array,
    h_group_sizes: jax.Array,
    w_gate_up: jax.Array,
    h_route_weights: jax.Array,
    w_down: jax.Array,
    queue_dst_ord: jax.Array,
    queue_entry: jax.Array,
    queue_row: jax.Array,
    route_combine_weights: jax.Array,
    route_valid_mask: jax.Array,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    progress: Callable[[str], None] | None = None,
    use_exact_expert_major: bool = False,
    implementation: SourcePushForwardImplementation = SOURCE_PUSH_FORWARD_IMPLEMENTATION_PALLAS_MGPU,
    execution_mode: str = FORWARD_EXECUTION_STAGED_HOST_SYNC,
) -> SourcePushForwardTiming:
    """Run full forward as staged JIT callables for ordering diagnostics."""

    stage_fns = _staged_source_push_forward_callables(
        mesh,
        config,
        use_exact_expert_major=use_exact_expert_major,
        implementation=implementation,
        execution_mode=execution_mode,
    )

    def call_stages(*, record_stage_times: bool = False):
        stage_times: dict[str, float] = {}

        stage_start = time.perf_counter()
        if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
            if stage_fns.destination_local_x_fn is None:
                raise AssertionError("Blackwell destination-local X callable was not initialized")
            destination_transport_expert_base = jax.device_put(expert_base, NamedSharding(mesh, P(None, None)))
            destination_transport_src_base_by_expert = jax.device_put(
                src_base_by_expert,
                NamedSharding(mesh, P(None, None, None)),
            )
            destination_x = stage_fns.destination_local_x_fn(
                x,
                send_meta,
                destination_transport_expert_base,
                destination_transport_src_base_by_expert,
            )
            if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
                _block_until_ready(destination_x)
            elif stage_fns.destination_x_barrier_fn is not None:
                destination_x = stage_fns.destination_x_barrier_fn(destination_x)
            h = stage_fns.w13_h_fn(destination_x, w_gate_up, h_group_sizes)
        else:
            _, h = stage_fns.w13_h_fn(
                x,
                send_meta,
                recv_meta,
                expert_base,
                src_base_by_expert,
                w_gate_up,
            )
        _block_until_ready(h)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W13] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        if implementation == SOURCE_PUSH_FORWARD_IMPLEMENTATION_BLACKWELL_STAGED:
            if stage_fns.w2_y_fn is None or stage_fns.return_fn is None:
                raise AssertionError("Blackwell W2/return callables were not initialized")
            y = stage_fns.w2_y_fn(h, h_route_weights, w_down, h_group_sizes)
            if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
                _block_until_ready(y)
            source_return = stage_fns.return_fn(
                y,
                recv_meta,
                expert_base,
                src_base_by_expert,
            )
        else:
            if stage_fns.w2_from_h_return_fn is None:
                raise AssertionError("Pallas W2 return callable was not initialized")
            source_return = stage_fns.w2_from_h_return_fn(
                h,
                h_route_weights,
                recv_meta,
                expert_base,
                src_base_by_expert,
                w_down,
            )
        if execution_mode == FORWARD_EXECUTION_STAGED_DEVICE_SYNC and stage_fns.source_return_barrier_fn is not None:
            source_return = stage_fns.source_return_barrier_fn(source_return)
        _block_until_ready(source_return)
        if record_stage_times:
            stage_times[FORWARD_STAGE_W2_RETURN] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        out = stage_fns.combine_fn(
            source_return,
            queue_dst_ord,
            queue_entry,
            queue_row,
            jnp.ones_like(route_combine_weights),
            route_valid_mask,
        )
        _block_until_ready(out)
        if record_stage_times:
            stage_times[FORWARD_STAGE_COMBINE] = time.perf_counter() - stage_start
        return out, stage_times

    if progress is not None:
        progress("first_call_start")
    start = time.perf_counter()
    out, stage_compile_times = call_stages(record_stage_times=True)
    compile_time = time.perf_counter() - start
    if progress is not None:
        progress("first_call_done")

    if progress is not None:
        progress("warmup_start")
    for _ in range(warmup):
        out, _ = call_stages()

    steady_state_times = []
    stage_steady_state_times: dict[str, list[float]] = {stage: [] for stage in FORWARD_STAGES}
    for _ in range(repeat_runs):
        if progress is not None:
            progress("steady_state_start")
        start = time.perf_counter()
        stage_elapsed = {stage: 0.0 for stage in FORWARD_STAGES}
        for _ in range(steps):
            out, step_stage_times = call_stages(record_stage_times=True)
            for stage in FORWARD_STAGES:
                stage_elapsed[stage] += step_stage_times[stage]
        steady_state_times.append((time.perf_counter() - start) / steps)
        for stage in FORWARD_STAGES:
            stage_steady_state_times[stage].append(stage_elapsed[stage] / steps)
        if progress is not None:
            progress("steady_state_done")

    return SourcePushForwardTiming(
        compile_time=compile_time,
        steady_state_times=steady_state_times,
        output=out,
        lower_compile_time=None,
        first_run_time=None,
        stage_steady_state_times=stage_steady_state_times,
        stage_compile_times=stage_compile_times,
    )


def _source_push_forward_timing_from_base(timing: TimingResult) -> SourcePushForwardTiming:
    return SourcePushForwardTiming(
        compile_time=timing.compile_time,
        steady_state_times=timing.steady_state_times,
        output=timing.output,
        lower_compile_time=timing.lower_compile_time,
        first_run_time=timing.first_run_time,
    )


def _validate_source_push_forward(
    config: PushInboxConfig,
    host_inputs: SourcePushForwardHostInputs,
    observed: jax.Array,
) -> SourcePushForwardValidationMetrics:
    expected = reference_source_push_forward_h(config, host_inputs)
    expected_host = np.asarray(jax.device_get(expected), dtype=np.float32)
    observed_host = np.asarray(jax.device_get(observed), dtype=np.float32)
    diff = np.abs(observed_host - expected_host)
    return SourcePushForwardValidationMetrics(
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
    )


def _run_forward_one(
    config: PushInboxConfig,
    settings: SourcePushInboxRunSettings,
    input_builder: Callable[
        [PushInboxConfig], SourcePushForwardHostInputs
    ] = make_source_push_forward_source_plan_inputs,
    *,
    execution_mode: str = FORWARD_EXECUTION_SINGLE_JIT,
) -> list[dict[str, Any]]:
    try:
        if settings.repeat_runs <= 0:
            raise ValueError(f"repeat_runs must be positive, got {settings.repeat_runs}")
        if execution_mode not in FORWARD_EXECUTION_MODES:
            raise ValueError(f"unknown execution_mode={execution_mode!r}; expected one of {FORWARD_EXECUTION_MODES}")
        validate_source_push_forward_config(config)
        _emit_progress(config, settings.progress_events, "mesh_start")
        mesh = _make_mesh(config.ep_size)
        _emit_progress(config, settings.progress_events, "make_inputs_start")
        host_inputs = input_builder(config)
        inputs = device_source_push_forward_inputs_from_host(config, host_inputs)
        _emit_progress(config, settings.progress_events, "device_put_start")
        inputs = _shard_source_push_forward_inputs(mesh, inputs)
        x = inputs.x
        send_meta = inputs.send_meta
        recv_meta = inputs.recv_meta
        expert_base = inputs.expert_base
        src_base_by_expert = inputs.src_base_by_expert
        h_group_sizes = inputs.h_group_sizes
        w_gate_up = inputs.w_gate_up
        h_route_weights = inputs.h_route_weights
        w_down = inputs.w_down
        queue_dst_ord = inputs.queue_dst_ord
        queue_entry = inputs.queue_entry
        queue_row = inputs.queue_row
        route_combine_weights = inputs.route_combine_weights
        route_valid_mask = inputs.route_valid_mask
        _emit_progress(config, settings.progress_events, "jit_start")
        if execution_mode == FORWARD_EXECUTION_STAGED_HOST_SYNC:
            timing = _time_staged_source_push_forward(
                mesh,
                config,
                x,
                send_meta,
                recv_meta,
                expert_base,
                src_base_by_expert,
                h_group_sizes,
                w_gate_up,
                h_route_weights,
                w_down,
                queue_dst_ord,
                queue_entry,
                queue_row,
                route_combine_weights,
                route_valid_mask,
                warmup=settings.warmup,
                steps=settings.steps,
                repeat_runs=settings.repeat_runs,
                progress=lambda event: _emit_progress(config, settings.progress_events, event),
                use_exact_expert_major=inputs.use_exact_expert_major,
            )
        else:
            fn = jax.jit(
                _sharded_source_push_forward_h_kernel(
                    mesh,
                    config,
                    use_exact_expert_major=inputs.use_exact_expert_major,
                )
            )
            timing = _source_push_forward_timing_from_base(
                _time_jitted(
                    fn,
                    x,
                    send_meta,
                    recv_meta,
                    expert_base,
                    src_base_by_expert,
                    w_gate_up,
                    h_route_weights,
                    w_down,
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
            )
        validation = _validate_source_push_forward(config, host_inputs, timing.output) if settings.check else None
        queue_stats = inputs.queue_stats
        rounded_rows = queue_stats["rounded_rows_per_rank_mean"]
        useful_rows = queue_stats["valid_rows_per_rank_mean"]
        w13_flops_per_rank = rounded_rows * config.hidden_dim * config.intermediate_dim * 4
        useful_w13_flops_per_rank = useful_rows * config.hidden_dim * config.intermediate_dim * 4
        w2_flops_per_rank = rounded_rows * config.intermediate_dim * config.hidden_dim * 2
        useful_forward_flops_per_rank = useful_rows * config.hidden_dim * config.intermediate_dim * 6
        rounded_forward_flops_per_rank = w13_flops_per_rank + w2_flops_per_rank
        send_bytes_per_rank = queue_stats["send_rounded_rows_per_rank_mean"] * config.hidden_dim * BYTES_PER_BF16
        w2_bytes_per_rank = rounded_rows * (config.intermediate_dim + config.hidden_dim) * BYTES_PER_BF16
        combine_bytes_per_rank = _combine_bytes_per_rank(config, queue_stats)
        bytes_per_rank = send_bytes_per_rank + w2_bytes_per_rank + combine_bytes_per_rank
        rows = []
        for repeat_run, steady_state_time in enumerate(timing.steady_state_times):
            row = {
                "kernel": KERNEL_NAME,
                "implementation": KERNEL_NAME,
                "row_type": "repeat",
                "stage": FORWARD_STAGE_TOTAL,
                "execution_mode": execution_mode,
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
                "forward_gbps_per_rank": bytes_per_rank / steady_state_time / 1e9,
                "rounded_forward_tflops_per_rank": rounded_forward_flops_per_rank / steady_state_time / 1e12,
                "useful_forward_tflops_per_rank": useful_forward_flops_per_rank / steady_state_time / 1e12,
                "w13_tflops_per_rank": w13_flops_per_rank / steady_state_time / 1e12,
                "rounded_w13_tflops_per_rank": w13_flops_per_rank / steady_state_time / 1e12,
                "useful_w13_tflops_per_rank": useful_w13_flops_per_rank / steady_state_time / 1e12,
                "w2_tflops_per_rank": w2_flops_per_rank / steady_state_time / 1e12,
                "max_abs_diff": None if validation is None else validation.max_abs_diff,
                "mean_abs_diff": None if validation is None else validation.mean_abs_diff,
                "error": None,
                "error_type": None,
                "error_message": None,
            }
            rows.append(row)
            if timing.stage_steady_state_times is None:
                continue
            for stage in FORWARD_STAGES:
                stage_steady_state_time = timing.stage_steady_state_times[stage][repeat_run]
                stage_row = {
                    "kernel": KERNEL_NAME,
                    "implementation": f"{KERNEL_NAME}_{stage}",
                    "row_type": "stage_repeat",
                    "stage": stage,
                    "execution_mode": execution_mode,
                    "config": asdict(config),
                    "queue_stats": queue_stats,
                    **queue_stats,
                    "compile_time": None if timing.stage_compile_times is None else timing.stage_compile_times[stage],
                    "lower_compile_time": None,
                    "first_run_time": None,
                    "repeat_run": repeat_run,
                    "repeat_runs": settings.repeat_runs,
                    "steady_state_time": stage_steady_state_time,
                    "bytes_per_rank": None,
                    "forward_gbps_per_rank": None,
                    "rounded_forward_tflops_per_rank": None,
                    "useful_forward_tflops_per_rank": None,
                    "w13_tflops_per_rank": None,
                    "rounded_w13_tflops_per_rank": None,
                    "useful_w13_tflops_per_rank": None,
                    "w2_tflops_per_rank": None,
                    "combine_gbps_per_rank": None,
                    "max_abs_diff": None,
                    "mean_abs_diff": None,
                    "error": None,
                    "error_type": None,
                    "error_message": None,
                }
                if stage == FORWARD_STAGE_W13:
                    stage_row["bytes_per_rank"] = send_bytes_per_rank
                    stage_row["forward_gbps_per_rank"] = send_bytes_per_rank / stage_steady_state_time / 1e9
                    stage_row["w13_tflops_per_rank"] = w13_flops_per_rank / stage_steady_state_time / 1e12
                    stage_row["rounded_w13_tflops_per_rank"] = w13_flops_per_rank / stage_steady_state_time / 1e12
                    stage_row["useful_w13_tflops_per_rank"] = (
                        useful_w13_flops_per_rank / stage_steady_state_time / 1e12
                    )
                elif stage == FORWARD_STAGE_W2_RETURN:
                    stage_row["bytes_per_rank"] = w2_bytes_per_rank
                    stage_row["forward_gbps_per_rank"] = w2_bytes_per_rank / stage_steady_state_time / 1e9
                    stage_row["w2_tflops_per_rank"] = w2_flops_per_rank / stage_steady_state_time / 1e12
                elif stage == FORWARD_STAGE_COMBINE:
                    stage_row["bytes_per_rank"] = combine_bytes_per_rank
                    stage_row["combine_gbps_per_rank"] = combine_bytes_per_rank / stage_steady_state_time / 1e9
                rows.append(stage_row)
        return _add_forward_summary_rows(rows)
    except Exception as exc:  # noqa: BLE001 - benchmark rows should capture unsupported candidates.
        if settings.debug_exceptions:
            raise
        return [
            {
                "kernel": KERNEL_NAME,
                "implementation": KERNEL_NAME,
                "row_type": "error",
                "stage": FORWARD_STAGE_TOTAL,
                "execution_mode": execution_mode,
                "config": asdict(config),
                "compile_time": None,
                "lower_compile_time": None,
                "first_run_time": None,
                "repeat_run": None,
                "repeat_runs": settings.repeat_runs,
                "steady_state_time": None,
                "bytes_per_rank": None,
                "forward_gbps_per_rank": None,
                "rounded_forward_tflops_per_rank": None,
                "useful_forward_tflops_per_rank": None,
                "w13_tflops_per_rank": None,
                "rounded_w13_tflops_per_rank": None,
                "useful_w13_tflops_per_rank": None,
                "w2_tflops_per_rank": None,
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


def _add_forward_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Append aggregate rows for full-forward target-gate reporting."""

    summaries = []
    for stage, repeat_row_type in (
        (FORWARD_STAGE_TOTAL, "repeat"),
        (FORWARD_STAGE_W13, "stage_repeat"),
        (FORWARD_STAGE_W2_RETURN, "stage_repeat"),
        (FORWARD_STAGE_COMBINE, "stage_repeat"),
    ):
        stage_rows = [
            row
            for row in rows
            if row.get("stage") == stage
            and row.get("row_type") == repeat_row_type
            and row.get("error_type") is None
            and row.get("steady_state_time") is not None
        ]
        if stage_rows:
            summaries.append(_forward_summary_row(stage_rows, stage=stage))
    return [*rows, *summaries]


def _forward_summary_row(rows: list[dict[str, Any]], *, stage: str) -> dict[str, Any]:
    first = rows[0]
    queue_stats = first.get("queue_stats")
    summary: dict[str, Any] = {
        "kernel": KERNEL_NAME,
        "implementation": first.get("implementation"),
        "row_type": "summary",
        "stage": stage,
        "execution_mode": first.get("execution_mode"),
        "config": first.get("config"),
        "queue_stats": queue_stats,
        "repeat_runs": first.get("repeat_runs"),
        "repeat_rows": len(rows),
        "error": None,
        "error_type": None,
        "error_message": None,
        "min_steady_state_time": _min_optional_field(rows, "steady_state_time"),
        "max_steady_state_time": _max_optional_field(rows, "steady_state_time"),
        "p90_steady_state_time": _percentile_field(rows, "steady_state_time", 0.90),
        "p95_steady_state_time": _percentile_field(rows, "steady_state_time", 0.95),
    }
    if isinstance(queue_stats, dict):
        summary.update(queue_stats)
    for metric in FORWARD_SUMMARY_METRICS:
        summary[f"median_{metric}"] = _median_optional_field(rows, metric)
    if stage == FORWARD_STAGE_W13:
        useful_w13_values = [
            row["useful_w13_tflops_per_rank"] for row in rows if row.get("useful_w13_tflops_per_rank") is not None
        ]
        slow_repeats = sum(value < SLOW_USEFUL_W13_TFLOPS_PER_RANK for value in useful_w13_values)
        summary.update(
            {
                "slow_useful_w13_threshold": SLOW_USEFUL_W13_TFLOPS_PER_RANK,
                "slow_useful_w13_repeats": slow_repeats,
                "slow_useful_w13_fraction": slow_repeats / len(useful_w13_values) if useful_w13_values else None,
                "min_useful_w13_tflops_per_rank": _min_optional_field(rows, "useful_w13_tflops_per_rank"),
            }
        )
    return summary


def _median_optional_field(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return median(values)


def _min_optional_field(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return min(values)


def _max_optional_field(rows: list[dict[str, Any]], field: str) -> float | int | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    if not values:
        return None
    return max(values)


def _percentile_field(rows: list[dict[str, Any]], field: str, percentile: float) -> float | int | None:
    values = sorted(row[field] for row in rows if row.get(field) is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def run_source_push_forward_source_plan(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
    execution_mode: str = FORWARD_EXECUTION_SINGLE_JIT,
) -> list[dict[str, Any]]:
    """Run W13, W2 return-copy, and deterministic source combine from one source-push plan."""

    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_forward_one(config, settings, execution_mode=execution_mode)


def run_source_push_forward_exact_source_plan(
    config: PushInboxConfig,
    *,
    warmup: int,
    steps: int,
    repeat_runs: int,
    check: bool,
    debug_exceptions: bool = False,
    separate_compile: bool = False,
    progress_events: bool = False,
    execution_mode: str = FORWARD_EXECUTION_SINGLE_JIT,
) -> list[dict[str, Any]]:
    """Run W13, W2 return, and deterministic combine using exact row starts."""

    settings = SourcePushInboxRunSettings(
        warmup=warmup,
        steps=steps,
        repeat_runs=repeat_runs,
        check=check,
        debug_exceptions=debug_exceptions,
        separate_compile=separate_compile,
        progress_events=progress_events,
    )
    return _run_forward_one(
        config,
        settings,
        input_builder=make_source_push_forward_exact_source_plan_inputs,
        execution_mode=execution_mode,
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
