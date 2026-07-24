# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Transformer Engine NCCL expert-parallel Grug MoE backend."""

from collections.abc import Callable
from functools import partial
import importlib
import math
from typing import Literal

import jax
import jax.numpy as jnp
from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from jax.sharding import AbstractMesh, Mesh, PartitionSpec as P
from jaxtyping import Array, Float, Int

from levanter.grug._moe.common import (
    _CHECKPOINT_DISPATCH_INPUT,
    _CHECKPOINT_DISPATCH_OUTPUT,
    _CHECKPOINT_EXPERT_HIDDEN,
    _zero_dropped_assignments,
    split_moe_w13_output,
)

_NCCLEP_DISPATCH_ALIGNMENT = 16
_EXPERT_AXIS = "expert"


def ncclep_receive_capacity(
    global_tokens: int,
    top_k: int,
    ep_size: int,
) -> int:
    """Return the fixed receive rows allocated on each NCCL_EP rank.

    ``global_tokens`` is the token count in one expert-parallel group. NCCL_EP's
    exact flat layout requires enough rows for every assignment in the group to
    target one rank.
    """
    if global_tokens <= 0:
        raise ValueError(f"global_tokens must be positive, got {global_tokens}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if global_tokens % ep_size != 0:
        raise ValueError(f"global_tokens={global_tokens} must be divisible by ep_size={ep_size}")
    return global_tokens * top_k


def ncclep_drop_receive_capacity(
    global_tokens: int,
    top_k: int,
    ep_size: int,
    num_experts: int,
    capacity_factor: float,
) -> int:
    """Return the bounded receive rows allocated on each approximate NCCL_EP rank."""
    _ = ncclep_receive_capacity(global_tokens, top_k, ep_size)
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by ep_size={ep_size}")
    if not math.isfinite(capacity_factor) or capacity_factor <= 0:
        raise ValueError(f"capacity_factor must be finite and positive, got {capacity_factor}")

    local_experts = num_experts // ep_size
    alignment = local_experts * _NCCLEP_DISPATCH_ALIGNMENT
    balanced_assignments = global_tokens * top_k // ep_size
    recv_capacity = math.ceil(balanced_assignments * capacity_factor / alignment) * alignment
    max_tokens_per_rank = global_tokens // ep_size
    if recv_capacity < max_tokens_per_rank:
        raise ValueError(
            "bounded NCCL_EP receive capacity must be at least max_tokens_per_rank; "
            f"got capacity={recv_capacity} and max_tokens_per_rank={max_tokens_per_rank}"
        )
    return recv_capacity


def _batch_leading_axes(batch_spec: P) -> tuple[str, ...]:
    if len(batch_spec) == 0 or batch_spec[0] is None:
        raise ValueError(f"NCCL_EP requires a sharded leading token dimension, got batch_spec={batch_spec}")
    if any(axis is not None for axis in batch_spec[1:]):
        raise ValueError(f"NCCL_EP supports sharding only the leading token dimension, got batch_spec={batch_spec}")

    leading = batch_spec[0]
    axes = leading if isinstance(leading, tuple) else (leading,)
    if _EXPERT_AXIS not in axes:
        raise ValueError(f"NCCL_EP batch_spec must include the {_EXPERT_AXIS!r} axis, got {batch_spec}")
    return axes


def _mask_unused_recv_rows(values: jax.Array, token_counts: jax.Array) -> jax.Array:
    valid_rows = jnp.arange(values.shape[-2], dtype=jnp.int32) < jnp.sum(token_counts, dtype=jnp.int32)
    return jnp.where(valid_rows[:, None], values, jnp.zeros((), dtype=values.dtype))


def _local_expert_ffn(
    recv_tokens: jax.Array,
    token_counts: jax.Array,
    w_up_gate: jax.Array,
    w_down: jax.Array,
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    dispatched = recv_tokens.reshape(recv_tokens.shape[-2], recv_tokens.shape[-1])
    group_sizes = token_counts.reshape(-1).astype(jnp.int32)
    dispatched = _mask_unused_recv_rows(dispatched, group_sizes)
    dispatched = tree_checkpoint_name(dispatched, _CHECKPOINT_DISPATCH_INPUT)

    w13_out = tree_checkpoint_name(
        _mask_unused_recv_rows(
            ragged_dot(dispatched, w_up_gate, group_sizes, implementation="triton"),
            group_sizes,
        ),
        _CHECKPOINT_EXPERT_HIDDEN,
    )
    intermediate_dim = w_down.shape[1]
    gate, up = split_moe_w13_output(w13_out, intermediate_dim=intermediate_dim, interleaved=False)
    expert_out = tree_checkpoint_name(
        _mask_unused_recv_rows(
            ragged_dot(activation_fn(gate) * up, w_down, group_sizes, implementation="triton"),
            group_sizes,
        ),
        _CHECKPOINT_DISPATCH_OUTPUT,
    )
    return expert_out.reshape(recv_tokens.shape)


def _apply_recv_weights(
    expert_out: jax.Array,
    recv_weights: jax.Array,
    token_counts: jax.Array,
) -> jax.Array:
    recv_rows = expert_out.shape[-2]
    # HT dispatch leaves the over-allocation tail undefined; mask it in both the primal and transpose.
    valid_rows = jnp.arange(recv_rows, dtype=jnp.int32) < jnp.sum(token_counts, dtype=jnp.int32)
    active = valid_rows & jnp.isfinite(recv_weights) & (recv_weights != 0)
    safe_weights = jnp.where(active, recv_weights, jnp.zeros((), dtype=recv_weights.dtype))
    return jnp.where(
        active[..., None],
        expert_out * safe_weights[..., None].astype(expert_out.dtype),
        jnp.zeros((), dtype=expert_out.dtype),
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 4))
def _ep_dispatch(
    layer_config,
    routes: jax.Array,
    tokens: jax.Array,
    weights: jax.Array,
    recv_capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    return _ep_dispatch_fwd(layer_config, routes, tokens, weights, recv_capacity)[0]


def _ep_dispatch_fwd(layer_config, routes, tokens, weights, recv_capacity):
    te_cpp_ep = importlib.import_module("transformer_engine.jax.cpp_extensions.ep")

    def dispatch():
        # The enclosing MoE shard_map gives these primitives physical rank
        # buffers. TE's outer custom partitioner retains global token extents
        # under JaxPP auto_axes and must not wrap this FFI boundary.
        token_counts, handle_memory = te_cpp_ep.EpPreparePrimitive.inner_primitive.bind(
            routes,
            top_k=int(layer_config.top_k),
            dispatch_output_per_expert_alignment=int(layer_config.dispatch_output_per_expert_alignment),
            is_outer=False,
        )
        recv_tokens, recv_weights = te_cpp_ep.EpDispatchPrimitive.inner_primitive.bind(
            handle_memory,
            routes,
            tokens,
            weights,
            top_k=int(layer_config.top_k),
            dispatch_output_per_expert_alignment=int(layer_config.dispatch_output_per_expert_alignment),
            recv_capacity_per_rank=recv_capacity,
            is_outer=False,
        )
        return recv_tokens, recv_weights, handle_memory, token_counts

    primal = te_cpp_ep._on_collective_stream(dispatch)()
    _, _, handle_memory, _ = primal
    return primal, (handle_memory, tuple(tokens.shape[:-1]))


def _ep_dispatch_bwd(layer_config, recv_capacity, residual, output_cotangents):
    del recv_capacity
    handle_memory, output_leading_shape = residual
    te_cpp_ep = importlib.import_module("transformer_engine.jax.cpp_extensions.ep")

    def dispatch_bwd():
        return te_cpp_ep.EpDispatchBwdPrimitive.inner_primitive.bind(
            handle_memory,
            output_cotangents[0],
            output_cotangents[1],
            top_k=int(layer_config.top_k),
            dispatch_output_per_expert_alignment=int(layer_config.dispatch_output_per_expert_alignment),
            out_leading_shape=output_leading_shape,
            out_partition_spec=None,
        )

    token_cotangent, weight_cotangent = te_cpp_ep._on_collective_stream(dispatch_bwd)()
    return None, token_cotangent, weight_cotangent


_ep_dispatch.defvjp(_ep_dispatch_fwd, _ep_dispatch_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0, 3))
def _ep_combine(
    layer_config,
    handle_memory: jax.Array,
    expert_out: jax.Array,
    output_leading_shape: tuple[int, ...],
) -> jax.Array:
    return _ep_combine_fwd(layer_config, handle_memory, expert_out, output_leading_shape)[0]


def _ep_combine_fwd(layer_config, handle_memory, expert_out, output_leading_shape):
    te_cpp_ep = importlib.import_module("transformer_engine.jax.cpp_extensions.ep")

    def combine():
        return te_cpp_ep.EpCombinePrimitive.inner_primitive.bind(
            handle_memory,
            expert_out,
            top_k=int(layer_config.top_k),
            dispatch_output_per_expert_alignment=int(layer_config.dispatch_output_per_expert_alignment),
            out_leading_shape=output_leading_shape,
            out_partition_spec=None,
        )

    output = te_cpp_ep._on_collective_stream(combine)()
    return output, (handle_memory, expert_out.shape[-2])


def _ep_combine_bwd(layer_config, output_leading_shape, residual, output_cotangent):
    del output_leading_shape
    handle_memory, recv_capacity = residual
    te_cpp_ep = importlib.import_module("transformer_engine.jax.cpp_extensions.ep")

    def combine_bwd():
        return te_cpp_ep.EpCombineBwdPrimitive.inner_primitive.bind(
            handle_memory,
            output_cotangent,
            top_k=int(layer_config.top_k),
            dispatch_output_per_expert_alignment=int(layer_config.dispatch_output_per_expert_alignment),
            recv_capacity_per_rank=recv_capacity,
            is_outer=False,
        )

    expert_out_cotangent = te_cpp_ep._on_collective_stream(combine_bwd)()
    return None, expert_out_cotangent


_ep_combine.defvjp(_ep_combine_fwd, _ep_combine_bwd)


def moe_mlp_ep_ncclep(
    x: Float[Array, "T H"],
    selected_experts: Int[Array, "T K"],
    combine_weights: Float[Array, "T K"],
    w_up_gate: Float[Array, "E H I2"],
    w_down: Float[Array, "E I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    overflow_policy: Literal["trap", "drop"],
    mesh: Mesh | AbstractMesh,
    batch_spec: P,
) -> tuple[Float[Array, "T H"], Int[Array, ""]]:
    """Run a routed MoE MLP through Transformer Engine's NCCL_EP transport.

    Transformer Engine is imported lazily because it is an optional GPU
    dependency. The process must register its FFI handlers and bootstrap the
    NCCL_EP communicator before tracing this function. Trap mode provisions the
    exact worst-case receive extent; drop mode uses a bounded receive extent and
    discards assignments that exceed it.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [T, H], got shape={x.shape}")
    if selected_experts.ndim != 2 or selected_experts.shape != combine_weights.shape:
        raise ValueError(
            "selected_experts and combine_weights must have identical rank-2 [T, K] shapes, "
            f"got {selected_experts.shape} and {combine_weights.shape}"
        )
    if selected_experts.shape[0] != x.shape[0]:
        raise ValueError(
            f"routing token dimension {selected_experts.shape[0]} must match x token dimension {x.shape[0]}"
        )
    if x.dtype != jnp.bfloat16 or w_up_gate.dtype != jnp.bfloat16 or w_down.dtype != jnp.bfloat16:
        raise TypeError("NCCL_EP requires bfloat16 activations and expert weights")
    if w_up_gate.ndim != 3 or w_down.ndim != 3:
        raise ValueError(
            f"expert weights must have shapes [E,H,2I] and [E,I,H], got {w_up_gate.shape} and {w_down.shape}"
        )
    if w_up_gate.shape[0] != num_experts or w_down.shape[0] != num_experts:
        raise ValueError(
            f"num_experts={num_experts} must match weight expert dimensions "
            f"{w_up_gate.shape[0]} and {w_down.shape[0]}"
        )
    if w_up_gate.shape[1] != x.shape[1] or w_down.shape[2] != x.shape[1]:
        raise ValueError("expert weight hidden dimensions must match the activation hidden dimension")
    if w_up_gate.shape[2] != 2 * w_down.shape[1]:
        raise ValueError("w_up_gate output dimension must be twice the w_down intermediate dimension")

    batch_leading_axes = _batch_leading_axes(batch_spec)
    ep_size = int(mesh.shape[_EXPERT_AXIS])
    if ep_size <= 1:
        raise ValueError(f"NCCL_EP requires expert axis size > 1, got {ep_size}")
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} must be divisible by expert axis size={ep_size}")
    outer_batch_size = math.prod(int(mesh.shape[axis]) for axis in batch_leading_axes if axis != _EXPERT_AXIS)
    if outer_batch_size != 1:
        raise ValueError(
            "NCCL_EP currently requires non-expert batch axes to have size one, "
            f"got batch_spec={batch_spec} on mesh={mesh}"
        )

    top_k = int(selected_experts.shape[1])
    if overflow_policy == "trap":
        recv_capacity = ncclep_receive_capacity(
            global_tokens=int(x.shape[0]),
            top_k=top_k,
            ep_size=ep_size,
        )
    elif overflow_policy == "drop":
        recv_capacity = ncclep_drop_receive_capacity(
            global_tokens=int(x.shape[0]),
            top_k=top_k,
            ep_size=ep_size,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
    else:
        raise ValueError(f"unknown NCCL_EP overflow policy: {overflow_policy!r}")

    te_ep = importlib.import_module("transformer_engine.jax.ep")
    layer_config = te_ep.EpLayerConfig(
        top_k=top_k,
        dispatch_output_per_expert_alignment=_NCCLEP_DISPATCH_ALIGNMENT,
    )
    expert_spec = P(_EXPERT_AXIS, None, None)

    def body(
        tokens: jax.Array,
        routes: jax.Array,
        weights: jax.Array,
        expert_w13: jax.Array,
        expert_w2: jax.Array,
    ) -> jax.Array:
        recv_tokens, recv_weights, handle_memory, token_counts = _ep_dispatch(
            layer_config,
            routes.astype(jnp.int32),
            tokens,
            weights.astype(jnp.float32),
            recv_capacity,
        )
        expert_out = _local_expert_ffn(
            recv_tokens,
            token_counts,
            expert_w13,
            expert_w2,
            activation_fn=activation_fn,
        )
        weighted_out = _apply_recv_weights(expert_out, recv_weights, token_counts)
        return _ep_combine(
            layer_config,
            handle_memory,
            weighted_out,
            tuple(tokens.shape[:-1]),
        ).astype(tokens.dtype)

    local_body = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, expert_spec, expert_spec),
        out_specs=batch_spec,
        check_vma=False,
    )
    output = local_body(x, selected_experts, combine_weights, w_up_gate, w_down)
    return output, _zero_dropped_assignments()
