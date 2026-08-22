# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend."""

import functools
import math
from collections.abc import Callable
from typing import TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import CapacityOverflow
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    _gather_dispatch_rows,
    _unpermute_from_global_expert,
)
from levanter.grug.sharding import _batch_axes

# QuACK's grouped GEMMs are written for SM100 and ship only with the CUDA 13 GPU extra.
_SM100_COMPUTE_CAPABILITY = 10.0

# An expert MLP takes both views of the receiver buffer's group sizes: the physical sizes,
# which charge trailing padding to the last expert, and the active sizes, which count only
# received rows. Which one a kernel needs depends on whether it covers the buffer or reads
# segment boundaries.
_ExpertMlp: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, Callable[[jax.Array], jax.Array]], jax.Array
]


def _ragged_dot_expert_mlp(
    x_dispatch: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    physical_group_sizes: jax.Array,
    active_group_sizes: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Portable expert MLP over XLA's `ragged_dot`, which covers the whole receiver buffer."""
    del active_group_sizes
    w13_out = ragged_dot(x_dispatch, moe_w13_local, physical_group_sizes)
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    return ragged_dot(activation_fn(gate) * up, moe_w2_local, physical_group_sizes)


def _cute_expert_mlp(
    x_dispatch: jax.Array,
    moe_w13_local: jax.Array,
    moe_w2_local: jax.Array,
    physical_group_sizes: jax.Array,
    active_group_sizes: jax.Array,
    activation_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Expert MLP on QuACK's SM100 grouped GEMMs plus cuDNN grouped weight gradients.

    The grouped kernels are driven by segment boundaries, so they take the active sizes and
    mask the receiver buffer's trailing padding rather than charging it to the last expert.
    """
    del activation_fn, physical_group_sizes

    # QuACK, cuDNN Frontend, and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
    from levanter.grug._moe.sonic_cute import _expert_mlp_cudnn, _interleave_gate_up  # noqa: PLC0415

    moe_dim = moe_w2_local.shape[1]
    w13_interleaved = _interleave_gate_up(moe_w13_local, moe_dim)
    cumulative_group_sizes = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)]
    )
    return _expert_mlp_cudnn(x_dispatch, w13_interleaved, moe_w2_local, active_group_sizes, cumulative_group_sizes)


@functools.cache
def _quack_grouped_gemm_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    if float(jax.devices("gpu")[0].compute_capability) < _SM100_COMPUTE_CAPABILITY:
        return False
    try:
        import levanter.grug._moe.sonic_cute  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def _select_expert_mlp(activation_fn: Callable[[jax.Array], jax.Array]) -> _ExpertMlp:
    """Pick the fastest expert-MLP kernel this process can actually run.

    QuACK's kernel fuses SwiGLU, so it only applies to SiLU. Everything else -- another
    activation, a non-SM100 GPU, a TPU or CPU, or a build without the GPU extra -- runs the
    portable `ragged_dot` path, which computes the same function.
    """
    if activation_fn is jax.nn.silu and _quack_grouped_gemm_available():
        return _cute_expert_mlp
    return _ragged_dot_expert_mlp


def _moe_mlp_ep_ragged_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    splits_per_peer: int = 1,
) -> tuple[Float[Array, "Tlocal H"], CapacityOverflow]:
    local_experts = moe_w13_local.shape[0]
    if num_experts % local_experts != 0:
        raise ValueError(
            f"num_experts={num_experts} must be divisible by local expert count={local_experts} in EP mode"
        )

    shard_id = jax.lax.axis_index("expert")
    ep_size = num_experts // local_experts
    tokens_per_shard = x_local.shape[0]
    topk = selected_experts_local.shape[1]
    assignments_per_shard = tokens_per_shard * topk
    local_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    local_capacity = max(local_experts, local_capacity)
    recv_capacity = local_capacity

    # One a2a update per (peer, local expert, split): reuse the peer-granular splits knob as
    # a per-expert-group split count with a comparable total update budget.
    splits_per_group = max(1, splits_per_peer // local_experts)

    with jax.named_scope("dispatch"):
        flat_selected = selected_experts_local.reshape(-1)
        sorted_indices = jnp.argsort(flat_selected)
        group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)
        sorted_x = _gather_dispatch_rows(x_local, sorted_indices, topk=topk)
        all_group_sizes = jax.lax.all_gather(group_sizes, "expert")
        clipped_group_sizes = _clip_receiver_group_sizes(
            all_group_sizes,
            local_expert_size=local_experts,
            receiver_capacity=local_capacity,
        )
        sender_group_sizes = clipped_group_sizes[shard_id]
        dispatch_params, return_params = _expert_granular_a2a_params(
            all_group_sizes,
            clipped_group_sizes,
            shard_id,
            local_expert_size=local_experts,
            splits_per_group=splits_per_group,
        )
        dispatch_out_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        # Accepted rows are the prefix of each unclipped expert group and receiver offsets
        # pack arrivals expert-major, so the received buffer feeds the grouped MLP directly:
        # no sender compaction and no receiver-side permute.
        x_dispatch = jax.lax.ragged_all_to_all(
            sorted_x,
            dispatch_out_shape,
            *dispatch_params,
            axis_name="expert",
        )
        active_group_sizes = jnp.sum(
            clipped_group_sizes.reshape(ep_size, ep_size, local_experts)[:, shard_id, :], axis=0
        )
        total_valid = jnp.sum(active_group_sizes, dtype=jnp.int32)
        physical_group_sizes = active_group_sizes.at[-1].add(recv_capacity - total_valid)

    with jax.named_scope("moe_up_down"):
        expert_mlp = _select_expert_mlp(activation_fn)

        def _mlp_call(x_d, w13, w2, physical, active):
            return expert_mlp(x_d, w13, w2, physical, active, activation_fn)

        # Recompute the [C, 2I]-class MLP intermediates in backward instead of saving them,
        # like the pooled backend's per-wave remat. The lean data path removed the pure
        # gather chains XLA's rematerializer used to trade away ~4 GiB of live buffers, and
        # without this the step's live set no longer fits next to NCCL's pools.
        remat = functools.partial(
            jax.checkpoint, prevent_cse=False, policy=jax.checkpoint_policies.nothing_saveable
        )
        out_dispatch = remat(_mlp_call)(
            x_dispatch,
            moe_w13_local,
            moe_w2_local,
            physical_group_sizes,
            active_group_sizes,
        )

    with jax.named_scope("combine"):
        return_out_shape = jnp.zeros((assignments_per_shard, x_local.shape[1]), dtype=out_dispatch.dtype)
        # The mirror of dispatch: valid prefixes land back at unclipped sorted positions,
        # dropped rows keep the output operand's zeros, so no expansion is needed and the
        # final gather-sum reads dropped slots as zero contributions.
        returned = jax.lax.ragged_all_to_all(
            out_dispatch,
            return_out_shape,
            *return_params,
            axis_name="expert",
        )
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(sender_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, CapacityOverflow(sender=dropped_total, receiver=jnp.zeros_like(dropped_total))
