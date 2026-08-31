# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Ragged all-to-all expert-parallel Grug MoE backend.

Axis names used in the shape annotations:

    Tlocal  tokens on this shard
    K       routed experts per token
    TK      routed assignments on this shard, Tlocal * K
    H       hidden size
    I       expert intermediate size
    I2      gate and up projections fused, 2 * I
    E       experts in the model
    Elocal  experts held by this shard
    Echunk  experts in one sequential chunk, Elocal / chunks
    C       rows in one chunk's receiver buffer, the per-chunk capacity
    S       shards on the expert axis
"""

import functools
import logging
import math
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import CapacityOverflow, _interleave_gate_up
from levanter.grug._moe.sonic import sonic_gather_sum, sonic_gather_sum_available
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    _sort_activations,
)
from levanter.grug.sharding import _batch_axes

logger = logging.getLogger(__name__)

# QuACK's grouped GEMMs are written for SM100 and ship only with the CUDA 13 GPU extra.
_SM100_COMPUTE_CAPABILITY = 10.0

# Sequential local-expert chunks per MoE layer; capacity splits evenly across chunks. Falls
# back to a single chunk when the local expert count is not divisible.
_EXPERT_CHUNKS = 2

# Selects the device-initiated ragged all-to-all kernel. The second entry is scoped to that op, so
# every other collective keeps NCCL's host-launched kernels.
RAGGED_REQUIRED_XLA_FLAGS = (
    "--xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true",
    "--xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall",
)


class _ExpertMlp(Protocol):
    """Runs the expert MLP over a receiver buffer laid out expert-major.

    Implementations take both views of the buffer's group sizes: the physical sizes, which charge
    trailing padding to the last expert, and the active sizes, which count only received rows.
    Which one a kernel reads depends on whether it covers the whole buffer or works from segment
    boundaries, so both are always passed and a kernel discards the one it does not use.
    """

    def __call__(
        self,
        x_dispatch: Float[Array, "C H"],
        moe_w13_local: Float[Array, "Echunk H I2"],
        moe_w2_local: Float[Array, "Echunk I H"],
        physical_group_sizes: Int[Array, "Echunk"],
        active_group_sizes: Int[Array, "Echunk"],
        activation_fn: Callable[[jax.Array], jax.Array],
    ) -> Float[Array, "C H"]: ...


def _ragged_dot_expert_mlp(
    x_dispatch: Float[Array, "C H"],
    moe_w13_local: Float[Array, "Echunk H I2"],
    moe_w2_local: Float[Array, "Echunk I H"],
    physical_group_sizes: Int[Array, "Echunk"],
    active_group_sizes: Int[Array, "Echunk"],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "C H"]:
    """Portable expert MLP over XLA's `ragged_dot`, which covers the whole receiver buffer."""
    del active_group_sizes
    w13_out = ragged_dot(x_dispatch, moe_w13_local, physical_group_sizes)
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    return ragged_dot(activation_fn(gate) * up, moe_w2_local, physical_group_sizes)


def _cute_expert_mlp(
    x_dispatch: Float[Array, "C H"],
    moe_w13_local: Float[Array, "Echunk H I2"],
    moe_w2_local: Float[Array, "Echunk I H"],
    physical_group_sizes: Int[Array, "Echunk"],
    active_group_sizes: Int[Array, "Echunk"],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "C H"]:
    """Expert MLP with all grouped GEMMs on QuACK's SM100 kernels.

    The grouped kernels are driven by segment boundaries, so they take the active sizes and
    mask the receiver buffer's trailing padding rather than charging it to the last expert.
    """
    del activation_fn, physical_group_sizes

    # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
    from levanter.grug._moe.sonic_cute import _expert_mlp_quack  # noqa: PLC0415

    moe_dim = moe_w2_local.shape[1]
    w13_interleaved = _interleave_gate_up(moe_w13_local, moe_dim)
    cumulative_group_sizes = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)]
    )
    return _expert_mlp_quack(x_dispatch, w13_interleaved, moe_w2_local, active_group_sizes, cumulative_group_sizes)


@functools.cache
def _quack_grouped_gemm_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    if float(jax.devices("gpu")[0].compute_capability) < _SM100_COMPUTE_CAPABILITY:
        return False
    try:
        import levanter.grug._moe.sonic_cute  # noqa: F401,PLC0415
    except (ImportError, AttributeError) as exc:
        logger.warning(
            "SM100 GPU present but the QuACK grouped-GEMM kernels did not import (%s). "
            "The ragged expert MLP falls back to ragged_dot, which computes the same function "
            "more slowly. Install levanter's `gpu` extra to use them.",
            exc,
        )
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


def _unpermute_from_global_expert(
    intermediate: Float[Array, "TK H"],
    sorted_indices: Int[Array, "TK"],
    combine_weights_local: Float[Array, "Tlocal K"],
    *,
    tokens_per_shard: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    """Weight each token's expert outputs by its routing weights and sum them."""
    positions = jnp.argsort(sorted_indices)
    if sonic_gather_sum_available():
        # One kernel for the gather and the sum, materializing neither the unpermuted
        # ``[TK, H]`` buffer nor the ``[T, K, H]`` view -- at top-8 that view is eight times
        # the output. It accumulates in fp32 like the einsum below and keeps the routing
        # weight in fp32 through the multiply, where the einsum has to cast it down to avoid
        # promoting the larger operand, so the two agree to a single rounding.
        return sonic_gather_sum(intermediate, positions.reshape(tokens_per_shard, topk), combine_weights_local)
    unsorted = _sort_activations(intermediate, positions)
    reshaped = unsorted.reshape(tokens_per_shard, topk, -1)
    return jnp.einsum(
        "tkd,tk->td", reshaped, combine_weights_local.astype(reshaped.dtype), preferred_element_type=jnp.float32
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def _gather_dispatch_rows(
    x_local: Float[Array, "Tlocal H"], sorted_indices: Int[Array, "TK"], topk: int
) -> Float[Array, "TK H"]:
    """Build the expert-sorted dispatch buffer with one gather.

    Equivalent to ``jnp.repeat(x_local, topk, axis=0)[sorted_indices]`` without
    materializing the repeated buffer or running a data-sized permute. The backward pass
    is the transpose: each token sums the cotangent rows of its ``topk`` sorted slots.
    """
    return x_local[sorted_indices // topk]


def _gather_dispatch_rows_fwd(
    x_local: Float[Array, "Tlocal H"], sorted_indices: Int[Array, "TK"], topk: int
) -> tuple[Float[Array, "TK H"], Int[Array, "TK"]]:
    return _gather_dispatch_rows(x_local, sorted_indices, topk), sorted_indices


def _gather_dispatch_rows_bwd(
    topk: int, sorted_indices: Int[Array, "TK"], cotangent: Float[Array, "TK H"]
) -> tuple[Float[Array, "Tlocal H"], None]:
    tokens_per_shard = sorted_indices.shape[0] // topk
    positions = jnp.argsort(sorted_indices).reshape(tokens_per_shard, topk)
    if sonic_gather_sum_available():
        ones = jnp.ones((tokens_per_shard, topk), dtype=jnp.float32)
        grad_x = sonic_gather_sum(cotangent, positions, ones)
    else:
        grad_x = jnp.sum(cotangent[positions], axis=1, dtype=jnp.float32)
    return grad_x.astype(cotangent.dtype), None


_gather_dispatch_rows.defvjp(_gather_dispatch_rows_fwd, _gather_dispatch_rows_bwd)


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

    # Local experts are processed in sequential chunks so only one chunk's transport buffers
    # are live at a time. The a2a outputs cannot be rematerialized (XLA never recomputes
    # collectives), so unchunked they pin [capacity, H] + [TK, H] per block window and the
    # hero step no longer fits next to NCCL's pools. Capacity splits evenly across chunks,
    # which also makes drop clipping per-chunk.
    chunks = _EXPERT_CHUNKS if local_experts % _EXPERT_CHUNKS == 0 and _EXPERT_CHUNKS > 1 else 1
    chunk_experts = local_experts // chunks
    chunk_capacity = max(chunk_experts, int(math.ceil(local_capacity / chunks)))
    hidden_dim = x_local.shape[1]

    with jax.named_scope("dispatch"):
        flat_selected = selected_experts_local.reshape(-1)  # [TK]
        sorted_indices = jnp.argsort(flat_selected)  # [TK]
        group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)  # [E]
        sorted_x = _gather_dispatch_rows(x_local, sorted_indices, topk)  # [TK, H]
        all_group_sizes = jax.lax.all_gather(group_sizes, "expert")  # [S, E]

    expert_mlp = _select_expert_mlp(activation_fn)
    chunk_of_expert = (jnp.arange(num_experts, dtype=jnp.int32) % local_experts) // chunk_experts  # [E]
    returned = jnp.zeros((assignments_per_shard, hidden_dim), dtype=x_local.dtype)  # [TK, H]
    accepted_local = jnp.zeros((), dtype=jnp.int32)
    for chunk_index in range(chunks):
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
            chunk_all_group_sizes = jnp.where(chunk_of_expert[None, :] == chunk_index, all_group_sizes, 0)  # [S, E]
            clipped_group_sizes = _clip_receiver_group_sizes(
                chunk_all_group_sizes,
                local_expert_size=local_experts,
                receiver_capacity=chunk_capacity,
            )
            # Sender starts come from the full (unmasked) sizes, so each chunk reads its
            # groups' accepted prefixes in place in the shared sorted buffer.
            dispatch_params, return_params = _expert_granular_a2a_params(
                all_group_sizes,
                clipped_group_sizes,
                shard_id,
                local_expert_size=local_experts,
            )
            # Serialize the chunks. Without this barrier, the scheduler can start the dispatch
            # of every chunk at the same time. This causes the high memory use that the chunks
            # prevent. A variant that overlaps one transport with the MLP stays within memory.
            # But it does not increase the speed. The transport and the MLP compete for the
            # same SMs.
            chunk_source, _ = jax.lax.optimization_barrier((sorted_x, returned))
            dispatch_out_shape = jnp.zeros((chunk_capacity, hidden_dim), dtype=x_local.dtype)  # [C, H]
            # Accepted rows are the prefix of each unclipped expert group and receiver offsets
            # pack arrivals expert-major, so the received buffer feeds the grouped MLP
            # directly: no sender compaction and no receiver-side permute.
            x_dispatch = jax.lax.ragged_all_to_all(  # [C, H]
                chunk_source,
                dispatch_out_shape,
                *dispatch_params,
                axis_name="expert",
            )
            active_all = jnp.sum(  # [Elocal]
                clipped_group_sizes.reshape(ep_size, ep_size, local_experts)[:, shard_id, :], axis=0
            )
            active_group_sizes = active_all[
                chunk_index * chunk_experts : (chunk_index + 1) * chunk_experts
            ]  # [Echunk]
            total_valid = jnp.sum(active_group_sizes, dtype=jnp.int32)
            physical_group_sizes = active_group_sizes.at[-1].add(chunk_capacity - total_valid)  # [Echunk]
            out_dispatch = expert_mlp(  # [C, H]
                x_dispatch,
                moe_w13_local[chunk_index * chunk_experts : (chunk_index + 1) * chunk_experts],
                moe_w2_local[chunk_index * chunk_experts : (chunk_index + 1) * chunk_experts],
                physical_group_sizes,
                active_group_sizes,
                activation_fn,
            )
            # The mirror of dispatch: valid prefixes land back at unclipped sorted positions.
            # Chaining every chunk through one output buffer composes the disjoint writes;
            # dropped rows keep the buffer's zeros, so the final gather-sum reads dropped
            # slots as zero contributions with no expansion step.
            returned = jax.lax.ragged_all_to_all(
                out_dispatch,
                returned,
                *return_params,
                axis_name="expert",
            )
            accepted_local = accepted_local + jnp.sum(clipped_group_sizes[shard_id], dtype=jnp.int32)

    with jax.named_scope("combine"):
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            combine_weights_local,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - accepted_local
        dropped_total = jax.lax.psum(dropped_local, _batch_axes(jax.sharding.get_abstract_mesh()))
    return out_local, CapacityOverflow(sender=dropped_total, receiver=jnp.zeros_like(dropped_total))
