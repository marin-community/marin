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
    U       expert-granular transfers on the expert axis
"""

import functools
import logging
import math
from collections.abc import Callable
from enum import auto, IntEnum
from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _assignment_validity,
    _interleave_gate_up,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.sonic import (
    sonic_gather_sum,
    sonic_gather_sum_available,
    sonic_scatter_rows,
    unwritten_buffer,
)
from levanter.grug._moe.ep_common import (
    ExpertA2aParams,
    _clip_receiver_group_sizes,
    _expert_granular_a2a_params,
    _sort_activations,
)

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
    Rows past the active count are unspecified, and may be non-finite, in ``x_dispatch`` and in
    the output's cotangent. Implementations keep them out of the active output rows and out of
    the weight gradients, and leave their own output rows past the active count unspecified.
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
    """Portable expert MLP over XLA's `ragged_dot`, including static trailing rows.

    `ragged_dot` covers the whole static buffer, so the rows past the active count are zeroed on
    the way in and on the way out. The output mask also zeroes their cotangent rows, which keeps
    them out of the weight gradients.
    """
    active = (jnp.arange(x_dispatch.shape[0]) < jnp.sum(active_group_sizes))[:, None]
    x_dispatch = jnp.where(active, x_dispatch, 0)
    w13_out = ragged_dot(x_dispatch, moe_w13_local, physical_group_sizes)
    moe_dim = moe_w2_local.shape[1]
    gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
    out = ragged_dot(activation_fn(gate) * up, moe_w2_local, physical_group_sizes)
    return jnp.where(active, out, 0)


def _cute_expert_mlp(
    x_dispatch: Float[Array, "C H"],
    moe_w13_local: Float[Array, "Echunk H I2"],
    moe_w2_local: Float[Array, "Echunk I H"],
    physical_group_sizes: Int[Array, "Echunk"],
    active_group_sizes: Int[Array, "Echunk"],
    activation_fn: Callable[[jax.Array], jax.Array],
) -> Float[Array, "C H"]:
    """Expert MLP on QuACK's SM100 grouped GEMMs, activation path and weight gradients alike.

    The grouped kernels are driven by segment boundaries, so they take the active sizes and
    leave trailing rows unspecified. The return transport reads only the active rows.
    """
    del activation_fn, physical_group_sizes

    # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
    from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad  # noqa: PLC0415

    moe_dim = moe_w2_local.shape[1]
    w13_interleaved = _interleave_gate_up(moe_w13_local, moe_dim)
    cumulative_group_sizes = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)]
    )
    return _expert_mlp_quack_wgrad(x_dispatch, w13_interleaved, moe_w2_local, cumulative_group_sizes)


@functools.cache
def _quack_grouped_gemm_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    device = jax.devices("gpu")[0]
    # Only NVIDIA devices report `compute_capability` as a CUDA major.minor version; ROCm reports
    # a gfx architecture name such as "gfx942".
    if "nvidia" not in device.device_kind.lower():
        return False
    if float(device.compute_capability) < _SM100_COMPUTE_CAPABILITY:
        return False
    try:
        # `sonic_cute` pulls in `quack_moe_cute`, which imports QuACK's varlen entry points at
        # module scope, so this covers a QuACK that is missing or has moved them.
        import levanter.grug._moe.sonic_cute  # noqa: F401,PLC0415
    except ImportError as exc:
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
    accepted: Bool[Array, "Tlocal K"],
    *,
    tokens_per_shard: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    """Weight each token's expert outputs by its routing weights and sum them.

    Dropped assignments, whose ``accepted`` is false, must have weight zero. Their rows may hold
    unspecified values: the sum never reads them, but the weight gradient may, so a caller zeroes
    their weights with ``jnp.where``, whose transpose discards those gradients. An accepted row's
    weight gets its gradient ``<dout, y>`` even at weight zero.
    """
    positions = jnp.argsort(sorted_indices)
    if sonic_gather_sum_available():
        # One kernel for the gather and the sum, materializing neither the unpermuted
        # ``[TK, H]`` buffer nor the ``[T, K, H]`` view -- at top-8 that view is eight times
        # the output. It accumulates in fp32 like the einsum below and keeps the routing
        # weight in fp32 through the multiply, where the einsum has to cast it down to avoid
        # promoting the larger operand, so the two agree to a single rounding.
        return sonic_gather_sum(intermediate, positions.reshape(tokens_per_shard, topk), combine_weights_local)
    unsorted = _sort_activations(intermediate, positions)
    reshaped = jnp.where(accepted[..., None], unsorted.reshape(tokens_per_shard, topk, -1), 0)
    return jnp.einsum(
        "tkd,tk->td", reshaped, combine_weights_local.astype(reshaped.dtype), preferred_element_type=jnp.float32
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def _gather_dispatch_rows(
    x_local: Float[Array, "Tlocal H"],
    sorted_indices: Int[Array, "TK"],
    accepted: Float[Array, "Tlocal K"],
    topk: int,
) -> Float[Array, "TK H"]:
    """Build the expert-sorted dispatch buffer, ``jnp.repeat(x_local, topk, axis=0)[sorted_indices]``.

    ``accepted`` is 1 for an assignment that reaches its expert and 0 otherwise. The transport
    reads only the accepted slots, so on GPU a kernel reads each token's row once and writes it to
    its accepted slots alone, leaving the others unspecified; elsewhere one gather fills every
    slot. The backward pass is the transpose: each token sums the cotangent rows of its accepted
    sorted slots, and the transport never writes the other slots' cotangent rows.
    """
    if sonic_gather_sum_available():
        tokens_per_shard = sorted_indices.shape[0] // topk
        positions = jnp.argsort(sorted_indices).reshape(tokens_per_shard, topk)
        return sonic_scatter_rows(x_local, positions, accepted != 0, rows=sorted_indices.shape[0])
    return x_local[sorted_indices // topk]


def _gather_dispatch_rows_fwd(
    x_local: Float[Array, "Tlocal H"],
    sorted_indices: Int[Array, "TK"],
    accepted: Float[Array, "Tlocal K"],
    topk: int,
) -> tuple[Float[Array, "TK H"], tuple[Int[Array, "TK"], Float[Array, "Tlocal K"]]]:
    return _gather_dispatch_rows(x_local, sorted_indices, accepted, topk), (sorted_indices, accepted)


def _gather_dispatch_rows_bwd(
    topk: int,
    residuals: tuple[Int[Array, "TK"], Float[Array, "Tlocal K"]],
    cotangent: Float[Array, "TK H"],
) -> tuple[Float[Array, "Tlocal H"], None, None]:
    sorted_indices, accepted = residuals
    tokens_per_shard = sorted_indices.shape[0] // topk
    positions = jnp.argsort(sorted_indices).reshape(tokens_per_shard, topk)
    if sonic_gather_sum_available():
        grad_x = sonic_gather_sum(cotangent, positions, accepted)
    else:
        grad_x = jnp.sum(jnp.where(accepted[..., None] != 0, cotangent[positions], 0), axis=1, dtype=jnp.float32)
    return grad_x.astype(cotangent.dtype), None, None


_gather_dispatch_rows.defvjp(_gather_dispatch_rows_fwd, _gather_dispatch_rows_bwd)


class _TransportBufferSite(IntEnum):
    DISPATCH_OUTPUT = auto()
    RETURN_OUTPUT = auto()
    OPERAND_COTANGENT = auto()


def _transport_buffer(
    rows: int, hidden_dim: int, dtype, tie: Int[Array, "N"], site: _TransportBufferSite
) -> Float[Array, "rows H"]:
    """Return an output buffer for an in-place ``ragged_all_to_all``, with unspecified contents.

    Every consumer of a transport output reads only the rows the collective writes, so the buffer
    needs no fill. On GPU it comes from a kernel that writes nothing; elsewhere it is zero-filled.

    A buffer with no inputs, such as ``jnp.zeros`` or ``jax.lax.empty``, is loop invariant. JAX or
    XLA hoists it out of the layer loop and merges equal-shaped ones, and CopyInsertion then copies
    it into every output slot on every layer. ``min(tie[0], -site) + site`` is zero for
    every non-negative ``tie`` but depends on a loop-carried value, so the buffer built from it
    stays in the loop. ``site`` makes each call's expression distinct, so CSE cannot merge two
    buffers into one.

    ``tie`` must contain non-negative integers. ``site`` must identify the call site.
    """
    marker = jnp.minimum(tie[0], -site) + site
    if sonic_gather_sum_available():
        return unwritten_buffer((rows, hidden_dim), dtype, marker)
    return jax.lax.broadcast(marker.astype(dtype), (rows, hidden_dim))


def _reverse_ragged_a2a(
    cotangent: Float[Array, "O H"], init: Float[Array, "R H"], params: ExpertA2aParams
) -> Float[Array, "R H"]:
    """Send each received row's cotangent back to the operand row it came from, into ``init``."""
    # Exchanged offsets reverse the collective, matching JAX's transpose rule.
    exchanged_output_offsets = jax.lax.all_to_all(params.output_offsets, "expert", 0, 0, tiled=True)
    exchanged_input_offsets = jax.lax.all_to_all(params.input_offsets, "expert", 0, 0, tiled=True)
    return jax.lax.ragged_all_to_all(
        cotangent,
        init,
        exchanged_output_offsets,
        params.recv_sizes,
        exchanged_input_offsets,
        params.send_sizes,
        axis_name="expert",
    )


# The two wrappers below write the received rows into ``output_init`` in place. Every caller
# passes an ``output_init`` whose rows at the written positions depend on no differentiated
# input: a fresh transport buffer, or one that earlier chunks wrote only at other rows. On such an
# input the overwrite has the derivative of adding the received rows to ``output_init``, so the
# transpose passes the output cotangent to ``output_init`` unchanged. JAX's transpose rule would
# zero the written rows first, one full pass over the buffer per call. The wrappers also build
# their backward buffers inside the layer loop, where JAX's rule would hoist them (#8822).


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ragged_a2a_add(
    operand_rows: int,
    operand: Float[Array, "R H"],
    output_init: Float[Array, "O H"],
    params: ExpertA2aParams,
) -> Float[Array, "O H"]:
    """Write the rows ``operand`` sends over the expert axis into ``output_init``.

    ``output_init`` must not depend on a differentiated input at the rows this call writes.
    ``operand_rows`` is ``operand.shape[0]``. The backward needs it and does not see the operand.
    Operand rows this call does not send get unspecified cotangent rows.
    """
    del operand_rows
    return jax.lax.ragged_all_to_all(operand, output_init, *params, axis_name="expert")


def _ragged_a2a_add_fwd(
    operand_rows: int,
    operand: Float[Array, "R H"],
    output_init: Float[Array, "O H"],
    params: ExpertA2aParams,
) -> tuple[Float[Array, "O H"], ExpertA2aParams]:
    return _ragged_a2a_add(operand_rows, operand, output_init, params), params


def _ragged_a2a_add_bwd(
    operand_rows: int,
    params: ExpertA2aParams,
    cotangent: Float[Array, "O H"],
) -> tuple[Float[Array, "R H"], Float[Array, "O H"], None]:
    init = _transport_buffer(
        operand_rows,
        cotangent.shape[1],
        cotangent.dtype,
        params.recv_sizes,
        site=_TransportBufferSite.OPERAND_COTANGENT,
    )
    return _reverse_ragged_a2a(cotangent, init, params), cotangent, None


_ragged_a2a_add.defvjp(_ragged_a2a_add_fwd, _ragged_a2a_add_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
def _ragged_a2a_add_forwarding(
    operand_rows: int,
    operand: Float[Array, "R H"],
    output_init: Float[Array, "O H"],
    params: ExpertA2aParams,
) -> tuple[Float[Array, "O H"], Float[Array, "R H"]]:
    """``_ragged_a2a_add`` that also returns ``operand`` for a later call to read.

    Later calls must read rows of the forwarded operand disjoint from the rows this call reads.
    The backward then writes this call's operand cotangent into the forwarded operand's
    cotangent, which is zero on those rows, where reading the operand directly in every call
    would give each call a zero-filled cotangent buffer and a sum over the calls.
    """
    del operand_rows
    return jax.lax.ragged_all_to_all(operand, output_init, *params, axis_name="expert"), operand


def _ragged_a2a_add_forwarding_fwd(
    operand_rows: int,
    operand: Float[Array, "R H"],
    output_init: Float[Array, "O H"],
    params: ExpertA2aParams,
) -> tuple[tuple[Float[Array, "O H"], Float[Array, "R H"]], ExpertA2aParams]:
    return _ragged_a2a_add_forwarding(operand_rows, operand, output_init, params), params


def _ragged_a2a_add_forwarding_bwd(
    operand_rows: int,
    params: ExpertA2aParams,
    cotangents: tuple[Float[Array, "O H"], Float[Array, "R H"]],
) -> tuple[Float[Array, "R H"], Float[Array, "O H"], None]:
    del operand_rows
    cotangent, forwarded_cotangent = cotangents
    return _reverse_ragged_a2a(cotangent, forwarded_cotangent, params), cotangent, None


_ragged_a2a_add_forwarding.defvjp(_ragged_a2a_add_forwarding_fwd, _ragged_a2a_add_forwarding_bwd)


def _accepted_assignments(
    flat_selected: Int[Array, "TK"],
    sorted_indices: Int[Array, "TK"],
    group_sizes: Int[Array, "E"],
    accepted_group_sizes: Int[Array, "E"],
) -> Bool[Array, "TK"]:
    """Mark, in assignment order, the assignments that reach their expert.

    Receivers accept a prefix of each expert's group in the expert-sorted buffer, so an
    assignment is accepted when its rank within its group is below the group's accepted size.
    Invalid assignments carry the out-of-range expert id ``E`` and are never accepted.
    """
    num_experts = group_sizes.shape[0]
    sorted_experts = flat_selected[sorted_indices]
    expert = jnp.minimum(sorted_experts, num_experts - 1)
    group_starts = jnp.cumsum(group_sizes) - group_sizes
    rank = jnp.arange(sorted_indices.shape[0], dtype=jnp.int32) - group_starts[expert]
    accepted_sorted = (sorted_experts < num_experts) & (rank < accepted_group_sizes[expert])
    return (
        jnp.zeros_like(accepted_sorted)
        .at[sorted_indices]
        .set(accepted_sorted, unique_indices=True, mode="promise_in_bounds")
    )


def _moe_mlp_ep_ragged_a2a_local(
    x_local: Float[Array, "Tlocal H"],
    selected_experts_local: Int[Array, "Tlocal K"],
    combine_weights_local: Float[Array, "Tlocal K"],
    token_valid_local: Bool[Array, "Tlocal"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    *,
    activation_fn: Callable[[jax.Array], jax.Array],
    num_experts: int,
    capacity_factor: float,
    token_sharding_axes: tuple[str, ...],
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
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
    physical_capacity = int(math.ceil(capacity_factor * assignments_per_shard))
    physical_capacity = max(local_experts, physical_capacity)

    # Local experts are processed in sequential chunks so only one chunk's transport buffers
    # are live at a time. The a2a outputs cannot be rematerialized (XLA never recomputes
    # collectives), so unchunked they pin [capacity, H] + [TK, H] per block window and the
    # hero step no longer fits next to NCCL's pools. Capacity splits evenly across chunks,
    # which also makes drop clipping per-chunk.
    chunks = _EXPERT_CHUNKS if local_experts % _EXPERT_CHUNKS == 0 and _EXPERT_CHUNKS > 1 else 1
    chunk_experts = local_experts // chunks
    chunk_capacity = max(chunk_experts, int(math.ceil(physical_capacity / chunks)))
    hidden_dim = x_local.shape[1]

    with jax.named_scope("dispatch"):
        assignment_valid = _assignment_validity(token_valid_local, tokens=tokens_per_shard, topk=topk)
        flat_selected = jnp.where(assignment_valid, selected_experts_local.reshape(-1), num_experts)  # [TK]
        sorted_indices = jnp.argsort(flat_selected)  # [TK]
        group_sizes = jnp.bincount(flat_selected, length=num_experts).astype(jnp.int32)  # [E]
        all_group_sizes = jax.lax.all_gather(group_sizes, "expert")  # [S, E]
        valid_assignments = jnp.sum(all_group_sizes, dtype=jnp.int32)
        logical_capacity = _scaled_capacity(
            valid_assignments,
            capacity_factor=capacity_factor,
            divisor=ep_size,
            minimum=local_experts,
            maximum=physical_capacity,
        )
        logical_chunk_capacity = jnp.maximum(
            (logical_capacity + chunks - 1) // chunks,
            chunk_experts,
        )
        chunk_of_expert = (jnp.arange(num_experts, dtype=jnp.int32) % local_experts) // chunk_experts  # [E]
        chunk_clipped_group_sizes = [  # [S, E] each
            _clip_receiver_group_sizes(
                jnp.where(chunk_of_expert[None, :] == chunk_index, all_group_sizes, 0),
                local_expert_size=local_experts,
                receiver_capacity=logical_chunk_capacity,
            )
            for chunk_index in range(chunks)
        ]
        # Every expert belongs to one chunk, so summing the chunks gives each expert's accepted
        # prefix of this shard's group.
        accepted_group_sizes = sum(clipped[shard_id] for clipped in chunk_clipped_group_sizes)  # [E]
        accepted = _accepted_assignments(flat_selected, sorted_indices, group_sizes, accepted_group_sizes)  # [TK]
        accepted = accepted.reshape(tokens_per_shard, topk)
        sorted_x = _gather_dispatch_rows(x_local, sorted_indices, accepted.astype(jnp.float32), topk)  # [TK, H]

    expert_mlp = _select_expert_mlp(activation_fn)
    # Rows no chunk writes are the dropped assignments, which the combine skips.
    returned = _transport_buffer(
        assignments_per_shard, hidden_dim, x_local.dtype, group_sizes, site=_TransportBufferSite.RETURN_OUTPUT
    )  # [TK, H]
    # Each chunk reads only its own experts' groups of the sorted buffer, so chunks read disjoint
    # rows and every chunk but the last forwards the buffer to the next.
    chunk_source = sorted_x
    for chunk_index, clipped_group_sizes in enumerate(chunk_clipped_group_sizes):
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
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
            chunk_source, _ = jax.lax.optimization_barrier((chunk_source, returned))
            # Accepted rows are the prefix of each unclipped expert group and receiver offsets
            # pack arrivals expert-major, so the received buffer feeds the grouped MLP
            # directly: no sender compaction and no receiver-side permute.
            dispatch_init = _transport_buffer(  # [C, H]
                chunk_capacity,
                hidden_dim,
                x_local.dtype,
                dispatch_params.send_sizes,
                site=_TransportBufferSite.DISPATCH_OUTPUT,
            )
            if chunk_index < chunks - 1:
                x_dispatch, chunk_source = _ragged_a2a_add_forwarding(  # [C, H]
                    assignments_per_shard, chunk_source, dispatch_init, dispatch_params
                )
            else:
                x_dispatch = _ragged_a2a_add(assignments_per_shard, chunk_source, dispatch_init, dispatch_params)
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
            # Chaining every chunk through one output buffer composes the disjoint writes, with
            # no expansion step.
            returned = _ragged_a2a_add(chunk_capacity, out_dispatch, returned, return_params)

    with jax.named_scope("combine"):
        # A dropped or padding assignment gets weight zero, so the gather-sum never reads its
        # unwritten row, and the `where` discards the weight gradient read from that row.
        out_local = _unpermute_from_global_expert(
            returned,
            sorted_indices,
            jnp.where(accepted, combine_weights_local, 0),
            accepted,
            tokens_per_shard=tokens_per_shard,
            topk=topk,
        ).astype(x_local.dtype)
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(accepted_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=dropped_total, receiver_dropped=jnp.zeros_like(dropped_total))
