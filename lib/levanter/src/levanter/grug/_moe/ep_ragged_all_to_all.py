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

import dataclasses
import functools
import logging
import math
from collections.abc import Callable
from enum import auto, IntEnum
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _assignment_validity,
    _interleave_gate_up,
    _invert_permutation,
    _scaled_capacity,
    CapacityDrops,
)
from levanter.grug._moe.sonic import sonic_gather_sum, sonic_gather_sum_available, unwritten_buffer
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
    """Runs the expert MLP over a receiver buffer laid out expert-major, with its own backward.

    Implementations take both views of the buffer's group sizes: the physical sizes, which charge
    trailing padding to the last expert, and the active sizes, which count only received rows.
    Rows past the active count are unspecified, and may be non-finite, in ``x_dispatch`` and in
    the output's cotangent. Implementations keep them out of the active output rows and out of
    the weight gradients, and leave their own output rows past the active count unspecified.

    ``backward`` also returns each row's ``<y, dy>`` in fp32, the gradient of a per-row scale
    applied to the output. The routed-expert backward turns it into the routing-weight
    gradient, so it never needs the output ``y`` itself.
    """

    def forward(
        self,
        x_dispatch: Float[Array, "C H"],
        moe_w13_local: Float[Array, "Echunk H I2"],
        moe_w2_local: Float[Array, "Echunk I H"],
        physical_group_sizes: Int[Array, "Echunk"],
        active_group_sizes: Int[Array, "Echunk"],
    ) -> tuple[Float[Array, "C H"], tuple[jax.Array, ...]]: ...

    def backward(
        self, residuals: tuple[jax.Array, ...], cotangent: Float[Array, "C H"]
    ) -> tuple[Float[Array, "C H"], Float[Array, "Echunk H I2"], Float[Array, "Echunk I H"], Float[Array, "C"]]: ...


@dataclasses.dataclass(frozen=True)
class _RaggedDotExpertMlp:
    """Portable expert MLP over XLA's `ragged_dot`, including static trailing rows.

    `ragged_dot` covers the whole static buffer, so the rows past the active count are zeroed on
    the way in and on the way out. The output mask also zeroes their cotangent rows, which keeps
    them out of the weight gradients.
    """

    activation_fn: Callable[[jax.Array], jax.Array]

    def _apply(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        active = (jnp.arange(x_dispatch.shape[0]) < jnp.sum(active_group_sizes))[:, None]
        x_dispatch = jnp.where(active, x_dispatch, 0)
        w13_out = ragged_dot(x_dispatch, moe_w13_local, physical_group_sizes)
        moe_dim = moe_w2_local.shape[1]
        gate, up = jnp.split(w13_out, [moe_dim], axis=-1)
        out = ragged_dot(self.activation_fn(gate) * up, moe_w2_local, physical_group_sizes)
        return jnp.where(active, out, 0)

    def forward(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        out = self._apply(x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes)
        return out, (x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes, out)

    def backward(self, residuals, cotangent):
        x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes, out = residuals
        _, vjp = jax.vjp(
            lambda x, w13, w2: self._apply(x, w13, w2, physical_group_sizes, active_group_sizes),
            x_dispatch,
            moe_w13_local,
            moe_w2_local,
        )
        dx, dw13, dw2 = vjp(cotangent)
        output_dot_cotangent = jnp.sum(out.astype(jnp.float32) * cotangent.astype(jnp.float32), axis=-1)
        return dx, dw13, dw2, output_dot_cotangent


@dataclasses.dataclass(frozen=True)
class _CuteExpertMlp:
    """Expert MLP on QuACK's SM100 grouped GEMMs, activation path and weight gradients alike.

    The grouped kernels are driven by segment boundaries, so they take the active sizes and
    leave trailing rows unspecified. SwiGLU is fused into the gate/up GEMM.
    """

    def forward(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        del physical_group_sizes
        # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
        from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad_fwd  # noqa: PLC0415

        moe_dim = moe_w2_local.shape[1]
        w13_interleaved = _interleave_gate_up(moe_w13_local, moe_dim)
        cumulative_group_sizes = jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)]
        )
        return _expert_mlp_quack_wgrad_fwd(x_dispatch, w13_interleaved, moe_w2_local, cumulative_group_sizes)

    def backward(self, residuals, cotangent):
        from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad_backward  # noqa: PLC0415

        dx, dw13_interleaved, dw2, output_dot_cotangent = _expert_mlp_quack_wgrad_backward(residuals, cotangent)
        w13_interleaved = residuals[1]
        moe_dim = w13_interleaved.shape[2] // 2
        # The interleave is a fixed permutation of the last axis; its transpose restores gate/up halves.
        _, deinterleave = jax.vjp(lambda w: _interleave_gate_up(w, moe_dim), jnp.zeros_like(w13_interleaved))
        (dw13,) = deinterleave(dw13_interleaved)
        return dx, dw13, dw2, output_dot_cotangent


@functools.cache
def _quack_grouped_gemm_available() -> bool:
    if jax.default_backend() != "gpu":
        return False
    if float(jax.devices("gpu")[0].compute_capability) < _SM100_COMPUTE_CAPABILITY:
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
        return _CuteExpertMlp()
    return _RaggedDotExpertMlp(activation_fn)


def _unpermute_from_global_expert(
    intermediate: Float[Array, "TK H"],
    sorted_indices: Int[Array, "TK"],
    combine_weights_local: Float[Array, "Tlocal K"],
    *,
    tokens_per_shard: int,
    topk: int,
) -> Float[Array, "Tlocal H"]:
    """Weight each token's expert outputs by its routing weights and sum them.

    Rows whose weight is zero never enter the sum, so they may hold unspecified values.
    """
    positions = _invert_permutation(sorted_indices)
    if sonic_gather_sum_available():
        # One kernel for the gather and the sum, materializing neither the unpermuted
        # ``[TK, H]`` buffer nor the ``[T, K, H]`` view -- at top-8 that view is eight times
        # the output. It accumulates in fp32 like the einsum below and keeps the routing
        # weight in fp32 through the multiply, where the einsum has to cast it down to avoid
        # promoting the larger operand, so the two agree to a single rounding.
        return sonic_gather_sum(intermediate, positions.reshape(tokens_per_shard, topk), combine_weights_local)
    unsorted = _sort_activations(intermediate, positions)
    reshaped = jnp.where(combine_weights_local[..., None] != 0, unsorted.reshape(tokens_per_shard, topk, -1), 0)
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
    """Build the expert-sorted dispatch buffer with one gather.

    Equivalent to ``jnp.repeat(x_local, topk, axis=0)[sorted_indices]`` without
    materializing the repeated buffer or running a data-sized permute. The backward pass
    is the transpose: each token sums the cotangent rows of its accepted sorted slots.
    ``accepted`` is 1 for an assignment that reaches its expert and 0 otherwise. The
    transport never reads the other slots, so their cotangent rows are unspecified.
    """
    del accepted
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
    positions = _invert_permutation(sorted_indices).reshape(tokens_per_shard, topk)
    if sonic_gather_sum_available():
        grad_x = sonic_gather_sum(cotangent, positions, accepted)
    else:
        grad_x = jnp.sum(jnp.where(accepted[..., None] != 0, cotangent[positions], 0), axis=1, dtype=jnp.float32)
    return grad_x.astype(cotangent.dtype), None, None


_gather_dispatch_rows.defvjp(_gather_dispatch_rows_fwd, _gather_dispatch_rows_bwd)


class _TransportBufferSite(IntEnum):
    DISPATCH_OUTPUT = auto()
    RETURN_OUTPUT = auto()
    DISPATCH_COTANGENT = auto()
    RETURN_COTANGENT = auto()
    OUTPUT_DOT_COTANGENT = auto()


def _transport_buffer(
    rows: int, hidden_dim: int, dtype, tie: Int[Array, "N"], site: _TransportBufferSite
) -> Float[Array, "rows H"]:
    """Return an output buffer for an in-place ``ragged_all_to_all``, with unspecified contents.

    Every consumer of a transport output reads only the rows the collective writes, so the buffer
    needs no fill. On GPU it comes from a kernel that writes nothing; elsewhere it is zero-filled.

    A buffer with no inputs, such as ``jnp.zeros`` or ``jax.lax.empty``, is loop invariant. JAX or
    XLA hoists it out of the layer loop and merges equal-shaped ones, and CopyInsertion then copies
    it into every output slot on every layer (#8822). ``min(tie[0], -site) + site`` is zero for
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


class _ExpertRouting(NamedTuple):
    """This shard's routing, all integer or boolean, shared by the forward and the backward."""

    sorted_indices: Int[Array, "TK"]
    accepted: Bool[Array, "Tlocal K"]
    group_sizes: Int[Array, "E"]
    all_group_sizes: Int[Array, "S E"]
    chunk_clipped_group_sizes: tuple[Int[Array, "S E"], ...]
    shard_id: Int[Array, ""]


@dataclasses.dataclass(frozen=True)
class _ExpertLayout:
    """Static shapes of the chunked expert pipeline on one shard."""

    tokens_per_shard: int
    topk: int
    local_experts: int
    ep_size: int
    chunk_experts: int
    chunk_capacity: int
    expert_mlp: _ExpertMlp

    @property
    def chunks(self) -> int:
        return self.local_experts // self.chunk_experts


class _ChunkResiduals(NamedTuple):
    dispatch_params: ExpertA2aParams
    return_params: ExpertA2aParams
    expert_mlp: tuple[jax.Array, ...]


def _routed_experts_forward(
    sorted_x: Float[Array, "TK H"],
    weights: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _ExpertRouting,
    layout: _ExpertLayout,
) -> tuple[Float[Array, "Tlocal H"], tuple[_ChunkResiduals, ...]]:
    assignments, hidden_dim = sorted_x.shape
    # Rows no chunk writes are the dropped assignments, which the combine skips.
    returned = _transport_buffer(
        assignments, hidden_dim, sorted_x.dtype, routing.group_sizes, site=_TransportBufferSite.RETURN_OUTPUT
    )  # [TK, H]
    chunk_source = sorted_x
    chunk_residuals = []
    for chunk_index, clipped_group_sizes in enumerate(routing.chunk_clipped_group_sizes):
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
            # Sender starts come from the full (unmasked) sizes, so each chunk reads its
            # groups' accepted prefixes in place in the shared sorted buffer.
            dispatch_params, return_params = _expert_granular_a2a_params(
                routing.all_group_sizes,
                clipped_group_sizes,
                routing.shard_id,
                local_expert_size=layout.local_experts,
            )
            if chunk_residuals:
                # Serialize the chunks. Without this barrier, the scheduler can start the dispatch
                # of every chunk at the same time, and the chunk buffers are all live at once,
                # which is the memory the chunks exist to save. The barrier waits for the previous
                # chunk's backward inputs rather than its return transport: the backward does not
                # need the return, so a recompute for the backward drops it and must not be held
                # to it.
                chunk_source, _ = jax.lax.optimization_barrier((chunk_source, chunk_residuals[-1].expert_mlp))
            # Accepted rows are the prefix of each unclipped expert group and receiver offsets
            # pack arrivals expert-major, so the received buffer feeds the grouped MLP
            # directly: no sender compaction and no receiver-side permute.
            dispatch_init = _transport_buffer(  # [C, H]
                layout.chunk_capacity,
                hidden_dim,
                sorted_x.dtype,
                dispatch_params.send_sizes,
                site=_TransportBufferSite.DISPATCH_OUTPUT,
            )
            x_dispatch = jax.lax.ragged_all_to_all(chunk_source, dispatch_init, *dispatch_params, axis_name="expert")
            active_all = jnp.sum(  # [Elocal]
                clipped_group_sizes.reshape(layout.ep_size, layout.ep_size, layout.local_experts)[
                    :, routing.shard_id, :
                ],
                axis=0,
            )
            experts = slice(chunk_index * layout.chunk_experts, (chunk_index + 1) * layout.chunk_experts)
            active_group_sizes = active_all[experts]  # [Echunk]
            total_valid = jnp.sum(active_group_sizes, dtype=jnp.int32)
            physical_group_sizes = active_group_sizes.at[-1].add(layout.chunk_capacity - total_valid)  # [Echunk]
            out_dispatch, expert_mlp_residuals = layout.expert_mlp.forward(  # [C, H]
                x_dispatch,
                moe_w13_local[experts],
                moe_w2_local[experts],
                physical_group_sizes,
                active_group_sizes,
            )
            # The mirror of dispatch: valid prefixes land back at unclipped sorted positions.
            # Chaining every chunk through one output buffer composes the disjoint writes, with
            # no expansion step.
            returned = jax.lax.ragged_all_to_all(out_dispatch, returned, *return_params, axis_name="expert")
            chunk_residuals.append(_ChunkResiduals(dispatch_params, return_params, expert_mlp_residuals))

    with jax.named_scope("combine"):
        out = _unpermute_from_global_expert(
            returned,
            routing.sorted_indices,
            weights,
            tokens_per_shard=layout.tokens_per_shard,
            topk=layout.topk,
        ).astype(sorted_x.dtype)
    return out, tuple(chunk_residuals)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _routed_experts(
    sorted_x: Float[Array, "TK H"],
    weights: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    routing: _ExpertRouting,
    layout: _ExpertLayout,
) -> Float[Array, "Tlocal H"]:
    """Dispatch the sorted rows to their experts, run the expert MLP, return and combine.

    ``weights`` must be zero for every assignment that ``routing.accepted`` marks as dropped.
    An accepted assignment whose weight is exactly zero gets a zero weight gradient, where the
    exact gradient is ``<dout, y>``: its output cotangent ``w * dout`` carries no information.
    Router weights that are sigmoids or softmaxes reach zero only below about 1e-38, where the
    derivative with respect to the logit is as small, so the router logits' gradient is unchanged.

    The backward computes the routing-weight gradient on the expert side. Each output row is
    ``y = h @ W2`` and its cotangent there is ``dy = w * dout``, so ``<dout, y> = <h, dh> / w``
    with ``dh = dy @ W2^T``, which the expert MLP backward computes anyway. The backward therefore
    reads neither ``y``, nor the return transport, nor the combined output. When the combined
    output is saved for the backward, a recompute for the backward runs only the dispatch and
    the gate/up projection.
    """
    out, _residuals = _routed_experts_forward(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout)
    return out


def _routed_experts_fwd(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout):
    out, chunk_residuals = _routed_experts_forward(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout)
    return out, (weights, routing, chunk_residuals)


def _routed_experts_bwd(layout, residuals, out_cotangent):
    weights, routing, chunk_residuals = residuals
    assignments = routing.sorted_indices.shape[0]
    hidden_dim = out_cotangent.shape[1]
    weights_f32 = weights.astype(jnp.float32)
    with jax.named_scope("combine"):
        # The combine's transpose: every accepted row's cotangent is its token's output cotangent
        # times its routing weight, rounded once, as the fused gather-sum's backward rounds it.
        sorted_weights = weights_f32.reshape(-1)[routing.sorted_indices]
        token_cotangent = out_cotangent[routing.sorted_indices // layout.topk].astype(jnp.float32)
        returned_cotangent = (token_cotangent * sorted_weights[:, None]).astype(out_cotangent.dtype)  # [TK, H]
    # Hold the backward transports until the recompute has dispatched its last chunk. Nothing
    # else orders them: the backward no longer reads the recomputed return. With more than one
    # collective in flight, a backward transport overlapping a recomputed dispatch gave
    # run-to-run different gradients on GB200.
    returned_cotangent, _ = jax.lax.optimization_barrier((returned_cotangent, chunk_residuals[-1].expert_mlp))

    dispatch_cotangent = _transport_buffer(
        assignments,
        hidden_dim,
        out_cotangent.dtype,
        routing.group_sizes,
        site=_TransportBufferSite.DISPATCH_COTANGENT,
    )  # [TK, H]
    output_dot = _transport_buffer(
        assignments, 1, jnp.float32, routing.group_sizes, site=_TransportBufferSite.OUTPUT_DOT_COTANGENT
    )  # [TK, 1]
    moe_w13_cotangents = []
    moe_w2_cotangents = []
    for chunk_index in reversed(range(layout.chunks)):
        dispatch_params, return_params, expert_mlp_residuals = chunk_residuals[chunk_index]
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
            out_dispatch_init = _transport_buffer(
                layout.chunk_capacity,
                hidden_dim,
                out_cotangent.dtype,
                return_params.recv_sizes,
                site=_TransportBufferSite.RETURN_COTANGENT,
            )
            out_dispatch_cotangent = _reverse_ragged_a2a(returned_cotangent, out_dispatch_init, return_params)
            x_dispatch_cotangent, w13_cotangent, w2_cotangent, row_output_dot = layout.expert_mlp.backward(
                expert_mlp_residuals, out_dispatch_cotangent
            )
            # Chunks read disjoint rows of the sorted buffer, so each writes its rows of one
            # shared cotangent buffer, and the rows no chunk writes are the dropped assignments.
            dispatch_cotangent = _reverse_ragged_a2a(x_dispatch_cotangent, dispatch_cotangent, dispatch_params)
            # Each row's <y, dy> travels back to its assignment's sorted position like y did.
            output_dot = jax.lax.ragged_all_to_all(
                row_output_dot[:, None], output_dot, *return_params, axis_name="expert"
            )
            moe_w13_cotangents.append(w13_cotangent)
            moe_w2_cotangents.append(w2_cotangent)

    with jax.named_scope("combine"):
        positions = _invert_permutation(routing.sorted_indices)
        assignment_output_dot = output_dot[:, 0][positions].reshape(weights.shape)
        # d/dw of w * <dout, y> is <dout, y> = <dy, y> / w. Dropped and padding assignments carry
        # no weight and get a zero gradient without the division.
        divisible = routing.accepted & (weights_f32 != 0)
        weights_cotangent = jnp.where(
            divisible, assignment_output_dot / jnp.where(divisible, weights_f32, 1), 0
        ).astype(weights.dtype)
    return (
        dispatch_cotangent,
        weights_cotangent,
        jnp.concatenate(moe_w13_cotangents[::-1], axis=0),
        jnp.concatenate(moe_w2_cotangents[::-1], axis=0),
        None,
    )


_routed_experts.defvjp(_routed_experts_fwd, _routed_experts_bwd)


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
        chunk_clipped_group_sizes = tuple(  # [S, E] each
            _clip_receiver_group_sizes(
                jnp.where(chunk_of_expert[None, :] == chunk_index, all_group_sizes, 0),
                local_expert_size=local_experts,
                receiver_capacity=logical_chunk_capacity,
            )
            for chunk_index in range(chunks)
        )
        # Every expert belongs to one chunk, so summing the chunks gives each expert's accepted
        # prefix of this shard's group.
        accepted_group_sizes = sum(clipped[shard_id] for clipped in chunk_clipped_group_sizes)  # [E]
        accepted = _accepted_assignments(flat_selected, sorted_indices, group_sizes, accepted_group_sizes)  # [TK]
        accepted = accepted.reshape(tokens_per_shard, topk)
        sorted_x = _gather_dispatch_rows(x_local, sorted_indices, accepted.astype(jnp.float32), topk)  # [TK, H]

    routing = _ExpertRouting(
        sorted_indices=sorted_indices,
        accepted=accepted,
        group_sizes=group_sizes,
        all_group_sizes=all_group_sizes,
        chunk_clipped_group_sizes=chunk_clipped_group_sizes,
        shard_id=shard_id,
    )
    layout = _ExpertLayout(
        tokens_per_shard=tokens_per_shard,
        topk=topk,
        local_experts=local_experts,
        ep_size=ep_size,
        chunk_experts=chunk_experts,
        chunk_capacity=chunk_capacity,
        expert_mlp=_select_expert_mlp(activation_fn),
    )
    # A dropped or padding assignment gets weight zero, so the combine never reads its unwritten
    # row, and the `where` discards any gradient for it.
    weights = jnp.where(accepted, combine_weights_local, 0)
    out_local = _routed_experts(sorted_x, weights, moe_w13_local, moe_w2_local, routing, layout)

    with jax.named_scope("combine"):
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(accepted_group_sizes, dtype=jnp.int32)
        dropped_total = jax.lax.psum(dropped_local, token_sharding_axes)
    return out_local, CapacityDrops(sender_dropped=dropped_total, receiver_dropped=jnp.zeros_like(dropped_total))
