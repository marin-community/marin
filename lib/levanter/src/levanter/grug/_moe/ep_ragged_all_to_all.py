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
from enum import auto, IntEnum, StrEnum
from typing import NamedTuple, Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from haliax.jax_utils import tree_checkpoint_name
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.common import (
    _assignment_validity,
    _deinterleave_gate_up,
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

    def apply(
        self,
        x_dispatch: Float[Array, "C H"],
        moe_w13_local: Float[Array, "Echunk H I2"],
        moe_w2_local: Float[Array, "Echunk I H"],
        physical_group_sizes: Int[Array, "Echunk"],
        active_group_sizes: Int[Array, "Echunk"],
    ) -> tuple[Float[Array, "C H"], object]:
        """`forward`'s output without the residuals, for a forward pass whose backward recomputes.

        Also returns the value the residuals' last-produced member stands for, so a caller can
        order later work after the same stage as when it waits on `forward`'s residuals.
        """
        ...

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

    def apply(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        out = self._apply(x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes)
        return out, out

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

    @staticmethod
    def _operands(moe_w13_local, moe_w2_local, active_group_sizes):
        """The interleaved gate/up weights and the cumulative active group sizes QuACK's GEMMs take."""
        moe_dim = moe_w2_local.shape[1]
        w13_interleaved = _interleave_gate_up(moe_w13_local, moe_dim)
        cumulative_group_sizes = jnp.concatenate(
            [jnp.zeros((1,), jnp.int32), jnp.cumsum(active_group_sizes).astype(jnp.int32)]
        )
        return w13_interleaved, cumulative_group_sizes

    def forward(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        del physical_group_sizes
        # QuACK and CUTLASS DSL are installed only with the CUDA 13 GPU extra.
        from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad_fwd  # noqa: PLC0415

        w13_interleaved, cumulative_group_sizes = self._operands(moe_w13_local, moe_w2_local, active_group_sizes)
        return _expert_mlp_quack_wgrad_fwd(x_dispatch, w13_interleaved, moe_w2_local, cumulative_group_sizes)

    def apply(self, x_dispatch, moe_w13_local, moe_w2_local, physical_group_sizes, active_group_sizes):
        del physical_group_sizes
        from levanter.grug._moe.sonic_cute import _expert_mlp_quack_apply  # noqa: PLC0415

        w13_interleaved, cumulative_group_sizes = self._operands(moe_w13_local, moe_w2_local, active_group_sizes)
        return _expert_mlp_quack_apply(x_dispatch, w13_interleaved, moe_w2_local, cumulative_group_sizes)

    def backward(self, residuals, cotangent):
        from levanter.grug._moe.sonic_cute import _expert_mlp_quack_wgrad_backward  # noqa: PLC0415

        dx, dw13_interleaved, dw2, output_dot_cotangent = _expert_mlp_quack_wgrad_backward(residuals, cotangent)
        return dx, _deinterleave_gate_up(dw13_interleaved), dw2, output_dot_cotangent


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


@jax.custom_vjp
def forward_barrier(values):
    """``optimization_barrier`` in the forward pass only; cotangents pass straight through.

    A plain barrier's transpose ties the cotangents together too: it would hold one input's
    cotangent until every other one exists, and between a weight gradient's all-reduce and the
    slice back to its FSDP shard it stops XLA from fusing the pair into a reduce-scatter.
    """
    return jax.lax.optimization_barrier(values)


def _forward_barrier_fwd(values):
    return forward_barrier(values), None


def _forward_barrier_bwd(_, cotangents):
    return (cotangents,)


forward_barrier.defvjp(_forward_barrier_fwd, _forward_barrier_bwd)


class _TransportSchedule(StrEnum):
    """How the chunked pipeline orders its transports against its compute."""

    # Data dependencies plus the barrier that keeps one chunk's buffers live at a time. Even this
    # order leaves a chunk's return and the next chunk's dispatch free to run together: only XLA's
    # latency-hiding scheduler with one collective in flight, or synchronous collectives, keeps
    # two device-initiated transports from being in flight at once, which can deadlock.
    DEPENDENCY = auto()
    # Also ties that put compute beside every transport under the latency-hiding scheduler with
    # one collective in flight. They free more transports from the compute between them, so the
    # program needs that scheduler.
    LATENCY_HIDING = auto()


class RoutingWeightGradient(StrEnum):
    """How the ragged expert-parallel backend differentiates the combine weights."""

    # <dout, y> for each assignment, read from the expert outputs, which the backward keeps.
    EXACT = auto()
    # <h, dh> / w for each expert row, from the expert MLP's own backward, where dh is the cotangent
    # of the activation h and the row's output cotangent is w * dout. The backward needs neither the
    # expert outputs nor their return transport, but the gradient is zero or inexact wherever
    # w * dout rounds to zero in the cotangent dtype: at w = 0, and in float16 also for normal weights
    # times small output cotangents.
    EXPERT_SIDE = auto()


# Small values the dispatch-overlap schedule computes before the first dispatch and ties to it. A
# recompute for the backward replays that tie, so it needs them again; saving them keeps their
# collectives out of the recompute.
DISPATCH_OVERLAP_SAVE_NAME = "grug_moe_dispatch_overlap_saved"


class DispatchOverlap(NamedTuple):
    """Caller compute that the ragged MoE runs while its first chunk's dispatch is in flight.

    ``fn(params, x)`` runs on this shard's tokens once the dispatch buffer is ready, and the first
    chunk's expert MLP waits for its result. With one collective in flight at a time, this puts
    the work beside the dispatch, which otherwise has no independent compute, in the forward and
    in the backward's recompute alike. ``fn`` must return a pytree of ``[Tlocal, ...]`` arrays.

    Passing one also orders the rest of the pipeline for XLA's latency-hiding scheduler with one
    collective in flight (`_TransportSchedule.LATENCY_HIDING`); the program needs that scheduler.
    It also moves the layer's capacity-drop count ahead of the first dispatch and tags it
    `DISPATCH_OVERLAP_SAVE_NAME`: a remat policy should save that name, or the recompute for the
    backward reruns the count's all-reduce to replay the ordering.
    """

    fn: Callable[[object, jax.Array], object]
    params: object
    x: jax.Array


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
    schedule: _TransportSchedule
    weight_gradient: RoutingWeightGradient

    @property
    def chunks(self) -> int:
        return self.local_experts // self.chunk_experts


class _ChunkPlan(NamedTuple):
    """One chunk's transfers and group sizes.

    ``_expert_granular_a2a_params`` builds the dispatch and return parameters as mirror
    transfers: each is the other's transpose (JAX's ragged_all_to_all transpose rule derives the
    same offsets with two offset all-to-alls), so the backward sends cotangents back along the
    mirror transfer's parameters.
    """

    dispatch_params: ExpertA2aParams
    return_params: ExpertA2aParams
    physical_group_sizes: Int[Array, "Echunk"]
    active_group_sizes: Int[Array, "Echunk"]


class _ChunkResiduals(NamedTuple):
    plan: _ChunkPlan
    expert_mlp: tuple[jax.Array, ...]


def _chunk_plans(routing: _ExpertRouting, layout: _ExpertLayout) -> tuple[_ChunkPlan, ...]:
    plans = []
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
            plans.append(_ChunkPlan(dispatch_params, return_params, physical_group_sizes, active_group_sizes))
    return tuple(plans)


def _dispatch_chunk(sorted_x: Float[Array, "TK H"], plan: _ChunkPlan, layout: _ExpertLayout) -> Float[Array, "C H"]:
    # Accepted rows are the prefix of each unclipped expert group and receiver offsets pack
    # arrivals expert-major, so the received buffer feeds the grouped MLP directly: no sender
    # compaction and no receiver-side permute.
    dispatch_init = _transport_buffer(  # [C, H]
        layout.chunk_capacity,
        sorted_x.shape[1],
        sorted_x.dtype,
        plan.dispatch_params.send_sizes,
        site=_TransportBufferSite.DISPATCH_OUTPUT,
    )
    return jax.lax.ragged_all_to_all(sorted_x, dispatch_init, *plan.dispatch_params, axis_name="expert")


# One chunk's expert MLP: its output, its residuals, and the value the next chunk's dispatch waits on.
_ChunkMlpCall = Callable[..., tuple[jax.Array, tuple[jax.Array, ...], object]]


def _forward_with_residuals(expert_mlp: _ExpertMlp) -> _ChunkMlpCall:
    def call(*args):
        out, residuals = expert_mlp.forward(*args)
        return out, residuals, residuals

    return call


def _forward_without_residuals(expert_mlp: _ExpertMlp) -> _ChunkMlpCall:
    def call(*args):
        out, ready = expert_mlp.apply(*args)
        return out, (), ready

    return call


def _routed_experts_forward(
    sorted_x: Float[Array, "TK H"],
    weights: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    staged: tuple[jax.Array, ...],
    routing: _ExpertRouting,
    layout: _ExpertLayout,
    chunk_mlp: _ChunkMlpCall,
) -> tuple[Float[Array, "Tlocal H"], tuple[jax.Array, ...], tuple[_ChunkResiduals, ...], Float[Array, "TK H"]]:
    assignments, hidden_dim = sorted_x.shape
    plans = _chunk_plans(routing, layout)
    # Rows no chunk writes are the dropped assignments, which the combine skips.
    returned = _transport_buffer(
        assignments, hidden_dim, sorted_x.dtype, routing.group_sizes, site=_TransportBufferSite.RETURN_OUTPUT
    )  # [TK, H]
    chunk_residuals = []
    previous_dispatch = None
    previous_ready = None
    for chunk_index, plan in enumerate(plans):
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
            source = sorted_x
            if previous_ready is not None and layout.schedule == _TransportSchedule.DEPENDENCY:
                # Serialize the chunks. Without this barrier, the scheduler can start the dispatch
                # of every chunk at the same time, and the chunk buffers are all live at once,
                # which is the memory the chunks exist to save. The barrier waits for the previous
                # chunk's backward inputs rather than its return transport: the backward does not
                # need the return, so a recompute for the backward drops it and must not be held
                # to it. (Dispatching chunk c+1 during chunk c's MLP measured 3 ms per layer
                # slower in a rematted four-GPU layer scan.)
                source, _ = jax.lax.optimization_barrier((sorted_x, previous_ready))
            elif previous_ready is not None:
                # Serialize the chunks' dispatches, as above, but on the previous chunk's dispatch
                # rather than its MLP: the backward's recompute has no other compute to put beside
                # this dispatch than the previous chunk's MLP. The dispatch also waits for the staged
                # overlap work, so that work runs beside the first dispatch rather than this one.
                source, _ = jax.lax.optimization_barrier((sorted_x, (previous_dispatch, staged)))
            x_dispatch = _dispatch_chunk(source, plan, layout)  # [C, H]
            previous_dispatch = x_dispatch
            if chunk_index == 0 and staged:
                # The first chunk's MLP waits for the staged overlap work, so the scheduler runs that
                # work while the dispatch is in flight rather than leaving the dispatch bare.
                x_dispatch, staged = jax.lax.optimization_barrier((x_dispatch, staged))
            if previous_ready is not None and layout.schedule == _TransportSchedule.LATENCY_HIDING:
                # The chunks' MLPs run in order, so the previous chunk's MLP is the compute beside
                # this chunk's dispatch, also in the backward's recompute.
                x_dispatch, _ = jax.lax.optimization_barrier((x_dispatch, previous_ready))
            experts = slice(chunk_index * layout.chunk_experts, (chunk_index + 1) * layout.chunk_experts)
            out_dispatch, expert_mlp_residuals, previous_ready = chunk_mlp(  # [C, H]
                x_dispatch,
                moe_w13_local[experts],
                moe_w2_local[experts],
                plan.physical_group_sizes,
                plan.active_group_sizes,
            )
            # The mirror of dispatch: valid prefixes land back at unclipped sorted positions.
            # Chaining every chunk through one output buffer composes the disjoint writes, with
            # no expansion step.
            returned = jax.lax.ragged_all_to_all(out_dispatch, returned, *plan.return_params, axis_name="expert")
            chunk_residuals.append(_ChunkResiduals(plan, expert_mlp_residuals))

    with jax.named_scope("combine"):
        out = _unpermute_from_global_expert(
            returned,
            routing.sorted_indices,
            weights,
            routing.accepted,
            tokens_per_shard=layout.tokens_per_shard,
            topk=layout.topk,
        ).astype(sorted_x.dtype)
    return out, staged, tuple(chunk_residuals), returned


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def _routed_experts(
    sorted_x: Float[Array, "TK H"],
    weights: Float[Array, "Tlocal K"],
    moe_w13_local: Float[Array, "Elocal H I2"],
    moe_w2_local: Float[Array, "Elocal I H"],
    staged: tuple[jax.Array, ...],
    routing: _ExpertRouting,
    layout: _ExpertLayout,
) -> tuple[Float[Array, "Tlocal H"], tuple[jax.Array, ...]]:
    """Dispatch the sorted rows to their experts, run the expert MLP, return and combine.

    ``weights`` must be zero for every assignment that ``routing.accepted`` marks as dropped.

    ``layout.weight_gradient`` selects the routing-weight gradient `RoutingWeightGradient`. EXACT
    keeps the returned expert outputs ``y`` for the backward and differentiates the combine. With
    EXPERT_SIDE, each output row is ``y = h @ W2`` and its cotangent there is ``dy = w * dout``, so
    ``<dout, y> = <h, dh> / w`` with ``dh = dy @ W2^T``, which the expert MLP backward computes
    anyway. That backward reads neither ``y``, nor the return transport, nor the combined output,
    and when the combined output is saved for the backward, a recompute for the backward runs only
    the dispatch and the gate/up projection. But where ``w * dout`` rounds to zero in the cotangent
    dtype, the row's cotangent carries no information, and the weight gradient is zero or inexact.

    ``staged`` (possibly empty) comes back unchanged, but only once the first chunk's dispatch has
    landed, and the first chunk's MLP waits for it; see `DispatchOverlap`. Its cotangent passes
    straight through.

    A forward pass whose residuals are recomputed for the backward rather than kept runs this
    primal instead of the fwd rule (``optimize_remat``), so it skips the residuals' stores, such
    as the gate/up pre-activations.
    """
    out, staged, _residuals, _returned = _routed_experts_forward(
        sorted_x,
        weights,
        moe_w13_local,
        moe_w2_local,
        staged,
        routing,
        layout,
        _forward_without_residuals(layout.expert_mlp),
    )
    return out, staged


def _routed_experts_fwd(sorted_x, weights, moe_w13_local, moe_w2_local, staged, routing, layout):
    out, staged, chunk_residuals, returned = _routed_experts_forward(
        sorted_x,
        weights,
        moe_w13_local,
        moe_w2_local,
        staged,
        routing,
        layout,
        _forward_with_residuals(layout.expert_mlp),
    )
    if layout.weight_gradient == RoutingWeightGradient.EXPERT_SIDE:
        returned = None
    return (out, staged), (weights, routing, chunk_residuals, returned)


def _routed_experts_bwd(layout, residuals, cotangents):
    out_cotangent, staged_cotangent = cotangents
    weights, routing, chunk_residuals, returned = residuals
    expert_side = layout.weight_gradient == RoutingWeightGradient.EXPERT_SIDE
    assignments = routing.sorted_indices.shape[0]
    hidden_dim = out_cotangent.shape[1]
    weights_f32 = weights.astype(jnp.float32)
    positions = jnp.argsort(routing.sorted_indices)
    with jax.named_scope("combine"):
        # The combine's transpose: every accepted row's cotangent is its token's output cotangent
        # times its routing weight, rounded once, as the fused gather-sum's backward rounds it.
        # The transport reads only the accepted rows.
        if sonic_gather_sum_available():
            returned_cotangent = sonic_scatter_rows(  # [TK, H]
                out_cotangent,
                positions.reshape(weights.shape),
                routing.accepted,
                rows=assignments,
                weights=weights_f32,
            )
        else:
            sorted_weights = weights_f32.reshape(-1)[routing.sorted_indices]
            token_cotangent = out_cotangent[routing.sorted_indices // layout.topk].astype(jnp.float32)
            returned_cotangent = (token_cotangent * sorted_weights[:, None]).astype(out_cotangent.dtype)
    latency_hiding = layout.schedule == _TransportSchedule.LATENCY_HIDING
    if latency_hiding:
        # Send the output cotangents only once the first chunk's MLP inputs exist. In a recompute for
        # the backward, that MLP is then the compute beside the second chunk's dispatch, and the
        # second chunk's MLP the compute beside the first cotangent dispatch.
        returned_cotangent, _ = jax.lax.optimization_barrier((returned_cotangent, chunk_residuals[0].expert_mlp))

    dispatch_cotangent = _transport_buffer(
        assignments,
        hidden_dim,
        out_cotangent.dtype,
        routing.group_sizes,
        site=_TransportBufferSite.DISPATCH_COTANGENT,
    )  # [TK, H]
    if expert_side:
        output_dot = _transport_buffer(
            assignments, 1, jnp.float32, routing.group_sizes, site=_TransportBufferSite.OUTPUT_DOT_COTANGENT
        )  # [TK, 1]
    moe_w13_cotangents = []
    moe_w2_cotangents = []
    previous_row_output_dot = None
    for chunk_index in reversed(range(layout.chunks)):
        plan, expert_mlp_residuals = chunk_residuals[chunk_index]
        with jax.named_scope(f"moe_chunk_{chunk_index}"):
            out_dispatch_init = _transport_buffer(
                layout.chunk_capacity,
                hidden_dim,
                out_cotangent.dtype,
                plan.return_params.recv_sizes,
                site=_TransportBufferSite.RETURN_COTANGENT,
            )
            chunk_cotangent = returned_cotangent
            if latency_hiding and previous_row_output_dot is not None:
                # This chunk's cotangent dispatch waits for the previous chunk's first backward GEMM,
                # so the rest of that chunk's backward is the compute beside it, rather than nothing.
                chunk_cotangent, _ = jax.lax.optimization_barrier((returned_cotangent, previous_row_output_dot))
            # Each transport's cotangent travels back along its mirror transfer.
            out_dispatch_cotangent = jax.lax.ragged_all_to_all(
                chunk_cotangent, out_dispatch_init, *plan.dispatch_params, axis_name="expert"
            )
            x_dispatch_cotangent, w13_cotangent, w2_cotangent, row_output_dot = layout.expert_mlp.backward(
                expert_mlp_residuals, out_dispatch_cotangent
            )
            sent_row_output_dot = row_output_dot
            if latency_hiding:
                # The row dots leave after the input cotangent's GEMM, so that GEMM is the compute
                # beside the next chunk's cotangent dispatch, which waits only for this chunk's first
                # GEMM.
                sent_row_output_dot, x_dispatch_cotangent = jax.lax.optimization_barrier(
                    (row_output_dot, x_dispatch_cotangent)
                )
            # Chunks read disjoint rows of the sorted buffer, so each writes its rows of one
            # shared cotangent buffer, and the rows no chunk writes are the dropped assignments.
            dispatch_cotangent = jax.lax.ragged_all_to_all(
                x_dispatch_cotangent, dispatch_cotangent, *plan.return_params, axis_name="expert"
            )
            if expert_side:
                # Each row's <y, dy> travels back to its assignment's sorted position like y did.
                output_dot = jax.lax.ragged_all_to_all(
                    sent_row_output_dot[:, None], output_dot, *plan.return_params, axis_name="expert"
                )
            previous_row_output_dot = row_output_dot
            moe_w13_cotangents.append(w13_cotangent)
            moe_w2_cotangents.append(w2_cotangent)

    with jax.named_scope("combine"):
        if expert_side:
            assignment_output_dot = output_dot[:, 0][positions].reshape(weights.shape)
            # d/dw of w * <dout, y> is <dout, y> = <dy, y> / w. Dropped and padding assignments
            # carry no weight and get a zero gradient without the division.
            divisible = routing.accepted & (weights_f32 != 0)
            weights_cotangent = jnp.where(divisible, assignment_output_dot / jnp.where(divisible, weights_f32, 1), 0)
        else:
            combined, combine_vjp = jax.vjp(
                lambda w: _unpermute_from_global_expert(
                    returned,
                    routing.sorted_indices,
                    w,
                    routing.accepted,
                    tokens_per_shard=layout.tokens_per_shard,
                    topk=layout.topk,
                ),
                weights_f32,
            )
            (weights_cotangent,) = combine_vjp(out_cotangent.astype(combined.dtype))
            # Dropped rows were never written, and their gradient may have read anything.
            weights_cotangent = jnp.where(routing.accepted, weights_cotangent, 0)
        weights_cotangent = weights_cotangent.astype(weights.dtype)
    return (
        dispatch_cotangent,
        weights_cotangent,
        jnp.concatenate(moe_w13_cotangents[::-1], axis=0),
        jnp.concatenate(moe_w2_cotangents[::-1], axis=0),
        staged_cotangent,
        None,
    )


_routed_experts.defvjp(_routed_experts_fwd, _routed_experts_bwd, optimize_remat=True)


def _dropped_total(
    group_sizes: Int[Array, "E"], accepted_group_sizes: Int[Array, "E"], token_sharding_axes: tuple[str, ...]
) -> Int[Array, ""]:
    with jax.named_scope("combine"):
        dropped_local = jnp.sum(group_sizes, dtype=jnp.int32) - jnp.sum(accepted_group_sizes, dtype=jnp.int32)
        return jax.lax.psum(dropped_local, token_sharding_axes)


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
    routing_weight_gradient: RoutingWeightGradient,
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops]:
    out_local, drops, _ = _ragged_a2a_local(
        x_local,
        selected_experts_local,
        combine_weights_local,
        token_valid_local,
        moe_w13_local,
        moe_w2_local,
        activation_fn=activation_fn,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
        token_sharding_axes=token_sharding_axes,
        routing_weight_gradient=routing_weight_gradient,
        overlap=None,
    )
    return out_local, drops


def _ragged_a2a_local(
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
    routing_weight_gradient: RoutingWeightGradient,
    overlap: DispatchOverlap | None,
) -> tuple[Float[Array, "Tlocal H"], CapacityDrops, object]:
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
        schedule=_TransportSchedule.DEPENDENCY if overlap is None else _TransportSchedule.LATENCY_HIDING,
        weight_gradient=routing_weight_gradient,
    )
    dropped_total = None
    if overlap is not None:
        # Every collective of the layer other than the transports completes before the first
        # dispatch: with one collective in flight, one issued between the transports takes the
        # slot, and the scheduler spends compute meant for a transport on it.
        dropped_total = tree_checkpoint_name(
            _dropped_total(group_sizes, accepted_group_sizes, token_sharding_axes), DISPATCH_OVERLAP_SAVE_NAME
        )
        sorted_x, dropped_total = forward_barrier((sorted_x, dropped_total))
    # A dropped or padding assignment gets weight zero, so the combine never reads its unwritten
    # row, and the `where` discards any gradient for it.
    weights = jnp.where(accepted, combine_weights_local, 0)
    staged: tuple[jax.Array, ...] = ()
    staged_tree = None
    if overlap is not None:
        # The overlap work starts no earlier than the dispatch can: it waits for the dispatch buffer
        # and the clipped group sizes the transfer sizes derive from.
        (sorted_x, chunk_clipped_group_sizes), overlap_x = forward_barrier(
            ((sorted_x, routing.chunk_clipped_group_sizes), overlap.x)
        )
        routing = routing._replace(chunk_clipped_group_sizes=chunk_clipped_group_sizes)
        staged_leaves, staged_tree = jax.tree.flatten(overlap.fn(overlap.params, overlap_x))
        staged = tuple(staged_leaves)
    out_local, staged = _routed_experts(sorted_x, weights, moe_w13_local, moe_w2_local, staged, routing, layout)
    if dropped_total is None:
        dropped_total = _dropped_total(group_sizes, accepted_group_sizes, token_sharding_axes)
    drops = CapacityDrops(sender_dropped=dropped_total, receiver_dropped=jnp.zeros_like(dropped_total))
    if staged_tree is None:
        return out_local, drops, None
    return out_local, drops, jax.tree.unflatten(staged_tree, staged)
