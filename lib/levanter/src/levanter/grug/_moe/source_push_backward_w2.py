# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Package-private W2/SwiGLU backward helpers for source-push MoE MLP.

These helpers cover backward steps 2-5 after ``dy`` has already been routed into
destination-local expert-major rows. They intentionally stop before W13
backward and dx return/combine so the same boundary can later be replaced by an
MGPU kernel inside the MLP-level custom VJP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from jax import shard_map
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as mgpu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float, Int

from levanter.grug._moe.source_push_plan import (
    SOURCE_PUSH_MESH_AXIS,
    _source_push_out_sharding,
)
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed

SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU = "reference_matmul_pallas_mgpu_swiglu"
SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_FUSED = "pallas_mgpu_fused"
SourcePushW2BackwardImplementation: TypeAlias = Literal[
    "reference",
    "reference_matmul_pallas_mgpu_swiglu",
    "pallas_mgpu_fused",
]

SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushW2MatmulBackwardImplementation: TypeAlias = Literal["reference", "pallas_mgpu"]

SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE = "reference"
SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU = "pallas_mgpu"
SourcePushW2SwiGLUBackwardImplementation: TypeAlias = Literal["reference", "pallas_mgpu"]

DEFAULT_SOURCE_PUSH_W2_SWIGLU_BACKWARD_ROW_BLOCK = 1
DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_ROW_BLOCK = 128
DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_INTERMEDIATE_BLOCK = 64
DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_HIDDEN_BLOCK = 128
MIN_MOSAIC_INT32_TRANSFER_ELEMENTS = 128
MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK = MIN_MOSAIC_INT32_TRANSFER_ELEMENTS
SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS = mgpu.LoweringSemantics.Lane
W2_WGMMA_SWIZZLE_BYTES = 128
W2_WGMMA_TILE_M = 8
SOURCE_PUSH_W2_MATMUL_BACKWARD_ROW_BLOCK_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)
SOURCE_PUSH_W2_MATMUL_BACKWARD_INTERMEDIATE_BLOCK_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
SOURCE_PUSH_W2_MATMUL_BACKWARD_HIDDEN_BLOCK_CANDIDATES = (128, 64, 32, 16, 8, 4, 2, 1)


@dataclass(frozen=True, slots=True)
class SourcePushW2SwiGLUBackwardPallasBlockSizes:
    """Tile sizes for the W2-backward SwiGLU/router derivative kernel."""

    row_block: int = DEFAULT_SOURCE_PUSH_W2_SWIGLU_BACKWARD_ROW_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushW2SwiGLUBackwardPallasBlockSizes":
        return cls()


@dataclass(frozen=True, slots=True)
class SourcePushW2MatmulBackwardPallasBlockSizes:
    """Tile sizes for the destination-local W2 matmul backward kernels."""

    row_block: int = DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_ROW_BLOCK
    intermediate_block: int = DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_INTERMEDIATE_BLOCK
    hidden_block: int = DEFAULT_SOURCE_PUSH_W2_MATMUL_BACKWARD_HIDDEN_BLOCK

    @classmethod
    def get_default(cls) -> "SourcePushW2MatmulBackwardPallasBlockSizes":
        return cls()


class _SourcePushW2BackwardOutput(NamedTuple):
    """Gradients produced by backward steps 2-5 from saved H."""

    d_h: Float[Array, "... twoI"]
    d_route_weight: Float[Array, "..."]
    dw2: Float[Array, "Dst E I D"]


class _SourcePushW2MatmulBackwardOutput(NamedTuple):
    """W2-matmul backward outputs before the SwiGLU/router derivative."""

    d_weighted_activation: Float[Array, "Dst E C I"]
    dw2: Float[Array, "Dst E I D"]


class _SourcePushW2SwiGLUBackwardOutput(NamedTuple):
    """SwiGLU and route-weight derivatives from ``d_weighted_activation``."""

    d_h: Float[Array, "Dst E C twoI"]
    d_route_weight: Float[Array, "Dst E C"]


def _source_push_w2_backward_for_expert_block_reference(
    h: Float[Array, "Dst C twoI"],
    route_weight: Float[Array, "Dst C"],
    dy: Float[Array, "Dst C D"],
    w2: Float[Array, "Dst I D"],
    valid: Bool[Array, "Dst C"] | Float[Array, "Dst C"],
) -> tuple[
    Float[Array, "Dst C twoI"],
    Float[Array, "Dst C"],
    Float[Array, "Dst I D"],
]:
    """Reference W2, route-weight, and SwiGLU backward for one local expert.

    Inputs are destination-local rows for a single local expert. Padding rows are
    ignored with ``valid`` so callers may pass source-padded expert-major spans
    without sanitizing invalid row payloads first.
    """

    valid_f = valid.astype(jnp.float32)
    h = h.astype(jnp.float32) * valid_f[..., None]
    route_weight = route_weight.astype(jnp.float32) * valid_f
    dy = dy.astype(jnp.float32) * valid_f[..., None]
    w2 = w2.astype(jnp.float32)

    gate, up = jnp.split(h, 2, axis=-1)
    silu_gate = jax.nn.silu(gate)
    activation = silu_gate * up
    weighted_activation = activation * route_weight[..., None]

    d_weighted_activation = jnp.einsum("dch,dih->dci", dy, w2)
    d_route_weight = jnp.sum(d_weighted_activation * activation, axis=-1) * valid_f
    dw2 = jnp.einsum("dci,dch->dih", weighted_activation, dy)

    d_activation = d_weighted_activation * route_weight[..., None]
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu_gate
    d_up = d_activation * silu_gate
    d_h = jnp.concatenate([d_gate, d_up], axis=-1) * valid_f[..., None]
    return d_h, d_route_weight, dw2


def _source_push_w2_backward_expert_blocks_reference(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> _SourcePushW2BackwardOutput:
    """Reference W2/SwiGLU backward over destination-local expert blocks."""

    _validate_expert_block_shapes(h, route_weight, dy, w2, valid)
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h,
        route_weight,
        valid,
    )
    matmul_output = _source_push_w2_matmul_backward_reference(weighted_activation, dy, w2, valid)
    swiglu_output = _source_push_w2_swiglu_backward_reference(
        h,
        route_weight,
        matmul_output.d_weighted_activation,
        valid,
    )
    return _SourcePushW2BackwardOutput(
        d_h=swiglu_output.d_h,
        d_route_weight=swiglu_output.d_route_weight,
        dw2=matmul_output.dw2,
    )


def _source_push_w2_backward_expert_blocks_pallas_mgpu_fused(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2BackwardOutput:
    """Fused W2 matmul/SwiGLU backward that avoids materializing d_weighted_activation."""

    if interpret:
        return _source_push_w2_backward_expert_blocks_reference(h, route_weight, dy, w2, valid)
    if jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU fused W2 backward requires a GPU backend; use the reference on CPU")
    if mesh is None:
        raise ValueError("Pallas/MGPU fused W2 backward requires a mesh so local-rank mgpu.kernel calls are sharded")
    block_sizes = SourcePushW2MatmulBackwardPallasBlockSizes.get_default()
    original_rows = h.shape[2]
    h, route_weight, dy, valid = _pad_w2_fused_rows_for_pallas(
        h,
        route_weight,
        dy,
        valid,
        row_multiple=block_sizes.row_block,
    )
    _validate_w2_matmul_pallas_request(
        h[..., : h.shape[-1] // 2],
        dy,
        w2,
        valid,
        block_sizes,
    )
    valid = valid.astype(jnp.float32)
    d_h, d_route_weight = _source_push_w2_dh_route_fused_sharded_mgpu_kernel(
        mesh,
        h,
        route_weight,
        dy.astype(w2.dtype),
        w2,
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
    )
    dw2 = _source_push_w2_dw2_from_h_fused_sharded_mgpu_kernel(
        mesh,
        h,
        route_weight,
        dy.astype(w2.dtype),
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
    )
    return _SourcePushW2BackwardOutput(
        d_h=d_h[:, :, :original_rows, :],
        d_route_weight=d_route_weight[:, :, :original_rows],
        dw2=dw2,
    )


def _pad_w2_fused_rows_for_pallas(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_multiple: int,
) -> tuple[
    Float[Array, "Dst E C twoI"],
    Float[Array, "Dst E C"],
    Float[Array, "Dst E C D"],
    Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
]:
    rows = h.shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return h, route_weight, dy, valid
    h = jnp.pad(h, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    route_weight = jnp.pad(route_weight, ((0, 0), (0, 0), (0, pad_rows)))
    dy = jnp.pad(dy, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad_rows)))
    return h, route_weight, dy, valid


def _source_push_w2_backward_expert_blocks(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    implementation: SourcePushW2BackwardImplementation = SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    matmul_implementation: SourcePushW2MatmulBackwardImplementation | None = None,
    swiglu_implementation: SourcePushW2SwiGLUBackwardImplementation | None = None,
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2BackwardOutput:
    """Package-private implementation boundary for W2 backward steps 2-5.

    ``reference_matmul_pallas_mgpu_swiglu`` is an explicit partial boundary: it
    keeps both W2 matmul gradients in JAX and only lowers the SwiGLU/router
    derivative from ``d_weighted_activation`` to Pallas.
    """

    if matmul_implementation is not None or swiglu_implementation is not None:
        if implementation != SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE:
            raise ValueError("stage-specific W2 backward selectors require implementation='reference'")
        return _source_push_w2_backward_expert_blocks_staged(
            h,
            route_weight,
            dy,
            w2,
            valid,
            matmul_implementation=matmul_implementation or SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
            swiglu_implementation=swiglu_implementation or SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    if implementation == SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE:
        return _source_push_w2_backward_expert_blocks_reference(h, route_weight, dy, w2, valid)
    if implementation == SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_FUSED:
        return _source_push_w2_backward_expert_blocks_pallas_mgpu_fused(
            h,
            route_weight,
            dy,
            w2,
            valid,
            interpret=interpret,
            mesh=mesh,
        )
    if implementation == SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU:
        return _source_push_w2_backward_expert_blocks_staged(
            h,
            route_weight,
            dy,
            w2,
            valid,
            matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
            swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    supported_implementations = (
        SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE_MATMUL_PALLAS_MGPU_SWIGLU,
        SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_FUSED,
    )
    raise ValueError(
        "source-push W2 backward implementation must be one of "
        f"{supported_implementations}, "
        f"got {implementation!r}"
    )


def _source_push_w2_backward_expert_blocks_staged(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    matmul_implementation: SourcePushW2MatmulBackwardImplementation,
    swiglu_implementation: SourcePushW2SwiGLUBackwardImplementation,
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2BackwardOutput:
    """Run W2 backward through independently selectable matmul and SwiGLU stages."""

    _validate_expert_block_shapes(h, route_weight, dy, w2, valid)
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h,
        route_weight,
        valid,
    )
    matmul_output = _source_push_w2_matmul_backward(
        weighted_activation,
        dy,
        w2,
        valid,
        implementation=matmul_implementation,
        interpret=interpret,
        mesh=mesh,
    )
    swiglu_output = _source_push_w2_swiglu_backward(
        h,
        route_weight,
        matmul_output.d_weighted_activation,
        valid,
        implementation=swiglu_implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )
    return _SourcePushW2BackwardOutput(
        d_h=swiglu_output.d_h,
        d_route_weight=swiglu_output.d_route_weight,
        dw2=matmul_output.dw2,
    )


def _source_push_w2_backward_from_flat_h_reference(
    expert_base: Int[Array, "Dst E"],
    h: Float[Array, "Dst rows twoI"],
    route_weight: Float[Array, "Dst rows"],
    dy: Float[Array, "Dst rows D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> _SourcePushW2BackwardOutput:
    """Reference W2/SwiGLU backward for the production flat-H residual layout.

    ``route_weight`` and ``dy`` must already be aligned with flat H rows. The
    returned ``d_h`` and ``d_route_weight`` use that same flat row layout, while
    ``dw2`` remains per destination/local expert.
    """

    return _source_push_w2_backward_from_flat_h(
        expert_base,
        h,
        route_weight,
        dy,
        w2,
        valid,
        implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    )


def _source_push_w2_backward_from_flat_h(
    expert_base: Int[Array, "Dst E"],
    h: Float[Array, "Dst rows twoI"],
    route_weight: Float[Array, "Dst rows"],
    dy: Float[Array, "Dst rows D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    implementation: SourcePushW2BackwardImplementation = SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    matmul_implementation: SourcePushW2MatmulBackwardImplementation | None = None,
    swiglu_implementation: SourcePushW2SwiGLUBackwardImplementation | None = None,
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
    contiguous_expert_gather: bool = False,
) -> _SourcePushW2BackwardOutput:
    """Implementation-selecting W2/SwiGLU backward for flat H rows."""

    _validate_flat_h_shapes(expert_base, h, route_weight, dy, w2, valid)
    if contiguous_expert_gather:
        h_blocks = _gather_flat_rows_by_expert_slice(h, expert_base, valid.shape[-1])
        route_weight_blocks = _gather_flat_rows_by_expert_slice(route_weight, expert_base, valid.shape[-1])
        dy_blocks = _gather_flat_rows_by_expert_slice(dy, expert_base, valid.shape[-1])
    else:
        flat_rows = _expert_flat_rows(expert_base, valid.shape[-1])
        h_blocks = _gather_flat_rows(h, flat_rows, fill_value=0)
        route_weight_blocks = _gather_flat_rows(route_weight, flat_rows, fill_value=0)
        dy_blocks = _gather_flat_rows(dy, flat_rows, fill_value=0)
    valid_blocks = _source_push_w2_valid_blocks_sharded(valid)
    block_output = _source_push_w2_backward_expert_blocks(
        h_blocks,
        route_weight_blocks,
        dy_blocks,
        w2,
        valid_blocks,
        implementation=implementation,
        matmul_implementation=matmul_implementation,
        swiglu_implementation=swiglu_implementation,
        block_sizes=block_sizes,
        interpret=interpret,
        mesh=mesh,
    )

    valid_f = valid_blocks.astype(block_output.d_h.dtype)
    flat_rows = _expert_flat_rows(expert_base, valid.shape[-1])
    dst_index = _dst_indices(expert_base.shape[0], expert_base.shape[1], valid.shape[-1])
    d_h = jnp.zeros(h.shape, dtype=block_output.d_h.dtype)
    d_route_weight = jnp.zeros(route_weight.shape, dtype=block_output.d_route_weight.dtype)
    d_h = d_h.at[dst_index, flat_rows].add(
        block_output.d_h * valid_f[..., None],
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None),
    )
    d_route_weight = d_route_weight.at[dst_index, flat_rows].add(
        block_output.d_route_weight * valid_f,
        out_sharding=_source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None),
    )
    return _SourcePushW2BackwardOutput(d_h=d_h, d_route_weight=d_route_weight, dw2=block_output.dw2)


def _source_push_w2_valid_blocks_sharded(
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> Bool[Array, "Dst E C"] | Float[Array, "Dst E C"]:
    sharding = _source_push_out_sharding(SOURCE_PUSH_MESH_AXIS, None, None)
    if sharding is None:
        return valid
    if jax.sharding.get_abstract_mesh().are_all_axes_explicit:
        return jax.sharding.reshard(valid, sharding)
    return jax.lax.with_sharding_constraint(valid, sharding)


def _source_push_destination_named_sharding(value: Array, ndim: int) -> NamedSharding | None:
    """Return destination-rank sharding using the mesh already attached to ``value``."""

    sharding = getattr(value, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        return None
    if SOURCE_PUSH_MESH_AXIS not in sharding.mesh.axis_names:
        return None
    return NamedSharding(sharding.mesh, P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(ndim - 1))))


def _with_source_push_destination_sharding(value: Array, like: Array | None = None) -> Array:
    """Constrain eager sharded arrays to destination-major source-push layout."""

    sharding = _source_push_destination_named_sharding(value, value.ndim)
    if sharding is None and like is not None:
        sharding = _source_push_destination_named_sharding(like, value.ndim)
    if sharding is None:
        return value
    return jax.device_put(value, sharding)


def _source_push_w2_activation_and_weighted_activation_reference(
    h: Float[Array, "... twoI"],
    route_weight: Float[Array, "..."],
    valid: Bool[Array, "..."] | Float[Array, "..."],
) -> tuple[Float[Array, "... I"], Float[Array, "... I"]]:
    valid_f = valid.astype(jnp.float32)
    h = h.astype(jnp.float32) * valid_f[..., None]
    route_weight = route_weight.astype(jnp.float32) * valid_f
    gate, up = jnp.split(h, 2, axis=-1)
    activation = jax.nn.silu(gate) * up
    weighted_activation = activation * route_weight[..., None]
    return activation, weighted_activation


def _source_push_w2_matmul_backward_reference(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> _SourcePushW2MatmulBackwardOutput:
    """Reference W2 matmul backward: ``dA_weighted`` and ``dw2`` only."""

    valid_f = valid.astype(jnp.float32)
    weighted_activation = weighted_activation.astype(jnp.float32) * valid_f[..., None]
    dy = dy.astype(jnp.float32) * valid_f[..., None]
    w2 = w2.astype(jnp.float32)
    d_weighted_activation = jnp.einsum("dech,deih->deci", dy, w2)
    dw2 = jnp.einsum("deci,dech->deih", weighted_activation, dy)
    return _SourcePushW2MatmulBackwardOutput(d_weighted_activation=d_weighted_activation, dw2=dw2)


def _source_push_w2_matmul_backward(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    implementation: SourcePushW2MatmulBackwardImplementation = SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushW2MatmulBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2MatmulBackwardOutput:
    """Implementation boundary for W2 matmul backward before SwiGLU/router derivatives."""

    if implementation == SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE:
        return _source_push_w2_matmul_backward_reference(weighted_activation, dy, w2, valid)
    if implementation == SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU:
        return _source_push_w2_matmul_backward_pallas_mgpu(
            weighted_activation,
            dy,
            w2,
            valid,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    supported_implementations = (
        SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    )
    raise ValueError(
        "source-push W2 matmul backward implementation must be one of "
        f"{supported_implementations}, "
        f"got {implementation!r}"
    )


def _source_push_w2_matmul_backward_pallas_mgpu(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW2MatmulBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2MatmulBackwardOutput:
    """Destination-local tiled Pallas W2 matmul backward.

    This is intentionally local-only: source-push transport already happened in
    the dy-routing stage, so no peer-id refs or Lane-lowered remote copies are
    needed here.
    """

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU W2 matmul backward requires a GPU backend; use the reference on CPU")
    _validate_w2_matmul_backward_shapes(weighted_activation, dy, w2, valid)
    if interpret:
        return _source_push_w2_matmul_backward_reference(weighted_activation, dy, w2, valid)
    original_rows = weighted_activation.shape[2]
    row_multiple = block_sizes.row_block if block_sizes is not None else MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK
    weighted_activation, dy, valid = _pad_w2_matmul_rows_for_pallas(
        weighted_activation,
        dy,
        valid,
        row_multiple=row_multiple,
    )
    block_sizes = (
        _source_push_w2_matmul_backward_inferred_block_sizes(weighted_activation, dy, w2)
        if block_sizes is None
        else block_sizes
    )
    _validate_w2_matmul_pallas_request(weighted_activation, dy, w2, valid, block_sizes)
    dy_for_wgmma = dy.astype(w2.dtype)
    weighted_activation_for_wgmma = weighted_activation.astype(w2.dtype)
    d_weighted_activation = _source_push_w2_d_weighted_activation_pallas_call(
        dy_for_wgmma,
        w2,
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )
    d_weighted_activation = d_weighted_activation[:, :, :original_rows, :]
    dw2 = _source_push_w2_dw2_pallas_call(
        weighted_activation_for_wgmma,
        dy_for_wgmma,
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=interpret,
        mesh=mesh,
    )
    return _SourcePushW2MatmulBackwardOutput(d_weighted_activation=d_weighted_activation, dw2=dw2)


def _pad_w2_matmul_rows_for_pallas(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_multiple: int,
) -> tuple[Float[Array, "Dst E C I"], Float[Array, "Dst E C D"], Bool[Array, "Dst E C"] | Float[Array, "Dst E C"]]:
    """Pad row axis so Mosaic W2 kernels avoid tiny irregular row tiles."""

    rows = weighted_activation.shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return weighted_activation, dy, valid
    weighted_activation = jnp.pad(weighted_activation, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    dy = jnp.pad(dy, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad_rows)))
    return weighted_activation, dy, valid


def _source_push_w2_d_weighted_activation_pallas_call(
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E C I"]:
    if mesh is not None and not interpret:
        return _source_push_w2_d_weighted_activation_sharded_pallas_call(
            mesh,
            dy,
            w2,
            valid,
            row_block=row_block,
            intermediate_block=intermediate_block,
            hidden_block=hidden_block,
        )

    output_shape = jax.ShapeDtypeStruct(dy.shape[:3] + (w2.shape[-2],), jnp.float32)
    output_zero = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    cost_estimate = _source_push_w2_d_weighted_activation_pallas_cost_estimate(dy, w2, valid, output_shape)
    kernel = _make_source_push_w2_d_weighted_activation_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
    )
    in_specs, out_spec = _source_push_w2_d_weighted_activation_block_specs(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=output_shape,
        grid=(
            dy.shape[0],
            dy.shape[1],
            dy.shape[2] // row_block,
            w2.shape[-2] // intermediate_block,
            dy.shape[-1] // hidden_block,
        ),
        interpret=interpret,
        name="source_push_w2_d_weighted_activation_pallas_mgpu",
        input_output_aliases={2: 0},
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(dy, w2, output_zero)


def _source_push_w2_dw2_pallas_call(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> Float[Array, "Dst E I D"]:
    if mesh is not None and not interpret:
        return _source_push_w2_dw2_sharded_pallas_call(
            mesh,
            weighted_activation,
            dy,
            valid,
            row_block=row_block,
            intermediate_block=intermediate_block,
            hidden_block=hidden_block,
        )

    output_shape = jax.ShapeDtypeStruct(
        weighted_activation.shape[:2] + (weighted_activation.shape[-1], dy.shape[-1]), jnp.float32
    )
    output_zero = jnp.zeros(output_shape.shape, dtype=output_shape.dtype)
    cost_estimate = _source_push_w2_dw2_pallas_cost_estimate(weighted_activation, dy, valid, output_shape)
    kernel = _make_source_push_w2_dw2_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )
    in_specs, out_spec = _source_push_w2_dw2_block_specs(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=output_shape,
        grid=(
            weighted_activation.shape[0],
            weighted_activation.shape[1],
            weighted_activation.shape[-1] // intermediate_block,
            dy.shape[-1] // hidden_block,
            weighted_activation.shape[2] // row_block,
        ),
        interpret=interpret,
        name="source_push_w2_dw2_pallas_mgpu",
        input_output_aliases={2: 0},
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(weighted_activation, dy, output_zero)


def _source_push_w2_d_weighted_activation_sharded_pallas_call(
    mesh: Mesh,
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E C I"]:
    kernel = _make_source_push_w2_d_weighted_activation_mgpu_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        experts_per_rank=dy.shape[1],
        rows=dy.shape[2],
        intermediate_dim=w2.shape[-2],
        hidden_dim=dy.shape[-1],
    )

    def local_fn(
        dy_local: Float[Array, "1 E C D"],
        w2_local: Float[Array, "1 E I D"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E C I"]:
        return kernel(dy_local[0], w2_local[0], valid_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(dy, w2, valid)


def _source_push_w2_dw2_sharded_pallas_call(
    mesh: Mesh,
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E I D"]:
    kernel = _make_source_push_w2_dw2_mgpu_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        experts_per_rank=weighted_activation.shape[1],
        rows=weighted_activation.shape[2],
        intermediate_dim=weighted_activation.shape[-1],
        hidden_dim=dy.shape[-1],
    )

    def local_fn(
        weighted_activation_local: Float[Array, "1 E C I"],
        dy_local: Float[Array, "1 E C D"],
        valid_local: Bool[Array, "1 E C"],
    ) -> Float[Array, "1 E I D"]:
        return kernel(weighted_activation_local[0], dy_local[0], valid_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(weighted_activation, dy, valid)


def _make_source_push_w2_d_weighted_activation_mgpu_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
):
    hidden_tiles = hidden_dim // hidden_block
    intermediate_tiles = intermediate_dim // intermediate_block
    row_tiles = rows // row_block

    def body(
        dy_ref: Float[pl.Ref, "E C D"],
        w2_ref: Float[pl.Ref, "E I D"],
        _valid_ref: Bool[pl.Ref, "E C"],
        d_weighted_activation_ref: Float[pl.Ref, "E C I"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        intermediate_tile = pl.program_id(2)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(dy_smem, w2_smem, ready_barrier) -> None:
                @pl.loop(0, hidden_tiles)
                def _hidden_loop(hidden_tile) -> None:
                    hidden_start = hidden_tile * hidden_block
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w2_ref.at[
                            expert,
                            pl.ds(intermediate_start, intermediate_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        w2_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, dy_smem, mgpu.transpose_ref(w2_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                dy_smem=_w2_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                w2_smem=_w2_wgmma_smem((intermediate_block, hidden_block), w2_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((row_block, intermediate_block)),
        )
        d_weighted_activation_ref[
            expert,
            pl.ds(row_start, row_block),
            pl.ds(intermediate_start, intermediate_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, rows, intermediate_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, intermediate_tiles),
        grid_names=("expert", "row_tile", "intermediate_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_w2_dw2_mgpu_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
):
    row_tiles = rows // row_block
    intermediate_tiles = intermediate_dim // intermediate_block
    hidden_tiles = hidden_dim // hidden_block

    def body(
        weighted_activation_ref: Float[pl.Ref, "E C I"],
        dy_ref: Float[pl.Ref, "E C D"],
        _valid_ref: Bool[pl.Ref, "E C"],
        dw2_ref: Float[pl.Ref, "E I D"],
    ) -> None:
        expert = pl.program_id(0)
        intermediate_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        intermediate_start = intermediate_tile * intermediate_block
        hidden_start = hidden_tile * hidden_block

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(weighted_activation_smem, dy_smem, ready_barrier) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    mgpu.copy_gmem_to_smem(
                        weighted_activation_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(intermediate_start, intermediate_block),
                        ],
                        weighted_activation_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            expert,
                            pl.ds(row_start, row_block),
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
                        ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(weighted_activation_smem, (1, 0)), dy_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                weighted_activation_smem=_w2_wgmma_smem(
                    (row_block, intermediate_block), weighted_activation_ref.dtype
                ),
                dy_smem=_w2_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((intermediate_block, hidden_block)),
        )
        dw2_ref[
            expert,
            pl.ds(intermediate_start, intermediate_block),
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, intermediate_dim, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, intermediate_tiles, hidden_tiles),
        grid_names=("expert", "intermediate_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _source_push_w2_dh_route_fused_sharded_mgpu_kernel(
    mesh: Mesh,
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    kernel = _make_source_push_w2_dh_route_fused_mgpu_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        experts_per_rank=h.shape[1],
        rows=h.shape[2],
        intermediate_dim=w2.shape[-2],
        hidden_dim=dy.shape[-1],
    )

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        route_weight_local: Float[Array, "1 E C"],
        dy_local: Float[Array, "1 E C D"],
        w2_local: Float[Array, "1 E I D"],
        valid_local: Float[Array, "1 E C"],
    ) -> tuple[Float[Array, "1 E C twoI"], Float[Array, "1 E C"]]:
        d_h_local, d_route_partial_local = kernel(
            h_local[0],
            route_weight_local[0],
            dy_local[0],
            w2_local[0],
            valid_local[0],
        )
        d_route_local = jnp.sum(d_route_partial_local, axis=-1)
        return d_h_local[None, ...], d_route_local[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(h, route_weight, dy, w2, valid)


def _source_push_w2_dw2_from_h_fused_sharded_mgpu_kernel(
    mesh: Mesh,
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> Float[Array, "Dst E I D"]:
    kernel = _make_source_push_w2_dw2_from_h_fused_mgpu_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        hidden_block=hidden_block,
        experts_per_rank=h.shape[1],
        rows=h.shape[2],
        intermediate_dim=h.shape[-1] // 2,
        hidden_dim=dy.shape[-1],
    )

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        route_weight_local: Float[Array, "1 E C"],
        dy_local: Float[Array, "1 E C D"],
        valid_local: Float[Array, "1 E C"],
    ) -> Float[Array, "1 E I D"]:
        return kernel(h_local[0], route_weight_local[0], dy_local[0], valid_local[0])[None, ...]

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=P(SOURCE_PUSH_MESH_AXIS, None, None, None),
        check_vma=False,
    )(h, route_weight, dy, valid)


def _make_source_push_w2_dh_route_fused_mgpu_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
):
    hidden_tiles = hidden_dim // hidden_block
    intermediate_tiles = intermediate_dim // intermediate_block
    row_tiles = rows // row_block

    def route_vector_smem(dtype):
        return mgpu.SMEM((row_block,), dtype=dtype)

    def body(
        h_ref: Float[pl.Ref, "E C twoI"],
        route_weight_ref: Float[pl.Ref, "E C"],
        dy_ref: Float[pl.Ref, "E C D"],
        w2_ref: Float[pl.Ref, "E I D"],
        valid_ref: Float[pl.Ref, "E C"],
        d_h_ref: Float[pl.Ref, "E C twoI"],
        d_route_partial_ref: Float[pl.Ref, "E C ITiles"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        intermediate_tile = pl.program_id(2)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block
        row_slice = pl.ds(row_start, row_block)
        intermediate_slice = pl.ds(intermediate_start, intermediate_block)

        def acc_scope(acc_ref) -> None:
            def smem_scope(
                dy_smem,
                w2_smem,
                gate_smem,
                up_smem,
                route_weight_smem,
                valid_smem,
                h_ready_barrier,
                route_ready_barrier,
                matmul_ready_barrier,
            ) -> None:
                zero_row_slice = pl.ds(0, row_block)
                zero_intermediate_slice = pl.ds(0, intermediate_block)
                mgpu.copy_gmem_to_smem(
                    h_ref.at[expert, row_slice, intermediate_slice],
                    gate_smem,
                    h_ready_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    h_ref.at[
                        expert,
                        row_slice,
                        pl.ds(intermediate_dim + intermediate_start, intermediate_block),
                    ],
                    up_smem,
                    h_ready_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    route_weight_ref.at[expert, row_slice],
                    route_weight_smem,
                    route_ready_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    valid_ref.at[expert, row_slice],
                    valid_smem,
                    route_ready_barrier,
                )
                mgpu.barrier_wait(h_ready_barrier)
                mgpu.barrier_wait(route_ready_barrier)

                route_weight_vec = mgpu.load(
                    route_weight_smem,
                    (zero_row_slice,),
                    layout=mgpu.Layout.WGMMA.reduce(1),
                ).astype(jnp.float32)
                valid_vec = mgpu.load(
                    valid_smem,
                    (zero_row_slice,),
                    layout=mgpu.Layout.WGMMA.reduce(1),
                ).astype(jnp.float32)
                route_weight = jax.lax.broadcast_in_dim(
                    route_weight_vec * valid_vec,
                    (row_block, intermediate_block),
                    (0,),
                )
                valid_f = jax.lax.broadcast_in_dim(valid_vec, (row_block, intermediate_block), (0,))

                @pl.loop(0, hidden_tiles)
                def _hidden_loop(hidden_tile) -> None:
                    hidden_start = hidden_tile * hidden_block
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            expert,
                            row_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
                        matmul_ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        w2_ref.at[
                            expert,
                            intermediate_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        w2_smem,
                        matmul_ready_barrier,
                    )
                    mgpu.barrier_wait(matmul_ready_barrier)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, dy_smem, mgpu.transpose_ref(w2_smem, (1, 0)))
                    mgpu.wgmma_wait(0)

                d_weighted_activation = acc_ref[...].astype(jnp.float32)
                gate = mgpu.load(
                    gate_smem,
                    (zero_row_slice, zero_intermediate_slice),
                    layout=mgpu.Layout.WGMMA,
                ).astype(jnp.float32)
                up = mgpu.load(
                    up_smem,
                    (zero_row_slice, zero_intermediate_slice),
                    layout=mgpu.Layout.WGMMA,
                ).astype(jnp.float32)
                silu_gate = jax.nn.silu(gate)
                activation = silu_gate * up
                d_route_partial = jnp.sum(d_weighted_activation * activation * valid_f, axis=-1)
                d_activation = d_weighted_activation * route_weight
                sigmoid_gate = jax.nn.sigmoid(gate)
                d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
                d_gate = d_activation * up * d_silu_gate * valid_f
                d_up = d_activation * silu_gate * valid_f

                d_h_ref[expert, row_slice, intermediate_slice] = d_gate
                d_h_ref[expert, row_slice, pl.ds(intermediate_dim + intermediate_start, intermediate_block)] = d_up
                d_route_partial_ref[expert, row_slice, intermediate_tile] = d_route_partial

            pl.run_scoped(
                smem_scope,
                dy_smem=_w2_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                w2_smem=_w2_wgmma_smem((intermediate_block, hidden_block), w2_ref.dtype),
                gate_smem=_w2_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                up_smem=_w2_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                route_weight_smem=route_vector_smem(route_weight_ref.dtype),
                valid_smem=route_vector_smem(valid_ref.dtype),
                h_ready_barrier=mgpu.Barrier(num_arrivals=2),
                route_ready_barrier=mgpu.Barrier(num_arrivals=2),
                matmul_ready_barrier=mgpu.Barrier(num_arrivals=2),
            )

        pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((row_block, intermediate_block)),
        )

    out_shape = (
        jax.ShapeDtypeStruct((experts_per_rank, rows, 2 * intermediate_dim), jnp.float32),
        jax.ShapeDtypeStruct((experts_per_rank, rows, intermediate_tiles), jnp.float32),
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, intermediate_tiles),
        grid_names=("expert", "row_tile", "intermediate_tile"),
        compiler_params=compiler_params,
    )


def _make_source_push_w2_dw2_from_h_fused_mgpu_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
    hidden_dim: int,
):
    row_tiles = rows // row_block
    intermediate_tiles = intermediate_dim // intermediate_block
    hidden_tiles = hidden_dim // hidden_block

    def route_vector_smem(dtype):
        return mgpu.SMEM((row_block,), dtype=dtype)

    def body(
        h_ref: Float[pl.Ref, "E C twoI"],
        route_weight_ref: Float[pl.Ref, "E C"],
        dy_ref: Float[pl.Ref, "E C D"],
        valid_ref: Float[pl.Ref, "E C"],
        dw2_ref: Float[pl.Ref, "E I D"],
    ) -> None:
        expert = pl.program_id(0)
        intermediate_tile = pl.program_id(1)
        hidden_tile = pl.program_id(2)
        intermediate_start = intermediate_tile * intermediate_block
        hidden_start = hidden_tile * hidden_block
        intermediate_slice = pl.ds(intermediate_start, intermediate_block)

        def acc_scope(acc_ref) -> jax.Array:
            def smem_scope(
                gate_smem,
                up_smem,
                activation_smem,
                dy_smem,
                route_weight_smem,
                valid_smem,
                ready_barrier,
                route_ready_barrier,
            ) -> None:
                @pl.loop(0, row_tiles)
                def _row_loop(row_tile) -> None:
                    row_start = row_tile * row_block
                    row_slice = pl.ds(row_start, row_block)
                    zero_row_slice = pl.ds(0, row_block)
                    mgpu.copy_gmem_to_smem(
                        h_ref.at[
                            expert,
                            row_slice,
                            intermediate_slice,
                        ],
                        gate_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        h_ref.at[
                            expert,
                            row_slice,
                            pl.ds(intermediate_dim + intermediate_start, intermediate_block),
                        ],
                        up_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        dy_ref.at[
                            expert,
                            row_slice,
                            pl.ds(hidden_start, hidden_block),
                        ],
                        dy_smem,
                        ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        route_weight_ref.at[expert, row_slice],
                        route_weight_smem,
                        route_ready_barrier,
                    )
                    mgpu.copy_gmem_to_smem(
                        valid_ref.at[expert, row_slice],
                        valid_smem,
                        route_ready_barrier,
                    )
                    mgpu.barrier_wait(ready_barrier)
                    mgpu.barrier_wait(route_ready_barrier)
                    route_weight_vec = mgpu.load(
                        route_weight_smem,
                        (zero_row_slice,),
                        layout=mgpu.Layout.WGMMA.reduce(1),
                    ).astype(jnp.float32)
                    valid_vec = mgpu.load(
                        valid_smem,
                        (zero_row_slice,),
                        layout=mgpu.Layout.WGMMA.reduce(1),
                    ).astype(jnp.float32)
                    route_weight = jax.lax.broadcast_in_dim(
                        route_weight_vec * valid_vec,
                        (row_block, intermediate_block),
                        (0,),
                    )

                    activation_smem[:, :] = (
                        jax.nn.silu(gate_smem[:, :].astype(jnp.float32))
                        * up_smem[:, :].astype(jnp.float32)
                        * route_weight
                    ).astype(activation_smem.dtype)
                    mgpu.commit_smem()
                    mgpu.wgmma(acc_ref, mgpu.transpose_ref(activation_smem, (1, 0)), dy_smem)
                    mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                gate_smem=_w2_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                up_smem=_w2_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                activation_smem=_w2_wgmma_smem((row_block, intermediate_block), h_ref.dtype),
                dy_smem=_w2_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                route_weight_smem=route_vector_smem(route_weight_ref.dtype),
                valid_smem=route_vector_smem(valid_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=3),
                route_ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        output = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((intermediate_block, hidden_block)),
        )
        dw2_ref[
            expert,
            intermediate_slice,
            pl.ds(hidden_start, hidden_block),
        ] = output

    out_shape = jax.ShapeDtypeStruct((experts_per_rank, intermediate_dim, hidden_dim), jnp.float32)
    compiler_params = mgpu.CompilerParams(lowering_semantics=SOURCE_PUSH_W2_MATMUL_LOWERING_SEMANTICS)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, intermediate_tiles, hidden_tiles),
        grid_names=("expert", "intermediate_tile", "hidden_tile"),
        compiler_params=compiler_params,
    )


def _source_push_w2_d_weighted_activation_block_specs(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    dy_spec = pl.BlockSpec(
        (None, None, row_block, hidden_block),
        lambda dst, expert, row_tile, _intermediate_tile, hidden_tile: (dst, expert, row_tile, hidden_tile),
    )
    w2_spec = pl.BlockSpec(
        (None, None, intermediate_block, hidden_block),
        lambda dst, expert, _row_tile, intermediate_tile, hidden_tile: (dst, expert, intermediate_tile, hidden_tile),
    )
    zero_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile, _hidden_tile: (dst, expert, row_tile, intermediate_tile),
    )
    out_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile, _hidden_tile: (dst, expert, row_tile, intermediate_tile),
    )
    return (dy_spec, w2_spec, zero_spec), out_spec


def _source_push_w2_dw2_block_specs(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], pl.BlockSpec]:
    weighted_activation_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, intermediate_tile, _hidden_tile, row_tile: (dst, expert, row_tile, intermediate_tile),
    )
    dy_spec = pl.BlockSpec(
        (None, None, row_block, hidden_block),
        lambda dst, expert, _intermediate_tile, hidden_tile, row_tile: (dst, expert, row_tile, hidden_tile),
    )
    zero_spec = pl.BlockSpec(
        (None, None, intermediate_block, hidden_block),
        lambda dst, expert, intermediate_tile, hidden_tile, _row_tile: (dst, expert, intermediate_tile, hidden_tile),
    )
    out_spec = pl.BlockSpec(
        (None, None, intermediate_block, hidden_block),
        lambda dst, expert, intermediate_tile, hidden_tile, _row_tile: (dst, expert, intermediate_tile, hidden_tile),
    )
    return (weighted_activation_spec, dy_spec, zero_spec), out_spec


def _make_source_push_w2_d_weighted_activation_kernel(
    *,
    row_block: int,
    intermediate_block: int,
):
    def kernel(
        dy_ref: Float[pl.Ref, "M H"],
        w2_ref: Float[pl.Ref, "I H"],
        _zero_ref: Float[pl.Ref, "M I"],
        d_weighted_activation_ref: Float[pl.Ref, "M I"],
    ) -> None:
        def acc_scope(acc_ref) -> None:
            def smem_scope(dy_smem, w2_smem, ready_barrier) -> None:
                mgpu.copy_gmem_to_smem(
                    dy_ref.at[
                        pl.ds(0, row_block),
                        pl.ds(0, dy_ref.shape[-1]),
                    ],
                    dy_smem,
                    ready_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    w2_ref.at[
                        pl.ds(0, intermediate_block),
                        pl.ds(0, dy_ref.shape[-1]),
                    ],
                    w2_smem,
                    ready_barrier,
                )
                mgpu.barrier_wait(ready_barrier)
                mgpu.commit_smem()
                mgpu.wgmma(acc_ref, dy_smem, mgpu.transpose_ref(w2_smem, (1, 0)))
                mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                dy_smem=_w2_wgmma_smem((row_block, dy_ref.shape[-1]), dy_ref.dtype),
                w2_smem=_w2_wgmma_smem((intermediate_block, dy_ref.shape[-1]), w2_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        acc = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((row_block, intermediate_block)),
        )
        mgpu.atomic_add(d_weighted_activation_ref, acc)

    return kernel


def _make_source_push_w2_dw2_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    hidden_block: int,
):
    def kernel(
        weighted_activation_ref: Float[pl.Ref, "M I"],
        dy_ref: Float[pl.Ref, "M H"],
        _zero_ref: Float[pl.Ref, "I H"],
        dw2_ref: Float[pl.Ref, "I H"],
    ) -> None:
        def acc_scope(acc_ref) -> None:
            def smem_scope(weighted_activation_smem, dy_smem, ready_barrier) -> None:
                mgpu.copy_gmem_to_smem(
                    weighted_activation_ref.at[
                        pl.ds(0, row_block),
                        pl.ds(0, intermediate_block),
                    ],
                    weighted_activation_smem,
                    ready_barrier,
                )
                mgpu.copy_gmem_to_smem(
                    dy_ref.at[
                        pl.ds(0, row_block),
                        pl.ds(0, hidden_block),
                    ],
                    dy_smem,
                    ready_barrier,
                )
                mgpu.barrier_wait(ready_barrier)
                mgpu.commit_smem()
                mgpu.wgmma(acc_ref, mgpu.transpose_ref(weighted_activation_smem, (1, 0)), dy_smem)
                mgpu.wgmma_wait(0)

            pl.run_scoped(
                smem_scope,
                weighted_activation_smem=_w2_wgmma_smem((row_block, intermediate_block), dy_ref.dtype),
                dy_smem=_w2_wgmma_smem((row_block, hidden_block), dy_ref.dtype),
                ready_barrier=mgpu.Barrier(num_arrivals=2),
            )
            return acc_ref[...]

        acc = pl.run_scoped(
            acc_scope,
            acc_ref=mgpu.ACC((intermediate_block, hidden_block)),
        )
        mgpu.atomic_add(dw2_ref, acc)

    return kernel


def _w2_wgmma_smem(shape: tuple[int, int], dtype):
    swizzle_elems = W2_WGMMA_SWIZZLE_BYTES // jnp.dtype(dtype).itemsize
    if shape[-2] % W2_WGMMA_TILE_M or shape[-1] % swizzle_elems:
        raise ValueError(
            "W2 WGMMA SMEM operands must be divisible by "
            f"tile=({W2_WGMMA_TILE_M}, {swizzle_elems}); got shape={shape}"
        )
    return mgpu.SMEM(
        shape,
        dtype=dtype,
        transforms=(
            mgpu.TilingTransform((W2_WGMMA_TILE_M, swizzle_elems)),
            mgpu.SwizzleTransform(W2_WGMMA_SWIZZLE_BYTES),
        ),
    )


def _source_push_w2_d_weighted_activation_pallas_reference(
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> Float[Array, "Dst E C I"]:
    valid_f = valid.astype(jnp.float32)
    return jnp.einsum("dech,deih->deci", dy.astype(jnp.float32) * valid_f[..., None], w2.astype(jnp.float32))


def _source_push_w2_dw2_pallas_reference(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> Float[Array, "Dst E I D"]:
    valid_f = valid.astype(jnp.float32)
    return jnp.einsum(
        "deci,dech->deih",
        weighted_activation.astype(jnp.float32) * valid_f[..., None],
        dy.astype(jnp.float32) * valid_f[..., None],
    )


def _source_push_w2_d_weighted_activation_pallas_cost_estimate(
    dy: Array,
    w2: Array,
    valid: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(dy.shape, dy.dtype),
        jax.ShapeDtypeStruct(w2.shape, w2.dtype),
        jax.ShapeDtypeStruct(valid.shape, valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_w2_d_weighted_activation_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_w2_dw2_pallas_cost_estimate(
    weighted_activation: Array,
    dy: Array,
    valid: Array,
    output_shape: jax.ShapeDtypeStruct,
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(weighted_activation.shape, weighted_activation.dtype),
        jax.ShapeDtypeStruct(dy.shape, dy.dtype),
        jax.ShapeDtypeStruct(valid.shape, valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_w2_dw2_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _source_push_w2_swiglu_backward_reference(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> _SourcePushW2SwiGLUBackwardOutput:
    """Reference SwiGLU and route-weight derivative after W2 matmul backward."""

    _validate_swiglu_backward_shapes(h, route_weight, d_weighted_activation, valid)
    valid_f = valid.astype(jnp.float32)
    activation, _weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        h,
        route_weight,
        valid,
    )
    h = h.astype(jnp.float32) * valid_f[..., None]
    route_weight = route_weight.astype(jnp.float32) * valid_f
    d_weighted_activation = d_weighted_activation.astype(jnp.float32) * valid_f[..., None]

    gate, up = jnp.split(h, 2, axis=-1)
    silu_gate = jax.nn.silu(gate)
    d_route_weight = jnp.sum(d_weighted_activation * activation, axis=-1) * valid_f

    d_activation = d_weighted_activation * route_weight[..., None]
    sigmoid_gate = jax.nn.sigmoid(gate)
    d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
    d_gate = d_activation * up * d_silu_gate
    d_up = d_activation * silu_gate
    d_h = jnp.concatenate([d_gate, d_up], axis=-1) * valid_f[..., None]
    return _SourcePushW2SwiGLUBackwardOutput(d_h=d_h, d_route_weight=d_route_weight)


def _source_push_w2_swiglu_backward(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    implementation: SourcePushW2SwiGLUBackwardImplementation = SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2SwiGLUBackwardOutput:
    """Implementation boundary for the non-matmul W2 backward stage."""

    if implementation == SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE:
        return _source_push_w2_swiglu_backward_reference(h, route_weight, d_weighted_activation, valid)
    if implementation == SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU:
        return _source_push_w2_swiglu_backward_pallas_mgpu(
            h,
            route_weight,
            d_weighted_activation,
            valid,
            block_sizes=block_sizes,
            interpret=interpret,
            mesh=mesh,
        )
    supported_implementations = (
        SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
        SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    )
    raise ValueError(
        "source-push W2 SwiGLU backward implementation must be one of "
        f"{supported_implementations}, "
        f"got {implementation!r}"
    )


def _source_push_w2_swiglu_backward_pallas_mgpu(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes | None = None,
    interpret: bool = False,
    mesh: Mesh | None = None,
) -> _SourcePushW2SwiGLUBackwardOutput:
    """Lane-lowered Pallas kernel for ``d_h`` and ``d_route_weight`` only."""

    if not interpret and jax.default_backend() != "gpu":
        raise NotImplementedError("Pallas/MGPU W2 SwiGLU backward requires a GPU backend; use the reference on CPU")
    block_sizes = SourcePushW2SwiGLUBackwardPallasBlockSizes.get_default() if block_sizes is None else block_sizes
    _validate_swiglu_pallas_request(h, route_weight, d_weighted_activation, valid, block_sizes)
    if mesh is not None and not interpret:
        matmul_block_sizes = SourcePushW2MatmulBackwardPallasBlockSizes.get_default()
        d_h, d_route_weight = _source_push_w2_swiglu_backward_tiled_sharded_pallas_call(
            mesh,
            h,
            route_weight,
            d_weighted_activation,
            valid.astype(jnp.float32),
            row_block=matmul_block_sizes.row_block,
            intermediate_block=matmul_block_sizes.intermediate_block,
        )
        return _SourcePushW2SwiGLUBackwardOutput(d_h=d_h, d_route_weight=d_route_weight)
    d_h, d_route_weight = _source_push_w2_swiglu_backward_pallas_call(
        h,
        route_weight,
        d_weighted_activation,
        valid,
        row_block=block_sizes.row_block,
        interpret=interpret,
        mesh=mesh,
    )
    return _SourcePushW2SwiGLUBackwardOutput(d_h=d_h, d_route_weight=d_route_weight)


def _source_push_w2_swiglu_backward_pallas_call(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
    interpret: bool,
    mesh: Mesh | None = None,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    if mesh is not None and not interpret:
        return _source_push_w2_swiglu_backward_sharded_pallas_call(
            mesh,
            h,
            route_weight,
            d_weighted_activation,
            valid,
            row_block=row_block,
        )

    output_shape = (
        jax.ShapeDtypeStruct(h.shape, jnp.float32),
        jax.ShapeDtypeStruct(route_weight.shape, jnp.float32),
    )
    cost_estimate = _source_push_w2_swiglu_backward_pallas_cost_estimate(
        h,
        route_weight,
        d_weighted_activation,
        valid,
        output_shape,
    )
    kernel = _make_source_push_w2_swiglu_backward_kernel(
        row_block=row_block,
        intermediate_dim=d_weighted_activation.shape[-1],
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    in_specs, out_specs = _source_push_w2_swiglu_backward_block_specs(
        row_block=row_block,
        intermediate_dim=d_weighted_activation.shape[-1],
    )
    return pl.pallas_call(
        kernel,
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(h.shape[0], h.shape[1], h.shape[2] // row_block),
        interpret=interpret,
        name="source_push_w2_swiglu_backward_pallas_mgpu",
        compiler_params=compiler_params,
        cost_estimate=cost_estimate,
    )(h, route_weight, d_weighted_activation, valid)


def _source_push_w2_swiglu_backward_tiled_sharded_pallas_call(
    mesh: Mesh,
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    original_rows = h.shape[2]
    h, route_weight, d_weighted_activation, valid = _pad_w2_swiglu_rows_for_pallas(
        h,
        route_weight,
        d_weighted_activation,
        valid,
        row_multiple=row_block,
    )

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        route_weight_local: Float[Array, "1 E C"],
        d_weighted_activation_local: Float[Array, "1 E C I"],
        valid_local: Float[Array, "1 E C"],
    ) -> tuple[Float[Array, "1 E C twoI"], Float[Array, "1 E C"]]:
        route_weight_tiles_local = jnp.repeat(route_weight_local[..., None], intermediate_block, axis=-1)
        d_gate_local, d_up_local, d_route_partial_local = _source_push_w2_swiglu_backward_tiled_pallas_call(
            h_local,
            route_weight_tiles_local,
            d_weighted_activation_local,
            valid_local,
            row_block=row_block,
            intermediate_block=intermediate_block,
            interpret=False,
        )
        d_h_local = jnp.concatenate([d_gate_local, d_up_local], axis=-1)
        d_route_local = jnp.sum(d_route_partial_local, axis=-1)
        return d_h_local, d_route_local

    d_h, d_route_weight = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(h, route_weight, d_weighted_activation, valid)
    return d_h[:, :, :original_rows, :], d_route_weight[:, :, :original_rows]


def _source_push_w2_swiglu_backward_tiled_pallas_call(
    h: Float[Array, "Dst E C twoI"],
    route_weight_tiles: Float[Array, "Dst E C BI"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
    interpret: bool,
) -> tuple[Float[Array, "Dst E C I"], Float[Array, "Dst E C I"], Float[Array, "Dst E C ITiles"]]:
    intermediate_dim = d_weighted_activation.shape[-1]
    intermediate_tiles = intermediate_dim // intermediate_block
    output_shape = (
        jax.ShapeDtypeStruct(d_weighted_activation.shape, jnp.float32),
        jax.ShapeDtypeStruct(d_weighted_activation.shape, jnp.float32),
        jax.ShapeDtypeStruct(d_weighted_activation.shape[:-1] + (intermediate_tiles,), jnp.float32),
    )
    in_specs, out_specs = _source_push_w2_swiglu_backward_tiled_block_specs(
        row_block=row_block,
        intermediate_block=intermediate_block,
        intermediate_dim=intermediate_dim,
    )
    return pl.pallas_call(
        _make_source_push_w2_swiglu_backward_tiled_pallas_kernel(),
        in_specs=in_specs,
        out_specs=out_specs,
        out_shape=output_shape,
        grid=(h.shape[0], h.shape[1], h.shape[2] // row_block, intermediate_tiles),
        interpret=interpret,
        name="source_push_w2_swiglu_backward_tiled_pallas_mgpu",
    )(h, h, route_weight_tiles, d_weighted_activation, valid)


def _source_push_w2_swiglu_backward_tiled_block_specs(
    *,
    row_block: int,
    intermediate_block: int,
    intermediate_dim: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], tuple[pl.BlockSpec, ...]]:
    gate_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile: (dst, expert, row_tile, intermediate_tile),
    )
    up_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile: (
            dst,
            expert,
            row_tile,
            intermediate_dim // intermediate_block + intermediate_tile,
        ),
    )
    route_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, _intermediate_tile: (dst, expert, row_tile, 0),
    )
    d_weighted_activation_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile: (dst, expert, row_tile, intermediate_tile),
    )
    valid_spec = pl.BlockSpec(
        (None, None, row_block),
        lambda dst, expert, row_tile, _intermediate_tile: (dst, expert, row_tile),
    )
    d_gate_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile: (dst, expert, row_tile, intermediate_tile),
    )
    d_up_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_block),
        lambda dst, expert, row_tile, intermediate_tile: (dst, expert, row_tile, intermediate_tile),
    )
    d_route_partial_spec = pl.BlockSpec(
        (None, None, row_block, 1),
        lambda dst, expert, row_tile, intermediate_tile: (dst, expert, row_tile, intermediate_tile),
    )
    return (
        gate_spec,
        up_spec,
        route_spec,
        d_weighted_activation_spec,
        valid_spec,
    ), (
        d_gate_spec,
        d_up_spec,
        d_route_partial_spec,
    )


def _make_source_push_w2_swiglu_backward_tiled_pallas_kernel():
    def kernel(
        gate_ref: Float[pl.Ref, "M I"],
        up_ref: Float[pl.Ref, "M I"],
        route_weight_ref: Float[pl.Ref, "M I"],
        d_weighted_activation_ref: Float[pl.Ref, "M I"],
        valid_ref: Float[pl.Ref, "M"],
        d_gate_ref: Float[pl.Ref, "M I"],
        d_up_ref: Float[pl.Ref, "M I"],
        d_route_partial_ref: Float[pl.Ref, "M one"],
    ) -> None:
        row_block = gate_ref.shape[0]
        intermediate_block = gate_ref.shape[1]
        row_slice = pl.ds(0, row_block)
        intermediate_slice = pl.ds(0, intermediate_block)

        _ = valid_ref
        gate = gate_ref[row_slice, intermediate_slice].astype(jnp.float32)
        up = up_ref[row_slice, intermediate_slice].astype(jnp.float32)
        route_weight = route_weight_ref[row_slice, intermediate_slice].astype(jnp.float32)
        d_weighted_activation = d_weighted_activation_ref[row_slice, intermediate_slice].astype(jnp.float32)

        silu_gate = jax.nn.silu(gate)
        activation = silu_gate * up
        d_route_partial = jnp.sum(d_weighted_activation * activation, axis=-1)
        d_activation = d_weighted_activation * route_weight
        sigmoid_gate = jax.nn.sigmoid(gate)
        d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        d_gate_ref[row_slice, intermediate_slice] = d_activation * up * d_silu_gate
        d_up_ref[row_slice, intermediate_slice] = d_activation * silu_gate
        d_route_partial_ref[row_slice, 0] = d_route_partial

    return kernel


def _source_push_w2_swiglu_backward_tiled_sharded_mgpu_kernel(
    mesh: Mesh,
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Float[Array, "Dst E C"],
    *,
    row_block: int,
    intermediate_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    original_rows = h.shape[2]
    h, route_weight, d_weighted_activation, valid = _pad_w2_swiglu_rows_for_pallas(
        h,
        route_weight,
        d_weighted_activation,
        valid,
        row_multiple=row_block,
    )
    kernel = _make_source_push_w2_swiglu_backward_tiled_mgpu_kernel(
        row_block=row_block,
        intermediate_block=intermediate_block,
        experts_per_rank=h.shape[1],
        rows=h.shape[2],
        intermediate_dim=d_weighted_activation.shape[-1],
    )

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        route_weight_local: Float[Array, "1 E C"],
        d_weighted_activation_local: Float[Array, "1 E C I"],
        valid_local: Float[Array, "1 E C"],
    ) -> tuple[Float[Array, "1 E C twoI"], Float[Array, "1 E C"]]:
        d_h_local, d_route_partial_local = kernel(
            h_local[0],
            route_weight_local[0],
            d_weighted_activation_local[0],
            valid_local[0],
        )
        d_route_local = jnp.sum(d_route_partial_local, axis=-1)
        return d_h_local[None, ...], d_route_local[None, ...]

    d_h, d_route_weight = shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(h, route_weight, d_weighted_activation, valid)
    return d_h[:, :, :original_rows, :], d_route_weight[:, :, :original_rows]


def _pad_w2_swiglu_rows_for_pallas(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Float[Array, "Dst E C"],
    *,
    row_multiple: int,
) -> tuple[
    Float[Array, "Dst E C twoI"],
    Float[Array, "Dst E C"],
    Float[Array, "Dst E C I"],
    Float[Array, "Dst E C"],
]:
    rows = h.shape[2]
    padded_rows = ((rows + row_multiple - 1) // row_multiple) * row_multiple
    pad_rows = padded_rows - rows
    if pad_rows == 0:
        return h, route_weight, d_weighted_activation, valid
    h = jnp.pad(h, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    route_weight = jnp.pad(route_weight, ((0, 0), (0, 0), (0, pad_rows)))
    d_weighted_activation = jnp.pad(d_weighted_activation, ((0, 0), (0, 0), (0, pad_rows), (0, 0)))
    valid = jnp.pad(valid, ((0, 0), (0, 0), (0, pad_rows)))
    return h, route_weight, d_weighted_activation, valid


def _make_source_push_w2_swiglu_backward_tiled_mgpu_kernel(
    *,
    row_block: int,
    intermediate_block: int,
    experts_per_rank: int,
    rows: int,
    intermediate_dim: int,
):
    intermediate_tiles = intermediate_dim // intermediate_block
    row_tiles = rows // row_block

    def body(
        h_ref: Float[pl.Ref, "E C twoI"],
        route_weight_ref: Float[pl.Ref, "E C"],
        d_weighted_activation_ref: Float[pl.Ref, "E C I"],
        valid_ref: Float[pl.Ref, "E C"],
        d_h_ref: Float[pl.Ref, "E C twoI"],
        d_route_partial_ref: Float[pl.Ref, "E C ITiles"],
    ) -> None:
        expert = pl.program_id(0)
        row_tile = pl.program_id(1)
        intermediate_tile = pl.program_id(2)
        row_start = row_tile * row_block
        intermediate_start = intermediate_tile * intermediate_block
        row_slice = pl.ds(row_start, row_block)
        intermediate_slice = pl.ds(intermediate_start, intermediate_block)

        gate = h_ref[expert, row_slice, intermediate_slice].astype(jnp.float32)
        up = h_ref[
            expert,
            row_slice,
            pl.ds(intermediate_dim + intermediate_start, intermediate_block),
        ].astype(jnp.float32)
        route_weight = route_weight_ref[expert, row_slice].astype(jnp.float32)
        valid_f = valid_ref[expert, row_slice].astype(jnp.float32)
        d_weighted_activation = d_weighted_activation_ref[expert, row_slice, intermediate_slice].astype(jnp.float32)

        silu_gate = jax.nn.silu(gate)
        activation = silu_gate * up
        d_route_partial = jnp.sum(d_weighted_activation * activation, axis=-1) * valid_f
        d_activation = d_weighted_activation * route_weight[:, None]
        sigmoid_gate = jax.nn.sigmoid(gate)
        d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        d_gate = d_activation * up * d_silu_gate
        d_up = d_activation * silu_gate

        d_h_ref[expert, row_slice, intermediate_slice] = d_gate
        d_h_ref[expert, row_slice, pl.ds(intermediate_dim + intermediate_start, intermediate_block)] = d_up
        d_route_partial_ref[expert, row_slice, intermediate_tile] = d_route_partial

    out_shape = (
        jax.ShapeDtypeStruct((experts_per_rank, rows, 2 * intermediate_dim), jnp.float32),
        jax.ShapeDtypeStruct((experts_per_rank, rows, intermediate_tiles), jnp.float32),
    )
    compiler_params = mgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
    return mgpu.kernel(
        body,
        out_shape=out_shape,
        grid=(experts_per_rank, row_tiles, intermediate_tiles),
        grid_names=("expert", "row_tile", "intermediate_tile"),
        compiler_params=compiler_params,
    )


def _source_push_w2_swiglu_backward_block_specs(
    *,
    row_block: int,
    intermediate_dim: int,
) -> tuple[tuple[pl.BlockSpec, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec], tuple[pl.BlockSpec, pl.BlockSpec]]:
    two_intermediate_dim = 2 * intermediate_dim
    h_spec = pl.BlockSpec(
        (None, None, row_block, two_intermediate_dim),
        lambda dst, expert, row_tile: (dst, expert, row_tile, 0),
    )
    route_spec = pl.BlockSpec(
        (None, None, row_block),
        lambda dst, expert, row_tile: (dst, expert, row_tile),
    )
    d_weighted_activation_spec = pl.BlockSpec(
        (None, None, row_block, intermediate_dim),
        lambda dst, expert, row_tile: (dst, expert, row_tile, 0),
    )
    valid_spec = pl.BlockSpec(
        (None, None, row_block),
        lambda dst, expert, row_tile: (dst, expert, row_tile),
    )
    d_h_spec = pl.BlockSpec(
        (None, None, row_block, two_intermediate_dim),
        lambda dst, expert, row_tile: (dst, expert, row_tile, 0),
    )
    d_route_weight_spec = pl.BlockSpec(
        (None, None, row_block),
        lambda dst, expert, row_tile: (dst, expert, row_tile),
    )
    return (
        h_spec,
        route_spec,
        d_weighted_activation_spec,
        valid_spec,
    ), (
        d_h_spec,
        d_route_weight_spec,
    )


def _source_push_w2_swiglu_backward_sharded_pallas_call(
    mesh: Mesh,
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    *,
    row_block: int,
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    """Run the destination-local SwiGLU derivative kernel on local shards."""

    def local_fn(
        h_local: Float[Array, "1 E C twoI"],
        route_weight_local: Float[Array, "1 E C"],
        d_weighted_activation_local: Float[Array, "1 E C I"],
        valid_local: Bool[Array, "1 E C"],
    ) -> tuple[Float[Array, "1 E C twoI"], Float[Array, "1 E C"]]:
        return _source_push_w2_swiglu_backward_pallas_call(
            h_local,
            route_weight_local,
            d_weighted_activation_local,
            valid_local,
            row_block=row_block,
            interpret=False,
            mesh=None,
        )

    return shard_map(
        local_fn,
        mesh=mesh,
        in_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        out_specs=(
            P(SOURCE_PUSH_MESH_AXIS, None, None, None),
            P(SOURCE_PUSH_MESH_AXIS, None, None),
        ),
        check_vma=False,
    )(h, route_weight, d_weighted_activation, valid)


def _make_source_push_w2_swiglu_backward_kernel(*, row_block: int, intermediate_dim: int):
    def kernel(
        h_ref: Float[pl.Ref, "M twoI"],
        route_weight_ref: Float[pl.Ref, "M"],
        d_weighted_activation_ref: Float[pl.Ref, "M I"],
        valid_ref: Bool[pl.Ref, "M"],
        d_h_ref: Float[pl.Ref, "M twoI"],
        d_route_weight_ref: Float[pl.Ref, "M"],
    ) -> None:
        row_slice = pl.ds(0, row_block)

        valid_f = valid_ref[row_slice].astype(jnp.float32)
        gate = h_ref[row_slice, pl.ds(0, intermediate_dim)].astype(jnp.float32) * valid_f[:, None]
        up = h_ref[row_slice, pl.ds(intermediate_dim, intermediate_dim)].astype(jnp.float32) * valid_f[:, None]
        route_weight = route_weight_ref[row_slice].astype(jnp.float32) * valid_f
        d_weighted_activation = (
            d_weighted_activation_ref[row_slice, pl.ds(0, intermediate_dim)].astype(jnp.float32) * valid_f[:, None]
        )

        silu_gate = jax.nn.silu(gate)
        activation = silu_gate * up
        d_route_weight = jnp.sum(d_weighted_activation * activation, axis=-1) * valid_f

        d_activation = d_weighted_activation * route_weight[:, None]
        sigmoid_gate = jax.nn.sigmoid(gate)
        d_silu_gate = sigmoid_gate * (1.0 + gate * (1.0 - sigmoid_gate))
        d_gate = d_activation * up * d_silu_gate
        d_up = d_activation * silu_gate

        d_h_ref[row_slice, pl.ds(0, intermediate_dim)] = d_gate
        d_h_ref[row_slice, pl.ds(intermediate_dim, intermediate_dim)] = d_up
        d_route_weight_ref[row_slice] = d_route_weight

    return kernel


def _source_push_w2_swiglu_backward_pallas_reference(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> tuple[Float[Array, "Dst E C twoI"], Float[Array, "Dst E C"]]:
    output = _source_push_w2_swiglu_backward_reference(h, route_weight, d_weighted_activation, valid)
    return output.d_h, output.d_route_weight


def _source_push_w2_swiglu_backward_pallas_cost_estimate(
    h: Array,
    route_weight: Array,
    d_weighted_activation: Array,
    valid: Array,
    output_shape: tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct],
) -> pl.CostEstimate:
    input_specs = (
        jax.ShapeDtypeStruct(h.shape, h.dtype),
        jax.ShapeDtypeStruct(route_weight.shape, route_weight.dtype),
        jax.ShapeDtypeStruct(d_weighted_activation.shape, d_weighted_activation.dtype),
        jax.ShapeDtypeStruct(valid.shape, valid.dtype),
    )
    body_cost = pl.estimate_cost(_source_push_w2_swiglu_backward_pallas_reference, *input_specs)
    return with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_specs,
        kernel_outputs_specs=output_shape,
    )


def _expert_flat_rows(expert_base: Int[Array, "Dst E"], rows_per_expert: int) -> Int[Array, "Dst E C"]:
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    row_offsets = jnp.arange(rows_per_expert, dtype=jnp.int32)
    return expert_base[..., None] + row_offsets[None, None, :]


def _dst_indices(dst_count: int, expert_count: int, rows_per_expert: int) -> Int[Array, "Dst E C"]:
    dst = jnp.arange(dst_count, dtype=jnp.int32)[:, None, None]
    return jnp.broadcast_to(dst, (dst_count, expert_count, rows_per_expert))


def _gather_flat_rows(
    rows: Array,
    flat_rows: Int[Array, "Dst E C"],
    *,
    fill_value: float,
) -> Array:
    dst_index = _dst_indices(flat_rows.shape[0], flat_rows.shape[1], flat_rows.shape[2])
    out_parts = (SOURCE_PUSH_MESH_AXIS, None, None) + (None,) * (rows.ndim - 2)
    return rows.at[dst_index, flat_rows].get(
        mode="fill",
        fill_value=fill_value,
        out_sharding=_source_push_out_sharding(*out_parts),
    )


def _gather_flat_rows_by_expert_slice(
    rows: Array,
    expert_base: Int[Array, "Dst E"],
    rows_per_expert: int,
) -> Array:
    """Gather source-padded contiguous expert slices from flat-H rows."""

    rows = jnp.asarray(rows)
    expert_base = jnp.asarray(expert_base, dtype=jnp.int32)
    rows_sharding = _source_push_destination_named_sharding(rows, rows.ndim)
    if rows_sharding is not None:
        return _gather_flat_rows_by_expert_slice_shard_map(
            rows,
            expert_base,
            rows_per_expert,
            rows_sharding.mesh,
        )

    rows = _with_source_push_destination_sharding(rows, expert_base)
    expert_base = _with_source_push_destination_sharding(expert_base, rows)
    slice_sizes = (rows_per_expert,) + rows.shape[2:]
    trailing_starts = (0,) * (rows.ndim - 2)

    def gather_dst(rows_dst, base_dst):
        def gather_expert(base):
            return jax.lax.dynamic_slice(rows_dst, (base, *trailing_starts), slice_sizes)

        return jax.vmap(gather_expert)(base_dst)

    return _with_source_push_destination_sharding(jax.vmap(gather_dst)(rows, expert_base), expert_base)


def _gather_flat_rows_by_expert_slice_shard_map(
    rows: Array,
    expert_base: Int[Array, "Dst E"],
    rows_per_expert: int,
    mesh: Mesh,
) -> Array:
    """Gather contiguous expert slices locally on each destination shard."""

    slice_sizes = (rows_per_expert,) + rows.shape[2:]
    trailing_starts = (0,) * (rows.ndim - 2)

    def local_gather(rows_local, expert_base_local):
        rows_dst = rows_local[0]
        base_dst = expert_base_local[0]

        def gather_expert(base):
            return jax.lax.dynamic_slice(rows_dst, (base, *trailing_starts), slice_sizes)

        return jax.vmap(gather_expert)(base_dst)[None, ...]

    rows_spec = P(SOURCE_PUSH_MESH_AXIS, *(None for _ in range(rows.ndim - 1)))
    out_spec = P(SOURCE_PUSH_MESH_AXIS, None, None, *(None for _ in range(rows.ndim - 2)))
    return shard_map(
        local_gather,
        mesh=mesh,
        in_specs=(rows_spec, P(SOURCE_PUSH_MESH_AXIS, None)),
        out_specs=out_spec,
        check_vma=False,
    )(rows, expert_base)


def _validate_expert_block_shapes(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> None:
    if h.ndim != 4:
        raise ValueError(f"h must have shape [Dst, E, C, 2I], got {h.shape}")
    if route_weight.shape != h.shape[:3]:
        raise ValueError(f"route_weight shape {route_weight.shape} must match h rows {h.shape[:3]}")
    if valid.shape != h.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match h rows {h.shape[:3]}")
    if dy.shape[:3] != h.shape[:3]:
        raise ValueError(f"dy rows {dy.shape[:3]} must match h rows {h.shape[:3]}")
    if w2.shape[:2] != h.shape[:2]:
        raise ValueError(f"w2 destination/expert shape {w2.shape[:2]} must match h {h.shape[:2]}")
    if h.shape[-1] != 2 * w2.shape[-2]:
        raise ValueError(f"h trailing dim {h.shape[-1]} must equal 2 * w2 intermediate dim {w2.shape[-2]}")
    if dy.shape[-1] != w2.shape[-1]:
        raise ValueError(f"dy hidden dim {dy.shape[-1]} must match w2 hidden dim {w2.shape[-1]}")


def _validate_swiglu_backward_shapes(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> None:
    if h.ndim != 4:
        raise ValueError(f"h must have shape [Dst, E, C, 2I], got {h.shape}")
    if h.shape[-1] % 2 != 0:
        raise ValueError(f"h trailing dim {h.shape[-1]} must be even")
    if route_weight.shape != h.shape[:3]:
        raise ValueError(f"route_weight shape {route_weight.shape} must match h rows {h.shape[:3]}")
    if valid.shape != h.shape[:3]:
        raise ValueError(f"valid shape {valid.shape} must match h rows {h.shape[:3]}")
    expected_d_weighted_activation = h.shape[:3] + (h.shape[-1] // 2,)
    if d_weighted_activation.shape != expected_d_weighted_activation:
        raise ValueError(
            f"d_weighted_activation shape {d_weighted_activation.shape} must be {expected_d_weighted_activation}"
        )


def _validate_w2_matmul_backward_shapes(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> None:
    if weighted_activation.ndim != 4:
        raise ValueError(f"weighted_activation must have shape [Dst, E, C, I], got {weighted_activation.shape}")
    if dy.shape[:3] != weighted_activation.shape[:3]:
        raise ValueError(f"dy rows {dy.shape[:3]} must match weighted_activation rows {weighted_activation.shape[:3]}")
    if valid.shape != weighted_activation.shape[:3]:
        raise ValueError(
            f"valid shape {valid.shape} must match weighted_activation rows {weighted_activation.shape[:3]}"
        )
    expected_w2_shape = weighted_activation.shape[:2] + (weighted_activation.shape[-1], dy.shape[-1])
    if w2.shape != expected_w2_shape:
        raise ValueError(f"w2 shape {w2.shape} must be {expected_w2_shape}")


def _source_push_w2_matmul_backward_inferred_block_sizes(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
) -> SourcePushW2MatmulBackwardPallasBlockSizes:
    _ = w2
    return SourcePushW2MatmulBackwardPallasBlockSizes(
        row_block=_largest_divisor_from_candidates(
            weighted_activation.shape[2],
            SOURCE_PUSH_W2_MATMUL_BACKWARD_ROW_BLOCK_CANDIDATES,
        ),
        intermediate_block=_largest_divisor_from_candidates(
            weighted_activation.shape[-1],
            SOURCE_PUSH_W2_MATMUL_BACKWARD_INTERMEDIATE_BLOCK_CANDIDATES,
        ),
        hidden_block=_largest_divisor_from_candidates(
            dy.shape[-1],
            SOURCE_PUSH_W2_MATMUL_BACKWARD_HIDDEN_BLOCK_CANDIDATES,
        ),
    )


def _largest_divisor_from_candidates(value: int, candidates: tuple[int, ...]) -> int:
    for candidate in candidates:
        if value % candidate == 0:
            return candidate
    raise ValueError(f"no candidate tile divides {value}")


def _validate_w2_matmul_pallas_request(
    weighted_activation: Float[Array, "Dst E C I"],
    dy: Float[Array, "Dst E C D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    block_sizes: SourcePushW2MatmulBackwardPallasBlockSizes,
) -> None:
    _validate_w2_matmul_backward_shapes(weighted_activation, dy, w2, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if block_sizes.row_block < MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK:
        raise ValueError(f"row_block {block_sizes.row_block} must be at least {MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK}")
    if block_sizes.intermediate_block <= 0:
        raise ValueError(f"intermediate_block must be positive, got {block_sizes.intermediate_block}")
    if block_sizes.hidden_block <= 0:
        raise ValueError(f"hidden_block must be positive, got {block_sizes.hidden_block}")
    if weighted_activation.shape[2] % block_sizes.row_block != 0:
        raise ValueError(
            f"row_block {block_sizes.row_block} must divide rows per expert {weighted_activation.shape[2]}"
        )
    if weighted_activation.shape[-1] % block_sizes.intermediate_block != 0:
        raise ValueError(
            "intermediate_block "
            f"{block_sizes.intermediate_block} must divide intermediate dim {weighted_activation.shape[-1]}"
        )
    if dy.shape[-1] % block_sizes.hidden_block != 0:
        raise ValueError(f"hidden_block {block_sizes.hidden_block} must divide hidden dim {dy.shape[-1]}")


def _validate_swiglu_pallas_request(
    h: Float[Array, "Dst E C twoI"],
    route_weight: Float[Array, "Dst E C"],
    d_weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
    block_sizes: SourcePushW2SwiGLUBackwardPallasBlockSizes,
) -> None:
    _validate_swiglu_backward_shapes(h, route_weight, d_weighted_activation, valid)
    if block_sizes.row_block <= 0:
        raise ValueError(f"row_block must be positive, got {block_sizes.row_block}")
    if h.shape[2] % block_sizes.row_block != 0:
        raise ValueError(f"row_block {block_sizes.row_block} must divide rows per expert {h.shape[2]}")


def _validate_flat_h_shapes(
    expert_base: Int[Array, "Dst E"],
    h: Float[Array, "Dst rows twoI"],
    route_weight: Float[Array, "Dst rows"],
    dy: Float[Array, "Dst rows D"],
    w2: Float[Array, "Dst E I D"],
    valid: Bool[Array, "Dst E C"] | Float[Array, "Dst E C"],
) -> None:
    if expert_base.shape != w2.shape[:2]:
        raise ValueError(f"expert_base shape {expert_base.shape} must match w2 destination/expert {w2.shape[:2]}")
    if h.shape[:2] != route_weight.shape:
        raise ValueError(f"route_weight shape {route_weight.shape} must match h rows {h.shape[:2]}")
    if dy.shape[:2] != h.shape[:2]:
        raise ValueError(f"dy rows {dy.shape[:2]} must match h rows {h.shape[:2]}")
    if valid.shape[:2] != w2.shape[:2]:
        raise ValueError(f"valid destination/expert shape {valid.shape[:2]} must match w2 {w2.shape[:2]}")
    if h.shape[-1] != 2 * w2.shape[-2]:
        raise ValueError(f"h trailing dim {h.shape[-1]} must equal 2 * w2 intermediate dim {w2.shape[-2]}")
    if dy.shape[-1] != w2.shape[-1]:
        raise ValueError(f"dy hidden dim {dy.shape[-1]} must match w2 hidden dim {w2.shape[-1]}")
