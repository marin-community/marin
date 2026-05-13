# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Optional CuTe/QuACK kernels for Grug MoE.

This module is imported lazily by `grug_moe.py` because it requires CUDA-only
packages from the Levanter/Marin GPU extras.
"""

from __future__ import annotations

from collections.abc import Sequence

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from cutlass import Float32, const_expr
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.grug_moe import _gather_sum_reference
from quack.activation import dgate_fn_map
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm90
from quack.gemm_dact import GemmDGatedMixin, GemmDGatedSm90
from quack.gemm_tvm_ffi_utils import make_scheduler_args, make_varlen_args

_TILE_M = 128
_TILE_N = 128
_TILE_K = 0
_CLUSTER_M = 2
_CLUSTER_N = 1
_PINGPONG = False
_MAX_ACTIVE_CLUSTERS = 66
_MAX_SWIZZLE_SIZE = 8


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _static_row_major_spec(shape: Sequence[int], *, inner_divisor: int | None = 128) -> cjax.TensorSpec:
    divisibility = inner_divisor if inner_divisor is not None and shape[-1] % inner_divisor == 0 else None
    return cjax.TensorSpec(static=True, divisibility=divisibility)


def _pack_bf16x2(value: jax.Array) -> jax.Array:
    return jax.lax.bitcast_convert_type(value.reshape(value.shape[0], value.shape[1] // 2, 2), jnp.float32)


def _unpack_bf16x2(value: jax.Array, dtype: jnp.dtype) -> jax.Array:
    unpacked = jax.lax.bitcast_convert_type(value, dtype)
    return unpacked.reshape(value.shape[0], value.shape[1] * 2)


@cute.jit
def _launch_quack_dgated_sm90(
    stream: cuda.CUstream,
    dout: cute.Tensor,
    w2_base_eih: cute.Tensor,
    h_packed: cute.Tensor,
    sorted_scores: cute.Tensor,
    cu_seqlens_m: cute.Tensor,
    x_gather_idx: cute.Tensor,
    dh_packed: cute.Tensor,
    a_prime: cute.Tensor,
    colvec_reduce_partial: cute.Tensor,
    *,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    tile_k: cutlass.Constexpr[int],
    cluster_m: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    pingpong: cutlass.Constexpr[bool],
    max_active_clusters: cutlass.Constexpr[int],
    max_swizzle_size: cutlass.Constexpr[int],
) -> None:
    w2_ihe = cute.make_tensor(
        w2_base_eih.iterator,
        cute.make_layout(
            (w2_base_eih.shape[1], w2_base_eih.shape[2], w2_base_eih.shape[0]),
            stride=(w2_base_eih.stride[1], w2_base_eih.stride[2], w2_base_eih.stride[0]),
        ),
    )
    tile_shape_mnk = (tile_m, tile_n)
    if const_expr(tile_k != 0):
        tile_shape_mnk = (tile_m, tile_n, tile_k)

    gemm = GemmDGatedSm90(
        Float32,
        dout.element_type,
        tile_shape_mnk,
        (cluster_m, cluster_n, 1),
        pingpong=pingpong,
        is_persistent=True,
        gather_A=True,
    )
    gemm.implicit_dtype = a_prime.element_type

    epilogue_args = GemmDGatedMixin.EpilogueArguments(
        a_prime,
        dgate_fn_map["swiglu"],
        mColVecBroadcast=sorted_scores,
        mColVecReduce=colvec_reduce_partial,
        rounding_mode=None,
        sr_seed=None,
    )
    scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle_size, None)
    varlen_args = make_varlen_args(cu_seqlens_m, None, x_gather_idx)
    gemm(dout, w2_ihe, dh_packed, h_packed, epilogue_args, scheduler_args, varlen_args, stream)


@cute.jit
def _launch_quack_dw2_sm90(
    stream: cuda.CUstream,
    dout: cute.Tensor,
    a_prime: cute.Tensor,
    cu_seqlens_k: cute.Tensor,
    x_gather_idx: cute.Tensor,
    dw2_eih: cute.Tensor,
    *,
    tile_m: cutlass.Constexpr[int],
    tile_n: cutlass.Constexpr[int],
    tile_k: cutlass.Constexpr[int],
    cluster_m: cutlass.Constexpr[int],
    cluster_n: cutlass.Constexpr[int],
    pingpong: cutlass.Constexpr[bool],
    max_active_clusters: cutlass.Constexpr[int],
    max_swizzle_size: cutlass.Constexpr[int],
) -> None:
    dout_ht = cute.make_tensor(
        dout.iterator,
        cute.make_layout((dout.shape[1], dout.shape[0]), stride=(dout.stride[1], dout.stride[0])),
    )
    a_prime_itk = cute.make_tensor(
        a_prime.iterator,
        cute.make_layout((a_prime.shape[1], a_prime.shape[0]), stride=(a_prime.stride[1], a_prime.stride[0])),
    )
    dw2_hie = cute.make_tensor(
        dw2_eih.iterator,
        cute.make_layout(
            (dw2_eih.shape[2], dw2_eih.shape[1], dw2_eih.shape[0]),
            stride=(dw2_eih.stride[2], dw2_eih.stride[1], dw2_eih.stride[0]),
        ),
    )

    tile_shape_mnk = (tile_m, tile_n)
    if const_expr(tile_k != 0):
        tile_shape_mnk = (tile_m, tile_n, tile_k)

    gemm = GemmDefaultSm90(
        Float32,
        dout.element_type,
        tile_shape_mnk,
        (cluster_m, cluster_n, 1),
        pingpong=pingpong,
        is_persistent=True,
        gather_A=True,
    )
    epilogue_args = GemmDefaultEpiMixin.EpilogueArguments()
    scheduler_args = make_scheduler_args(max_active_clusters, max_swizzle_size, None)
    varlen_args = make_varlen_args(None, cu_seqlens_k, x_gather_idx)
    gemm(dout_ht, a_prime_itk, dw2_hie, None, epilogue_args, scheduler_args, varlen_args, stream)


def _jax_quack_dgated(
    dout: jax.Array,
    h: jax.Array,
    w2_base_eih: jax.Array,
    topk_scores: jax.Array,
    x_gather_idx: jax.Array,
    s_scatter_idx: jax.Array,
    expert_frequency_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    assignments = h.shape[0]
    intermediate = h.shape[1] // 2
    colvec_tiles = _ceil_div(intermediate, _TILE_N)
    h_packed = _pack_bf16x2(h)
    sorted_scores = topk_scores.reshape(assignments)[s_scatter_idx].astype(jnp.float32)

    call = cjax.cutlass_call(
        _launch_quack_dgated_sm90,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((assignments, intermediate), jnp.float32),
            jax.ShapeDtypeStruct((assignments, intermediate), h.dtype),
            jax.ShapeDtypeStruct((assignments, colvec_tiles), jnp.float32),
        ),
        input_spec=(
            _static_row_major_spec(dout.shape),
            _static_row_major_spec(w2_base_eih.shape),
            _static_row_major_spec(h_packed.shape),
            _static_row_major_spec(sorted_scores.shape),
            _static_row_major_spec(expert_frequency_offset.shape, inner_divisor=None),
            _static_row_major_spec(x_gather_idx.shape),
        ),
        output_spec=(
            _static_row_major_spec((assignments, intermediate)),
            _static_row_major_spec((assignments, intermediate)),
            _static_row_major_spec((assignments, colvec_tiles), inner_divisor=None),
        ),
        allow_cuda_graph=False,
        use_static_tensors=True,
        tile_m=_TILE_M,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        cluster_m=_CLUSTER_M,
        cluster_n=_CLUSTER_N,
        pingpong=_PINGPONG,
        max_active_clusters=_MAX_ACTIVE_CLUSTERS,
        max_swizzle_size=_MAX_SWIZZLE_SIZE,
    )
    dh_packed, a_prime, colvec_reduce_partial = call(
        dout,
        w2_base_eih,
        h_packed,
        sorted_scores,
        expert_frequency_offset,
        x_gather_idx,
    )
    dh = _unpack_bf16x2(dh_packed, h.dtype)
    ds_scattered = jnp.sum(colvec_reduce_partial, axis=-1)
    ds = jnp.zeros_like(sorted_scores).at[s_scatter_idx].set(ds_scattered)
    return dh, a_prime, ds, colvec_reduce_partial


def _jax_quack_dw2(
    dout: jax.Array,
    a_prime: jax.Array,
    x_gather_idx: jax.Array,
    expert_frequency_offset: jax.Array,
) -> jax.Array:
    call = cjax.cutlass_call(
        _launch_quack_dw2_sm90,
        output_shape_dtype=jax.ShapeDtypeStruct(
            (expert_frequency_offset.shape[0] - 1, a_prime.shape[1], dout.shape[1]),
            a_prime.dtype,
        ),
        input_spec=(
            _static_row_major_spec(dout.shape),
            _static_row_major_spec(a_prime.shape),
            _static_row_major_spec(expert_frequency_offset.shape, inner_divisor=None),
            _static_row_major_spec(x_gather_idx.shape),
        ),
        output_spec=_static_row_major_spec((expert_frequency_offset.shape[0] - 1, a_prime.shape[1], dout.shape[1])),
        allow_cuda_graph=False,
        use_static_tensors=True,
        tile_m=_TILE_M,
        tile_n=_TILE_N,
        tile_k=_TILE_K,
        cluster_m=_CLUSTER_M,
        cluster_n=_CLUSTER_N,
        pingpong=_PINGPONG,
        max_active_clusters=_MAX_ACTIVE_CLUSTERS,
        max_swizzle_size=_MAX_SWIZZLE_SIZE,
    )
    return call(dout, a_prime, expert_frequency_offset, x_gather_idx)


def _jax_quack_down_bwd(
    dout: jax.Array,
    h_interleaved: jax.Array,
    w_down: jax.Array,
    combine_weights: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    expert_frequency_offset: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    dh, a_prime, ds, colvec_reduce_partial = _jax_quack_dgated(
        dout,
        h_interleaved,
        w_down,
        combine_weights,
        token_ids_sort,
        sorted_assignment_ids,
        expert_frequency_offset,
    )
    dw2 = _jax_quack_dw2(dout, a_prime, token_ids_sort, expert_frequency_offset)
    return dh, a_prime, ds, colvec_reduce_partial, dw2


@jax.custom_vjp
def quack_interleaved_down_gather_sum(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> jax.Array:
    out, _ = _quack_interleaved_down_gather_sum_forward(
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        sorted_assignment_ids,
        dispatch_positions,
        group_sizes,
    )
    return out


def _quack_interleaved_down_gather_sum_forward(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    del sorted_assignment_ids
    hidden = jax.nn.silu(w13_out_interleaved[:, 0::2]) * w13_out_interleaved[:, 1::2]
    dispatch_output = ragged_dot(hidden, w_down, group_sizes)
    out = _gather_sum_reference(dispatch_output, dispatch_positions, combine_weights)
    expert_frequency_offset = jnp.concatenate(
        [jnp.zeros((1,), dtype=jnp.int32), jnp.cumsum(group_sizes, dtype=jnp.int32)]
    )
    return out, (w13_out_interleaved, combine_weights, w_down, token_ids_sort, expert_frequency_offset)


def _quack_interleaved_down_gather_sum_fwd(
    w13_out_interleaved: jax.Array,
    combine_weights: jax.Array,
    w_down: jax.Array,
    token_ids_sort: jax.Array,
    sorted_assignment_ids: jax.Array,
    dispatch_positions: jax.Array,
    group_sizes: jax.Array,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    out, residuals = _quack_interleaved_down_gather_sum_forward(
        w13_out_interleaved,
        combine_weights,
        w_down,
        token_ids_sort,
        sorted_assignment_ids,
        dispatch_positions,
        group_sizes,
    )
    return out, (*residuals, sorted_assignment_ids)


def _quack_interleaved_down_gather_sum_bwd(
    residuals: tuple[jax.Array, ...],
    dout: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, None, None, None, None]:
    w13_out_interleaved, combine_weights, w_down, token_ids_sort, expert_frequency_offset, sorted_assignment_ids = (
        residuals
    )
    dh_interleaved, _, d_scores_flat, _, d_w_down = _jax_quack_down_bwd(
        dout,
        w13_out_interleaved,
        w_down,
        combine_weights,
        token_ids_sort,
        sorted_assignment_ids,
        expert_frequency_offset,
    )
    d_combine_weights = d_scores_flat.reshape(combine_weights.shape).astype(combine_weights.dtype)
    return dh_interleaved, d_combine_weights, d_w_down.astype(w_down.dtype), None, None, None, None


quack_interleaved_down_gather_sum.defvjp(
    _quack_interleaved_down_gather_sum_fwd,
    _quack_interleaved_down_gather_sum_bwd,
)
