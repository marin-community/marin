#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX bridges for the SM100 RMS-GatedNorm reverse kernels."""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug._moe.quack_moe_cute import _cute_dtype
from levanter.grug._moe.rms_gated_norm import exact_rms_backward_consumer
from levanter.kernels.pallas.cost_estimate_utils import with_io_bytes_accessed
from quack.activation import dact_fn_map
from quack.cute_dsl_utils import get_max_active_clusters, mlir_namedtuple
from quack.epi_ops import (
    ColVecLoad,
    ColVecReduce,
    RowVecLoad,
    TileLoad,
    TileStore,
    colvec_reduce_accumulate,
    vec_multiply,
)
from quack.gemm_act import GemmActMixin
from quack.gemm_dact import GemmDActSm100
from quack.gemm_sm100 import GemmSm100
from quack.gemm_tvm_ffi_utils import make_scheduler_args
from quack.rounding import RoundingMode

_ACCUMULATOR_DTYPE = cutlass.Float32
_DEFAULT_TILE_MN = (256, 128)
_DEFAULT_CLUSTER_MNK = (2, 1, 1)
_DEFAULT_BACKWARD_TILE_MN = (256, 128)
_DEFAULT_BACKWARD_CLUSTER_MNK = (2, 1, 1)
_DEFAULT_MAX_SWIZZLE = 8
_FALLBACK_MAX_ACTIVE_CLUSTERS = 148
_MATRIX_MODE = (1, 2, 0)
_MATRIX_DIVISIBILITY = (1, 1, 8)
_VECTOR_DIVISIBILITY = (1, 4)
_CONSUMER_BLOCK_M = 128
_CONSUMER_BLOCK_N = 128


def _max_active_clusters(cluster_mnk: tuple[int, int, int]) -> int:
    """Size the persistent grid from the live device, as the sibling QuACK bridges do."""
    if jax.default_backend() == "cpu":
        return _FALLBACK_MAX_ACTIVE_CLUSTERS
    return get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])


class _GemmRmsBackwardMixin(GemmActMixin):
    """CuTe epilogue that returns an unweighted cotangent and RMS row partials."""

    _epi_ops = (  # pyrefly: ignore[bad-override-mutable-attribute]
        RowVecLoad("mNormWeight"),
        ColVecLoad("mInverseRms"),
        TileLoad("mDirectCotangent"),
        TileLoad("mX"),
        ColVecReduce("mRowDotPartial"),
        TileStore("mAuxOut"),
    )
    _extra_param_fields = ()

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):  # pyrefly: ignore[bad-override]
        mNormWeight: cute.Tensor | None = None
        mInverseRms: cute.Tensor | None = None
        mDirectCotangent: cute.Tensor | None = None
        mX: cute.Tensor | None = None
        mRowDotPartial: cute.Tensor | None = None
        mAuxOut: cute.Tensor | None = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        self.aux_out_dtype = args.mAuxOut.element_type
        self.aux_out_layout = cutlass.utils.LayoutEnum.from_tensor(args.mAuxOut)
        self.cta_tile_shape_aux_out_mn = self.cta_tile_shape_mnk[:2]
        values = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**values)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrNormWeight = epi_loop_tensors.get("mNormWeight")
        tDrInverseRms = epi_loop_tensors.get("mInverseRms")
        tRS_rDirectCotangent = epi_loop_tensors.get("mDirectCotangent")
        tRS_rX = epi_loop_tensors.get("mX")
        tDrRowDot = epi_loop_tensors.get("mRowDotPartial")

        tRS_rUnweightedCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rUnweightedCotangent.store(tRS_rD.load().to(self.a_dtype))
        tRS_rUnweightedCotangent.store(tRS_rUnweightedCotangent.load() + tRS_rDirectCotangent.load())
        tRS_rD.store(tRS_rUnweightedCotangent.load().to(tRS_rD.element_type))

        tRS_rXHat = cute.make_rmem_tensor_like(tRS_rD)
        for i in cutlass.range(cute.size(tRS_rXHat), unroll_full=True):
            tRS_rXHat[i] = tRS_rX[i].to(tRS_rD.element_type) * tDrInverseRms[i]

        tRS_rUnweightedCotangent = cute.make_rmem_tensor_like(tRS_rD)
        tRS_rUnweightedCotangent.store(tRS_rD.load())
        vec_multiply(self, tRS_rD, None, tDrNormWeight)
        tRS_rRowDotProduct = cute.make_rmem_tensor_like(tRS_rD)
        tRS_rRowDotProduct.store(tRS_rD.load() * tRS_rXHat.load())
        colvec_reduce_accumulate(self, tDrRowDot, tRS_rRowDotProduct)
        return (tRS_rUnweightedCotangent,)


class _GemmRmsBackwardSm100(_GemmRmsBackwardMixin, GemmSm100):  # pyrefly: ignore[inconsistent-inheritance]
    pass


@cute_launcher_factory
def _build_aliased_backward_producer_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build the producer with ``direct`` serving as its output buffer."""

    @cute.jit
    def launcher(
        stream,
        mGatePreactivationCotangent,
        mWDown,
        mDirectAndUnweightedCotangent,
        mX,
        mNormWeight,
        mInverseRms,
        mRowDotPartial,
    ):
        gemm = _GemmRmsBackwardSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = _GemmRmsBackwardMixin.EpilogueArguments(
            mNormWeight=mNormWeight,
            mInverseRms=mInverseRms,
            mDirectCotangent=mDirectAndUnweightedCotangent,
            mX=mX,
            mRowDotPartial=mRowDotPartial,
            mAuxOut=mDirectAndUnweightedCotangent,
        )
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(
            mGatePreactivationCotangent,
            mWDown,
            None,
            None,
            epilogue,
            scheduler,
            None,
            stream,
        )

    return launcher


@cute_launcher_factory
def _build_silu_backward_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build ``dpreact = (a @ b.T) * silu'(preact)``."""

    @cute.jit
    def launcher(stream, mA, mB, mPreAct, mDPreAct, mPostAct):
        gemm = GemmDActSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = GemmActMixin.EpilogueArguments(mPostAct, dact_fn_map["silu"])
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mA, mB, mDPreAct, mPreAct, epilogue, scheduler, None, stream)

    return launcher


def quack_coda_rms_backward_producer(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    direct_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_BACKWARD_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_BACKWARD_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Produce BF16 ``du`` and RMS-row partials.

    The canonical shapes are ``gate_preactivation_cotangent[M,R]``,
    ``w_down[D,R]``, full-width tensors ``[M,D]``, ``norm_weight[D]``, and
    ``inverse_rms[M]``. Inputs must already be local to one device.
    """
    if gate_preactivation_cotangent.ndim != 2 or w_down.ndim != 2:
        raise ValueError(
            "expected gate_preactivation_cotangent[M,R] and w_down[D,R], got "
            f"{gate_preactivation_cotangent.shape} and {w_down.shape}"
        )
    rows, rank = gate_preactivation_cotangent.shape
    hidden_dim, w_rank = w_down.shape
    if w_rank != rank:
        raise ValueError(f"contracting dimensions differ: {rank} != {w_rank}")
    expected_full_shape = (rows, hidden_dim)
    if direct_cotangent.shape != expected_full_shape or x.shape != expected_full_shape:
        raise ValueError(
            f"expected direct_cotangent and x shape {expected_full_shape}, got {direct_cotangent.shape} and {x.shape}"
        )
    if norm_weight.shape != (hidden_dim,) or inverse_rms.shape != (rows,):
        raise ValueError(
            f"expected norm_weight[{hidden_dim}] and inverse_rms[{rows}], got {norm_weight.shape} and {inverse_rms.shape}"
        )
    if not (gate_preactivation_cotangent.dtype == w_down.dtype == direct_cotangent.dtype == x.dtype):
        raise ValueError("GEMM and full-width tensor inputs must have the same dtype")
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"RMS backward producer requires BF16 inputs, got {x.dtype}")
    # The epilogue reads inverse_rms as the exact float32 reciprocal the forward retained; a
    # narrower dtype has already lost the precision the reverse algebra assumes. norm_weight is
    # unconstrained because both it and the reference promote to float32 before the row dot.
    if inverse_rms.dtype != jnp.float32:
        raise ValueError(f"inverse_rms must be float32, got {inverse_rms.dtype}")

    _, tile_n = tile_mn
    hidden_tiles = (hidden_dim + tile_n - 1) // tile_n
    launcher = _build_aliased_backward_producer_launcher(
        a_dtype=_cute_dtype(gate_preactivation_cotangent.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )

    tensor_spec = cjax.TensorSpec
    a_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    w_down_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    matrix_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    vector_spec = tensor_spec(divisibility=_VECTOR_DIVISIBILITY, static=False)
    partial_spec = tensor_spec(divisibility=(1, 1, 1), static=False)
    output_shape_dtype = (
        jax.ShapeDtypeStruct((1, rows, hidden_dim), x.dtype),
        jax.ShapeDtypeStruct((1, rows, hidden_tiles), jnp.float32),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=(a_spec, w_down_spec, matrix_spec, matrix_spec, vector_spec, vector_spec),
        output_spec=(matrix_spec, partial_spec),
        input_output_aliases={2: 0},
        use_static_tensors=False,
    )
    unweighted_cotangent, row_dot_partial = call(
        gate_preactivation_cotangent[None, :, :],
        w_down[None, :, :],
        direct_cotangent[None, :, :],
        x[None, :, :],
        norm_weight[None, :],
        inverse_rms[None, :],
    )
    return unweighted_cotangent[0], row_dot_partial[0]


def _rms_backward_consumer_kernel(
    unweighted_ref,
    row_dot_ref,
    x_ref,
    norm_weight_ref,
    inverse_rms_ref,
    output_ref,
):
    unweighted = plgpu.load(unweighted_ref).astype(jnp.float32)
    row_mean = plgpu.load(row_dot_ref).astype(jnp.float32)
    x = plgpu.load(x_ref).astype(jnp.float32)
    norm_weight = plgpu.load(norm_weight_ref).astype(jnp.float32)
    inverse_rms = plgpu.load(inverse_rms_ref).astype(jnp.float32)
    normalized_x = x * inverse_rms[:, None]
    weighted_cotangent = unweighted * norm_weight[None, :]
    output = (weighted_cotangent - normalized_x * row_mean[:, None]) * inverse_rms[:, None]
    plgpu.store(output_ref, output.astype(output_ref.dtype))


@functools.lru_cache(maxsize=None)
def _rms_backward_consumer_call(rows: int, hidden_dim: int, dtype, norm_weight_dtype):
    output_shape = jax.ShapeDtypeStruct((rows, hidden_dim), dtype)
    input_shapes = (
        output_shape,
        jax.ShapeDtypeStruct((rows,), jnp.float32),
        output_shape,
        jax.ShapeDtypeStruct((hidden_dim,), norm_weight_dtype),
        jax.ShapeDtypeStruct((rows,), jnp.float32),
    )
    body_cost = pl.estimate_cost(exact_rms_backward_consumer, *input_shapes)
    cost_estimate = with_io_bytes_accessed(
        body_cost,
        kernel_inputs_specs=input_shapes,
        kernel_outputs_specs=(output_shape,),
    )
    return pl.pallas_call(
        _rms_backward_consumer_kernel,
        out_shape=output_shape,
        in_specs=(
            pl.BlockSpec((_CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N), lambda i, j: (i, j)),
            pl.BlockSpec((_CONSUMER_BLOCK_M,), lambda i, j: (i,)),
            pl.BlockSpec((_CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N), lambda i, j: (i, j)),
            pl.BlockSpec((_CONSUMER_BLOCK_N,), lambda i, j: (j,)),
            pl.BlockSpec((_CONSUMER_BLOCK_M,), lambda i, j: (i,)),
        ),
        out_specs=pl.BlockSpec((_CONSUMER_BLOCK_M, _CONSUMER_BLOCK_N), lambda i, j: (i, j)),
        grid=(pl.cdiv(rows, _CONSUMER_BLOCK_M), pl.cdiv(hidden_dim, _CONSUMER_BLOCK_N)),
        input_output_aliases={0: 0},
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=2),
        cost_estimate=cost_estimate,
        name="rms_backward_consumer",
    )


def quack_coda_rms_backward_consumer(
    unweighted_cotangent: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
) -> jax.Array:
    """Overwrite ``unweighted_cotangent`` with the final RMS input cotangent.

    This function is called inside the RMS-GatedNorm ``shard_map``; every argument is already
    local to one device. The aliased input/output contract lets XLA reuse the producer buffer.
    """
    if unweighted_cotangent.ndim != 2 or x.shape != unweighted_cotangent.shape:
        raise ValueError("unweighted_cotangent and x must be rank-2 arrays with matching shapes")
    rows, hidden_dim = x.shape
    if row_dot.shape != (rows,) or inverse_rms.shape != (rows,) or norm_weight.shape != (hidden_dim,):
        raise ValueError("RMS backward consumer vector dimensions do not agree")
    if unweighted_cotangent.dtype != x.dtype or x.dtype != jnp.bfloat16:
        raise ValueError("RMS backward consumer requires matching BF16 full-width tensors")
    if row_dot.dtype != jnp.float32 or inverse_rms.dtype != jnp.float32:
        raise ValueError("row_dot and inverse_rms must be float32")
    if rows % _CONSUMER_BLOCK_M != 0 or hidden_dim % _CONSUMER_BLOCK_N != 0:
        raise ValueError(
            f"rows and hidden_dim must be multiples of {_CONSUMER_BLOCK_M} and {_CONSUMER_BLOCK_N}, "
            f"got {rows} and {hidden_dim}"
        )
    call = _rms_backward_consumer_call(rows, hidden_dim, x.dtype, norm_weight.dtype)
    row_mean = row_dot / hidden_dim
    return call(unweighted_cotangent, row_mean, x, norm_weight, inverse_rms)


def quack_silu_backward_gemm(
    output_cotangent: jax.Array,
    w_up: jax.Array,
    preactivation: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Return the SiLU-input cotangent and recomputed SiLU output."""
    if output_cotangent.ndim != 2 or w_up.ndim != 2 or preactivation.ndim != 2:
        raise ValueError(
            f"expected output_cotangent[M,D], w_up[R,D], preactivation[M,R], got "
            f"{output_cotangent.shape}, {w_up.shape}, and {preactivation.shape}"
        )
    rows, hidden_dim = output_cotangent.shape
    rank, w_hidden_dim = w_up.shape
    if w_hidden_dim != hidden_dim or preactivation.shape != (rows, rank):
        raise ValueError("SiLU backward GEMM dimensions do not agree")
    if output_cotangent.dtype != w_up.dtype:
        raise ValueError("output_cotangent and w_up must have the same dtype")
    if output_cotangent.dtype != jnp.bfloat16:
        raise ValueError(f"SiLU backward GEMM requires BF16 inputs, got {output_cotangent.dtype}")

    launcher = _build_silu_backward_launcher(
        a_dtype=_cute_dtype(output_cotangent.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    matrix_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    output_shape_dtype = (
        jax.ShapeDtypeStruct((1, rows, rank), output_cotangent.dtype),
        jax.ShapeDtypeStruct((1, rows, rank), output_cotangent.dtype),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=(matrix_spec, matrix_spec, matrix_spec),
        output_spec=(matrix_spec, matrix_spec),
        use_static_tensors=False,
    )
    preactivation_cotangent, postactivation = call(
        output_cotangent[None, :, :],
        w_up[None, :, :],
        preactivation[None, :, :],
    )
    return preactivation_cotangent[0], postactivation[0]
