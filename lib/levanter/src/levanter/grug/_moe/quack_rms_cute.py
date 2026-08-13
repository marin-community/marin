#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX bridges for the SM100 RMS-GatedNorm reverse kernels."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from levanter.grug._moe.quack_moe_cute import _cute_dtype
from quack.activation import dact_fn_map
from quack.cute_dsl_utils import mlir_namedtuple
from quack.epi_ops import (
    ColVecLoad,
    ColVecReduce,
    RowVecLoad,
    RowVecReduce,
    TileLoad,
    colvec_reduce_accumulate,
    rowvec_reduce_accumulate,
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
_SM100_MULTIPROCESSORS = 148


def _max_active_clusters(cluster_mnk: tuple[int, int, int]) -> int:
    """Size the SM100 persistent grid without relying on PyTorch's CUDA runtime."""
    if jax.default_backend() == "cpu":
        return _FALLBACK_MAX_ACTIVE_CLUSTERS
    device = jax.local_devices()[0]
    compute_capability = getattr(device, "compute_capability", None)
    if callable(compute_capability):
        compute_capability = compute_capability()
    if isinstance(compute_capability, str):
        major, _, minor = compute_capability.partition(".")
        compute_capability = (int(major), int(minor or 0))
    if not isinstance(compute_capability, tuple) or compute_capability[:2] != (10, 0):
        raise ValueError(f"RMS-GatedNorm CuTe kernels require SM100, got {device.device_kind} ({compute_capability})")
    cluster_size = cluster_mnk[0] * cluster_mnk[1]
    return _SM100_MULTIPROCESSORS // cluster_size


class _GemmRmsBackwardPartialsMixin(GemmActMixin):
    """CuTe epilogue that emits only RMS row and norm-gain partials."""

    _epi_ops = (  # pyrefly: ignore[bad-override-mutable-attribute]
        RowVecLoad("mNormWeight"),
        ColVecLoad("mInverseRms"),
        TileLoad("mOutputCotangent"),
        TileLoad("mGate"),
        TileLoad("mX"),
        ColVecReduce("mRowDotPartial"),
        RowVecReduce("mNormWeightPartial"),
    )
    _extra_param_fields = ()

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):  # pyrefly: ignore[bad-override]
        mNormWeight: cute.Tensor | None = None
        mInverseRms: cute.Tensor | None = None
        mOutputCotangent: cute.Tensor | None = None
        mGate: cute.Tensor | None = None
        mX: cute.Tensor | None = None
        mRowDotPartial: cute.Tensor | None = None
        mNormWeightPartial: cute.Tensor | None = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        values = self._epi_ops_to_params_dict(args)
        return self.EpilogueParams(**values)

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrNormWeight = epi_loop_tensors.get("mNormWeight")
        tDrInverseRms = epi_loop_tensors.get("mInverseRms")
        tRS_rOutputCotangent = epi_loop_tensors.get("mOutputCotangent")
        tRS_rGate = epi_loop_tensors.get("mGate")
        tRS_rX = epi_loop_tensors.get("mX")
        tDrRowDot = epi_loop_tensors.get("mRowDotPartial")
        tDrNormWeightPartial = epi_loop_tensors.get("mNormWeightPartial")

        tRS_rUnweightedCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rUnweightedCotangent.store(tRS_rD.load().to(self.a_dtype))
        tRS_rDirectCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rDirectCotangent.store(tRS_rOutputCotangent.load() * tRS_rGate.load())
        tRS_rUnweightedCotangent.store(tRS_rUnweightedCotangent.load() + tRS_rDirectCotangent.load())
        tRS_rD.store(tRS_rUnweightedCotangent.load().to(tRS_rD.element_type))

        tRS_rXHat = cute.make_rmem_tensor_like(tRS_rD)
        for i in cutlass.range(cute.size(tRS_rXHat), unroll_full=True):
            tRS_rXHat[i] = tRS_rX[i].to(tRS_rD.element_type) * tDrInverseRms[i]

        tRS_rNormWeightProduct = cute.make_rmem_tensor_like(tRS_rD)
        tRS_rNormWeightProduct.store(tRS_rD.load() * tRS_rXHat.load())
        rowvec_reduce_accumulate(self, tDrNormWeightPartial, tRS_rNormWeightProduct)
        vec_multiply(self, tRS_rD, None, tDrNormWeight)
        tRS_rRowDotProduct = cute.make_rmem_tensor_like(tRS_rD)
        tRS_rRowDotProduct.store(tRS_rD.load() * tRS_rXHat.load())
        colvec_reduce_accumulate(self, tDrRowDot, tRS_rRowDotProduct)
        return ()


class _GemmRmsBackwardPartialsSm100(  # pyrefly: ignore[inconsistent-inheritance]
    _GemmRmsBackwardPartialsMixin, GemmSm100
):
    pass


@cute_launcher_factory
def _build_backward_producer_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build the CODA backward producer GEMM and its row partials."""

    @cute.jit
    def launcher(
        stream,
        mGatePreactivationCotangent,
        mWDown,
        mOutputCotangent,
        mGate,
        mX,
        mNormWeight,
        mInverseRms,
        mRowDotPartial,
        mNormWeightPartial,
    ):
        gemm = _GemmRmsBackwardPartialsSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = _GemmRmsBackwardPartialsMixin.EpilogueArguments(
            mNormWeight=mNormWeight,
            mInverseRms=mInverseRms,
            mOutputCotangent=mOutputCotangent,
            mGate=mGate,
            mX=mX,
            mRowDotPartial=mRowDotPartial,
            mNormWeightPartial=mNormWeightPartial,
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


class _GemmRmsBackwardConsumerMixin(GemmActMixin):
    """Recompute the low-rank product and emit final RMS input cotangents."""

    _epi_ops = (  # pyrefly: ignore[bad-override-mutable-attribute]
        RowVecLoad("mNormWeight"),
        ColVecLoad("mInverseRms"),
        ColVecLoad("mRowMean"),
        TileLoad("mOutputCotangent"),
        TileLoad("mGate"),
        TileLoad("mX"),
    )
    _extra_param_fields = ()

    @mlir_namedtuple
    class EpilogueArguments(NamedTuple):  # pyrefly: ignore[bad-override]
        mNormWeight: cute.Tensor | None = None
        mInverseRms: cute.Tensor | None = None
        mRowMean: cute.Tensor | None = None
        mOutputCotangent: cute.Tensor | None = None
        mGate: cute.Tensor | None = None
        mX: cute.Tensor | None = None
        rounding_mode: cutlass.Constexpr[int] = RoundingMode.RN

    def epi_to_underlying_arguments(self, args, *, loc=None, ip=None):
        self.rounding_mode = args.rounding_mode
        return self.EpilogueParams(**self._epi_ops_to_params_dict(args))

    @cute.jit
    def epi_visit_subtile(self, params, epi_loop_tensors, tRS_rD, tRS_rC=None):
        tDrNormWeight = epi_loop_tensors.get("mNormWeight")
        tDrInverseRms = epi_loop_tensors.get("mInverseRms")
        tDrRowMean = epi_loop_tensors.get("mRowMean")
        tRS_rOutputCotangent = epi_loop_tensors.get("mOutputCotangent")
        tRS_rGate = epi_loop_tensors.get("mGate")
        tRS_rX = epi_loop_tensors.get("mX")

        tRS_rUnweightedCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rUnweightedCotangent.store(tRS_rD.load().to(self.a_dtype))
        tRS_rDirectCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rDirectCotangent.store(tRS_rOutputCotangent.load() * tRS_rGate.load())
        tRS_rUnweightedCotangent.store(tRS_rUnweightedCotangent.load() + tRS_rDirectCotangent.load())

        for i in cutlass.range(cute.size(tRS_rD), unroll_full=True):
            x_hat = tRS_rX[i].to(tRS_rD.element_type) * tDrInverseRms[i]
            weighted = tRS_rUnweightedCotangent[i].to(tRS_rD.element_type) * tDrNormWeight[i]
            tRS_rD[i] = (weighted - x_hat * tDrRowMean[i]) * tDrInverseRms[i]
        return ()


class _GemmRmsBackwardConsumerSm100(  # pyrefly: ignore[inconsistent-inheritance]
    _GemmRmsBackwardConsumerMixin, GemmSm100
):
    pass


@cute_launcher_factory
def _build_backward_consumer_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    @cute.jit
    def launcher(
        stream,
        mGatePreactivationCotangent,
        mWDown,
        mOutputCotangent,
        mGate,
        mRowMean,
        mX,
        mNormWeight,
        mInverseRms,
        mDx,
    ):
        gemm = _GemmRmsBackwardConsumerSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = _GemmRmsBackwardConsumerMixin.EpilogueArguments(
            mNormWeight=mNormWeight,
            mInverseRms=mInverseRms,
            mRowMean=mRowMean,
            mOutputCotangent=mOutputCotangent,
            mGate=mGate,
            mX=mX,
        )
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mGatePreactivationCotangent, mWDown, mDx, None, epilogue, scheduler, None, stream)

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
    output_cotangent: jax.Array,
    gate: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_BACKWARD_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_BACKWARD_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Produce only RMS row-dot and norm-gain partials.

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
    if (
        output_cotangent.shape != expected_full_shape
        or gate.shape != expected_full_shape
        or x.shape != expected_full_shape
    ):
        raise ValueError(
            f"expected full-width tensors with shape {expected_full_shape}, got "
            f"{output_cotangent.shape}, {gate.shape}, and {x.shape}"
        )
    if norm_weight.shape != (hidden_dim,) or inverse_rms.shape != (rows,):
        raise ValueError(
            f"expected norm_weight[{hidden_dim}] and inverse_rms[{rows}], got {norm_weight.shape} and {inverse_rms.shape}"
        )
    if not (gate_preactivation_cotangent.dtype == w_down.dtype == output_cotangent.dtype == gate.dtype == x.dtype):
        raise ValueError("GEMM and full-width tensor inputs must have the same dtype")
    if x.dtype != jnp.bfloat16:
        raise ValueError(f"RMS backward producer requires BF16 inputs, got {x.dtype}")
    # The epilogue reads inverse_rms as the exact float32 reciprocal the forward retained; a
    # narrower dtype has already lost the precision the reverse algebra assumes. norm_weight is
    # unconstrained because both it and the reference promote to float32 before the row dot.
    if inverse_rms.dtype != jnp.float32:
        raise ValueError(f"inverse_rms must be float32, got {inverse_rms.dtype}")

    tile_m, tile_n = tile_mn
    row_tiles = (rows + tile_m - 1) // tile_m
    hidden_tiles = (hidden_dim + tile_n - 1) // tile_n
    launcher = _build_backward_producer_launcher(
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
        jax.ShapeDtypeStruct((1, rows, hidden_tiles), jnp.float32),
        jax.ShapeDtypeStruct((1, row_tiles, hidden_dim), jnp.float32),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=(a_spec, w_down_spec, matrix_spec, matrix_spec, matrix_spec, vector_spec, vector_spec),
        output_spec=(partial_spec, partial_spec),
        use_static_tensors=False,
    )
    row_dot_partial, norm_weight_partial = call(
        gate_preactivation_cotangent[None, :, :],
        w_down[None, :, :],
        output_cotangent[None, :, :],
        gate[None, :, :],
        x[None, :, :],
        norm_weight[None, :],
        inverse_rms[None, :],
    )
    return row_dot_partial[0], norm_weight_partial[0]


def quack_coda_rms_backward_consumer(
    gate_preactivation_cotangent: jax.Array,
    w_down: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    row_dot: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    inverse_rms: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_BACKWARD_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_BACKWARD_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> jax.Array:
    """Recompute the low-rank product and emit final RMS input cotangents."""
    if gate_preactivation_cotangent.ndim != 2 or w_down.ndim != 2:
        raise ValueError("gate_preactivation_cotangent and w_down must be rank-2")
    rows, rank = gate_preactivation_cotangent.shape
    hidden_dim, w_rank = w_down.shape
    if rank != w_rank:
        raise ValueError("RMS backward consumer contracting dimensions do not agree")
    full_shape = (rows, hidden_dim)
    if output_cotangent.shape != full_shape or gate.shape != full_shape or x.shape != full_shape:
        raise ValueError("RMS backward consumer full-width dimensions do not agree")
    rows, hidden_dim = x.shape
    if row_dot.shape != (rows,) or inverse_rms.shape != (rows,) or norm_weight.shape != (hidden_dim,):
        raise ValueError("RMS backward consumer vector dimensions do not agree")
    if not (
        gate_preactivation_cotangent.dtype
        == w_down.dtype
        == output_cotangent.dtype
        == gate.dtype
        == x.dtype
        == jnp.bfloat16
    ):
        raise ValueError("RMS backward consumer requires matching BF16 tensors")
    if row_dot.dtype != jnp.float32 or inverse_rms.dtype != jnp.float32:
        raise ValueError("row_dot and inverse_rms must be float32")

    launcher = _build_backward_consumer_launcher(
        a_dtype=_cute_dtype(x.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    matrix_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    vector_spec = tensor_spec(divisibility=_VECTOR_DIVISIBILITY, static=False)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((1, rows, hidden_dim), x.dtype),
        input_spec=(
            matrix_spec,
            matrix_spec,
            matrix_spec,
            matrix_spec,
            vector_spec,
            matrix_spec,
            vector_spec,
            vector_spec,
        ),
        output_spec=(matrix_spec,),
        use_static_tensors=False,
    )
    row_mean = row_dot / hidden_dim
    return call(
        gate_preactivation_cotangent[None, :, :],
        w_down[None, :, :],
        output_cotangent[None, :, :],
        gate[None, :, :],
        row_mean[None, :],
        x[None, :, :],
        norm_weight[None, :],
        inverse_rms[None, :],
    )[0]


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
