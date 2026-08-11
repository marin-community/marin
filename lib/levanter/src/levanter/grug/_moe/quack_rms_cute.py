#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX bridge for QuACK's SM100 GEMM with inverse-RMS scaling and SiLU epilogue."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import quack.utils as quack_utils
from levanter.cutlass_kernel_cache import cute_launcher_factory, cutlass_call
from quack.activation import act_fn_map, dact_fn_map
from quack.cute_dsl_utils import get_max_active_clusters, mlir_namedtuple
from quack.epi_ops import (
    ColVecLoad,
    ColVecReduce,
    RowVecLoad,
    Scalar,
    TileLoad,
    TileStore,
    colvec_reduce_accumulate,
    vec_multiply,
)
from quack.gemm_act import GemmActMixin
from quack.gemm_dact import GemmDActSm100
from quack.gemm_norm_act import GemmNormActSm100
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
_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}


class _GemmRmsBackwardMixin(GemmActMixin):
    """GEMM producer for CODA RMS backward partials.

    The GEMM result is rounded to BF16 before the BF16 residual add, matching
    stock JAX. The epilogue computes ``wdy = du * gamma`` and emits row
    partials for ``sum(x_hat * wdy)``. It stores BF16 ``du`` so the consumer
    can apply ``gamma`` while emitting the input gradient.
    """

    _epi_ops = (  # pyrefly: ignore[bad-override-mutable-attribute]
        Scalar("alpha"),
        Scalar("beta"),
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
        alpha: cutlass.Float32 | cute.Tensor | None = None
        beta: cutlass.Float32 | cute.Tensor | None = None
        mNormWeight: cute.Tensor | None = None
        mInverseRms: cute.Tensor | None = None
        mDirectCotangent: cute.Tensor | None = None
        mX: cute.Tensor | None = None
        mRowDotPartial: cute.Tensor | None = None
        mAuxOut: cute.Tensor | None = None
        add_to_output: cutlass.Constexpr[bool] = False
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

        rD = tRS_rD.load()
        if cutlass.const_expr(hasattr(params, "alpha") and params.alpha is not None):
            rD *= quack_utils.load_scalar_or_pointer(params.alpha)
        tRS_rUnweightedCotangent = cute.make_rmem_tensor_like(tRS_rD, self.a_dtype)
        tRS_rUnweightedCotangent.store(rD.to(self.a_dtype))
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


def _cute_dtype(dtype: jnp.dtype) -> type[cutlass.Numeric]:
    return _JAX_TO_CUTE[jnp.dtype(dtype)]


def _max_active_clusters(cluster_mnk: tuple[int, int, int]) -> int:
    if jax.default_backend() == "cpu":
        return _FALLBACK_MAX_ACTIVE_CLUSTERS
    try:
        return get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    except AssertionError:
        # CoreWeave's JAX image has CPU-only Torch. QuACK queries occupancy
        # through torch.cuda, so use the measured GB200 fallback in that image.
        return _FALLBACK_MAX_ACTIVE_CLUSTERS


@cute_launcher_factory
def _build_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build ``post = silu((a @ b) * inverse_rms)`` without storing the preactivation."""

    @cute.jit
    def launcher(stream, mA, mB, mInverseRms, mPostAct):
        gemm = GemmNormActSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = GemmActMixin.EpilogueArguments(
            mPostAct,
            act_fn_map["silu"],
            mRowVecBroadcast=None,
            mColVecBroadcast=mInverseRms,
        )
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mA, mB, None, None, epilogue, scheduler, None, stream)

    return launcher


@cute_launcher_factory
def _build_training_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build the training variant, retaining the normalized preactivation for reverse mode."""

    @cute.jit
    def launcher(stream, mA, mB, mInverseRms, mPreAct, mPostAct):
        gemm = GemmNormActSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        epilogue = GemmActMixin.EpilogueArguments(
            mPostAct,
            act_fn_map["silu"],
            mRowVecBroadcast=None,
            mColVecBroadcast=mInverseRms,
        )
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        gemm(mA, mB, None, mPreAct, epilogue, scheduler, None, stream)

    return launcher


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
        mDirectCotangent,
        mX,
        mNormWeight,
        mInverseRms,
        mWeightedCotangent,
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
            mDirectCotangent=mDirectCotangent,
            mX=mX,
            mRowDotPartial=mRowDotPartial,
            mAuxOut=mWeightedCotangent,
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


def quack_rms_scaled_silu_gemm(
    a: jax.Array,
    b: jax.Array,
    inverse_rms: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> jax.Array:
    """Return ``silu((a @ b) * inverse_rms[:, None])`` using QuACK on SM100.

    All inputs must be device-local. Call this function inside an explicit
    :func:`jax.shard_map` when the corresponding global arrays are sharded.
    """
    if a.ndim != 2 or b.ndim != 2 or inverse_rms.ndim != 1:
        raise ValueError(f"expected a[M,K], b[K,N], inverse_rms[M], got {a.shape}, {b.shape}, {inverse_rms.shape}")
    rows, contracting = a.shape
    if b.shape[0] != contracting:
        raise ValueError(f"contracting dimensions differ: {a.shape} and {b.shape}")
    if inverse_rms.shape[0] != rows:
        raise ValueError(f"inverse_rms rows differ: {inverse_rms.shape[0]} != {rows}")
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")

    max_active_clusters = _max_active_clusters(cluster_mnk)
    launcher = _build_launcher(
        a_dtype=_cute_dtype(a.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    # QuACK's fixed-M ABI carries a trailing batch mode even for one GEMM. The
    # physical singleton-first views below expose logical [M,K,L], [N,K,L],
    # and [M,N,L] without materializing transposes.
    a_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    b_spec = tensor_spec(mode=(2, 1, 0), divisibility=(1, 1, 8), static=False)
    inverse_rms_spec = tensor_spec(divisibility=(1, 4), static=False)
    output_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(jax.ShapeDtypeStruct((1, rows, b.shape[1]), a.dtype),),
        input_spec=(a_spec, b_spec, inverse_rms_spec),
        output_spec=(output_spec,),
        use_static_tensors=False,
    )
    (postact,) = call(a[None, :, :], b[None, :, :], inverse_rms[None, :])
    return postact[0]


def quack_rms_scaled_silu_gemm_with_preactivation(
    a: jax.Array,
    b: jax.Array,
    inverse_rms: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Return the FP32 preactivation and activated BF16 output for reverse mode."""
    if a.ndim != 2 or b.ndim != 2 or inverse_rms.ndim != 1:
        raise ValueError(f"expected a[M,K], b[K,N], inverse_rms[M], got {a.shape}, {b.shape}, {inverse_rms.shape}")
    rows, contracting = a.shape
    if b.shape[0] != contracting:
        raise ValueError(f"contracting dimensions differ: {a.shape} and {b.shape}")
    if inverse_rms.shape[0] != rows:
        raise ValueError(f"inverse_rms rows differ: {inverse_rms.shape[0]} != {rows}")
    if a.dtype != b.dtype:
        raise ValueError(f"a and b must have the same dtype, got {a.dtype} and {b.dtype}")

    max_active_clusters = _max_active_clusters(cluster_mnk)
    launcher = _build_training_launcher(
        a_dtype=_cute_dtype(a.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    a_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    b_spec = tensor_spec(mode=(2, 1, 0), divisibility=(1, 1, 8), static=False)
    inverse_rms_spec = tensor_spec(divisibility=(1, 4), static=False)
    output_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    output_shape_dtype = (
        jax.ShapeDtypeStruct((1, rows, b.shape[1]), jnp.float32),
        jax.ShapeDtypeStruct((1, rows, b.shape[1]), a.dtype),
    )
    call = cutlass_call(
        launcher,
        output_shape_dtype=output_shape_dtype,
        input_spec=(a_spec, b_spec, inverse_rms_spec),
        output_spec=(output_spec, output_spec),
        use_static_tensors=False,
    )
    preactivation, postactivation = call(a[None, :, :], b[None, :, :], inverse_rms[None, :])
    return preactivation[0], postactivation[0]


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
    """Produce BF16 ``du`` plus RMS-row partials.

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

    _, tile_n = tile_mn
    hidden_tiles = (hidden_dim + tile_n - 1) // tile_n
    max_active_clusters = _max_active_clusters(cluster_mnk)
    launcher = _build_backward_producer_launcher(
        a_dtype=_cute_dtype(gate_preactivation_cotangent.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )

    tensor_spec = cjax.TensorSpec
    a_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    w_down_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    matrix_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    vector_spec = tensor_spec(divisibility=(1, 4), static=False)
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


def quack_silu_backward_gemm(
    output_cotangent: jax.Array,
    w_up: jax.Array,
    preactivation: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Fuse ``output_cotangent @ w_up.T`` with the SiLU derivative epilogue."""
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

    max_active_clusters = _max_active_clusters(cluster_mnk)
    launcher = _build_silu_backward_launcher(
        a_dtype=_cute_dtype(output_cotangent.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=max_active_clusters,
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    matrix_spec = tensor_spec(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
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


__all__ = [
    "quack_coda_rms_backward_producer",
    "quack_rms_scaled_silu_gemm",
    "quack_rms_scaled_silu_gemm_with_preactivation",
    "quack_silu_backward_gemm",
]
