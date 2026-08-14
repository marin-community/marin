#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""JAX bridges for the SM100 RMS-GatedNorm reverse kernels."""

from __future__ import annotations

import functools
import math
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
import quack.copy_utils as copy_utils
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
from quack.gemm_default_epi import GemmDefaultEpiMixin, GemmDefaultSm100
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
_GATE_REVERSE_BLOCK_N = 32
_RMS_REVERSE_BLOCK_M = 64
_RMS_REVERSE_BLOCK_D = 64


class _GateAccumulatorInplace:
    """Compute the Sigmoid-output cotangent into a donated normalized buffer."""

    def __init__(self, dtype: type[cutlass.Numeric]):
        self.dtype = dtype
        self.num_threads = 256
        self.threads_per_row = 16
        self.vector_size = 128 // dtype.width
        self.tile_mn = (
            self.num_threads // self.threads_per_row,
            self.threads_per_row * self.vector_size,
        )

    @cute.jit
    def __call__(
        self,
        mNormalized: cute.Tensor,
        mOutputCotangent: cute.Tensor,
        mGate: cute.Tensor,
        mGateAccumulator: cute.Tensor,
        stream,
    ):
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype,
            self.threads_per_row,
            self.num_threads,
            num_copy_elems=self.vector_size,
        )
        self.kernel(mNormalized, mOutputCotangent, mGate, mGateAccumulator, tiled_copy).launch(
            grid=[
                cute.ceil_div(mNormalized.shape[0], self.tile_mn[0]),
                cute.ceil_div(mNormalized.shape[1], self.tile_mn[1]),
                mNormalized.shape[2],
            ],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mNormalized: cute.Tensor,
        mOutputCotangent: cute.Tensor,
        mGate: cute.Tensor,
        mGateAccumulator: cute.Tensor,
        tiled_copy: cute.TiledCopy,
    ):
        thread_idx, _, _ = cute.arch.thread_idx()
        tile_m, tile_n, batch_idx = cute.arch.block_idx()
        normalized, output_cotangent, gate, gate_accumulator = (
            cute.local_tile(tensor[None, None, batch_idx], self.tile_mn, (tile_m, tile_n))
            for tensor in (mNormalized, mOutputCotangent, mGate, mGateAccumulator)
        )
        thread_copy = tiled_copy.get_slice(thread_idx)
        normalized_gmem = thread_copy.partition_S(normalized)
        output_cotangent_gmem = thread_copy.partition_S(output_cotangent)
        gate_gmem = thread_copy.partition_S(gate)
        gate_accumulator_gmem = thread_copy.partition_D(gate_accumulator)
        normalized_rmem = cute.make_rmem_tensor_like(normalized_gmem)
        output_cotangent_rmem = cute.make_rmem_tensor_like(output_cotangent_gmem)
        gate_rmem = cute.make_rmem_tensor_like(gate_gmem)
        gate_accumulator_rmem = cute.make_rmem_tensor_like(gate_accumulator_gmem)
        copy_utils.copy(normalized_gmem, normalized_rmem)
        copy_utils.copy(output_cotangent_gmem, output_cotangent_rmem)
        copy_utils.copy(gate_gmem, gate_rmem)
        one = self.dtype(1)
        for i in cutlass.range(cute.size(gate_accumulator_rmem), unroll_full=True, vectorize=True):
            gate_cotangent = (output_cotangent_rmem[i] * normalized_rmem[i]).to(self.dtype)
            sigmoid_cotangent = (gate_rmem[i] * (one - gate_rmem[i])).to(self.dtype)
            gate_accumulator_rmem[i] = (gate_cotangent * sigmoid_cotangent).to(self.dtype)
        copy_utils.copy(gate_accumulator_rmem, gate_accumulator_gmem)


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


@cute_launcher_factory
def _build_cute_gate_accumulator_launcher(*, a_dtype: type[cutlass.Numeric]):
    @cute.jit
    def launcher(stream, mNormalizedAndGateAccumulator, mOutputCotangent, mGate):
        _GateAccumulatorInplace(a_dtype)(
            mNormalizedAndGateAccumulator,
            mOutputCotangent,
            mGate,
            mNormalizedAndGateAccumulator,
            stream,
        )

    return launcher


@cute_launcher_factory
def _build_gate_silu_reverse_launcher(
    *,
    a_dtype: type[cutlass.Numeric],
    tile_mn: tuple[int, int],
    cluster_mnk: tuple[int, int, int],
    max_active_clusters: int,
    max_swizzle: int,
):
    """Build output-gate reverse, SiLU reverse, and up-weight reverse as one call."""

    @cute.jit
    def launcher(
        stream,
        mNormalizedAndGateAccumulator,
        mOutputCotangent,
        mGate,
        mWUp,
        mGatePreactivation,
        mGatePreactivationCotangent,
        mGateHidden,
        mWUpCotangent,
    ):
        _GateAccumulatorInplace(a_dtype)(
            mNormalizedAndGateAccumulator,
            mOutputCotangent,
            mGate,
            mNormalizedAndGateAccumulator,
            stream,
        )

        dact_gemm = GemmDActSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        dact_epilogue = GemmActMixin.EpilogueArguments(mGateHidden, dact_fn_map["silu"])
        scheduler = make_scheduler_args(max_active_clusters, max_swizzle, None)
        dact_gemm(
            mNormalizedAndGateAccumulator,
            mWUp,
            mGatePreactivationCotangent,
            mGatePreactivation,
            dact_epilogue,
            scheduler,
            None,
            stream,
        )

        gate_hidden_transposed = cute.make_tensor(
            mGateHidden.iterator,
            cute.select(mGateHidden.layout, mode=[1, 0, 2]),
        )
        gate_accumulator_transposed = cute.make_tensor(
            mNormalizedAndGateAccumulator.iterator,
            cute.select(mNormalizedAndGateAccumulator.layout, mode=[1, 0, 2]),
        )
        weight_gemm = GemmDefaultSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        weight_gemm(
            gate_hidden_transposed,
            gate_accumulator_transposed,
            mWUpCotangent,
            None,
            GemmDefaultEpiMixin.EpilogueArguments(),
            scheduler,
            None,
            stream,
        )

    return launcher


@cute_launcher_factory
def _build_gate_silu_dact_launcher(
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
        mNormalized,
        mOutputCotangent,
        mGate,
        mWUp,
        mGatePreactivation,
        mGateAccumulator,
        mGatePreactivationCotangent,
        mGateHidden,
    ):
        _GateAccumulatorInplace(a_dtype)(
            mNormalized,
            mOutputCotangent,
            mGate,
            mGateAccumulator,
            stream,
        )
        dact_gemm = GemmDActSm100(
            _ACCUMULATOR_DTYPE,
            a_dtype,
            tile_mn,
            cluster_mnk,
            gather_A=False,
            use_clc_persistence=False,
        )
        dact_gemm(
            mGateAccumulator,
            mWUp,
            mGatePreactivationCotangent,
            mGatePreactivation,
            GemmActMixin.EpilogueArguments(mGateHidden, dact_fn_map["silu"]),
            make_scheduler_args(max_active_clusters, max_swizzle, None),
            None,
            stream,
        )

    return launcher


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
def _build_aliased_backward_consumer_launcher(
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
        # ``cutlass_call`` removes aliased outputs from the launcher ABI. Use the gate input
        # as the GEMM destination so the epilogue can load each gate tile before replacing it
        # with the corresponding RMS input cotangent.
        gemm(mGatePreactivationCotangent, mWDown, mGate, None, epilogue, scheduler, None, stream)

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


def _gate_accumulator_tile(output_cotangent, normalized, gate):
    gate_cotangent = (output_cotangent * normalized).astype(jnp.bfloat16)
    sigmoid_cotangent = (gate * (jnp.array(1, gate.dtype) - gate)).astype(jnp.bfloat16)
    return (gate_cotangent * sigmoid_cotangent).astype(jnp.bfloat16)


def _rms_gated_norm_reverse_kernel(
    output_cotangent_ref,
    x_ref,
    norm_weight_ref,
    w_down_ref,
    w_up_ref,
    inverse_rms_ref,
    gate_preactivation_ref,
    norm_weight_zero_ref,
    w_down_zero_ref,
    w_up_zero_ref,
    x_cotangent_ref,
    norm_weight_cotangent_ref,
    w_down_cotangent_ref,
    w_up_cotangent_ref,
    *,
    block_m: int,
    block_d: int,
):
    """Reverse one row tile and atomically accumulate all replicated parameter gradients."""
    del norm_weight_zero_ref, w_down_zero_ref, w_up_zero_ref
    row_tile = pl.program_id(0)
    start_m = row_tile * block_m
    span_m = pl.ds(start_m, block_m)
    rank = w_down_ref.shape[1]
    span_r = pl.ds(0, rank)
    inverse_rms = plgpu.load(inverse_rms_ref.at[span_m]).astype(jnp.float32)
    hidden_tiles = pl.cdiv(x_ref.shape[1], block_d)
    gate_preactivation = plgpu.load(gate_preactivation_ref.at[span_m, span_r])
    gate_hidden = jax.nn.silu(gate_preactivation.astype(jnp.float32)).astype(jnp.bfloat16)
    gate_preactivation_cotangent = jnp.zeros((block_m, rank), dtype=jnp.float32)

    def gate_body(i, accumulator):
        start_d = i * block_d
        span_d = pl.ds(start_d, block_d)
        output_cotangent = plgpu.load(output_cotangent_ref.at[span_m, span_d])
        x = plgpu.load(x_ref.at[span_m, span_d]).astype(jnp.float32)
        norm_weight = plgpu.load(norm_weight_ref.at[span_d]).astype(jnp.float32)
        normalized = (x * inverse_rms[:, None] * norm_weight[None, :]).astype(jnp.bfloat16)
        w_up = plgpu.load(w_up_ref.at[span_r, span_d])
        gate_preactivation_up = pl.dot(gate_hidden, w_up).astype(jnp.bfloat16)
        gate = jax.nn.sigmoid(gate_preactivation_up.astype(jnp.float32)).astype(jnp.bfloat16)
        gate_accumulator = _gate_accumulator_tile(output_cotangent, normalized, gate)
        direct = (output_cotangent * gate).astype(jnp.bfloat16)
        plgpu.store(x_cotangent_ref.at[span_m, span_d], direct)
        w_up_partial = pl.dot(gate_hidden.T, gate_accumulator)
        plgpu.atomic_add(w_up_cotangent_ref, (span_r, span_d), w_up_partial)
        return accumulator + pl.dot(gate_accumulator, w_up.T)

    gate_preactivation_cotangent = jax.lax.fori_loop(0, hidden_tiles, gate_body, gate_preactivation_cotangent)
    sigmoid = jax.nn.sigmoid(gate_preactivation.astype(jnp.float32))
    silu_derivative = sigmoid * (1 + gate_preactivation.astype(jnp.float32) * (1 - sigmoid))
    gate_preactivation_cotangent = (gate_preactivation_cotangent * silu_derivative).astype(jnp.bfloat16)
    row_dot = jnp.zeros((block_m,), dtype=jnp.float32)

    def rms_partials_body(i, row_dot):
        start_d = i * block_d
        span_d = pl.ds(start_d, block_d)
        w_down = plgpu.load(w_down_ref.at[span_d, span_r])
        x = plgpu.load(x_ref.at[span_m, span_d]).astype(jnp.float32)
        norm_weight = plgpu.load(norm_weight_ref.at[span_d]).astype(jnp.float32)
        normalized = (x * inverse_rms[:, None] * norm_weight[None, :]).astype(jnp.bfloat16)
        w_down_partial = pl.dot(normalized.T, gate_preactivation_cotangent)
        plgpu.atomic_add(w_down_cotangent_ref, (span_d, span_r), w_down_partial)
        low_rank = pl.dot(gate_preactivation_cotangent, w_down.T).astype(jnp.bfloat16)
        direct = plgpu.load(x_cotangent_ref.at[span_m, span_d])
        unweighted = (low_rank + direct).astype(jnp.bfloat16)
        plgpu.store(x_cotangent_ref.at[span_m, span_d], unweighted)
        x_hat = x * inverse_rms[:, None]
        weighted = unweighted.astype(jnp.float32) * norm_weight[None, :]
        norm_weight_partial = jnp.sum(unweighted.astype(jnp.float32) * x_hat, axis=0)
        plgpu.atomic_add(norm_weight_cotangent_ref, span_d, norm_weight_partial)
        return row_dot + jnp.sum(weighted * x_hat, axis=1)

    row_dot = jax.lax.fori_loop(0, hidden_tiles, rms_partials_body, row_dot)
    row_mean = row_dot / x_ref.shape[1]

    def output_body(i, _):
        start_d = i * block_d
        span_d = pl.ds(start_d, block_d)
        unweighted = plgpu.load(x_cotangent_ref.at[span_m, span_d])
        x = plgpu.load(x_ref.at[span_m, span_d]).astype(jnp.float32)
        x_hat = x * inverse_rms[:, None]
        norm_weight = plgpu.load(norm_weight_ref.at[span_d]).astype(jnp.float32)
        weighted = unweighted.astype(jnp.float32) * norm_weight[None, :]
        x_cotangent = (weighted - x_hat * row_mean[:, None]) * inverse_rms[:, None]
        plgpu.store(x_cotangent_ref.at[span_m, span_d], x_cotangent.astype(x_cotangent_ref.dtype))

    jax.lax.fori_loop(0, hidden_tiles, output_body, None)


def _matrix_bytes(shape: tuple[int, int], dtype) -> int:
    return math.prod(shape) * jnp.dtype(dtype).itemsize


def _gate_accumulator_inplace_kernel(
    normalized_ref,
    output_cotangent_ref,
    gate_ref,
    gate_accumulator_ref,
):
    # BlockSpec has already sliced every ref to this program's tile.
    normalized = plgpu.load(normalized_ref)
    output_cotangent = plgpu.load(output_cotangent_ref)
    gate = plgpu.load(gate_ref)
    gate_cotangent = (output_cotangent * normalized).astype(jnp.bfloat16)
    sigmoid_cotangent = (gate * (jnp.array(1, gate.dtype) - gate)).astype(jnp.bfloat16)
    plgpu.store(gate_accumulator_ref, (gate_cotangent * sigmoid_cotangent).astype(gate_accumulator_ref.dtype))


@functools.lru_cache(maxsize=None)
def _gate_accumulator_inplace_call(rows: int, hidden_dim: int, dtype):
    block_m = _RMS_REVERSE_BLOCK_M
    block_d = _RMS_REVERSE_BLOCK_D
    shape = jax.ShapeDtypeStruct((rows, hidden_dim), dtype)
    block_spec = pl.BlockSpec((block_m, block_d), lambda i, j: (i, j))
    return pl.pallas_call(
        _gate_accumulator_inplace_kernel,
        out_shape=shape,
        in_specs=(block_spec, block_spec, block_spec),
        out_specs=block_spec,
        input_output_aliases={0: 0},
        grid=(pl.cdiv(rows, block_m), pl.cdiv(hidden_dim, block_d)),
        compiler_params=plgpu.CompilerParams(num_warps=4, num_stages=1),
        cost_estimate=pl.CostEstimate(
            flops=4 * rows * hidden_dim,
            transcendentals=0,
            bytes_accessed=4 * _matrix_bytes((rows, hidden_dim), dtype),
        ),
        name="gate_accumulator_inplace",
    )


def quack_gate_accumulator_inplace(
    normalized: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
) -> jax.Array:
    """Overwrite ``normalized`` with the exact BF16 output-gate accumulator."""
    if normalized.shape != output_cotangent.shape or normalized.shape != gate.shape:
        raise ValueError("gate accumulator inputs must have matching shapes")
    if normalized.ndim != 2 or normalized.dtype != jnp.bfloat16:
        raise ValueError("gate accumulator requires rank-2 BF16 inputs")
    if output_cotangent.dtype != normalized.dtype or gate.dtype != normalized.dtype:
        raise ValueError("gate accumulator inputs must have matching dtypes")
    rows, hidden_dim = normalized.shape
    if rows % _RMS_REVERSE_BLOCK_M or hidden_dim % _RMS_REVERSE_BLOCK_D:
        raise ValueError("gate accumulator dimensions must align with the RMS reverse tiles")
    return _gate_accumulator_inplace_call(rows, hidden_dim, normalized.dtype)(
        normalized,
        output_cotangent,
        gate,
    )


def _quack_cute_gate_accumulator_inplace(
    normalized: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
) -> jax.Array:
    """Run the CuTe gate-accumulator stage alone for accelerator parity checks."""
    if normalized.ndim != 2 or normalized.shape != output_cotangent.shape or normalized.shape != gate.shape:
        raise ValueError("gate accumulator inputs must be matching rank-2 arrays")
    if (
        normalized.dtype != jnp.bfloat16
        or output_cotangent.dtype != normalized.dtype
        or gate.dtype != normalized.dtype
    ):
        raise ValueError("gate accumulator requires matching BF16 inputs")
    rows, hidden_dim = normalized.shape
    launcher = _build_cute_gate_accumulator_launcher(a_dtype=_cute_dtype(normalized.dtype))
    matrix_spec = cjax.TensorSpec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    call = cutlass_call(
        launcher,
        input_output_aliases={0: 0},
        output_shape_dtype=jax.ShapeDtypeStruct((1, rows, hidden_dim), normalized.dtype),
        input_spec=(matrix_spec,) * 3,
        output_spec=(matrix_spec,),
        use_static_tensors=False,
    )
    return call(normalized[None, :, :], output_cotangent[None, :, :], gate[None, :, :])[0]


def _quack_coda_gate_silu_dact_components(
    normalized: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    w_up: jax.Array,
    gate_preactivation: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Run the fused elementwise and dSiLU stages without the weight reverse."""
    rows, hidden_dim = normalized.shape
    rank = w_up.shape[0]
    launcher = _build_gate_silu_dact_launcher(
        a_dtype=_cute_dtype(normalized.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )
    matrix_spec = cjax.TensorSpec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    call = cutlass_call(
        launcher,
        output_shape_dtype=(
            jax.ShapeDtypeStruct((1, rows, hidden_dim), normalized.dtype),
            jax.ShapeDtypeStruct((1, rows, rank), normalized.dtype),
            jax.ShapeDtypeStruct((1, rows, rank), normalized.dtype),
        ),
        input_spec=(matrix_spec,) * 5,
        output_spec=(matrix_spec,) * 3,
        use_static_tensors=False,
    )
    gate_accumulator, gate_preactivation_cotangent, gate_hidden = call(
        normalized[None, :, :],
        output_cotangent[None, :, :],
        gate[None, :, :],
        w_up[None, :, :],
        gate_preactivation[None, :, :],
    )
    return gate_accumulator[0], gate_preactivation_cotangent[0], gate_hidden[0]


def _quack_coda_gate_silu_reverse_components(
    normalized: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    w_up: jax.Array,
    gate_preactivation: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if normalized.ndim != 2 or normalized.shape != output_cotangent.shape or normalized.shape != gate.shape:
        raise ValueError("gate reverse full-width inputs must be matching rank-2 arrays")
    rows, hidden_dim = normalized.shape
    if w_up.ndim != 2:
        raise ValueError("w_up must be rank 2")
    rank, w_hidden_dim = w_up.shape
    if w_hidden_dim != hidden_dim or gate_preactivation.shape != (rows, rank):
        raise ValueError("gate reverse contraction dimensions do not agree")
    if not (
        normalized.dtype
        == output_cotangent.dtype
        == gate.dtype
        == w_up.dtype
        == gate_preactivation.dtype
        == jnp.bfloat16
    ):
        raise ValueError("gate reverse requires matching BF16 inputs")
    if rows % _RMS_REVERSE_BLOCK_M or hidden_dim % _RMS_REVERSE_BLOCK_D:
        raise ValueError("gate reverse dimensions must align with the elementwise tiles")

    launcher = _build_gate_silu_reverse_launcher(
        a_dtype=_cute_dtype(normalized.dtype),
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_active_clusters=_max_active_clusters(cluster_mnk),
        max_swizzle=max_swizzle,
    )
    tensor_spec = cjax.TensorSpec
    matrix_spec = tensor_spec(mode=_MATRIX_MODE, divisibility=_MATRIX_DIVISIBILITY, static=False)
    output_shape_dtype = (
        jax.ShapeDtypeStruct((1, rows, hidden_dim), normalized.dtype),
        jax.ShapeDtypeStruct((1, rows, rank), normalized.dtype),
        jax.ShapeDtypeStruct((1, rows, rank), normalized.dtype),
        jax.ShapeDtypeStruct((1, rank, hidden_dim), normalized.dtype),
    )
    call = cutlass_call(
        launcher,
        input_output_aliases={0: 0},
        output_shape_dtype=output_shape_dtype,
        input_spec=(matrix_spec,) * 5,
        output_spec=(matrix_spec,) * 4,
        use_static_tensors=False,
    )
    gate_accumulator, gate_preactivation_cotangent, gate_hidden, w_up_cotangent = call(
        normalized[None, :, :],
        output_cotangent[None, :, :],
        gate[None, :, :],
        w_up[None, :, :],
        gate_preactivation[None, :, :],
    )
    return gate_accumulator[0], gate_preactivation_cotangent[0], gate_hidden[0], w_up_cotangent[0]


def quack_coda_gate_silu_reverse(
    normalized: jax.Array,
    output_cotangent: jax.Array,
    gate: jax.Array,
    w_up: jax.Array,
    gate_preactivation: jax.Array,
    *,
    tile_mn: tuple[int, int] = _DEFAULT_TILE_MN,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER_MNK,
    max_swizzle: int = _DEFAULT_MAX_SWIZZLE,
) -> tuple[jax.Array, jax.Array]:
    """Return gate-preactivation and up-weight cotangents without exposing gate dY."""
    _, gate_preactivation_cotangent, _, w_up_cotangent = _quack_coda_gate_silu_reverse_components(
        normalized,
        output_cotangent,
        gate,
        w_up,
        gate_preactivation,
        tile_mn=tile_mn,
        cluster_mnk=cluster_mnk,
        max_swizzle=max_swizzle,
    )
    return gate_preactivation_cotangent, w_up_cotangent


@functools.lru_cache(maxsize=None)
def _rms_gated_norm_reverse_call(rows: int, hidden_dim: int, rank: int, dtype):
    block_m = _RMS_REVERSE_BLOCK_M
    block_d = _RMS_REVERSE_BLOCK_D
    cost = pl.CostEstimate(
        flops=10 * rows * hidden_dim * rank + 18 * rows * hidden_dim + 8 * rows * rank,
        transcendentals=rows * rank,
        bytes_accessed=(
            8 * _matrix_bytes((rows, hidden_dim), dtype)
            + 4 * _matrix_bytes((rows, rank), dtype)
            + 4 * _matrix_bytes((rank, hidden_dim), dtype)
            + 2 * _matrix_bytes((hidden_dim, rank), dtype)
        ),
    )
    bf16 = jax.ShapeDtypeStruct
    return pl.pallas_call(
        functools.partial(
            _rms_gated_norm_reverse_kernel,
            block_m=block_m,
            block_d=block_d,
        ),
        out_shape=(
            bf16((rows, hidden_dim), dtype),
            bf16((hidden_dim,), jnp.float32),
            bf16((hidden_dim, rank), jnp.float32),
            bf16((rank, hidden_dim), jnp.float32),
        ),
        in_specs=(pl.no_block_spec,) * 10,
        out_specs=(pl.no_block_spec,) * 4,
        input_output_aliases={7: 1, 8: 2, 9: 3},
        grid=(pl.cdiv(rows, block_m),),
        compiler_params=plgpu.CompilerParams(num_warps=8, num_stages=2),
        cost_estimate=cost,
        name="rms_gated_norm_reverse",
    )


def quack_rms_gated_norm_reverse(
    output_cotangent: jax.Array,
    x: jax.Array,
    norm_weight: jax.Array,
    w_down: jax.Array,
    w_up: jax.Array,
    inverse_rms: jax.Array,
    gate_preactivation: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Reverse RMSNorm-GatedNorm behind one device-local custom-call boundary."""
    if output_cotangent.ndim != 2:
        raise ValueError(f"output_cotangent must be rank 2, got {output_cotangent.shape}")
    rows, hidden_dim = output_cotangent.shape
    if x.shape != (rows, hidden_dim):
        raise ValueError("full-width reverse inputs must have matching shapes")
    if w_down.ndim != 2:
        raise ValueError(f"w_down must be rank 2, got {w_down.shape}")
    rank = w_down.shape[1]
    if w_down.shape != (hidden_dim, rank) or norm_weight.shape != (hidden_dim,) or inverse_rms.shape != (rows,):
        raise ValueError("RMS reverse inputs have inconsistent dimensions")
    if w_up.shape != (rank, hidden_dim):
        raise ValueError("GatedNorm weight dimensions do not agree")
    if gate_preactivation.shape != (rows, rank):
        raise ValueError("retained GatedNorm residual dimensions do not agree")
    dtypes = {value.dtype for value in (output_cotangent, x, w_down, w_up, gate_preactivation)}
    if dtypes != {jnp.dtype(jnp.bfloat16)}:
        raise ValueError(f"gate-SiLU reverse requires matching BF16 inputs, got {sorted(map(str, dtypes))}")
    if inverse_rms.dtype != jnp.float32:
        raise ValueError(f"inverse_rms must be float32, got {inverse_rms.dtype}")
    divisors = (_RMS_REVERSE_BLOCK_M, _RMS_REVERSE_BLOCK_D, _GATE_REVERSE_BLOCK_N)
    if rows % divisors[0] or hidden_dim % divisors[1] or rank % divisors[2]:
        raise ValueError(
            "gate-SiLU reverse requires rows, hidden_dim, and rank divisible by "
            f"{divisors}, got {(rows, hidden_dim, rank)}"
        )

    call = _rms_gated_norm_reverse_call(rows, hidden_dim, rank, output_cotangent.dtype)
    x_cotangent, norm_weight_cotangent, w_down_cotangent, w_up_cotangent = call(
        output_cotangent,
        x,
        norm_weight,
        w_down,
        w_up,
        inverse_rms,
        gate_preactivation,
        jnp.zeros(norm_weight.shape, dtype=jnp.float32),
        jnp.zeros(w_down.shape, dtype=jnp.float32),
        jnp.zeros(w_up.shape, dtype=jnp.float32),
    )
    return (
        x_cotangent,
        norm_weight_cotangent.astype(norm_weight.dtype),
        w_down_cotangent.astype(w_down.dtype),
        w_up_cotangent.astype(w_up.dtype),
    )


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

    launcher = _build_aliased_backward_consumer_launcher(
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
        input_output_aliases={3: 0},
        output_shape_dtype=jax.ShapeDtypeStruct(
            (1, rows, hidden_dim),
            x.dtype,
            # CuTe custom calls do not infer shard_map's varying-manual-axis annotation from
            # their operands. The RMS input cotangent must carry the same annotation as x for
            # the enclosing custom VJP transpose contract.
            manual_axis_type=jax.typeof(x).manual_axis_type,
        ),
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
