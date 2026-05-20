# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Portions of this file are adapted from:
# /tmp/cutlass-codex/examples/python/CuTeDSL/cute/ampere/kernel/attention/flash_attention_v2.py

"""CuTe DSL segmented FlashAttention forward kernel skeleton for Grug.

This module intentionally has no import-time dependency on CUTLASS/CuTe. The
kernel classes are built inside :func:`segmented_flash_attention_forward_launcher`
after the backend has already imported the optional CUDA-only modules.

The first target is BSHD BF16/FP16 causal self-attention with packed segment
metadata:

    valid[b, q] and lower_bounds[b, q] <= k <= q

The dynamic segment mask is applied in every N tile. That is simpler than the
FlashAttention v2 sample's split between masked and unmasked tiles and avoids
materializing a [B, S, S] mask.
"""

import importlib
import math
from types import SimpleNamespace
from typing import Any, Callable


def _module_attr(modules: Any, name: str) -> Any:
    if isinstance(modules, dict):
        return modules.get(name)
    return getattr(modules, name, None)


def _import_cute_dependencies(modules: Any) -> SimpleNamespace:
    """Load optional CuTe dependencies only when a launcher is requested."""
    cute = _module_attr(modules, "cute")
    cuda = _module_attr(modules, "cuda")
    if cute is None:
        cute = importlib.import_module("cutlass.cute")
    if cuda is None:
        cuda = importlib.import_module("cuda.bindings.driver")
    nvgpu = importlib.import_module("cutlass.cute.nvgpu")

    return SimpleNamespace(
        cutlass=importlib.import_module("cutlass"),
        cute=cute,
        cuda=cuda,
        pipeline=importlib.import_module("cutlass.pipeline"),
        utils=importlib.import_module("cutlass.utils"),
        warp=nvgpu.warp,
    )


def _validate_config(
    *,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    tile_m: int,
    tile_n: int,
    num_threads: int,
) -> None:
    if head_dim <= 0 or head_dim_v <= 0:
        raise ValueError(f"head_dim and head_dim_v must be positive, got {head_dim=} {head_dim_v=}")
    if head_dim_v != head_dim:
        raise NotImplementedError(
            f"segmented FA4/CuTe v1 requires head_dim_v == head_dim, got {head_dim_v=} {head_dim=}"
        )
    if qhead_per_kvhead <= 0:
        raise ValueError(f"qhead_per_kvhead must be positive, got {qhead_per_kvhead}")
    if tile_m <= 0 or tile_n <= 0:
        raise ValueError(f"tile_m and tile_n must be positive, got {tile_m=} {tile_n=}")
    if num_threads <= 0 or num_threads % 32 != 0:
        raise ValueError(f"num_threads must be a positive multiple of 32, got {num_threads}")
    if (tile_m * 2) % num_threads != 0:
        raise ValueError(f"tile_m * 2 must be divisible by num_threads, got {tile_m=} {num_threads=}")


def segmented_flash_attention_forward_launcher(
    modules: Any,
    *,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    tile_m: int = 128,
    tile_n: int = 64,
    num_threads: int = 128,
) -> Any:
    """Build a JAX/CUTLASS-callable segmented FlashAttention forward launcher.

    Args:
        modules: Optional dependency bundle from ``fa4_cute_backend``. It may
            provide ``cute`` and ``cuda`` attributes; other CUTLASS modules are
            imported lazily here.
        head_dim: Q/K head dimension.
        head_dim_v: V/O head dimension.
        qhead_per_kvhead: Number of query heads sharing each K/V head.
        tile_m: Query tile size.
        tile_n: Key/value tile size.
        num_threads: CUDA threads per CTA.

    Returns:
        A ``cute.jit`` launcher with the JAX ``cutlass_call`` signature:
        ``(stream, q, k, v, lower_bounds, valid, out, lse, *, softmax_scale)``.
    """
    _validate_config(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
    )
    deps = _import_cute_dependencies(modules)
    cutlass = deps.cutlass
    cute = deps.cute
    cuda = deps.cuda
    pipeline = deps.pipeline
    utils = deps.utils
    warp = deps.warp

    class SegmentedFlashAttentionForwardAmpere:
        def __init__(
            self,
            head_dim: int,
            head_dim_v: int,
            qhead_per_kvhead: int,
            m_block_size: int,
            n_block_size: int,
            num_threads: int,
        ):
            self._head_dim = head_dim
            self._head_dim_v = head_dim_v
            self._qhead_per_kvhead = qhead_per_kvhead
            self._m_block_size = m_block_size
            self._n_block_size = n_block_size
            self._head_dim_padded = (head_dim + 31) // 32 * 32
            self._head_dim_v_padded = (head_dim_v + 31) // 32 * 32
            self._head_dim_qo_padded = max(self._head_dim_padded, self._head_dim_v_padded)
            self._num_threads = num_threads
            self.cta_sync_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=num_threads)

        @staticmethod
        def can_implement(
            dtype: Any,
            head_dim: int,
            head_dim_v: int,
            m_block_size: int,
            n_block_size: int,
            num_threads: int,
        ) -> bool:
            if dtype != cutlass.Float16 and dtype != cutlass.BFloat16:
                return False
            if head_dim != head_dim_v or head_dim % 8 != 0:
                return False
            if (m_block_size * 2) % num_threads != 0:
                return False
            head_dim_padded = (head_dim + 31) // 32 * 32
            head_dim_v_padded = (head_dim_v + 31) // 32 * 32
            smem_usage = (
                m_block_size * max(head_dim_padded, head_dim_v_padded)
                + n_block_size * (head_dim_padded + head_dim_v_padded)
            ) * 2
            return smem_usage <= utils.get_smem_capacity_in_bytes("sm_80")

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mLowerBounds: cute.Tensor,
            mValid: cute.Tensor,
            mO: cute.Tensor,
            mLSE: cute.Tensor,
            softmax_scale: cutlass.Float32,
            stream: cuda.CUstream,
        ):
            if cutlass.const_expr(mQ.element_type != mK.element_type or mQ.element_type != mV.element_type):
                raise TypeError("q/k/v tensors must have the same element type")
            if cutlass.const_expr(mO.element_type != mQ.element_type):
                raise TypeError("output tensor must have the same element type as q/k/v")
            if cutlass.const_expr(mLSE.element_type != cutlass.Float32):
                raise TypeError("LSE tensor must be float32")
            if cutlass.const_expr(mQ.element_type != cutlass.Float16 and mQ.element_type != cutlass.BFloat16):
                raise TypeError("only Float16 and BFloat16 are supported")
            if cutlass.const_expr(mLowerBounds.element_type != cutlass.Int32):
                raise TypeError("lower_bounds must be int32")
            if cutlass.const_expr(mValid.element_type != cutlass.Int32):
                raise TypeError("valid must be int32")
            if cutlass.const_expr(mQ.shape[1] != self._head_dim or mK.shape[1] != self._head_dim):
                raise ValueError("q/k trailing dimension must match head_dim")
            if cutlass.const_expr(mV.shape[1] != self._head_dim_v or mO.shape[1] != self._head_dim_v):
                raise ValueError("v/o trailing dimension must match head_dim_v")
            if cutlass.const_expr(mQ.shape[2] != mK.shape[2] * self._qhead_per_kvhead):
                raise ValueError("q heads must equal k/v heads times qhead_per_kvhead")
            if cutlass.const_expr(mK.shape[2] != mV.shape[2]):
                raise ValueError("k/v head counts must match")
            if cutlass.const_expr(mLowerBounds.shape[0] != mQ.shape[3] or mLowerBounds.shape[1] != mQ.shape[0]):
                raise ValueError("lower_bounds must have shape [B, S]")
            if cutlass.const_expr(mValid.shape[0] != mQ.shape[3] or mValid.shape[1] != mQ.shape[0]):
                raise ValueError("valid must have shape [B, S]")
            if cutlass.const_expr(
                mLSE.shape[0] != mQ.shape[3] or mLSE.shape[1] != mQ.shape[2] or mLSE.shape[2] != mQ.shape[0]
            ):
                raise ValueError("LSE must have shape [B, Hq, S]")
            if cutlass.const_expr(
                not self.can_implement(
                    mQ.element_type,
                    self._head_dim,
                    self._head_dim_v,
                    self._m_block_size,
                    self._n_block_size,
                    self._num_threads,
                )
            ):
                raise ValueError("unsupported segmented FA4/CuTe tile configuration")

            self._dtype = mQ.element_type
            smem_k_block_size = 64 if self._head_dim_qo_padded % 64 == 0 else 32
            swizzle_bits = 3 if smem_k_block_size == 64 else 2
            sQO_layout_atom = cute.make_composed_layout(
                cute.make_swizzle(swizzle_bits, 3, 3),
                0,
                cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
            )
            sQ_layout = cute.tile_to_shape(sQO_layout_atom, (self._m_block_size, self._head_dim_padded), (0, 1))
            sO_layout = cute.tile_to_shape(sQO_layout_atom, (self._m_block_size, self._head_dim_v_padded), (0, 1))
            sQO_layout = cute.tile_to_shape(
                sQO_layout_atom,
                (self._m_block_size, self._head_dim_qo_padded),
                (0, 1),
            )
            sK_layout = cute.tile_to_shape(sQO_layout_atom, (self._n_block_size, self._head_dim_padded), (0, 1))
            sV_layout = cute.tile_to_shape(sQO_layout_atom, (self._n_block_size, self._head_dim_v_padded), (0, 1))

            @cute.struct
            class SharedStorage:
                sQO: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sQO_layout)], 1024]
                sK: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sK_layout)], 1024]
                sV: cute.struct.Align[cute.struct.MemRange[self._dtype, cute.cosize(sV_layout)], 1024]

            universal_copy_bits = 128
            async_copy_elems = universal_copy_bits // self._dtype.width
            atom_universal_copy = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self._dtype,
                num_bits_per_copy=universal_copy_bits,
            )
            tQKV_shape_dim_1 = sQO_layout_atom.outer.shape[1] // async_copy_elems
            tQKV_layout = cute.make_layout(
                (self._num_threads // tQKV_shape_dim_1, tQKV_shape_dim_1),
                stride=(tQKV_shape_dim_1, 1),
            )
            vQKV_layout = cute.make_layout((1, async_copy_elems))
            gmem_tiled_copy_QKV = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)
            gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tQKV_layout, vQKV_layout)
            tiled_mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(self._dtype, cutlass.Float32, (16, 8, 16)),
                (self._num_threads // 32, 1, 1),
                permutation_mnk=(self._num_threads // 32 * 16, 16, 16),
            )

            grid_dim = (
                cute.ceil_div(mQ.shape[0], self._m_block_size),
                cute.size(mQ.shape[3]),
                cute.size(mQ.shape[2]),
            )
            softmax_scale_log2 = softmax_scale * 1.4426950408889634
            self.kernel(
                mQ,
                mK,
                mV,
                mLowerBounds,
                mValid,
                mO,
                mLSE,
                softmax_scale_log2,
                sQ_layout,
                sK_layout,
                sV_layout,
                sO_layout,
                gmem_tiled_copy_QKV,
                gmem_tiled_copy_O,
                tiled_mma,
                SharedStorage,
            ).launch(grid=grid_dim, block=[self._num_threads, 1, 1], stream=stream)

        @cute.kernel
        def kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mLowerBounds: cute.Tensor,
            mValid: cute.Tensor,
            mO: cute.Tensor,
            mLSE: cute.Tensor,
            softmax_scale_log2: cutlass.Float32,
            sQ_layout: cute.ComposedLayout,
            sK_layout: cute.ComposedLayout,
            sV_layout: cute.ComposedLayout,
            sO_layout: cute.ComposedLayout,
            gmem_tiled_copy_QKV: cute.TiledCopy,
            gmem_tiled_copy_O: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
            SharedStorage: cutlass.Constexpr,
        ):
            tidx, _, _ = cute.arch.thread_idx()
            m_block, batch_idx, q_head = cute.arch.block_idx()
            kv_head = q_head // self._qhead_per_kvhead

            n_block_max = min(
                cute.ceil_div((m_block + 1) * self._m_block_size, self._n_block_size),
                cute.ceil_div(mK.shape[0], self._n_block_size),
            )
            n_block = n_block_max - 1
            first_query = cutlass.min(m_block * self._m_block_size, mQ.shape[0] - 1)
            n_block_min = mLowerBounds[batch_idx, first_query] // self._n_block_size

            gQ = cute.local_tile(
                mQ[None, None, q_head, batch_idx],
                (self._m_block_size, self._head_dim_padded),
                (m_block, 0),
            )
            gK = cute.local_tile(
                mK[None, None, kv_head, batch_idx],
                (self._n_block_size, self._head_dim_padded),
                (None, 0),
            )
            gV = cute.local_tile(
                mV[None, None, kv_head, batch_idx],
                (self._n_block_size, self._head_dim_v_padded),
                (None, 0),
            )

            smem = cutlass.utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sQ = storage.sQO.get_tensor(sQ_layout)
            sK = storage.sK.get_tensor(sK_layout)
            sV = storage.sV.get_tensor(sV_layout)
            sVt = cute.composition(
                sV,
                cute.make_layout((self._head_dim_v_padded, self._n_block_size), stride=(self._n_block_size, 1)),
            )

            gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_slice(tidx)
            tQgQ = gmem_thr_copy_QKV.partition_S(gQ)
            tQsQ = gmem_thr_copy_QKV.partition_D(sQ)
            tKgK = gmem_thr_copy_QKV.partition_S(gK)
            tKsK = gmem_thr_copy_QKV.partition_D(sK)
            tVgV = gmem_thr_copy_QKV.partition_S(gV)
            tVsV = gmem_thr_copy_QKV.partition_D(sV)

            thr_mma = tiled_mma.get_slice(tidx)
            tSrQ = thr_mma.make_fragment_A(thr_mma.partition_A(sQ))
            tSrK = thr_mma.make_fragment_B(thr_mma.partition_B(sK))
            tOrVt = thr_mma.make_fragment_B(thr_mma.partition_B(sVt))
            acc_O = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self._m_block_size, self._head_dim_v_padded)),
                cutlass.Float32,
            )
            acc_O.fill(0.0)

            smem_copy_atom_Q = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            )
            smem_copy_atom_K = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            )
            smem_copy_atom_V = cute.make_copy_atom(warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4), self._dtype)
            smem_tiled_copy_Q = cute.make_tiled_copy_A(smem_copy_atom_Q, tiled_mma)
            smem_tiled_copy_K = cute.make_tiled_copy_B(smem_copy_atom_K, tiled_mma)
            smem_tiled_copy_V = cute.make_tiled_copy_B(smem_copy_atom_V, tiled_mma)

            smem_thr_copy_Q = smem_tiled_copy_Q.get_slice(tidx)
            smem_thr_copy_K = smem_tiled_copy_K.get_slice(tidx)
            smem_thr_copy_V = smem_tiled_copy_V.get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tSrK_copy_view = smem_thr_copy_K.retile(tSrK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)
            tOrVt_copy_view = smem_thr_copy_V.retile(tOrVt)

            mcQ = cute.make_identity_tensor(mQ.layout.shape)
            mcKV = cute.make_identity_tensor(mK.layout.shape)
            cQ = cute.local_tile(
                mcQ[None, None, q_head, batch_idx],
                (self._m_block_size, self._head_dim_padded),
                (m_block, 0),
            )
            cKV = cute.local_tile(
                mcKV[None, None, kv_head, batch_idx],
                (self._n_block_size, self._head_dim_padded),
                (n_block, 0),
            )
            tQcQ = gmem_thr_copy_QKV.partition_S(cQ)
            tKVcKV = gmem_thr_copy_QKV.partition_S(cKV)
            tQpQ = cute.make_rmem_tensor(
                cute.make_layout(
                    (tQsQ.shape[0][1], cute.size(tQsQ, mode=[1]), cute.size(tQsQ, mode=[2])),
                    stride=(cute.size(tQsQ, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            tKVpKV = cute.make_rmem_tensor(
                cute.make_layout(
                    (tKsK.shape[0][1], cute.size(tKsK, mode=[1]), cute.size(tKsK, mode=[2])),
                    stride=(cute.size(tKsK, mode=[2]), 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tQpQ.shape[0]):
                for rest_k in cutlass.range_constexpr(tQpQ.shape[2]):
                    tQpQ[rest_v, 0, rest_k] = cute.elem_less(tQcQ[(0, rest_v), 0, rest_k][1], mQ.layout.shape[1])
            for rest_v in cutlass.range_constexpr(tKVpKV.shape[0]):
                for rest_k in cutlass.range_constexpr(tKVpKV.shape[2]):
                    tKVpKV[rest_v, 0, rest_k] = cute.elem_less(tKVcKV[(0, rest_v), 0, rest_k][1], mK.layout.shape[1])

            for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
                if cute.elem_less(tQcQ[0, m, 0][0], mQ.layout.shape[0]):
                    cute.copy(gmem_tiled_copy_QKV, tQgQ[None, m, None], tQsQ[None, m, None], pred=tQpQ[None, m, None])
                else:
                    tQsQ[None, m, None].fill(0)
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if cute.elem_less(tKVcKV[0, n, 0][0], mK.layout.shape[0]):
                    cute.copy(
                        gmem_tiled_copy_QKV,
                        tKgK[None, n, None, n_block],
                        tKsK[None, n, None],
                        pred=tKVpKV[None, n, None],
                    )
                else:
                    tKsK[None, n, None].fill(0)
            cute.arch.cp_async_commit_group()

            row_max = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_sum = cute.make_rmem_tensor((acc_O.shape[0][0] * acc_O.shape[1]), cutlass.Float32)
            row_max.fill(-cutlass.Float32.inf)
            row_sum.fill(0.0)

            basic_params = SimpleNamespace(
                m_block=m_block,
                n_block=n_block,
                batch_idx=batch_idx,
                q_head=q_head,
                mQ=mQ,
                mK=mK,
                mLowerBounds=mLowerBounds,
                mValid=mValid,
            )
            mma_params = SimpleNamespace(
                thr_mma=thr_mma, tiled_mma=tiled_mma, tSrQ=tSrQ, tSrK=tSrK, tOrVt=tOrVt, acc_O=acc_O
            )
            gmem_copy_params = SimpleNamespace(
                gmem_tiled_copy_QKV=gmem_tiled_copy_QKV,
                tKVcKV=tKVcKV,
                tKgK=tKgK,
                tKsK=tKsK,
                tVgV=tVgV,
                tVsV=tVsV,
                tKVpKV=tKVpKV,
            )
            smem_copy_params = SimpleNamespace(
                smem_tiled_copy_Q=smem_tiled_copy_Q,
                smem_tiled_copy_K=smem_tiled_copy_K,
                smem_tiled_copy_V=smem_tiled_copy_V,
                tSsQ=tSsQ,
                tSrQ_copy_view=tSrQ_copy_view,
                tSsK=tSsK,
                tSrK_copy_view=tSrK_copy_view,
                tOsVt=tOsVt,
                tOrVt_copy_view=tOrVt_copy_view,
            )
            softmax_params = SimpleNamespace(row_max=row_max, row_sum=row_sum, softmax_scale_log2=softmax_scale_log2)

            if n_block_min < n_block_max:
                basic_params.n_block = n_block_max - 1
                self.compute_one_n_block(
                    basic_params,
                    mma_params,
                    gmem_copy_params,
                    smem_copy_params,
                    softmax_params,
                    is_first_n_block=True,
                )
                for n_tile in range(1, n_block_max - n_block_min, 1):
                    basic_params.n_block = n_block_max - n_tile - 1
                    self.compute_one_n_block(
                        basic_params,
                        mma_params,
                        gmem_copy_params,
                        smem_copy_params,
                        softmax_params,
                        is_first_n_block=False,
                    )

            self.normalize_softmax_and_store_lse(
                acc_O,
                row_max,
                row_sum,
                mLSE,
                mQ,
                thr_mma,
                batch_idx,
                q_head,
                m_block,
                softmax_scale_log2,
            )
            rO = cute.make_fragment_like(acc_O, self._dtype)
            rO.store(acc_O.load().to(self._dtype))
            sO = storage.sQO.get_tensor(sO_layout)
            smem_copy_atom_O = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self._dtype)
            smem_tiled_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma)
            smem_thr_copy_O = smem_tiled_copy_O.get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

            gO = cute.local_tile(
                mO[None, None, q_head, batch_idx],
                (self._m_block_size, self._head_dim_v_padded),
                (m_block, 0),
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOrO = cute.make_fragment_like(tOgO, self._dtype)
            self.cta_sync_barrier.arrive_and_wait()
            cute.copy(gmem_tiled_copy_O, tOsO, tOrO)

            mcO = cute.make_identity_tensor(mO.layout.shape)
            cO = cute.local_tile(
                mcO[None, None, q_head, batch_idx],
                (self._m_block_size, self._head_dim_v_padded),
                (m_block, 0),
            )
            tOcO = gmem_thr_copy_O.partition_D(cO)
            tOpO = cute.make_rmem_tensor(
                cute.make_layout(
                    (tOgO.shape[0][1], tOgO.shape[1], tOgO.shape[2]),
                    stride=(tOgO.shape[2], 0, 1),
                ),
                cutlass.Boolean,
            )
            for rest_v in cutlass.range_constexpr(tOpO.shape[0]):
                for rest_n in cutlass.range_constexpr(cute.size(tOpO.shape[2])):
                    tOpO[rest_v, 0, rest_n] = cute.elem_less(tOcO[(0, rest_v), 0, rest_n][1], mO.layout.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOpO.shape[1])):
                if cute.elem_less(tOcO[0, rest_m, 0][0], mO.layout.shape[0]):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None],
                    )

        @cute.jit
        def compute_one_n_block(
            self,
            basic_params: SimpleNamespace,
            mma_params: SimpleNamespace,
            gmem_copy_params: SimpleNamespace,
            smem_copy_params: SimpleNamespace,
            softmax_params: SimpleNamespace,
            is_first_n_block: cutlass.Constexpr,
        ):
            acc_S = cute.make_rmem_tensor(
                mma_params.thr_mma.partition_shape_C((self._m_block_size, self._n_block_size)),
                cutlass.Float32,
            )
            acc_S.fill(0.0)

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            if is_first_n_block:
                for n in cutlass.range_constexpr(cute.size(gmem_copy_params.tVsV.shape[1])):
                    if cute.elem_less(gmem_copy_params.tKVcKV[0, n, 0][0], basic_params.mK.layout.shape[0]):
                        cute.copy(
                            gmem_copy_params.gmem_tiled_copy_QKV,
                            gmem_copy_params.tVgV[None, n, None, basic_params.n_block],
                            gmem_copy_params.tVsV[None, n, None],
                            pred=gmem_copy_params.tKVpKV[None, n, None],
                        )
                    else:
                        gmem_copy_params.tVsV[None, n, None].fill(0.0)
            else:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tVgV[None, None, None, basic_params.n_block],
                    gmem_copy_params.tVsV,
                    pred=gmem_copy_params.tKVpKV,
                )
            cute.arch.cp_async_commit_group()

            cute.copy(
                smem_copy_params.smem_tiled_copy_Q,
                smem_copy_params.tSsQ[None, None, 0],
                smem_copy_params.tSrQ_copy_view[None, None, 0],
            )
            cute.copy(
                smem_copy_params.smem_tiled_copy_K,
                smem_copy_params.tSsK[None, None, 0],
                smem_copy_params.tSrK_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(smem_copy_params.tSsQ.shape[2])):
                k_next = (k + 1) % cute.size(smem_copy_params.tSsQ.shape[2])
                cute.copy(
                    smem_copy_params.smem_tiled_copy_Q,
                    smem_copy_params.tSsQ[None, None, k_next],
                    smem_copy_params.tSrQ_copy_view[None, None, k_next],
                )
                cute.copy(
                    smem_copy_params.smem_tiled_copy_K,
                    smem_copy_params.tSsK[None, None, k_next],
                    smem_copy_params.tSrK_copy_view[None, None, k_next],
                )
                cute.gemm(
                    mma_params.tiled_mma,
                    acc_S,
                    mma_params.tSrQ[None, None, k],
                    mma_params.tSrK[None, None, k],
                    acc_S,
                )

            cute.arch.cp_async_wait_group(0)
            self.cta_sync_barrier.arrive_and_wait()
            if basic_params.n_block > 0:
                cute.copy(
                    gmem_copy_params.gmem_tiled_copy_QKV,
                    gmem_copy_params.tKgK[None, None, None, basic_params.n_block - 1],
                    gmem_copy_params.tKsK,
                    pred=gmem_copy_params.tKVpKV,
                )
                cute.arch.cp_async_commit_group()

            self.softmax_rescale_O(basic_params, mma_params, softmax_params, acc_S, is_first_n_block)

            rP = cute.make_fragment_like(acc_S, self._dtype)
            rP.store(acc_S.load().to(self._dtype))
            rP_layout_divided = cute.logical_divide(rP.layout, (None, None, 2))
            rP_mma_view = cute.make_layout(
                (
                    (rP_layout_divided.shape[0], rP_layout_divided.shape[2][0]),
                    rP_layout_divided.shape[1],
                    rP_layout_divided.shape[2][1],
                ),
                stride=(
                    (rP_layout_divided.stride[0], rP_layout_divided.stride[2][0]),
                    rP_layout_divided.stride[1],
                    rP_layout_divided.stride[2][1],
                ),
            )
            tOrS = cute.make_tensor(rP.iterator, rP_mma_view)
            cute.copy(
                smem_copy_params.smem_tiled_copy_V,
                smem_copy_params.tOsVt[None, None, 0],
                smem_copy_params.tOrVt_copy_view[None, None, 0],
            )
            for k in cutlass.range_constexpr(cute.size(tOrS.shape[2])):
                k_next = (k + 1) % cute.size(tOrS.shape[2])
                cute.copy(
                    smem_copy_params.smem_tiled_copy_V,
                    smem_copy_params.tOsVt[None, None, k_next],
                    smem_copy_params.tOrVt_copy_view[None, None, k_next],
                )
                cute.gemm(
                    mma_params.tiled_mma,
                    mma_params.acc_O,
                    tOrS[None, None, k],
                    mma_params.tOrVt[None, None, k],
                    mma_params.acc_O,
                )

        @cute.jit
        def softmax_rescale_O(
            self,
            basic_params: SimpleNamespace,
            mma_params: SimpleNamespace,
            softmax_params: SimpleNamespace,
            acc_S: cute.Tensor,
            is_first_n_block: cutlass.Constexpr,
        ):
            acc_S_mn = self._make_acc_tensor_mn_view(acc_S)
            acc_O_mn = self._make_acc_tensor_mn_view(mma_params.acc_O)
            row_max_prev = cute.make_fragment_like(softmax_params.row_max, cutlass.Float32)
            if cutlass.const_expr(not is_first_n_block):
                cute.basic_copy(softmax_params.row_max, row_max_prev)

            mcS = cute.make_identity_tensor(
                (
                    basic_params.mQ.shape[3],
                    basic_params.mQ.shape[0],
                    basic_params.mQ.shape[2],
                    basic_params.mK.shape[0],
                )
            )
            cS = cute.local_tile(
                mcS[basic_params.batch_idx, None, basic_params.q_head, None],
                (self._m_block_size, self._n_block_size),
                (basic_params.m_block, basic_params.n_block),
            )
            tScS = mma_params.thr_mma.partition_C(cS)
            tScS_mn = self._make_acc_tensor_mn_view(tScS)

            for r in cutlass.range_constexpr(cute.size(softmax_params.row_max)):
                query_idx = tScS_mn[r, 0][1]
                query_in_bounds = cute.elem_less(query_idx, basic_params.mQ.shape[0])
                query_meta_idx = cutlass.min(query_idx, basic_params.mQ.shape[0] - 1)
                query_valid = basic_params.mValid[basic_params.batch_idx, query_meta_idx] != 0
                query_lower_bound = basic_params.mLowerBounds[basic_params.batch_idx, query_meta_idx]

                for c in cutlass.range_constexpr(cute.size(tScS_mn.shape[1])):
                    key_idx = tScS_mn[0, c][3]
                    key_in_bounds = cute.elem_less(key_idx, basic_params.mK.shape[0])
                    key_after_lower_bound = cute.elem_less(query_lower_bound, key_idx + 1)
                    key_before_query = cute.elem_less(key_idx, query_idx + 1)
                    if not (
                        query_in_bounds
                        and query_valid
                        and key_in_bounds
                        and key_after_lower_bound
                        and key_before_query
                    ):
                        acc_S_mn[r, c] = -cutlass.Float32.inf

                acc_S_row = acc_S_mn[r, None].load()
                row_max_cur_row = acc_S_row.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
                row_max_cur_row = self._threadquad_reduce_max(row_max_cur_row)
                row_max_prev_row = None
                if cutlass.const_expr(not is_first_n_block):
                    row_max_prev_row = row_max_prev[r]
                    row_max_cur_row = cute.arch.fmax(row_max_prev_row, row_max_cur_row)
                row_max_cur_row = 0.0 if row_max_cur_row == -cutlass.Float32.inf else row_max_cur_row

                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * softmax_params.softmax_scale_log2
                    - row_max_cur_row * softmax_params.softmax_scale_log2,
                    fastmath=True,
                )
                acc_S_row_sum = acc_S_row_exp.reduce(cute.ReductionOp.ADD, cutlass.Float32.zero, 0)
                if cutlass.const_expr(not is_first_n_block):
                    prev_minus_cur_exp = cute.math.exp2(
                        row_max_prev_row * softmax_params.softmax_scale_log2
                        - row_max_cur_row * softmax_params.softmax_scale_log2,
                        fastmath=True,
                    )
                    acc_S_row_sum = acc_S_row_sum + softmax_params.row_sum[r] * prev_minus_cur_exp
                    acc_O_mn[r, None] = acc_O_mn[r, None].load() * prev_minus_cur_exp

                softmax_params.row_max[r] = row_max_cur_row
                softmax_params.row_sum[r] = acc_S_row_sum
                acc_S_mn[r, None] = acc_S_row_exp

        @cute.jit
        def normalize_softmax_and_store_lse(
            self,
            acc_O: cute.Tensor,
            row_max: cute.Tensor,
            row_sum: cute.Tensor,
            mLSE: cute.Tensor,
            mQ: cute.Tensor,
            thr_mma: cute.TiledMma,
            batch_idx: cutlass.Int32,
            q_head: cutlass.Int32,
            m_block: cutlass.Int32,
            softmax_scale_log2: cutlass.Float32,
        ):
            acc_O_mn = self._make_acc_tensor_mn_view(acc_O)
            mcO = cute.make_identity_tensor((mQ.shape[3], mQ.shape[0], mQ.shape[2], self._head_dim_v_padded))
            cO = cute.local_tile(
                mcO[batch_idx, None, q_head, None],
                (self._m_block_size, self._head_dim_v_padded),
                (m_block, 0),
            )
            tOcO = thr_mma.partition_C(cO)
            tOcO_mn = self._make_acc_tensor_mn_view(tOcO)
            ln2 = cutlass.Float32(math.log(2.0))
            for r in cutlass.range_constexpr(cute.size(row_sum)):
                row_sum[r] = self._threadquad_reduce_sum(row_sum[r])
                row_sum_is_zero_or_nan = row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
                scale = 1.0 if row_sum_is_zero_or_nan else cute.arch.rcp_approx(row_sum[r])
                acc_O_mn[r, None] = acc_O_mn[r, None].load() * scale
                query_idx = tOcO_mn[r, 0][1]
                if cute.elem_less(query_idx, mLSE.shape[2]):
                    lse = (
                        (row_max[r] * softmax_scale_log2 + cute.math.log2(row_sum[r], fastmath=True)) * ln2
                        if not row_sum_is_zero_or_nan
                        else -cutlass.Float32.inf
                    )
                    mLSE[batch_idx, q_head, query_idx] = lse

        def _make_acc_tensor_mn_view(self, acc: cute.Tensor) -> cute.Tensor:
            acc_layout_col_major = cute.make_layout(acc.layout.shape)
            acc_layout_mn = cute.make_layout(
                (
                    (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),
                    (acc_layout_col_major.shape[0][0], acc_layout_col_major.shape[2]),
                ),
                stride=(
                    (acc_layout_col_major.stride[0][1], acc_layout_col_major.stride[1]),
                    (acc_layout_col_major.stride[0][0], acc_layout_col_major.stride[2]),
                ),
            )
            acc_layout_mn = cute.composition(acc.layout, acc_layout_mn)
            return cute.make_tensor(acc.iterator, acc_layout_mn)

        def _threadquad_reduce(self, val: Any, op: Callable[[Any, Any], Any]) -> Any:
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, mask=-1, mask_and_clamp=31))
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, mask=-1, mask_and_clamp=31))
            return val

        def _threadquad_reduce_max(self, val: Any) -> Any:
            return self._threadquad_reduce(val, lambda x, y: cute.arch.fmax(x, y))

        def _threadquad_reduce_sum(self, val: Any) -> Any:
            return self._threadquad_reduce(val, lambda x, y: x + y)

    kernel = SegmentedFlashAttentionForwardAmpere(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        m_block_size=tile_m,
        n_block_size=tile_n,
        num_threads=num_threads,
    )

    @cute.jit
    def _launch_segmented_flash_attention_forward(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        lower_bounds: cute.Tensor,
        valid: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        kernel(q, k, v, lower_bounds, valid, out, lse, softmax_scale, stream)

    return _launch_segmented_flash_attention_forward


def segmented_flash_attention_backward_launcher(
    modules: Any,
    *,
    dtype: Any,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    tile_m: int = 64,
    tile_n: int = 64,
    num_threads: int = 128,
) -> Any:
    """Build the FA4/CuTe segmented backward launcher.

    This path reuses the upstream FA4 SM80/SM120 backward pipeline shape:
    preprocess computes dPsum/LSE-log2 and zeros dQ accumulators, the main
    tiled kernel accumulates dQ/dK/dV, then postprocess converts accumulators
    to the requested BF16/FP16 gradients.
    """
    _validate_config(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
    )
    deps = _import_cute_dependencies(modules)
    cutlass = deps.cutlass
    cute = deps.cute
    cuda = deps.cuda

    if str(dtype) == "bfloat16":
        cute_dtype = cutlass.BFloat16
    elif str(dtype) == "float16":
        cute_dtype = cutlass.Float16
    else:
        raise TypeError(f"segmented FA4/CuTe backward expects bf16/fp16, got {dtype}")

    preprocess_module = importlib.import_module("flash_attn.cute.flash_bwd_preprocess")
    postprocess_module = importlib.import_module("flash_attn.cute.flash_bwd_postprocess")
    # JAX cutlass_call exposes nested scratch buffers as generic memrefs. Upstream
    # postprocess uses cp.async, which verifies only for global memrefs.
    postprocess_module.assume_tensor_aligned = lambda tensor: tensor
    postprocess_module.cpasync.CopyG2SOp = lambda *args, **kwargs: cute.nvgpu.CopyUniversalOp()
    segmented_bwd_module = importlib.import_module("levanter.grug.fa4_cute_segmented_bwd")
    FlashAttentionBackwardPreprocess = preprocess_module.FlashAttentionBackwardPreprocess
    FlashAttentionBackwardPostprocess = postprocess_module.FlashAttentionBackwardPostprocess
    SegmentedFlashAttentionBackwardSm80 = segmented_bwd_module.SegmentedFlashAttentionBackwardSm80
    SegmentedFlashAttentionBackwardSm120 = segmented_bwd_module.SegmentedFlashAttentionBackwardSm120
    if qhead_per_kvhead <= 1:
        raise NotImplementedError("optimized segmented FA4/CuTe backward currently requires GQA")

    is_sm120_config = tile_m == 64 and tile_n == 64 and num_threads == 128
    arch = 120 if is_sm120_config else 80
    if arch == 120:
        num_stages_q = 2 if head_dim <= 64 else 1
        num_stages_do = 2 if head_dim <= 64 else 1
        atom_layout_m_sdp = 4
        atom_layout_n_dkv = 4
        atom_layout_m_dq = 4
        bwd_cls = SegmentedFlashAttentionBackwardSm120
    else:
        num_stages_q = 2
        num_stages_do = 2
        atom_layout_m_sdp = 2
        atom_layout_n_dkv = 2
        atom_layout_m_dq = 2
        bwd_cls = SegmentedFlashAttentionBackwardSm80

    postprocess_threads = 128
    preprocess = FlashAttentionBackwardPreprocess(
        cute_dtype,
        head_dim,
        head_dim_v,
        tile_m,
        num_threads=postprocess_threads,
        use_padded_offsets=False,
    )
    dq_postprocess = FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim,
        arch,
        tile_m,
        num_threads=postprocess_threads,
        AtomLayoutMdQ=atom_layout_m_dq,
    )
    dk_postprocess = FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim,
        arch,
        tile_n,
        num_threads=postprocess_threads,
        AtomLayoutMdQ=atom_layout_n_dkv,
    )
    dv_postprocess = FlashAttentionBackwardPostprocess(
        cute_dtype,
        head_dim_v,
        arch,
        tile_n,
        num_threads=postprocess_threads,
        AtomLayoutMdQ=atom_layout_n_dkv,
    )
    backward = bwd_cls(
        cute_dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        tile_m,
        tile_n,
        num_stages_q,
        num_stages_do,
        num_threads,
        False,
        True,
        False,
        False,
        False,
        atom_layout_m_sdp,
        atom_layout_n_dkv,
        atom_layout_m_dq,
        V_in_regs=False,
        score_mod=None,
        score_mod_bwd=None,
    )

    class _Float32ZeroFill:
        def __init__(self, num_threads: int):
            self._num_threads = num_threads

        @cute.jit
        def __call__(self, tensor: cute.Tensor, stream: cuda.CUstream):
            self.kernel(tensor).launch(
                grid=[cute.ceil_div(cute.size(tensor), self._num_threads), 1, 1],
                block=[self._num_threads, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def kernel(self, tensor: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bidx, _, _ = cute.arch.block_idx()
            flat = cute.make_tensor(tensor.iterator, cute.make_layout(cute.size(tensor)))
            idx = bidx * self._num_threads + tidx
            if idx < cute.size(flat):
                flat[idx] = cutlass.Float32(0.0)

    zero_fill = _Float32ZeroFill(postprocess_threads)

    @cute.jit
    def _launch_segmented_flash_attention_backward(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        dout: cute.Tensor,
        lse: cute.Tensor,
        lower_bounds: cute.Tensor,
        valid: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dpsum: cute.Tensor,
        lse_log2: cute.Tensor,
        dq_accum: cute.Tensor,
        dk_accum: cute.Tensor,
        dv_accum: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        preprocess(out, dout, dpsum, lse, lse_log2, dq_accum, None, None, None, stream)
        zero_fill(dk_accum, stream)
        zero_fill(dv_accum, stream)
        backward(
            q,
            k,
            v,
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk_accum,
            dv_accum,
            lower_bounds,
            valid,
            softmax_scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            stream,
        )
        dq_postprocess(dq_accum, dq, softmax_scale, None, None, stream)
        dk_postprocess(dk_accum, dk, softmax_scale, None, None, stream)
        dv_postprocess(dv_accum, dv, cutlass.Float32(1.0), None, None, stream)

    return _launch_segmented_flash_attention_backward


def segmented_flash_attention_backward_scalar_launcher(
    modules: Any,
    *,
    head_dim: int,
    head_dim_v: int,
    qhead_per_kvhead: int,
    tile_m: int = 128,
    tile_n: int = 64,
    num_threads: int = 128,
) -> Any:
    """Build the JAX/CUTLASS-callable segmented backward launcher.

    This is a correctness-first scalar CuTe implementation for the narrow Grug
    target: BF16/FP16, causal packed segments, ``Dv == D``, and no dropout. It
    recomputes probabilities from ``q/k/lse`` and returns per-query-head dK/dV;
    the JAX boundary reduces those intermediates across GQA groups.
    """
    _validate_config(
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        qhead_per_kvhead=qhead_per_kvhead,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
    )
    deps = _import_cute_dependencies(modules)
    cutlass = deps.cutlass
    cute = deps.cute
    cuda = deps.cuda

    class SegmentedFlashAttentionBackwardScalar:
        def __init__(
            self,
            *,
            head_dim: int,
            qhead_per_kvhead: int,
        ):
            self._head_dim = head_dim
            self._qhead_per_kvhead = qhead_per_kvhead

        @cute.jit
        def __call__(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mdO: cute.Tensor,
            mLSE: cute.Tensor,
            mLowerBounds: cute.Tensor,
            mValid: cute.Tensor,
            mdQ: cute.Tensor,
            mdKByQHead: cute.Tensor,
            mdVByQHead: cute.Tensor,
            softmax_scale: cutlass.Float32,
            stream: cuda.CUstream,
        ):
            if cutlass.const_expr(mQ.element_type != mK.element_type or mQ.element_type != mV.element_type):
                raise TypeError("q/k/v tensors must have the same element type")
            if cutlass.const_expr(mO.element_type != mQ.element_type or mdO.element_type != mQ.element_type):
                raise TypeError("out/dout tensors must have the same element type as q/k/v")
            if cutlass.const_expr(mdQ.element_type != mQ.element_type):
                raise TypeError("dQ tensor must have the same element type as q")
            if cutlass.const_expr(mdKByQHead.element_type != mQ.element_type):
                raise TypeError("dKByQHead tensor must have the same element type as q")
            if cutlass.const_expr(mdVByQHead.element_type != mQ.element_type):
                raise TypeError("dVByQHead tensor must have the same element type as q")
            if cutlass.const_expr(mLSE.element_type != cutlass.Float32):
                raise TypeError("LSE tensor must be float32")
            if cutlass.const_expr(mLowerBounds.element_type != cutlass.Int32):
                raise TypeError("lower_bounds must be int32")
            if cutlass.const_expr(mValid.element_type != cutlass.Int32):
                raise TypeError("valid must be int32")
            if cutlass.const_expr(mQ.element_type != cutlass.Float16 and mQ.element_type != cutlass.BFloat16):
                raise TypeError("only Float16 and BFloat16 are supported")
            if cutlass.const_expr(mQ.shape[1] != self._head_dim or mK.shape[1] != self._head_dim):
                raise ValueError("q/k head dimension must match head_dim")
            if cutlass.const_expr(mV.shape[1] != self._head_dim or mO.shape[1] != self._head_dim):
                raise ValueError("v/o head dimension must match head_dim")
            if cutlass.const_expr(mQ.shape[2] != mK.shape[2] * self._qhead_per_kvhead):
                raise ValueError("q heads must equal k/v heads times qhead_per_kvhead")
            if cutlass.const_expr(mK.shape[2] != mV.shape[2]):
                raise ValueError("k/v head counts must match")
            if cutlass.const_expr(mO.shape != mQ.shape or mdO.shape != mQ.shape or mdQ.shape != mQ.shape):
                raise ValueError("out/dout/dq must have q shape")
            if cutlass.const_expr(mdKByQHead.shape != mQ.shape or mdVByQHead.shape != mQ.shape):
                raise ValueError("dKByQHead/dVByQHead must have q shape")
            if cutlass.const_expr(mLowerBounds.shape[0] != mQ.shape[3] or mLowerBounds.shape[1] != mQ.shape[0]):
                raise ValueError("lower_bounds must have shape [B, S]")
            if cutlass.const_expr(mValid.shape[0] != mQ.shape[3] or mValid.shape[1] != mQ.shape[0]):
                raise ValueError("valid must have shape [B, S]")
            if cutlass.const_expr(
                mLSE.shape[0] != mQ.shape[3] or mLSE.shape[1] != mQ.shape[2] or mLSE.shape[2] != mQ.shape[0]
            ):
                raise ValueError("LSE must have shape [B, Hq, S]")

            self._dtype = mQ.element_type
            self.dq_kernel(
                mQ,
                mK,
                mV,
                mO,
                mdO,
                mLSE,
                mLowerBounds,
                mValid,
                mdQ,
                softmax_scale,
            ).launch(
                grid=[mQ.shape[0], mQ.shape[2], mQ.shape[3]],
                block=[1, 1, 1],
                stream=stream,
            )
            self.dkdv_kernel(
                mQ,
                mK,
                mV,
                mO,
                mdO,
                mLSE,
                mLowerBounds,
                mValid,
                mdKByQHead,
                mdVByQHead,
                softmax_scale,
            ).launch(
                grid=[mK.shape[0], mQ.shape[2], mQ.shape[3]],
                block=[1, 1, 1],
                stream=stream,
            )

        @cute.kernel
        def dq_kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mdO: cute.Tensor,
            mLSE: cute.Tensor,
            mLowerBounds: cute.Tensor,
            mValid: cute.Tensor,
            mdQ: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            query_idx, q_head, batch_idx = cute.arch.block_idx()
            kv_head = q_head // self._qhead_per_kvhead
            q_valid = mValid[batch_idx, query_idx] != 0
            lower_bound = mLowerBounds[batch_idx, query_idx]
            softmax_scale_log2 = softmax_scale * cutlass.Float32(math.log2(math.e))

            acc_dq = cute.make_rmem_tensor((self._head_dim,), cutlass.Float32)
            acc_dq.fill(0.0)
            if q_valid:
                row_max = -cutlass.Float32.inf
                for key_idx in cutlass.range(lower_bound, query_idx + 1, unroll=1):
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self._head_dim):
                        score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                    row_max = cute.arch.fmax(row_max, score * softmax_scale_log2)
                row_sum = cutlass.Float32(0.0)
                for key_idx in cutlass.range(lower_bound, query_idx + 1, unroll=1):
                    score = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self._head_dim):
                        score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                    row_sum += cute.math.exp2(score * softmax_scale_log2 - row_max)
                delta = cutlass.Float32(0.0)
                for key_idx in cutlass.range(lower_bound, query_idx + 1, unroll=1):
                    score = cutlass.Float32(0.0)
                    d_p = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self._head_dim):
                        score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                        d_p += mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mV[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                    prob = cute.math.exp2(score * softmax_scale_log2 - row_max) * cute.arch.rcp_approx(row_sum)
                    delta += prob * d_p
                for key_idx in cutlass.range(lower_bound, query_idx + 1, unroll=1):
                    score = cutlass.Float32(0.0)
                    d_p = cutlass.Float32(0.0)
                    for d in cutlass.range_constexpr(self._head_dim):
                        score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                        d_p += mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mV[
                            key_idx, d, kv_head, batch_idx
                        ].to(cutlass.Float32)
                    prob = cute.math.exp2(score * softmax_scale_log2 - row_max) * cute.arch.rcp_approx(row_sum)
                    d_s = prob * (d_p - delta) * softmax_scale
                    for d in cutlass.range_constexpr(self._head_dim):
                        acc_dq[d] = acc_dq[d] + d_s * mK[key_idx, d, kv_head, batch_idx].to(cutlass.Float32)

            for d in cutlass.range_constexpr(self._head_dim):
                mdQ[query_idx, d, q_head, batch_idx] = acc_dq[d].to(self._dtype)

        @cute.kernel
        def dkdv_kernel(
            self,
            mQ: cute.Tensor,
            mK: cute.Tensor,
            mV: cute.Tensor,
            mO: cute.Tensor,
            mdO: cute.Tensor,
            mLSE: cute.Tensor,
            mLowerBounds: cute.Tensor,
            mValid: cute.Tensor,
            mdKByQHead: cute.Tensor,
            mdVByQHead: cute.Tensor,
            softmax_scale: cutlass.Float32,
        ):
            key_idx, q_head, batch_idx = cute.arch.block_idx()
            kv_head = q_head // self._qhead_per_kvhead
            key_valid = mValid[batch_idx, key_idx] != 0

            acc_dk = cute.make_rmem_tensor((self._head_dim,), cutlass.Float32)
            acc_dv = cute.make_rmem_tensor((self._head_dim,), cutlass.Float32)
            acc_dk.fill(0.0)
            acc_dv.fill(0.0)
            if key_valid:
                softmax_scale_log2 = softmax_scale * cutlass.Float32(math.log2(math.e))
                for query_idx in cutlass.range(0, mQ.shape[0], unroll=1):
                    query_lower_bound = mLowerBounds[batch_idx, query_idx]
                    if mValid[batch_idx, query_idx] != 0 and query_idx >= key_idx and query_lower_bound <= key_idx:
                        if query_lower_bound == query_idx:
                            for d in cutlass.range_constexpr(self._head_dim):
                                acc_dv[d] = acc_dv[d] + mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32)
                        else:
                            row_max = -cutlass.Float32.inf
                            for row_key_idx in cutlass.range(query_lower_bound, query_idx + 1, unroll=1):
                                row_score = cutlass.Float32(0.0)
                                for d in cutlass.range_constexpr(self._head_dim):
                                    row_score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                                        row_key_idx, d, kv_head, batch_idx
                                    ].to(cutlass.Float32)
                                row_max = cute.arch.fmax(row_max, row_score * softmax_scale_log2)
                            row_sum = cutlass.Float32(0.0)
                            for row_key_idx in cutlass.range(query_lower_bound, query_idx + 1, unroll=1):
                                row_score = cutlass.Float32(0.0)
                                for d in cutlass.range_constexpr(self._head_dim):
                                    row_score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                                        row_key_idx, d, kv_head, batch_idx
                                    ].to(cutlass.Float32)
                                row_sum += cute.math.exp2(row_score * softmax_scale_log2 - row_max)
                            delta = cutlass.Float32(0.0)
                            for row_key_idx in cutlass.range(query_lower_bound, query_idx + 1, unroll=1):
                                row_score = cutlass.Float32(0.0)
                                row_d_p = cutlass.Float32(0.0)
                                for d in cutlass.range_constexpr(self._head_dim):
                                    row_score += mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mK[
                                        row_key_idx, d, kv_head, batch_idx
                                    ].to(cutlass.Float32)
                                    row_d_p += mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32) * mV[
                                        row_key_idx, d, kv_head, batch_idx
                                    ].to(cutlass.Float32)
                                row_prob = cute.math.exp2(
                                    row_score * softmax_scale_log2 - row_max
                                ) * cute.arch.rcp_approx(row_sum)
                                delta += row_prob * row_d_p
                            score = cutlass.Float32(0.0)
                            d_p = cutlass.Float32(0.0)
                            for d in cutlass.range_constexpr(self._head_dim):
                                q_value = mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32)
                                k_value = mK[key_idx, d, kv_head, batch_idx].to(cutlass.Float32)
                                do_value = mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32)
                                v_value = mV[key_idx, d, kv_head, batch_idx].to(cutlass.Float32)
                                score += q_value * k_value
                                d_p += do_value * v_value
                            prob = cute.math.exp2(score * softmax_scale_log2 - row_max) * cute.arch.rcp_approx(row_sum)
                            d_s = prob * (d_p - delta) * softmax_scale
                            for d in cutlass.range_constexpr(self._head_dim):
                                acc_dv[d] = acc_dv[d] + prob * mdO[query_idx, d, q_head, batch_idx].to(cutlass.Float32)
                                acc_dk[d] = acc_dk[d] + d_s * mQ[query_idx, d, q_head, batch_idx].to(cutlass.Float32)

            for d in cutlass.range_constexpr(self._head_dim):
                mdKByQHead[key_idx, d, q_head, batch_idx] = acc_dk[d].to(self._dtype)
                mdVByQHead[key_idx, d, q_head, batch_idx] = acc_dv[d].to(self._dtype)

    kernel = SegmentedFlashAttentionBackwardScalar(
        head_dim=head_dim,
        qhead_per_kvhead=qhead_per_kvhead,
    )

    @cute.jit
    def _launch_segmented_flash_attention_backward(
        stream: cuda.CUstream,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        out: cute.Tensor,
        dout: cute.Tensor,
        lse: cute.Tensor,
        lower_bounds: cute.Tensor,
        valid: cute.Tensor,
        dq: cute.Tensor,
        dk_by_q_head: cute.Tensor,
        dv_by_q_head: cute.Tensor,
        *,
        softmax_scale: cutlass.Float32,
    ):
        kernel(q, k, v, out, dout, lse, lower_bounds, valid, dq, dk_by_q_head, dv_by_q_head, softmax_scale, stream)

    return _launch_segmented_flash_attention_backward


__all__ = [
    "segmented_flash_attention_backward_launcher",
    "segmented_flash_attention_backward_scalar_launcher",
    "segmented_flash_attention_forward_launcher",
]
