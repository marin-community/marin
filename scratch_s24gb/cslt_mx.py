# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""cuSPARSELt 2:4 GEMMs through nvmath-python's low-level bindings: bf16, e4m3 (tensor scale) and block-scaled e4m3
(VEC64_UE8M0 = MXFP8 with 64-element blocks, the only block-scaled e4m3 mode of sparse SM100 MMA).

``SparseMM(M, K, N, kind)`` is ``D[M, N] = A[M, K] (2:4 along K, row-major) @ B[K, N] (column-major)``, bf16 out."""

import ctypes

import torch
from nvmath.bindings import cusparseLt as cl

CUDA_R_32F, CUDA_R_16BF, CUDA_R_8F_E4M3 = 0, 14, 28
ORDER_COL, ORDER_ROW = 1, 2
OP_N = 0


def _set(handle, desc, attr, value: int):
    v = ctypes.c_int32(value)
    cl.matmul_desc_set_attribute(handle, desc, attr, ctypes.addressof(v), 4)


def _set_ptr(handle, desc, attr, ptr: int):
    v = ctypes.c_void_p(ptr)
    cl.matmul_desc_set_attribute(handle, desc, attr, ctypes.addressof(v), 8)


_HANDLE = None


def handle():
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = cl.init()
    return _HANDLE


def version() -> int:
    return cl.get_version(handle())


class SparseMM:
    def __init__(self, M, K, N, kind, d_order=ORDER_ROW):
        h = handle()
        self.M, self.K, self.N, self.kind = M, K, N, kind
        dt = {"bf16": CUDA_R_16BF, "fp8": CUDA_R_8F_E4M3, "mxfp8": CUDA_R_8F_E4M3}[kind]
        tdt = torch.bfloat16 if kind == "bf16" else torch.float8_e4m3fn
        self.A = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
        self.matA = cl.structured_descriptor_init(h, M, K, K, 16, dt, ORDER_ROW, cl.Sparsity.SPARSITY_50_PERCENT)
        self.matB = cl.dense_descriptor_init(h, K, N, K, 16, dt, ORDER_COL)
        ld_d = N if d_order == ORDER_ROW else M
        self.matC = cl.dense_descriptor_init(h, M, N, ld_d, 16, CUDA_R_16BF, d_order)
        self.matD = cl.dense_descriptor_init(h, M, N, ld_d, 16, CUDA_R_16BF, d_order)
        self.desc = cl.matmul_descriptor_init(
            h, OP_N, OP_N, self.matA, self.matB, self.matC, self.matD, cl.ComputeType.COMPUTE_32F
        )
        self.tdt = tdt
        if kind == "mxfp8":
            # Scale buffers: one byte per (row, 64-element K block), padded to 128 x 4 atoms; all 2^0.
            def scales(rows):
                n = (-(-rows // 128) * 128) * (-(-(K // 64) // 4) * 4)
                return torch.full((n,), 127, device="cuda", dtype=torch.uint8)

            self.sfa, self.sfb = scales(M), scales(N)
            md = cl.MatmulDescAttribute
            _set(h, self.desc, md.A_SCALE_MODE, cl.MatmulMatrixScale.VEC64_UE8M0)
            _set(h, self.desc, md.B_SCALE_MODE, cl.MatmulMatrixScale.VEC64_UE8M0)
            _set_ptr(h, self.desc, md.A_SCALE_POINTER, self.sfa.data_ptr())
            _set_ptr(h, self.desc, md.B_SCALE_POINTER, self.sfb.data_ptr())
        self.alg = cl.matmul_alg_selection_init(h, self.desc, cl.MatmulAlg.DEFAULT)
        self.plan = cl.matmul_plan_init(h, self.desc, self.alg)
        ws = cl.matmul_get_workspace(h, self.plan)
        self.ws = torch.empty(max(ws, 1), device="cuda", dtype=torch.uint8)
        size, buf = cl.sp_mma_compressed_size2(h, self.matA)
        self.comp = torch.empty(size, device="cuda", dtype=torch.uint8)
        self.comp_buf = torch.empty(max(buf, 1), device="cuda", dtype=torch.uint8)
        self.alpha, self.beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        self.D = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        self.stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    def load(self, a_dense_masked: torch.Tensor, b: torch.Tensor):
        """a: [M, K] already 2:4 along K; b: [K, N]. Compresses A and stores B column-major."""
        self.a_dense = a_dense_masked.to(self.tdt).contiguous()
        self.b_col = b.t().contiguous().to(self.tdt)  # [N, K] = column-major [K, N]
        self.compress()

    def compress(self):
        cl.sp_mma_compress2(
            handle(),
            self.matA,
            1,
            OP_N,
            self.a_dense.data_ptr(),
            self.comp.data_ptr(),
            self.comp_buf.data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )

    def _args(self):
        return (
            handle(),
            self.plan,
            ctypes.addressof(self.alpha),
            self.comp.data_ptr(),
            self.b_col.data_ptr(),
            ctypes.addressof(self.beta),
            self.D.data_ptr(),
            self.D.data_ptr(),
            self.ws.data_ptr(),
            ctypes.addressof(self.stream),
            1,
        )

    def search(self):
        self.stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        cl.matmul_search(*self._args())

    def __call__(self):
        cl.matmul(*self._args())
        return self.D
