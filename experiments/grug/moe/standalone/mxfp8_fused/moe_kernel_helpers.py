# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

"""
Shared utilities for discrete-weight block-scaled grouped GEMM kernels.

This module contains:
- Constants shared across kernel variants
- PTX/DSL utility functions (fmin, fmax, warp reduction, atomics)
- Activation functions (sigmoid, silu, geglu-scaled silu)
- CPU-side reference and validation utilities
- Kernel configuration and validation functions
- Kernel helper functions that don't depend on kernel instance state
"""

from typing import Type, Tuple, Union


import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects.nvvm import AtomicOpKind
from cutlass.cute.typing import Float32, Int32, BFloat16, AddressSpace
from cutlass._mlir.dialects import math, nvvm, llvm, vector, arith
from .moe_persistent_scheduler import MoESchedulerParams
from .utils import atomicrmw_compat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIX_PAD_SIZE = 256
"""Fixed pad size for user-side padding, decoupled from the kernel tile size."""


# ---------------------------------------------------------------------------
# PTX / DSL utility functions
# ---------------------------------------------------------------------------


@dsl_user_op
def make_dual_ptr_tensors_from_param(
    ptx_first_param_name: str,
    first_cnt: int,
    second_cnt: int,
    *,
    loc=None,
    ip=None,
):
    """Create two Int64 tensors from consecutive kernel param-space pointers.

    Kernel parameters are laid out contiguously in param space (8 bytes each
    for .u64). Given the PTX name of the first parameter, this function:
    1. Gets the param-space address via ``mov`` + ``cvta.param.u64``
    2. Creates tensor_a of shape (first_cnt,) starting at offset 0
    3. Creates tensor_b of shape (second_cnt,) starting at offset first_cnt * 8

    :param ptx_first_param_name: PTX name of the very first pointer param
    :param first_cnt: Number of pointers in the first group (e.g., expert_cnt for b_ptrs)
    :param second_cnt: Number of pointers in the second group (e.g., expert_cnt for sfb_ptrs)
    :return: (first_tensor, second_tensor) — two (cnt,) Int64 tensors
    """
    generic_addr = cutlass.Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            f"""{{
                .reg .u64 %paddr, %gaddr;
                mov.u64    %paddr, {ptx_first_param_name};
                cvta.param.u64 %gaddr, %paddr;
                mov.u64    $0, %gaddr;
            }}""",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )
    base_ptr = cute.make_ptr(
        cutlass.Int64,
        generic_addr,
        AddressSpace.generic,
        assumed_align=8,
        loc=loc,
        ip=ip,
    )

    first_layout = cute.make_layout((first_cnt,), loc=loc, ip=ip)
    first_tensor = cute.make_tensor(base_ptr, first_layout, loc=loc, ip=ip)

    second_ptr = base_ptr + first_cnt
    second_layout = cute.make_layout((second_cnt,), loc=loc, ip=ip)
    second_tensor = cute.make_tensor(second_ptr, second_layout, loc=loc, ip=ip)

    return first_tensor, second_tensor


def fmin(a: Union[float, Float32], b: Union[float, Float32], *, nan=True, loc=None, ip=None) -> Float32:
    if nan:
        ptx_instr = f"min.NaN.f32 $0, $1, $2;"
    else:
        ptx_instr = f"min.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def fmax(a: Union[float, Float32], b: Union[float, Float32], *, nan=True, loc=None, ip=None) -> Float32:
    if nan:
        ptx_instr = f"max.NaN.f32 $0, $1, $2;"
    else:
        ptx_instr = f"max.f32 $0, $1, $2;"
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            f"{ptx_instr}",
            f"=f,f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def fmin_bf16x2(
    a0: BFloat16,
    a1: BFloat16,
    b0: BFloat16,
    b1: BFloat16,
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Tuple[BFloat16, BFloat16]:
    vec_bf16x2_type = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
    a_vec = vector.from_elements(
        vec_bf16x2_type,
        [BFloat16(a0).ir_value(loc=loc, ip=ip), BFloat16(a1).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    b_vec = vector.from_elements(
        vec_bf16x2_type,
        [BFloat16(b0).ir_value(loc=loc, ip=ip), BFloat16(b1).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    a_packed = llvm.bitcast(Int32.mlir_type, a_vec, loc=loc, ip=ip)
    b_packed = llvm.bitcast(Int32.mlir_type, b_vec, loc=loc, ip=ip)

    if nan:
        ptx_instr = f"min.NaN.bf16x2 $0, $1, $2;"
    else:
        ptx_instr = f"min.bf16x2 $0, $1, $2;"

    res_packed = llvm.inline_asm(
        Int32.mlir_type,
        [a_packed, b_packed],
        ptx_instr,
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    res_vec = llvm.bitcast(vec_bf16x2_type, res_packed, loc=loc, ip=ip)

    res0 = BFloat16(vector.extract(res_vec, [], [0], loc=loc, ip=ip))
    res1 = BFloat16(vector.extract(res_vec, [], [1], loc=loc, ip=ip))
    return (res0, res1)


def fmax_bf16x2(
    a0: BFloat16,
    a1: BFloat16,
    b0: BFloat16,
    b1: BFloat16,
    *,
    nan: bool = True,
    loc=None,
    ip=None,
) -> Tuple[BFloat16, BFloat16]:
    vec_bf16x2_type = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
    a_vec = vector.from_elements(
        vec_bf16x2_type,
        [BFloat16(a0).ir_value(loc=loc, ip=ip), BFloat16(a1).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    b_vec = vector.from_elements(
        vec_bf16x2_type,
        [BFloat16(b0).ir_value(loc=loc, ip=ip), BFloat16(b1).ir_value(loc=loc, ip=ip)],
        loc=loc,
        ip=ip,
    )
    a_packed = llvm.bitcast(Int32.mlir_type, a_vec, loc=loc, ip=ip)
    b_packed = llvm.bitcast(Int32.mlir_type, b_vec, loc=loc, ip=ip)

    if nan:
        ptx_instr = f"max.NaN.bf16x2 $0, $1, $2;"
    else:
        ptx_instr = f"max.bf16x2 $0, $1, $2;"

    res_packed = llvm.inline_asm(
        Int32.mlir_type,
        [a_packed, b_packed],
        ptx_instr,
        "=r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    res_vec = llvm.bitcast(vec_bf16x2_type, res_packed, loc=loc, ip=ip)

    res0 = BFloat16(vector.extract(res_vec, [], [0], loc=loc, ip=ip))
    res1 = BFloat16(vector.extract(res_vec, [], [1], loc=loc, ip=ip))
    return (res0, res1)


def atomic_add_bf16x2(ptr, val_fp32_lo, val_fp32_hi, *, loc=None, ip=None):
    """Packed BF16x2 atomic reduction to global memory."""
    lo_ir = val_fp32_lo.ir_value(loc=loc, ip=ip)
    hi_ir = val_fp32_hi.ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr, hi_ir, lo_ir],
        "{ .reg .b32 packed; cvt.rn.bf16x2.f32 packed, $1, $2; red.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def atomic_max_float32(
    ptr,
    value: Float32,
    *,
    positive_only: bool = True,
    loc=None,
    ip=None,
) -> Float32:
    value_int = llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)

    old_value_int = atomicrmw_compat(
        T.i32(),  # res (ignored by wheels whose atomicrmw infers it)
        op=cutlass._mlir.dialects.nvvm.AtomicOpKind.MAX,
        ptr=ptr,
        a=value_int,
        loc=loc,
        ip=ip,
    )

    return Float32(llvm.bitcast(T.f32(), old_value_int, loc=loc, ip=ip))


def atomic_add_float32(
    ptr,
    value: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Atomic FP32 addition in global memory (used for dprob gradient accumulation)."""
    old_value = atomicrmw_compat(
        T.f32(),  # res (ignored by wheels whose atomicrmw infers it)
        op=AtomicOpKind.FADD,
        ptr=ptr,
        a=value.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )

    return Float32(llvm.bitcast(T.f32(), old_value, loc=loc, ip=ip))


def cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8_type, loc=None, ip=None):
    fp32x4 = fp32x4.load()
    src_vec4 = fp32x4.ir_value(loc=loc, ip=ip) if hasattr(fp32x4, "ir_value") else fp32x4

    src0 = Float32(vector.extract(src_vec4, [], [0])).ir_value(loc=loc, ip=ip)
    src1 = Float32(vector.extract(src_vec4, [], [1])).ir_value(loc=loc, ip=ip)
    src2 = Float32(vector.extract(src_vec4, [], [2])).ir_value(loc=loc, ip=ip)
    src3 = Float32(vector.extract(src_vec4, [], [3])).ir_value(loc=loc, ip=ip)

    cvt_instruction = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
        cvt_instruction = "cvt.rp.satfinite.ue8m0x2.f32"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction = "cvt.rn.satfinite.e4m3x2.f32"
    else:
        with cute.arch.elect_one():
            cute.printf("error: unsupported fp8 element type")
        return

    asm_tmpl = (
        "{\n" "  .reg .b16 lo;\n" "  .reg .b16 hi;\n" f"  {cvt_instruction} lo, $2, $1;\n" f"  {cvt_instruction} hi, $4, $3;\n" "  mov.b32 $0, {lo, hi};\n" "}"
    )
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src0, src1, src2, src3],
        asm_tmpl,
        "=r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    return packed_i32


def cvt_f32x4_to_f8x4(fp32x4, fp8x4, loc=None, ip=None):
    packed_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, fp8x4.element_type)
    fp8x4_i32 = cute.recast_tensor(fp8x4, cutlass.Int32)
    fp8x4_i32[0] = cutlass.Int32(packed_i32)


def cvt_f32_to_f8_to_f32(fp32x1, fp8_type, loc=None, ip=None):
    src_fp32 = Float32(fp32x1).ir_value(loc=loc, ip=ip)

    cvt_instruction_downcast = ""
    cvt_instruction_upcast = ""
    if cutlass.const_expr(fp8_type == cutlass.Float8E8M0FNU):
        cvt_instruction_downcast = "cvt.rp.satfinite.ue8m0x2.f32"
        cvt_instruction_upcast = "cvt.rn.bf16x2.ue8m0x2"
    elif cutlass.const_expr(fp8_type == cutlass.Float8E4M3FN):
        cvt_instruction_downcast = "cvt.rn.satfinite.e4m3x2.f32"
        cvt_instruction_upcast = "cvt.rn.bf16x2.e4m3x2"
    else:
        with cute.arch.elect_one():
            cute.printf("error: unsupported fp8 element type")
        return

    asm_tmpl = "{\n" "  .reg .b16 bf_lo;\n" f"  {cvt_instruction_downcast} bf_lo, 0f00000000, $1;\n" f"  {cvt_instruction_upcast}  $0, bf_lo;\n" "}"
    packed_i32 = llvm.inline_asm(
        T.i32(),
        [src_fp32],
        asm_tmpl,
        "=r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )

    vec_bf16_ty = ir.Type.parse("vector<2xbf16>")
    bf2_lo = llvm.bitcast(vec_bf16_ty, packed_i32, loc=loc, ip=ip)
    h0 = vector.extract(bf2_lo, [], [0], loc=loc, ip=ip)
    return arith.extf(Float32.mlir_type, h0, loc=loc, ip=ip)


def ceil_div(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Activation functions (device-side, used inside kernels)
# ---------------------------------------------------------------------------


def sigmoid_f32(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Compute the sigmoid of the input value."""
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-a, fastmath=fastmath))


def silu_f32(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Compute the SiLU (Swish) of the input value."""
    return a * sigmoid_f32(a, fastmath=fastmath)


def silu_f32_scaled(
    a: Union[float, Float32],
    alpha: Union[float, Float32] = 1.702,
    fastmath: bool = False,
) -> Union[float, Float32]:
    """Compute the scaled SiLU ``a * sigmoid(alpha * a)`` of the input."""
    return a * sigmoid_f32(a * alpha, fastmath=fastmath)


def silu_f32_geglu_scaled(a: Union[float, Float32], fastmath: bool = False) -> Union[float, Float32]:
    """Backwards-compatible wrapper for :func:`silu_f32_scaled` with ``alpha=1.702``."""
    return silu_f32_scaled(a, alpha=1.702, fastmath=fastmath)


# ---------------------------------------------------------------------------
# CPU-side reference and validation utilities
# ---------------------------------------------------------------------------


def get_amax_smem_size(num_epilog_warps=4):
    """Shared memory size (bytes) needed for per-warp amax scratch space."""
    return num_epilog_warps * cute.size_in_bytes(cutlass.Float32, cute.make_layout((1,)))


def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
    """
    Reciprocal of the maximum representable absolute value for a given data type.

    :param dtype: Data type
    :return: 1 / max_abs_value
    """
    if dtype == cutlass.Float4E2M1FN:
        return 1 / 6.0
    if dtype == cutlass.Float8E4M3FN:
        return 1 / 448.0
    if dtype == cutlass.Float8E5M2:
        return 1 / 128.0
    return 1.0


def is_valid_dtypes_and_scale_factor_vec_size(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    acc_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
) -> bool:
    """
    Check if the data type / scale-factor vector-size combination is valid.

    :return: True if valid, False otherwise
    """
    is_valid = True
    if ab_dtype not in {
        cutlass.Float4E2M1FN,
        cutlass.Float8E5M2,
        cutlass.Float8E4M3FN,
    }:
        is_valid = False

    if sf_vec_size not in {16, 32}:
        is_valid = False

    if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
        is_valid = False

    if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
        is_valid = False
    if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
        is_valid = False

    if acc_dtype not in {cutlass.Float32}:
        is_valid = False

    if d_dtype not in {
        cutlass.Float32,
        cutlass.Float16,
        cutlass.BFloat16,
        cutlass.Float8E5M2,
        cutlass.Float8E4M3FN,
        cutlass.Float4E2M1FN,
    }:
        is_valid = False

    return is_valid


def is_valid_layouts(
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    cd_major: str,
) -> bool:
    """Check if layouts and dtypes are valid combinations."""
    is_valid = True

    if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
        is_valid = False
    # TODO: Currently we don't support m major output for Float4E2M1FN,
    # Need to support it in the future.
    if d_dtype is cutlass.Float4E2M1FN and cd_major == "m":
        is_valid = False
    return is_valid


def is_valid_mma_tiler_and_cluster_shape(
    use_2cta_instrs: bool,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    m_aligned: int,
    fix_pad_size: int = FIX_PAD_SIZE,
) -> bool:
    """
    Check if the MMA tiler and cluster shape are valid.

    :param fix_pad_size: The fixed pad size used by the kernel (default: FIX_PAD_SIZE).
    :return: True if valid, False otherwise
    """
    is_valid = True

    if not ((not use_2cta_instrs and mma_tiler_mn[0] in [128]) or (use_2cta_instrs and mma_tiler_mn[0] in [256])):
        is_valid = False
    # Needs to have even iterations with Epi Tile N 64 for swiGeLU fusion
    if mma_tiler_mn[1] not in [256]:
        is_valid = False
    if cluster_shape_mn[0] % (2 if use_2cta_instrs else 1) != 0:
        is_valid = False
    is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
    if (
        cluster_shape_mn[0] * cluster_shape_mn[1] > 16
        or cluster_shape_mn[0] <= 0
        or cluster_shape_mn[1] <= 0
        or cluster_shape_mn[0] > 4
        or cluster_shape_mn[1] > 4
        or not is_power_of_2(cluster_shape_mn[0])
        or not is_power_of_2(cluster_shape_mn[1])
    ):
        is_valid = False
    cluster_tiler_m = (cluster_shape_mn[0] // (2 if use_2cta_instrs else 1)) * mma_tiler_mn[0]

    if cluster_tiler_m not in [128, 256]:
        is_valid = False

    if m_aligned % mma_tiler_mn[0] != 0:
        is_valid = False

    if m_aligned != fix_pad_size:
        is_valid = False

    return is_valid


def is_valid_tensor_alignment(
    m: int,
    n: int,
    k: int,
    l: int,
    ab_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    a_major: str,
    b_major: str,
    cd_major: str,
) -> bool:
    """Check if the tensor alignment requirements are met for TMA loads/stores."""
    is_valid = True

    def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
        major_mode_idx = 0 if is_mode0_major else 1
        num_major_elements = tensor_shape[major_mode_idx]
        num_contiguous_elements = 16 * 8 // dtype.width
        return num_major_elements % num_contiguous_elements == 0

    if (
        not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
        or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
        or not check_contigous_16B_alignment(d_dtype, cd_major == "m", (m, n, l))
    ):
        is_valid = False
    return is_valid


def can_implement(
    ab_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    acc_dtype: Type[cutlass.Numeric],
    d_dtype: Type[cutlass.Numeric],
    use_2cta_instrs: bool,
    mma_tiler_mn: Tuple[int, int],
    cluster_shape_mn: Tuple[int, int],
    m: int,
    n: int,
    k: int,
    l: int,
    a_major: str,
    b_major: str,
    cd_major: str,
    m_aligned: int,
    fix_pad_size: int = FIX_PAD_SIZE,
) -> bool:
    """
    Check if the grouped GEMM can be implemented with the given parameters.

    :param fix_pad_size: The fixed pad size used by the kernel (default: FIX_PAD_SIZE).
    :return: True if implementable, False otherwise
    """
    result = True

    if m_aligned != fix_pad_size:
        result = False

    if not is_valid_dtypes_and_scale_factor_vec_size(ab_dtype, sf_dtype, sf_vec_size, acc_dtype, d_dtype):
        result = False

    if not is_valid_layouts(ab_dtype, d_dtype, a_major, b_major, cd_major):
        result = False

    if not is_valid_mma_tiler_and_cluster_shape(use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, m_aligned, fix_pad_size):
        result = False

    if not is_valid_tensor_alignment(m, n, k, l, ab_dtype, d_dtype, a_major, b_major, cd_major):
        result = False

    if not (a_major == "k" and b_major == "k"):
        result = False

    if n % 64 != 0 or m % 256 != 0:
        result = False

    return result


def compute_stages(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    epi_tile: cute.Tile,
    epi_tile_c: cute.Tile,
    c_dtype: Type[cutlass.Numeric],
    c_layout: utils.LayoutEnum,
    d_dtype: Type[cutlass.Numeric],
    d_layout: utils.LayoutEnum,
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    num_smem_capacity: int,
    occupancy: int,
    generate_sfd: bool,
    num_epilog_warps: int = 4,
    bias_dtype=None,
) -> Tuple[int, int, int, int, int, int]:
    """Compute the number of pipeline stages for A/B/D operands based on heuristics.

    :param num_epilog_warps: Number of epilogue warps (default 4, may differ for DGLU).
    :param bias_dtype: Bias element type (e.g. cutlass.BFloat16) when bias enabled, None otherwise.
    :return: (num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage, num_bias_stage)
    """
    num_acc_stage = 1 if mma_tiler_mnk[1] == 256 else 2

    num_c_stage = 2 if generate_sfd else 1
    num_d_stage = 2 if generate_sfd else 1

    num_tile_stage = 2

    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        1,
    )
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
        tiled_mma,
        mma_tiler_mnk,
        b_dtype,
        1,
    )

    sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        1,
    )

    sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
        tiled_mma,
        mma_tiler_mnk,
        sf_vec_size,
        1,
    )

    c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
        c_dtype,
        c_layout,
        epi_tile_c,
        1,
    )

    d_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
        d_dtype,
        d_layout,
        epi_tile,
        1,
    )

    ab_bytes_per_stage = (
        cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
        + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
        + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
    )
    mbar_helpers_bytes = 1024
    sinfo_bytes = 4 * 4 * num_tile_stage
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
    c_bytes = c_bytes_per_stage * num_c_stage
    d_bytes_per_stage = cute.size_in_bytes(d_dtype, d_smem_layout_staged_one)
    d_bytes = d_bytes_per_stage * num_d_stage * (2 if generate_sfd else 1)
    amax_bytes = get_amax_smem_size(num_epilog_warps) if d_dtype == cutlass.BFloat16 else 0
    # Bias SMEM stages
    if bias_dtype is not None:
        num_bias_stage = 2  # double buffer, each stage = full tile_N
        bias_epi_tile_n = mma_tiler_mnk[1]
        bias_bytes = bias_epi_tile_n * num_bias_stage * (bias_dtype.width // 8)
    else:
        num_bias_stage = 0
        bias_bytes = 0

    epi_bytes = c_bytes + d_bytes + amax_bytes + bias_bytes

    num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes + sinfo_bytes)) // ab_bytes_per_stage

    total_bytes = occupancy * (ab_bytes_per_stage * num_ab_stage + epi_bytes + sinfo_bytes + mbar_helpers_bytes)

    return num_acc_stage, num_ab_stage, num_c_stage, num_d_stage, num_tile_stage, num_bias_stage


def compute_grid(
    sched_params: MoESchedulerParams,
    max_active_clusters: cutlass.Constexpr,
    use_2cta_instrs: bool,
) -> Tuple[MoESchedulerParams, Tuple[int, int, int]]:
    """Compute grid shape for MoE persistent tile scheduling.

    The grid Z dimension indexes persistent clusters. Grid X/Y cover
    the cluster shape (including 2CTA factor in X).
    """
    grid = (
        sched_params.cluster_shape_mn[0],
        sched_params.cluster_shape_mn[1],
        max_active_clusters,
    )
    return sched_params, grid


def compute_stages_wgrad(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: Tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    epi_tile: cute.Tile,
    c_dtype: Type[cutlass.Numeric],
    c_layout: utils.LayoutEnum,
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    num_smem_capacity: int,
    occupancy: int,
) -> Tuple[int, int, int]:
    """Compute pipeline stages for the grouped GEMM wgrad kernel."""
    num_acc_stage = 2
    num_c_stage = 2
    num_tile_stage = 2

    a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
    b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
    sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
    sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler_mnk, sf_vec_size, 1)
    c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(c_dtype, c_layout, epi_tile, 1)

    ab_bytes_per_stage = (
        cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
        + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
        + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
    )
    mbar_helpers_bytes = 1024
    sinfo_bytes = 4 * 4 * num_tile_stage
    c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
    c_bytes = c_bytes_per_stage * num_c_stage

    num_ab_stage = (num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes + sinfo_bytes)) // ab_bytes_per_stage

    return num_acc_stage, num_ab_stage, num_c_stage


def get_tma_atom_kind(atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean) -> Union[cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp]:
    """
    Select the appropriate TMA copy atom based on SM count and multicast flag.

    :raises ValueError: If the atom_sm_cnt / mcast combination is invalid
    """
    if atom_sm_cnt == 2 and mcast:
        return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
    elif atom_sm_cnt == 2 and not mcast:
        return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
    elif atom_sm_cnt == 1 and mcast:
        return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
    elif atom_sm_cnt == 1 and not mcast:
        return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

    raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")


# ---------------------------------------------------------------------------
# Kernel helper functions (no kernel instance state needed)
# ---------------------------------------------------------------------------


@cute.jit
def amax_reduction_per_thread(vec_fp32, amax_fp32):
    """Per-thread amax reduction over an FP32 register fragment."""
    vec_fp32_ssa = vec_fp32.load()
    abs_acc_values_ir = cutlass._mlir.dialects.math.absf(vec_fp32_ssa.ir_value())
    abs_acc_values = type(vec_fp32_ssa)(abs_acc_values_ir, vec_fp32_ssa.shape, vec_fp32_ssa.dtype)
    subtile_amax = abs_acc_values.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
    return cute.arch.fmax(amax_fp32, subtile_amax)


@cute.jit
def quant_sfd_row(
    tile_idx,
    tiled_copy_r2s,
    src,
    pvscale,
    norm_const,
    rcp_limit,
    tRSrD,
    sf_vec_size,
    vectorized_f32,
    sf_dtype,
    d_dtype,
    use_fp8_ptx_cvt,
):
    tTR_rAcc_frg = cute.logical_divide(src, cute.make_layout(sf_vec_size))
    acc_frg = tTR_rAcc_frg.load()
    abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
    abs_acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    avg_fp32 = abs_acc_frg[None, 0].reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0) * rcp_limit * norm_const
    if tile_idx == 0:
        pvscale[0] = avg_fp32
    elif tile_idx == 1:
        pvscale[1] = avg_fp32
    elif tile_idx == 2:
        pvscale[2] = avg_fp32
    elif tile_idx == 3:
        pvscale[3] = avg_fp32
    qpvscale_up = cvt_f32_to_f8_to_f32(avg_fp32, sf_dtype)
    fp32_max = cutlass.Float32(3.40282346638528859812e38)
    acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_up)
    acc_scale = fmin(acc_scale, fp32_max, nan=True)
    if cutlass.const_expr(vectorized_f32):
        vec = tTR_rAcc_frg[None, 0]
        for ei in cutlass.range_constexpr(0, sf_vec_size, 2):
            vec[ei], vec[ei + 1] = cute.arch.mul_packed_f32x2((vec[ei], vec[ei + 1]), (acc_scale, acc_scale), rnd="rn", ftz=False)
    else:
        vec = tTR_rAcc_frg[None, 0]
        for ei in cutlass.range_constexpr(sf_vec_size):
            vec[ei] = vec[ei] * acc_scale
    acc_vec = tiled_copy_r2s.retile(src).load()
    if cutlass.const_expr(not use_fp8_ptx_cvt):
        tRSrD.store(acc_vec.to(d_dtype))
    else:
        tRSrD_i32 = cute.recast_tensor(tRSrD, cutlass.Int32)
        for ei in cutlass.range_constexpr(0, sf_vec_size, 4):
            fp32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
            fp32x4[0] = acc_vec[ei + 0]
            fp32x4[1] = acc_vec[ei + 1]
            fp32x4[2] = acc_vec[ei + 2]
            fp32x4[3] = acc_vec[ei + 3]
            fp8x4_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, d_dtype)
            tRSrD_i32[ei // 4] = cutlass.Int32(fp8x4_i32)


@cute.jit
def quant_sfd_col(
    tile_idx,
    tiled_copy_r2s,
    src,
    pvscale,
    norm_const,
    rcp_limit,
    tRSrD,
    sf_vec_size,
    sf_dtype,
    d_dtype,
    use_fp8_ptx_cvt,
):
    tTR_rAcc_frg = cute.logical_divide(src, cute.make_layout(sf_vec_size))
    acc_frg = tTR_rAcc_frg.load()
    abs_acc_frg_ir = cutlass._mlir.dialects.math.absf(acc_frg.ir_value())
    acc_frg = type(acc_frg)(abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype)
    avg_fp32 = cutlass.Float32(0.0)
    fp32_max = cutlass.Float32(3.40282346638528859812e38)
    tidx, _, _ = cute.arch.thread_idx()
    for vi in cutlass.range_constexpr(0, acc_frg.shape[0], 4):
        max_value0 = cutlass.Float32(cute.arch.warp_redux_sync(value=acc_frg[vi, 0], kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True))
        max_value1 = cutlass.Float32(cute.arch.warp_redux_sync(value=acc_frg[vi + 1, 0], kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True))
        max_value2 = cutlass.Float32(cute.arch.warp_redux_sync(value=acc_frg[vi + 2, 0], kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True))
        max_value3 = cutlass.Float32(cute.arch.warp_redux_sync(value=acc_frg[vi + 3, 0], kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=True))

        scale = rcp_limit * norm_const
        max_value0, max_value1 = cute.arch.mul_packed_f32x2((max_value0, max_value1), (scale, scale), rnd="rn", ftz=False)
        max_value2, max_value3 = cute.arch.mul_packed_f32x2((max_value2, max_value3), (scale, scale), rnd="rn", ftz=False)

        if tidx % 32 == vi:
            avg_fp32 = max_value0
        if tidx % 32 == vi + 1:
            avg_fp32 = max_value1
        if tidx % 32 == vi + 2:
            avg_fp32 = max_value2
        if tidx % 32 == vi + 3:
            avg_fp32 = max_value3

        max_value_tensor = cute.make_rmem_tensor(4, cutlass.Float32)
        max_value_tensor[0] = max_value0
        max_value_tensor[1] = max_value1
        max_value_tensor[2] = max_value2
        max_value_tensor[3] = max_value3

        if cutlass.const_expr(not use_fp8_ptx_cvt):
            max_value_vec_f8 = max_value_tensor.load().to(sf_dtype)
        else:
            max_value_vec_f8 = cute.make_rmem_tensor(4, sf_dtype)
            cvt_f32x4_to_f8x4(max_value_tensor, max_value_vec_f8)
            max_value_vec_f8 = max_value_vec_f8.load()
        max_value_vec_f32_chunked = max_value_vec_f8.to(cutlass.Float32)
        max_value0 = max_value_vec_f32_chunked[0]
        max_value1 = max_value_vec_f32_chunked[1]
        max_value2 = max_value_vec_f32_chunked[2]
        max_value3 = max_value_vec_f32_chunked[3]

        max_value_rcp0 = cute.arch.rcp_approx(max_value0)
        max_value_rcp1 = cute.arch.rcp_approx(max_value1)
        max_value_rcp2 = cute.arch.rcp_approx(max_value2)
        max_value_rcp3 = cute.arch.rcp_approx(max_value3)

        max_value_rcp0 = fmin(max_value_rcp0, fp32_max, nan=True)
        max_value_rcp1 = fmin(max_value_rcp1, fp32_max, nan=True)
        max_value_rcp2 = fmin(max_value_rcp2, fp32_max, nan=True)
        max_value_rcp3 = fmin(max_value_rcp3, fp32_max, nan=True)

        acc_scale_col0, acc_scale_col1 = cute.arch.mul_packed_f32x2((norm_const, norm_const), (max_value_rcp0, max_value_rcp1), rnd="rn", ftz=False)
        acc_scale_col2, acc_scale_col3 = cute.arch.mul_packed_f32x2((norm_const, norm_const), (max_value_rcp2, max_value_rcp3), rnd="rn", ftz=False)

        tTR_rAcc_frg[vi], tTR_rAcc_frg[vi + 1] = cute.arch.mul_packed_f32x2(
            (tTR_rAcc_frg[vi], tTR_rAcc_frg[vi + 1]), (acc_scale_col0, acc_scale_col1), rnd="rn", ftz=False
        )
        tTR_rAcc_frg[vi + 2], tTR_rAcc_frg[vi + 3] = cute.arch.mul_packed_f32x2(
            (tTR_rAcc_frg[vi + 2], tTR_rAcc_frg[vi + 3]), (acc_scale_col2, acc_scale_col3), rnd="rn", ftz=False
        )

    pvscale[None, None, tile_idx][0] = avg_fp32
    acc_vec = tiled_copy_r2s.retile(src).load()
    if cutlass.const_expr(not use_fp8_ptx_cvt):
        tRSrD.store(acc_vec.to(d_dtype))
    else:
        tRSrD_i32 = cute.recast_tensor(tRSrD, cutlass.Int32)
        for ei in cutlass.range_constexpr(0, sf_vec_size, 4):
            fp32x4 = cute.make_rmem_tensor(4, cutlass.Float32)
            fp32x4[0] = acc_vec[ei + 0]
            fp32x4[1] = acc_vec[ei + 1]
            fp32x4[2] = acc_vec[ei + 2]
            fp32x4[3] = acc_vec[ei + 3]
            fp8x4_i32 = cvt_f32x4_to_f8x4_pack_i32(fp32x4, d_dtype)
            tRSrD_i32[ei // 4] = cutlass.Int32(fp8x4_i32)


def epilog_gmem_copy_and_partition(
    tidx: cutlass.Int32,
    atom: Union[cute.CopyAtom, cute.TiledCopy],
    gD_mnl: cute.Tensor,
    epi_tile: cute.Tile,
    sD: cute.Tensor,
) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
    """Partition shared memory and global memory for TMA epilogue store.

    :param tidx: Thread index in epilogue warp groups (unused, kept for interface compat)
    :param atom: TMA copy atom
    :param gD_mnl: Global tensor D
    :param epi_tile: Epilogue tiler
    :param sD: Shared memory tensor
    :return: (tma_atom_d, bSG_sD, bSG_gD)
    """
    gD_epi = cute.flat_divide(gD_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
    tma_atom_d = atom
    sD_for_tma_partition = cute.group_modes(sD, 0, 2)
    gD_for_tma_partition = cute.group_modes(gD_epi, 0, 2)
    bSG_sD, bSG_gD = cpasync.tma_partition(
        tma_atom_d,
        0,
        cute.make_layout(1),
        sD_for_tma_partition,
        gD_for_tma_partition,
    )
    return tma_atom_d, bSG_sD, bSG_gD
