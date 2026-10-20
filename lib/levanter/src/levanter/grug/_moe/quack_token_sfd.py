# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 quantization along the token axis (M) of a varlen-m grouped GEMM's output, in the epilogue.

The MXFP8 expert weight gradients contract over tokens per expert, so their operands are quantized in 32-row blocks
that start at each expert's first row (``quack_mxfp8.token_scale_blocks``). A varlen-m GEMM's tiles also start at
each group's first row, and SM100's tmem epilogue gives each warp 32 consecutive rows with lane == row, so a token
block is exactly one warp: its amax per column is one ``redux.sync.max.abs`` across the warp. QuACK's
``BlockScaleFactorStore(direction="col")`` does this for plain GEMMs; ``TokenBlockScaleStore`` extends it to varlen-m
outputs (the scales land in the per-group tile-padded layout the varlen-k weight-gradient GEMM reads, group ``e``
from scale tile ``cu[e] // 128 + e`` on) and to non-gated aux outputs, and masks the rows past a group's end, which
belong to the next group, out of its last block.

The scale tensor is the blocked ``(1, ceil(N / 128), padded_scale_tiles(M, E), 32, 4, 4)`` e8m0 layout that
``blocked_scales(exponents.T[None])`` builds from ``quantize_token_blocks``'s exponents. Slots of blocks that no
group row reaches are left unwritten; the weight-gradient GEMM never reads them.
"""

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from quack.epilogue.ops import EpiOp, _get_lane_warp_layouts
from quack.epilogue.quantize_out import BlockScaleFactorStore

TOKEN_BLOCK = 32
_SCALE_ATOM_ROWS = 128


class _TokenSfdState(NamedTuple):
    tDgSFD: cute.Tensor
    tDcSFD: cute.Tensor
    limit_m: Int32
    limit_n: Int32
    valid_m: Int32


class _TokenSfdLoopState(NamedTuple):
    tDgSFD_cur: cute.Tensor
    tDcSFD_cur: cute.Tensor
    limit_m: Int32
    limit_n: Int32
    valid_m: Int32


class TokenBlockScaleStore(BlockScaleFactorStore):
    """Token-axis (column-direction) e8m0 scale factors for an e4m3 output of a varlen-m SM100 GEMM.

    ``output`` names the quantized output: ``"D"`` or a non-gated aux ``TileStore``. Each 32-row block of a group
    gets one scale per column from the amax of its rows inside the group, as ``mx_quantize_row_blocks`` over
    ``token_scale_blocks`` computes it (bit-exact on the same fp32 values)."""

    def __init__(self, name, output="D"):
        EpiOp.__init__(self, name)
        self.direction = "col"
        self.quant_output = output

    def to_params(self, gemm, args):
        mSFD = getattr(args, self.name)
        assert gemm.arch == 100, "TokenBlockScaleStore runs on the SM100 tmem epilogue"
        assert getattr(gemm, "varlen_m", False), "TokenBlockScaleStore quantizes a varlen-m output"
        assert gemm.acc_dtype is Float32, "token quantization needs fp32 accumulation"
        assert mSFD.element_type is cutlass.Float8E8M0FNU, "token scales are e8m0 (MXFP8)"
        if self.quant_output == "D":
            val_dtype = gemm.d_dtype
        else:
            store = next(op for op in gemm._epi_ops if op.is_tile_store() and op.name == self.quant_output)
            assert not store.gated, "token quantization of a gated (halved) output is unsupported"
            val_dtype = getattr(args, self.quant_output).element_type
        assert val_dtype is cutlass.Float8E4M3FN, f"token quantization writes e4m3 values, got {val_dtype}"
        # The MXFP8 scales take no norm constant (a row-direction store of the same epilogue may: sfd_norm_const).
        assert gemm.cta_tile_shape_mnk[0] % _SCALE_ATOM_ROWS == 0, "the CTA tile must cover whole 128-row atoms"
        setattr(gemm, self._vec_attr(), TOKEN_BLOCK)
        setattr(gemm, self._sf_dtype_attr(), mSFD.element_type)
        setattr(gemm, self._val_dtype_attr(), val_dtype)
        setattr(gemm, self._acc_mult_attr(), 1)
        s = mSFD.stride
        # (1, rn, rm_k, 32, 4, 4) -> rank-2 (M_pad, N_pad): M = ((32 rows sharing a byte, 4 slots), 128-row atoms),
        # N = ((32, 4) columns of an atom, 128-column atoms). Rank 2 so varlen's offset_batch_SFA shifts the atom
        # mode by this group's padded tile offset.
        layout = cute.make_layout(
            (((TOKEN_BLOCK, 4), mSFD.shape[2]), ((32, 4), mSFD.shape[1])),
            stride=(((0, 1), s[2]), ((16, 4), s[1])),
        )
        return {self.name: cute.make_tensor(mSFD.iterator, layout)}

    @cute.jit
    def begin(self, gemm, param, smem_tensor, ctx):
        lane_layout_MN, _ = _get_lane_warp_layouts(ctx.tiled_copy_t2r, reference_src=False)
        assert lane_layout_MN[0].stride == 1, "token quantization requires lane == row"
        assert cute.size(lane_layout_MN[0]) % TOKEN_BLOCK == 0, "a token block must be one warp's rows"
        tDgSFD, limit_m, limit_n = self._sfd_gmem_view(gemm, param, ctx, atom_cols=128)
        tDcSFD = ctx.partition_for_epilogue_fn(cute.make_identity_tensor((ctx.tile_M, ctx.tile_N)))
        valid_m = Int32(ctx.varlen_manager.len_m(ctx.batch_idx) - ctx.tile_coord_mnkl[0] * ctx.tile_M)
        return _TokenSfdState(tDgSFD, tDcSFD, limit_m, limit_n, valid_m)

    @cute.jit
    def begin_loop(self, gemm, state, epi_coord):
        tDgSFD_cur = cute.group_modes(state.tDgSFD, 3, cute.rank(state.tDgSFD))[None, None, None, epi_coord]
        tDcSFD_cur = cute.group_modes(state.tDcSFD, 3, cute.rank(state.tDcSFD))[None, None, None, epi_coord]
        return _TokenSfdLoopState(tDgSFD_cur, tDcSFD_cur, state.limit_m, state.limit_n, state.valid_m)

    @cute.jit
    def quantize(self, gemm, loop_state, frag):
        _token_quantize_subtile(
            loop_state.tDgSFD_cur,
            loop_state.tDcSFD_cur,
            loop_state.limit_m,
            loop_state.limit_n,
            loop_state.valid_m,
            frag,
        )

    @cute.jit
    def end(self, gemm, param, state, epi_tile, tiled_copy_t2r, tiled_copy_r2s, tile_coord_mnkl, varlen_manager, tidx):
        # The scale bytes are stored per subtile in _token_quantize_subtile.
        return


@dsl_user_op
def _f32_bits(value: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(llvm.bitcast(T.i32(), value.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bits_f32(bits: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), bits.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@cute.jit
def _token_quantize_subtile(tDgSFD_cur, tDcSFD_cur, limit_m, limit_n, valid_m, frag):
    """One subtile's token quantization. Each column's block amax is one integer ``redux.sync.max`` over the bits
    of ``|x|`` across the warp (non-negative floats order like their bits), the rows at or past ``valid_m`` (the
    group's end) contributing zero. The e8m0 exponent is ``ceil(log2(amax / 448))`` read from ``amax``'s bits:
    ``amax = m 2^E`` with ``m`` in [1, 2) gives ``E - 8``, plus one when ``m > 1.75`` (448 = 1.75 * 2^8), clipped
    to [1, 254], 127 for an all-zero block: ``mx_quantize_row_blocks``'s rounding, bit for bit."""
    tDg_u8 = cute.recast_tensor(tDgSFD_cur, cutlass.Uint8)
    lane = cute.arch.lane_idx()
    # Lane == row: every element of this thread's fragment is in the same row.
    row_ok = tDcSFD_cur[0][0] < valid_m
    size = cute.size(frag)
    assert size % TOKEN_BLOCK == 0, f"a subtile fragment of {size} elements does not split into lane groups"
    # Every lane computes every column's exponent (the redux broadcasts the amax) and keeps the one column of
    # each group of 32 that it stores: one byte store per 32 columns instead of one predicated store per column.
    kept = [Int32(0)] * (size // TOKEN_BLOCK)
    for i in cutlass.range_constexpr(size):
        value = Float32(cutlass.select_(row_ok, frag[i], Float32(0.0)))
        magnitude = _f32_bits(value) & 0x7FFFFFFF
        amax_bits = cute.arch.warp_redux_sync(magnitude, "max", 0xFFFFFFFF)
        above = Int32(cutlass.select_((amax_bits & 0x7FFFFF) > 0x600000, Int32(1), Int32(0)))
        exponent = ((amax_bits >> 23) & 0xFF) - 8 + above
        exponent = Int32(cutlass.select_(exponent < 1, Int32(1), exponent))
        exponent = Int32(cutlass.select_(exponent > 254, Int32(254), exponent))
        exponent = Int32(cutlass.select_(amax_bits == 0, Int32(127), exponent))
        group = i // TOKEN_BLOCK
        kept[group] = Int32(cutlass.select_(lane == i % TOKEN_BLOCK, exponent, kept[group]))
        # The exact reciprocal of 2^(e - 127) has the biased exponent 254 - e.
        frag[i] = value * _bits_f32((254 - exponent) << 23)
    if limit_m > 0:
        for group in cutlass.range_constexpr(size // TOKEN_BLOCK):
            index = group * TOKEN_BLOCK + lane
            if tDcSFD_cur[index][1] < limit_n:
                tDg_u8[index] = cutlass.Uint8(kept[group])
