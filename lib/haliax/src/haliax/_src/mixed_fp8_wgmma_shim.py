# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Enable mixed E4M3/E5M2 Hopper ``wgmma`` on stock jax/jaxlib 0.10.0 — no forked wheel.

The FP8 ragged-dot hybrid (:mod:`haliax._src.fp8_ragged`) runs the Transformer-Engine recipe:
E4M3 forward operands, E5M2 output-gradient. That makes both backward grouped GEMMs *mixed*
FP8 — dgrad ``e5m2 x e4m3`` (jax ``ragged_dot_mgpu``) and wgrad ``e4m3 x e5m2`` (haliax
``transposed_ragged_dot_mgpu``). Hopper ``wgmma.mma_async`` supports this natively (independent
``.atype``/``.btype``), but stock Mosaic-GPU (jax/jaxlib 0.10.0) rejects a mixed pair in four
places, each assuming ``a`` and ``b`` share one element type:

  1. jaxlib C++  ``mosaic_gpu.cc`` ``WGMMAOp::verify``       — dialect verifier (gate)
  2. jax python  ``mosaic/gpu/wgmma.py``                     — PTX emitter (functional)
  3. jax python  ``pallas/mosaic_gpu/primitives.py:wgmma``   — primitive dtype gate
  4. jax python  ``pallas/ops/gpu/ragged_dot_mgpu.py``       — grouped-GEMM dlhs guard

The forked-jaxlib branch (``grug-fp8-fork``) relaxes all four in a rebuilt jax+jaxlib. This shim
reaches the *identical* lowering on stock wheels, entirely from Python:

  * (2) is the only *functional* change — the emitter must write independent ``.atype.btype``
    PTX. We vendor the two patched emitter functions verbatim from the fork
    (``mcwitt/jax@mixed-fp8-wgmma-0.10.0``) and rebind their globals onto the stock module.
    This piece is irreducible: the Warpgroup dialect path emits its PTX through this same
    Python emitter, so without it a mixed pair either raises or emits wrong same-type PTX.
  * (3) and (4) are one-line dtype gates; we relax each in place (read the function's source,
    widen the guard, re-exec), so no function body is duplicated.
  * (1) is compiled into jaxlib and cannot be patched from Python. The verifier is a pure gate
    (the PTX is produced entirely by (2)), so we disable IR verification *only for the dynamic
    extent of Mosaic-GPU module construction* — by wrapping ``mosaic.gpu.core._kernel_to_module``
    and ``_run_serde_pass``. Every other MLIR verification in the process (XLA, other pallas
    backends) is untouched. This is the tightest scope reachable from Python: a single op's C++
    verifier cannot be disabled, only verification for a code region.

:func:`activate` is idempotent and must run before the FP8 mosaic kernels are traced;
:mod:`haliax.nn.ragged_dot` calls it when it imports the mosaic backend. The patched emitter is
a strict superset of stock (identical PTX for same-dtype operands) and the verifier-disable is
per-call-scoped, so global activation is safe.

This is a research shim: it monkeypatches jax internals and skips IR verification for the mosaic
kernels, so it is pinned hard to jax/jaxlib 0.10.0 and fails fast on any drift. The durable fix
is upstreaming the verifier/emitter relaxation (the fork is upstreamable as-is).
"""

from __future__ import annotations  # vendored emitter signatures (np/ir/fa) stay lazy strings

import contextlib
import functools
import importlib
import logging
import re
import types

import jax

logger = logging.getLogger(__name__)

_SUPPORTED_JAX_VERSION = "0.10.0"

# FP8 formats Hopper wgmma accepts as an independent .atype/.btype pair.
_F8_DTYPE_NAMES = ("float8_e4m3fn", "float8_e5m2")


def _is_mixed_f8(x, y) -> bool:
    """True when ``x``/``y`` are both FP8 (the e4m3/e5m2 pair wgmma may mix)."""
    return str(x.dtype) in _F8_DTYPE_NAMES and str(y.dtype) in _F8_DTYPE_NAMES


def _check_jax_version() -> None:
    if jax.__version__ != _SUPPORTED_JAX_VERSION:
        raise RuntimeError(
            f"mixed-fp8 wgmma shim supports jax {_SUPPORTED_JAX_VERSION} only (got "
            f"{jax.__version__}); it reaches into jax internals (wgmma emitter, pallas dtype "
            "guards, Mosaic-GPU verification). Pin jax, or use the forked-jaxlib path."
        )


def _rebind_globals(fn, new_globals):
    """A copy of ``fn`` whose global-name lookups resolve against ``new_globals``.

    The vendored emitter functions are authored here but must resolve ``ir``/``fa``/``mma_utils``/
    ``bytewidth``/... from the *stock* ``mosaic.gpu.wgmma`` module, exactly as the fork's in-tree
    copies do. Swapping ``__globals__`` does that without re-importing every helper.
    """
    return types.FunctionType(fn.__code__, new_globals, fn.__name__, fn.__defaults__, fn.__closure__)


# ======================================================================================
# (2) Vendored PTX emitter — verbatim from mcwitt/jax@mixed-fp8-wgmma-0.10.0
#     jax/experimental/mosaic/gpu/wgmma.py. The only change vs stock is threading an
#     independent ``a_element_type`` so the instruction emits ``.{a_el_ty}.{b_el_ty}``.
#     DO NOT edit these bodies by hand — re-splice from the fork if jax is bumped.
# --------------------------------------------------------------------------------------
def wgmma_m64(
    acc: np.ndarray,  # of register Values
    a,
    b_descriptor: ir.Value,
    a_transpose: bool | None,
    b_transpose: bool,
    a_k_stride: int | None,
    b_k_stride: int,
    n: int,
    swizzle: int,
    a_element_type: ir.Type,
    b_element_type: ir.Type,
):
  out_ty = ir.VectorType(acc.flat[0].type).element_type
  if not _supported_wgmma_types(out_ty, b_element_type):
    raise ValueError(f"Unsupported wgmma types {(out_ty, b_element_type)=}")
  if n % 8:
    raise ValueError

  bf16 = ir.BF16Type.get()
  f16 = ir.F16Type.get()
  i8 = ir.IntegerType.get_signless(8)
  i32 = ir.IntegerType.get_signless(32)
  i64 = ir.IntegerType.get_signless(64)
  f8e5m2 = ir.Float8E5M2Type.get()
  f8e4m3fn = ir.Float8E4M3FNType.get()
  if b_k_stride % 16:
    raise ValueError
  # Only 16-bit types support transposes
  supports_transpose = bytewidth(b_element_type) == 2
  if not supports_transpose and (a_transpose or b_transpose):
    raise ValueError("Only f16 WGMMA supports transposes")
  if a_in_regs := isinstance(a, fa.FragmentedArray):
    if a.mlir_dtype not in {bf16, f16, i8, f8e5m2, f8e4m3fn}:
      raise ValueError(f"Unsupported A register array dtype: {a.mlir_dtype}")
    # Column count must be equal to swizzle // bytewidth.
    elt_bytewidth = utils.bytewidth(b_element_type)
    swizzle_elems = swizzle // elt_bytewidth
    if a.shape != (64, swizzle_elems):
      raise ValueError("Unsupported A register array shape")
    if a.layout not in {fa.WGMMA_LAYOUT, fa.WGMMA_LAYOUT_8BIT}:
      raise ValueError("Unsupported A register array layout")
    if a_k_stride is not None or a_transpose is not None:
      raise ValueError("Unsupported WGMMA features with A in registers")
  else:
    if a_k_stride is None or a_k_stride % 16:
      raise ValueError
    if a_transpose is None:
      raise ValueError

  if isinstance(out_ty, ir.F32Type) or out_ty == i32:
    num_acc_regs = n // 2
    out_ty_field = ir.VectorType.get((1,), out_ty)
    acc_regs = list(acc.flat)
    assert acc_regs[0].type == ir.VectorType.get((1,), out_ty)
    to_acc_vec_regs = lambda regs: np.array(regs).reshape(acc.shape)
    acc_constraint = "r" if isinstance(out_ty, ir.IntegerType) else "f"
  elif isinstance(out_ty, ir.F16Type):
    num_acc_regs = n // 4
    out_ty_field = i32
    acc_regs = [_as_i32_reg(reg) for reg in acc.flat]
    vec_ty = ir.VectorType(acc.flat[0].type)
    to_acc_vec_regs = lambda regs: np.array([_unpack_i32(vec_ty, reg) for reg in regs]).reshape(acc.shape)
    acc_constraint = "r"
  else:
    raise ValueError(
        f"WGMMA instruction only supports f32, f16 and s32 out (got {out_ty})")

  if supports_transpose:
    num_imm_regs = 4
  elif out_ty == i32:
    num_imm_regs = 0
  else:
    num_imm_regs = 2

  if a_in_regs:
    a_reg_constraints = ["r"] * 4  # 4x (b)f16x2 or s8x4 registers
    if supports_transpose:
      num_imm_regs -= 1  # transpose not supported for a in registers
  else:
    a_reg_constraints = ["l"]  # descriptor
  # Reference for i/o aliasing: https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html
  # Seems like it's not actually documented in LLVM IR docs.
  reg_constraints_list = (
      [f"={acc_constraint}"] * num_acc_regs  # accumulator registers
      + [str(i) for i in range(num_acc_regs)]  # we alias outputs as inputs, too.
      + a_reg_constraints  # a descriptor / registers
      + ["l"] * 1  # b descriptor
      + ["n"] * (1 + num_imm_regs)  # literal constants
  )
  reg_constraints = ",".join(reg_constraints_list)
  reg_count = itertools.count()

  def take_regs(n):
    return (f"${i}" for i in itertools.islice(reg_count, n))

  acc_reg_vector = "{" + ",".join(take_regs(num_acc_regs)) + "}"
  for _ in take_regs(num_acc_regs):  # Ignore next entries: aliasing.
    pass
  if a_in_regs:
    a_regs = "{" + ",".join(take_regs(len(a_reg_constraints))) + "}"
  else:
    a_regs, = take_regs(1)
  b_desc_reg, use_out_reg = take_regs(2)
  # Immediate regs (scale, ...).
  imm_regs = "".join(f", {r}" for r in take_regs(num_imm_regs))
  assert next(reg_count) == len(reg_constraints_list)
  k_instr = 32 // bytewidth(b_element_type)

  def ptx_operand_type(ty):
    if isinstance(ty, ir.Float8E5M2Type):
      return "e5m2"
    if isinstance(ty, ir.Float8E4M3FNType):
      return "e4m3"
    if isinstance(ty, ir.IntegerType):
      # TODO(bchetioui): add u8 support in the future. Currently we always
      # assume that 8-bit integers are s8, and we would need to change the
      # signature of `wgmma` to indicate whether the input should be treated as
      # signed or not.
      return "s8"
    return str(ty)

  # `a` and `b` share an element type except for the FP8 e4m3/e5m2 pair, which
  # may be mixed: PTX `wgmma` takes independent `.atype`/`.btype`.
  a_el_ty = ptx_operand_type(a_element_type)
  b_el_ty = ptx_operand_type(b_element_type)

  out_ty_str = str(out_ty)
  if out_ty == i32:
    out_ty_str = "s32"

  wgmma_instr = (
      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty_str}.{a_el_ty}.{b_el_ty} "
      f"{acc_reg_vector}, {a_regs}, {b_desc_reg}, p{imm_regs};"
  )
  ptx = f"{{ .reg .pred p; setp.ne.b32 p, {use_out_reg}, 0; {wgmma_instr} }}\n"

  def lc(x):
    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result

  use_out = scale_a = scale_b = lc(1)
  if out_ty == i32:
    imms = [use_out]
  else:
    imms = [use_out, scale_a, scale_b]

  if supports_transpose and a_transpose is not None:
    imms += [lc(int(a_transpose)), lc(int(b_transpose))]
  elif supports_transpose:
    imms += [lc(int(b_transpose))]

  assert len(imms) == num_imm_regs + 1  # +1 for the use_out_reg in setp.ne.b32

  expected_dim = 10 if utils.bitwidth(out_ty) == 32 else 9
  expected_regs_per_tile = 4 if utils.bitwidth(out_ty) == 32 else 2
  if acc.ndim != expected_dim or acc.shape[0] != 1 or math.prod(acc.shape[2:]) != expected_regs_per_tile:
    raise ValueError(acc.shape)
  acc_struct_type = ir.Type.parse(
      f"!llvm.struct<({','.join(str(out_ty_field) for _ in acc_regs)})>"
  )
  for i in range((swizzle // bytewidth(b_element_type)) // k_instr):
    # Slice out the relevant part of A or advance the A descriptor.
    if a_in_regs:
      a_slice = a[:, (i * k_instr) : ((i + 1) * k_instr)]
      a_args = [_as_i32_reg(v) for v in a_slice.registers.flat]
    else:
      if i > 0:
        assert a_k_stride is not None
        a = _llvm_add(
            a,
            llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, a_k_stride >> 4)),
        )
      a_args = [a]
    # Advance the B descriptor.
    if i > 0:
      b_descriptor = _llvm_add(
          b_descriptor,
          llvm.ConstantOp(i64, ir.IntegerAttr.get(i64, b_k_stride >> 4)),
      )
    assert len(a_args) == len(a_reg_constraints)
    acc_struct = llvm.inline_asm(
        acc_struct_type,
        [*acc_regs, *a_args, b_descriptor, *imms],
        ptx,
        reg_constraints,
        asm_dialect=0,
        has_side_effects=True,
    )
    assert isinstance(acc_struct, ir.Value)
    acc_regs = [
        llvm.extractvalue(out_ty_field, acc_struct, [i]) for i in range(len(acc_regs))
    ]
  return to_acc_vec_regs(acc_regs)


def wgmma(
    acc: WGMMAAccumulator,
    a: fa.FragmentedArray | ir.Value,
    b: ir.Value,
    *,
    swizzle: int = 128,
):
  """Perform acc += a @ b using the WGMMA instruction.

  `a` may be passed in registers, or as a memref. `b` must be a memref.

  The expected (logical) memref shapes are:
    a: (m // tile_m, k // tile_k, tile_m, tile_k)
    b: (k // tile_k, n // tile_n, tile_k, tile_n).

  While the shapes may be physically transposed, when considering the row-major
  physical shape, the tile dimensions must be the two minor dimensions and must
  have the shape (8, S) where S = swizzle // bytewidth(element_type).
  """
  if swizzle == 16:
    raise NotImplementedError("No swizzle is not supported")
  # Step 1. Establish the shape and element type of the operation.
  if not isinstance(b.type, ir.MemRefType):
    raise ValueError(f"B must be a memref, got: {b.type}")
  bf16 = ir.BF16Type.get()
  f32 = ir.F32Type.get()
  f16 = ir.F16Type.get()
  i32 = ir.IntegerType.get_signless(32)
  i8 = ir.IntegerType.get_signless(8)
  f8e5m2 = ir.Float8E5M2Type.get()
  f8e4m3fn = ir.Float8E4M3FNType.get()
  (k, n), element_type = mma_utils.tiled_memref_shape(b)
  if a_in_regs := isinstance(a, fa.FragmentedArray):
    m, k2 = a.shape
    element_type2 = a.mlir_dtype
    if element_type2 not in {f16, bf16, i8, f8e5m2, f8e4m3fn}:
      raise ValueError(
          "Only f16, bf16, i8, f8e5m2, f8e4m3fn are supported for A "
          f"in registers, got {element_type2}"
      )
    if element_type2 == i8 and swizzle == 32:
      # TODO(bchetioui): relax this when ptxas is fixed. As of ptxas 12.8,
      # optimizations eliminate MMA instructions, leading to only the first tile
      # of the result being computed correctly.
      raise NotImplementedError("swizzle=32 not supported for s8 lhs in registers")
  elif isinstance(a.type, ir.MemRefType):
    (m, k2), element_type2 = mma_utils.tiled_memref_shape(a)
  else:
    raise ValueError(f"Unsupported A type: {type(a)}")
  if k != k2:
    raise ValueError(
        "WGMMA requires A and B to have the same contraction dimension (K),"
        f" got: {k2} and {k}"
    )
  is_mixed_fp8 = (
      isinstance(element_type, SUPPORTED_F8_TYPES)
      and isinstance(element_type2, SUPPORTED_F8_TYPES)
  )
  # A and B must share an element type, except that the e4m3/e5m2 FP8 pair may
  # be mixed (PTX `wgmma` takes independent `.atype`/`.btype`). See
  # https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma
  if element_type != element_type2 and not is_mixed_fp8:
    raise ValueError(
        "WGMMA requires A and B to have the same element type (or be a mix of"
        f" the e4m3/e5m2 FP8 types), got: {element_type2} and {element_type}"
    )
  if acc._value.shape != (m, n):
    raise ValueError(
        f"Accumulator shape mismatch: expected {(m, n)}, got {acc._value.shape}"
    )
  if element_type == f32 or element_type == ir.BF16Type.get():
    if acc._value.mlir_dtype != f32:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators"
          f" of type f32, but got: {acc._value.mlir_dtype}"
      )
  elif any(
      isinstance(element_type, t)
      for t in {ir.F16Type, ir.Float8E5M2Type, ir.Float8E4M3FNType}
  ):
    if acc._value.mlir_dtype != f16 and acc._value.mlir_dtype != f32:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators "
          f"of type f32 or f16, but got: {acc._value.mlir_dtype}"
      )
  elif element_type == i8:
    if a_in_regs and not a.is_signed:  # pyrefly: ignore[missing-attribute]
      raise NotImplementedError("WGMMA with lhs of type u8")
    if acc._value.mlir_dtype != i32 or not acc._value.is_signed:
      raise ValueError(
          f"WGMMA with element type {element_type} only supports accumulators "
          f"of type s32, but got: {acc._value.mlir_dtype}"
      )
  else:
    raise NotImplementedError(f"Unsupported element type: {element_type}")

  # Step 2. Decide on the instruction shapes we'll use. Note that with swizzles,
  # instructions must be issued in groups of the same width as the swizzle.
  m_group_elems = 64  # Hopper has a fixed M instruction shape.
  k_group_elems = swizzle // utils.bytewidth(element_type)
  if n > 256 or n % 8:
    raise ValueError(f"N must be a multiple of 8 and <= 256, got: {n}")
  n_group_elems = n  # We assume only one N group below.
  if m % m_group_elems:
    raise ValueError(f"M must be a multiple of {m_group_elems}, got: {m}")
  if k % k_group_elems:
    raise ValueError(f"K must be a multiple of {k_group_elems}, got: {k}")
  m_groups = m // m_group_elems
  k_groups = k // k_group_elems
  # TODO(apaszke): Require users to bitcast input refs to tf32 before WGMMA.
  wgmma_element_type = (
      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type
  )
  # `a`'s PTX operand type, which differs from `b`'s only for the mixed FP8 case.
  a_wgmma_element_type = (
      ir.FloatTF32Type.get()
      if element_type2 == ir.F32Type.get()
      else element_type2
  )

  # Step 3. Compute the operand descriptors.
  if a_in_regs:
    a_desc_base = a_m_group_stride = a_k_group_stride = None
    a_instr_params = dict(a_transpose=None, a_k_stride=None)
  else:
    assert isinstance(a, ir.Value)
    (
        (a_desc_base, a_k_instr_stride),
        (a_m_group_stride, a_k_group_stride),
        a_fastest,
    ) = mma_utils.create_descriptor(
        a,
        swizzle=swizzle,
        large_tile=(m_group_elems, k_group_elems),
        group_size=(m_group_elems, k_group_elems),
        logical_k_major=False,
    )
    assert not a_k_instr_stride[0]  # We'd need separate a/b swizzles.
    a_k_instr_stride = a_k_instr_stride[1][0]
    a_instr_params = dict(a_transpose=a_fastest != mma_utils.Dim.K,
                          a_k_stride=a_k_instr_stride)
  (
      (b_desc_base, b_k_instr_stride),
      (b_n_group_stride, b_k_group_stride),
      b_fastest,
  ) = mma_utils.create_descriptor(
      b,
      swizzle=swizzle,
      large_tile=(k_group_elems,) * 2,  # It's not a typo that we use k for n.
      group_size=(k_group_elems, n_group_elems),
      logical_k_major=True,
  )
  assert not b_k_instr_stride[0]  # We'd need separate a/b swizzles.
  b_k_instr_stride = b_k_instr_stride[1][0]
  del b_n_group_stride  # We only support one N group.

  # Step 4. Issue the instructions.
  if a_in_regs:
    assert isinstance(a, fa.FragmentedArray)
    a = wgmma_fence(a)  # Make sure the registers are ready.

  i64 = ir.IntegerType.get_signless(64)
  new_acc_regs = acc._value.registers.copy()
  for mi in range(m_groups):
    for ki in range(k_groups):
      if a_in_regs:
        assert isinstance(a, fa.FragmentedArray)
        a_mk = a[
            mi * m_group_elems : (mi + 1) * m_group_elems,
            ki * k_group_elems : (ki + 1) * k_group_elems,
        ]
      else:
        assert a_m_group_stride is not None and a_k_group_stride is not None
        a_group_offset = mi * a_m_group_stride + ki * a_k_group_stride
        a_mk = _llvm_add(
            a_desc_base, c(mma_utils.encode_addr(a_group_offset), i64),
        )
      b_k = _llvm_add(
          b_desc_base, c(mma_utils.encode_addr(ki * b_k_group_stride), i64)
      )
      new_acc_regs[mi : mi + 1] = wgmma_m64(
          new_acc_regs[mi : mi + 1],
          a_mk,
          b_k,
          swizzle=swizzle,
          n=n_group_elems,
          a_element_type=a_wgmma_element_type,
          b_element_type=wgmma_element_type,
          b_transpose=b_fastest != mma_utils.Dim.K,
          b_k_stride=b_k_instr_stride,
          **a_instr_params,
      )
  return WGMMAAccumulator(
      _value=fa.FragmentedArray(
          _registers=new_acc_regs,
          _layout=acc._value.layout,
          _is_signed=acc._value.is_signed,
      ),
      _original_layout=acc._original_layout,
      _sync=False,
  )
# ======================================================================================


def _install_emitter() -> None:
    """Rebind the stock ``wgmma``/``wgmma_m64`` to the mixed-FP8-capable vendored copies."""
    # ``import_module`` returns the submodule from sys.modules; ``import x.y.wgmma`` would instead
    # walk attributes and hit the same-named ``wgmma``/``mma`` *functions* the package re-exports.
    _mma = importlib.import_module("jax.experimental.mosaic.gpu.mma")
    _w = importlib.import_module("jax.experimental.mosaic.gpu.wgmma")

    # The fork adds ``from .mma import SUPPORTED_F8_TYPES``; stock ``wgmma.py`` lacks it.
    _w.__dict__.setdefault("SUPPORTED_F8_TYPES", _mma.SUPPORTED_F8_TYPES)
    _w.wgmma_m64 = _rebind_globals(wgmma_m64, _w.__dict__)
    _w.wgmma = _rebind_globals(wgmma, _w.__dict__)


# ======================================================================================
# (3) + (4) One-line dtype gates — relaxed in place (no body duplication).
# --------------------------------------------------------------------------------------


def _top_level_function_source(fn) -> str:
    """Read a top-level function's source straight from its file.

    ``inspect.getsource`` goes through ``linecache``, which under heavy import load was observed
    to intermittently return truncated/guard-less content for jax internals — making exact source
    matching non-deterministic. Reading the file at ``co_firstlineno`` and taking the def block by
    indentation is deterministic. Assumes a column-0 ``def`` (true for the jax 0.10.0 functions
    this shim targets — the version gate enforces that).
    """
    with open(fn.__code__.co_filename) as handle:
        lines = handle.readlines()
    start = fn.__code__.co_firstlineno - 1
    block = [lines[start]]
    for line in lines[start + 1:]:
        # Stop at the next top-level definition. A bare column-0 line that is not a def/class/
        # decorator (e.g. a multi-line signature's closing paren, or a trailing constant) stays
        # in the block; re-execing those extra statements is harmless since only the target
        # function is read back out.
        if re.match(r"(def |class |@)", line):
            break
        block.append(line)
    return "".join(block)


def _relax_inline_dtype_guard(module, func_name: str, lhs: str, rhs: str) -> None:
    """Widen the ``if <lhs>.dtype != <rhs>.dtype:`` guard in ``module.func_name`` to allow the
    mixed FP8 pair, by re-execing the function's own source with the guard widened in place.

    The guard is located with a regex anchored on the operand names and tolerant of indentation,
    trailing whitespace, and line endings — exact-substring matching is too brittle across the
    minor source-formatting differences that occur even within one jax release.
    """
    fn = getattr(module, func_name)
    src = _top_level_function_source(fn)
    guard = re.compile(rf"(?m)^([ \t]*)if {re.escape(lhs)}\.dtype != {re.escape(rhs)}\.dtype:")
    matches = guard.findall(src)
    if len(matches) != 1:
        raise RuntimeError(
            f"mixed-fp8 shim: expected exactly one `if {lhs}.dtype != {rhs}.dtype:` guard in "
            f"{module.__name__}.{func_name}, found {len(matches)} (jax {jax.__version__} drift?)"
        )
    module.__dict__.setdefault("_haliax_is_mixed_f8", _is_mixed_f8)
    new_src = guard.sub(
        rf"\1if {lhs}.dtype != {rhs}.dtype and not _haliax_is_mixed_f8({lhs}, {rhs}):",
        src,
        count=1,
    )
    namespace: dict = {}
    exec(compile(new_src, module.__file__, "exec"), module.__dict__, namespace)
    setattr(module, func_name, namespace[func_name])


def _install_dtype_guards() -> None:
    _primitives = importlib.import_module("jax._src.pallas.mosaic_gpu.primitives")
    _plgpu = importlib.import_module("jax.experimental.pallas.mosaic_gpu")
    _ragged = importlib.import_module("jax.experimental.pallas.ops.gpu.ragged_dot_mgpu")

    # (3) pallas wgmma primitive: ``if a.dtype != b.dtype:``
    _relax_inline_dtype_guard(_primitives, "wgmma", "a", "b")
    # ``plgpu.wgmma`` is a by-value re-export of ``primitives.wgmma`` — repoint it too.
    _plgpu.wgmma = _primitives.wgmma

    # (4) ragged_dot_mgpu grouped GEMM (the dlhs is mixed e5m2 x e4m3): ``if lhs.dtype != rhs.dtype:``
    _relax_inline_dtype_guard(_ragged, "ragged_dot", "lhs", "rhs")


# ======================================================================================
# (1) C++ dialect verifier — bypassed only during Mosaic-GPU module construction.
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _verification_disabled():
    """Disable MLIR IR verification for the dynamic extent of this context.

    Patches the two Python-reachable verification entrypoints Mosaic-GPU uses: explicit
    ``ir.Operation.verify()`` calls and the post-pass verifier inside ``PassManager``. Restored
    on exit. Scoped to Mosaic lowering by the wrappers in :func:`_install_scoped_verify_disable`.
    """
    from jaxlib.mlir import ir, passmanager

    orig_verify = ir.Operation.verify
    orig_parse = passmanager.PassManager.parse

    def _noop_verify(self, *args, **kwargs):
        return True

    def _parse_no_verify(*args, **kwargs):
        pm = orig_parse(*args, **kwargs)
        pm.enable_verifier(False)
        return pm

    ir.Operation.verify = _noop_verify
    passmanager.PassManager.parse = _parse_no_verify
    try:
        yield
    finally:
        ir.Operation.verify = orig_verify
        passmanager.PassManager.parse = orig_parse


# Mosaic-GPU lowering entrypoints that invoke MLIR verification. Wrapping these disables
# verification for the dynamic extent of Mosaic-GPU lowering, which is where the C++
# ``WGMMAOp::verify`` gate rejects a mixed FP8 pair. SCOPE CAVEAT: this is NOT limited to the
# mixed-fp8 kernel — any Mosaic-GPU kernel lowered while a wrapped call is on the stack also skips
# verification (a single op's C++ verifier cannot be disabled from Python). It is, however, scoped
# to Mosaic lowering only: XLA and every non-mosaic compilation keep verification, and it is
# restored the moment the wrapped call returns. The forked-jaxlib path has no such caveat — it
# fixes WGMMAOp::verify precisely.
_VERIFY_DISABLE_TARGETS = (
    # The pallas mosaic path our pl.pallas_call kernels take: encloses the whole build, including
    # core._lower_as_gpu_kernel's verify (core.py:918) and any post-dialect-lowering verify.
    ("jax._src.pallas.mosaic_gpu.lowering", "lower_pipelined_jaxpr_to_module"),
    # core build entrypoints: _lower_as_gpu_kernel holds the verify the pallas path hits directly;
    # _kernel_to_module / _run_serde_pass cover the as_gpu_kernel + serialization paths.
    ("jax.experimental.mosaic.gpu.core", "_lower_as_gpu_kernel"),
    ("jax.experimental.mosaic.gpu.core", "_run_serde_pass"),
    ("jax.experimental.mosaic.gpu.core", "_kernel_to_module"),
)


def _install_scoped_verify_disable() -> None:
    """Wrap the Mosaic-GPU lowering entrypoints so MLIR verification — and with it the C++
    ``WGMMAOp::verify`` gate that rejects a mixed FP8 pair — is disabled for the dynamic extent of
    Mosaic-GPU lowering. See :data:`_VERIFY_DISABLE_TARGETS` for the scope caveat."""
    for module_path, name in _VERIFY_DISABLE_TARGETS:
        module = importlib.import_module(module_path)
        orig = getattr(module, name, None)
        if orig is None or getattr(orig, "_haliax_mixed_f8_wrapped", False):
            continue

        @functools.wraps(orig)
        def wrapper(*args, _orig=orig, **kwargs):
            with _verification_disabled():
                return _orig(*args, **kwargs)

        wrapper._haliax_mixed_f8_wrapped = True
        setattr(module, name, wrapper)


_activated = False


def activate() -> None:
    """Install the mixed-FP8 wgmma overlay on stock jax/jaxlib. Idempotent."""
    global _activated
    if _activated:
        return
    _check_jax_version()
    _install_emitter()
    _install_dtype_guards()
    _install_scoped_verify_disable()
    _activated = True


def activate_if_supported() -> bool:
    """Activate the overlay when running on the supported jax, else log and no-op.

    The auto-activation hook for the FP8 ragged path: an unsupported jax must not break import,
    but the mixed wgmma simply will not be enabled (the kernels then hit stock's own dtype error
    if a mixed pair is attempted). Use :func:`activate` directly to require the overlay.
    """
    if jax.__version__ != _SUPPORTED_JAX_VERSION:
        logger.warning(
            "mixed-fp8 wgmma overlay not activated: needs jax %s, found %s. The FP8 ragged "
            "hybrid's mixed E4M3/E5M2 backward will be unavailable on this jax.",
            _SUPPORTED_JAX_VERSION,
            jax.__version__,
        )
        return False
    activate()
    return True


def is_active() -> bool:
    return _activated
