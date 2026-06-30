# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""In-place patch enabling mixed E4M3xE5M2 FP8 ``wgmma`` on Hopper in JAX/Mosaic-GPU.

Stock JAX (>=0.10.0, and current ``main``) rejects ``wgmma`` with mismatched operand
element types at two layers, and the Mosaic PTX emitter hardcodes a single element type
into both operand slots of the ``wgmma.mma_async`` instruction. But the PTX ISA defines
FP8 ``wgmma`` as ``.dtype.atype.btype`` with ``.atype``/``.btype`` independently in
``{.e4m3, .e5m2}`` -- FP8 is the explicit exception to the floating "atype == btype" rule
(CUTLASS emits exactly ``wgmma.mma_async...f32.e4m3.e5m2`` on SM90). So mixed FP8 is a real
Hopper capability that JAX simply never wired up (logbook GFP8-034/035).

This module applies the minimal change needed to reach that capability:

  1. ``_src/pallas/mosaic_gpu/primitives.py`` -- relax the Pallas trace-time gate
     ("Mixed input dtypes for matrix multiplication unsupported") to allow the
     ``{e4m3, e5m2}`` FP8 pair.
  2. ``experimental/mosaic/gpu/wgmma.py`` -- relax the emitter gate ("WGMMA requires A and
     B to have the same element type"), derive the A operand's PTX type string separately,
     and emit ``.{a_el_ty}.{b_el_ty}`` instead of ``.{el_ty}.{el_ty}``.

The K-step / swizzle / descriptor math is purely byte-width driven, and e4m3/e5m2 are both
1 byte, so it is identical for the pair; the f32 accumulator path already accepts both FP8
types. No algorithmic change.

The patch is written as anchored string replacements (idempotent, version-tolerant) rather
than a context diff, so it applies cleanly inside an ephemeral cluster pod whose JAX line
numbers may drift. ``apply()`` returns the list of files it modified; it raises if any
anchor is missing (fail fast rather than silently no-op).
"""

from __future__ import annotations

import importlib.util
import os


# (file-relative-path, anchor, replacement). Anchors are unique, stable substrings.
_PATCHES: list[tuple[str, str, str]] = [
    # --- 1. Pallas trace-time gate (primitives.py) ---
    (
        "_src/pallas/mosaic_gpu/primitives.py",
        '  if a.dtype != b.dtype:\n'
        '    raise ValueError(f"Mixed input dtypes for matrix multiplication unsupported: lhs={a.dtype}, rhs={b.dtype}")',
        '  _MIXED_FP8 = (jnp.float8_e4m3fn, jnp.float8_e5m2)\n'
        '  if a.dtype != b.dtype and not (a.dtype in _MIXED_FP8 and b.dtype in _MIXED_FP8):\n'
        '    raise ValueError(f"Mixed input dtypes for matrix multiplication unsupported: lhs={a.dtype}, rhs={b.dtype}")',
    ),
    # --- 2a. Emitter gate (wgmma.py: wgmma()) ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '  if element_type != element_type2:\n'
        '    raise ValueError(\n'
        '        "WGMMA requires A and B to have the same element type, got:"\n'
        '        f" {element_type2} and {element_type}"\n'
        '    )',
        '  _fp8 = (f8e4m3fn, f8e5m2)\n'
        '  if element_type != element_type2 and not (element_type in _fp8 and element_type2 in _fp8):\n'
        '    raise ValueError(\n'
        '        "WGMMA requires A and B to have the same element type, got:"\n'
        '        f" {element_type2} and {element_type}"\n'
        '    )',
    ),
    # --- 2b. Derive A operand wgmma element type alongside B's (wgmma.py: wgmma()) ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '  wgmma_element_type = (\n'
        '      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type\n'
        '  )',
        '  wgmma_element_type = (\n'
        '      ir.FloatTF32Type.get() if element_type == ir.F32Type.get() else element_type\n'
        '  )\n'
        '  # A operand may differ from B for the FP8 e4m3/e5m2 pair (PTX .atype != .btype).\n'
        '  a_wgmma_element_type = (\n'
        '      ir.FloatTF32Type.get() if element_type2 == ir.F32Type.get() else element_type2\n'
        '  )',
    ),
    # --- 2c. Thread A's type into wgmma_m64 at the call site (wgmma.py: wgmma()) ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '          n=n_group_elems,\n'
        '          element_type=wgmma_element_type,\n'
        '          b_transpose=b_fastest != mma_utils.Dim.K,',
        '          n=n_group_elems,\n'
        '          element_type=wgmma_element_type,\n'
        '          a_element_type=a_wgmma_element_type,\n'
        '          b_transpose=b_fastest != mma_utils.Dim.K,',
    ),
    # --- 2d. wgmma_m64 signature: accept a separate A element type (wgmma.py) ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '    n: int,\n'
        '    swizzle: int,\n'
        '    element_type: ir.Type,\n'
        '):',
        '    n: int,\n'
        '    swizzle: int,\n'
        '    element_type: ir.Type,\n'
        '    a_element_type: ir.Type | None = None,\n'
        '):',
    ),
    # --- 2f. wgmma_m64: env-gated debug log of the emitted instruction (test aid; NOT in the
    #         canonical PR diff). Lets the H100 job print the literal `.atype.btype` string. ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '  def lc(x):\n'
        '    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result',
        '  if __import__("os").environ.get("MIXED_FP8_LOG_PTX"):\n'
        '    print(f"[mixed-fp8 wgmma emit] {wgmma_instr}", flush=True)\n'
        '\n'
        '  def lc(x):\n'
        '    return llvm.ConstantOp(i32, ir.IntegerAttr.get(i32, x)).result',
    ),
    # --- 2e. wgmma_m64: compute A's PTX type string and emit .atype.btype (wgmma.py) ---
    (
        "experimental/mosaic/gpu/wgmma.py",
        '  out_ty_str = str(out_ty)\n'
        '  if out_ty == i32:\n'
        '    out_ty_str = "s32"\n'
        '\n'
        '  wgmma_instr = (\n'
        '      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty_str}.{el_ty}.{el_ty} "',
        '  out_ty_str = str(out_ty)\n'
        '  if out_ty == i32:\n'
        '    out_ty_str = "s32"\n'
        '\n'
        '  # .atype (A operand). Defaults to B\'s type for the same-type case; differs only\n'
        '  # for the FP8 e4m3/e5m2 pair, the sole PTX-ISA exception to .atype == .btype.\n'
        '  a_element_type = element_type if a_element_type is None else a_element_type\n'
        '  a_el_ty = str(a_element_type)\n'
        '  if isinstance(a_element_type, ir.Float8E5M2Type):\n'
        '    a_el_ty = "e5m2"\n'
        '  elif isinstance(a_element_type, ir.Float8E4M3FNType):\n'
        '    a_el_ty = "e4m3"\n'
        '  elif isinstance(a_element_type, ir.IntegerType):\n'
        '    a_el_ty = "s8"\n'
        '\n'
        '  wgmma_instr = (\n'
        '      f"wgmma.mma_async.sync.aligned.m64n{n}k{k_instr}.{out_ty_str}.{a_el_ty}.{el_ty} "',
    ),
]


def _jax_root() -> str:
    """Locate the installed jax package WITHOUT importing it, so the on-disk patch lands
    before any jax submodule (primitives.py / wgmma.py) is imported and cached."""
    spec = importlib.util.find_spec("jax")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot locate installed jax package")
    return os.path.dirname(spec.origin)


def apply(verbose: bool = True) -> list[str]:
    """Apply the mixed-FP8 wgmma patch in place. Idempotent. Returns modified file paths."""
    root = _jax_root()
    changed: list[str] = []
    for rel, anchor, replacement in _PATCHES:
        path = os.path.join(root, rel)
        with open(path) as f:
            src = f.read()
        if replacement in src:
            continue  # already applied (replacement blocks carry unique markers)
        if anchor not in src:
            raise RuntimeError(f"patch anchor not found in {path}:\n{anchor[:120]}...")
        src = src.replace(anchor, replacement, 1)
        with open(path, "w") as f:
            f.write(src)
        if path not in changed:
            changed.append(path)
        if verbose:
            print(f"patched: {path}")
    return changed


if __name__ == "__main__":
    apply()
