# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""H100 derisk for the mixed E4M3xE5M2 FP8 wgmma JAX patch (logbook GFP8-035).

Applies ``mixed_fp8_wgmma_patch`` to the live JAX install, then exercises mixed-FP8 wgmma
on a real Hopper GPU across the two lowering paths and checks three things per case:

  1. it COMPILES (no gate / no MLIR-verifier rejection),
  2. it LOWERS to the expected PTX -- the dumped PTX contains
     ``wgmma.mma_async...f32.e4m3.e5m2`` (independent A/B operand types), and
  3. it is NUMERICALLY correct vs an f32 reference computed from the FP8 values.

Cases:
  * dense  Lane       e4m3 x e4m3  (baseline: patch must not regress same-type)
  * dense  Lane       e4m3 x e5m2  (mixed, Python PTX-emitter path)
  * dense  Warpgroup  e4m3 x e4m3  (baseline)
  * dense  Warpgroup  e4m3 x e5m2  (mixed, MLIR-dialect path -> WGMMAOp verifier)
  * wgrad  Warpgroup  e4m3 x e5m2  (the PRODUCTION path: haliax transposed_ragged_dot,
                                    i.e. E4M3 activations contracting an E5M2 output-grad)
  * negative control: dense e4m3 x bf16 must STILL raise (we opened only the fp8 pair)

Mosaic-GPU is H100-only; run on an H100 via Iris (see logbook submit recipe). Each sub-case
is isolated in try/except so one failure (e.g. a C++ WGMMAOp verifier blocking the Warpgroup
path) still yields a full diagnostic matrix from a single job.
"""

import argparse
import functools
import glob
import os
import shutil
import sys
import tempfile
import traceback


def _glob_first(name: str) -> str | None:
    """First match of ``nvidia/**/<name>`` across sys.path site-packages dirs."""
    for base in sys.path:
        if base and os.path.isdir(base):
            hits = glob.glob(os.path.join(base, "nvidia", "**", name), recursive=True)
            if hits:
                return hits[0]
    return None


def _ensure_cuda_toolchain() -> bool:
    """Assemble a CUDA dir XLA/Mosaic can use, before jax is imported (see bench_ragged_mosaic_f8)."""
    ptxas = shutil.which("ptxas") or _glob_first("ptxas")
    libdevice = _glob_first("libdevice.10.bc")
    if not ptxas or not libdevice:
        print(f"cuda toolchain: incomplete (ptxas={ptxas}, libdevice={libdevice})")
        return False
    nvvm_parent = os.path.dirname(os.path.dirname(os.path.dirname(libdevice)))
    root = tempfile.mkdtemp(prefix="xla_cuda_")
    os.symlink(os.path.dirname(ptxas), os.path.join(root, "bin"))
    os.symlink(os.path.join(nvvm_parent, "nvvm"), os.path.join(root, "nvvm"))
    os.environ["PATH"] = os.path.dirname(ptxas) + os.pathsep + os.environ.get("PATH", "")
    flags = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_cuda_data_dir={root}".strip()
    for var in ("CUDA_DIR", "CUDA_HOME", "CUDA_PATH"):
        os.environ[var] = root
    cwd_link = os.path.join(os.getcwd(), "libdevice.10.bc")
    if not os.path.exists(cwd_link):
        os.symlink(libdevice, cwd_link)
    print(f"cuda toolchain: ptxas={ptxas}\n  libdevice={libdevice}")
    return True


# --- toolchain + PTX dump dir + patch, all BEFORE importing jax submodules ----------------
_PTX_DUMP = tempfile.mkdtemp(prefix="ptx_dump_")
_ensure_cuda_toolchain()
os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} --xla_dump_to={_PTX_DUMP}".strip()

os.environ.setdefault("MIXED_FP8_LOG_PTX", "1")  # have the patched emitter log each wgmma instr

sys.path.insert(0, os.path.dirname(__file__))
import mixed_fp8_wgmma_patch  # noqa: E402

_PATCH_ENABLED = os.environ.get("MIXED_FP8_PATCH", "1") == "1"
if _PATCH_ENABLED:
    print("applying mixed-fp8 wgmma patch...")
    mixed_fp8_wgmma_patch.apply()
else:
    print("MIXED_FP8_PATCH=0 -> running UNPATCHED (expect mixed cases to raise the gate)")

import jax  # noqa: E402

print(f"jax version: {jax.__version__}")
import jax.numpy as jnp  # noqa: E402
import jax.experimental.pallas as pl  # noqa: E402
import jax.experimental.pallas.mosaic_gpu as plgpu  # noqa: E402
import numpy as np  # noqa: E402

from haliax._src.transposed_ragged_dot_mgpu import transposed_ragged_dot  # noqa: E402

E4M3 = jnp.float8_e4m3fn
E5M2 = jnp.float8_e5m2
_DT = {"e4m3": E4M3, "e5m2": E5M2, "bf16": jnp.bfloat16}


def _rand_fp8(key, shape, dtype):
    """Small-magnitude inputs that are exactly representable in both e4m3 and e5m2."""
    x = jax.random.randint(key, shape, -4, 5).astype(jnp.float32) * 0.5  # {-2,-1.5,...,2}
    return x.astype(dtype)


# --- minimal single-tile dense wgmma kernel, runnable under either lowering semantics ------
# RHS is passed K-major (bT, shape (N, K)) and presented to wgmma via a logical transpose_ref,
# so b_fastest==K and no hardware operand transpose is requested -- FP8 wgmma forbids transposes
# (only 16-bit types support them). Computes a @ bT.T.
def _dense_mma(a, bT, *, semantics):
    """acc[M,N] = a[M,K] @ bT[N,K].T, single CTA, FP8 operands, f32 accumulate."""
    M, K = a.shape
    N, K2 = bT.shape
    assert K == K2
    swizzle = 128
    swizzle_elems = swizzle // jnp.dtype(a.dtype).itemsize  # fp8: 128
    transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))
    out_swizzle_elems = swizzle // jnp.dtype(jnp.float32).itemsize  # f32: 32
    out_transforms = (plgpu.TilingTransform((8, out_swizzle_elems)), plgpu.SwizzleTransform(swizzle))

    def body(a_gmem, b_gmem, o_gmem):
        def acc_scope(acc_ref):
            def block_mm(_, a_smem, b_smem):
                plgpu.wgmma(acc_ref, a_smem, plgpu.transpose_ref(b_smem, (1, 0)))
                plgpu.wgmma_wait(0)

            plgpu.emit_pipeline(
                block_mm,
                grid=(1,),
                in_specs=[
                    plgpu.BlockSpec((M, K), lambda i: (0, 0), transforms=transforms),
                    plgpu.BlockSpec((N, K), lambda i: (0, 0), transforms=transforms),
                ],
                max_concurrent_steps=1,
            )(a_gmem, b_gmem)
            return acc_ref[...]

        acc = pl.run_scoped(acc_scope, plgpu.ACC((M, N)))

        @functools.partial(pl.run_scoped, o_smem=plgpu.SMEM((M, N), dtype=jnp.float32, transforms=out_transforms))
        def store(o_smem):
            o_smem[...] = acc.astype(jnp.float32)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem, o_gmem)
            plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    kernel = plgpu.kernel(
        body,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        grid=(1,),
        grid_names=("x",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=semantics),
    )
    return kernel(a, bT)


def _scan_ptx_for(*needles) -> dict[str, bool]:
    found = {n: False for n in needles}
    for path in glob.glob(os.path.join(_PTX_DUMP, "**", "*.ptx"), recursive=True) + glob.glob(
        os.path.join(_PTX_DUMP, "*.ptx")
    ):
        try:
            txt = open(path).read()
        except OSError:
            continue
        for n in needles:
            if n in txt:
                found[n] = True
    return found


def _check_numerics(name, out, ref, results, ptx_needle=None):
    out = np.asarray(out, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    max_abs = float(np.max(np.abs(out - ref)))
    denom = float(np.max(np.abs(ref))) or 1.0
    rel = max_abs / denom
    ptx = _scan_ptx_for(ptx_needle).get(ptx_needle, None) if ptx_needle else None
    ok = rel < 1e-2
    results[name] = {"status": "OK" if ok else "NUMERIC-FAIL", "max_abs": max_abs, "rel": rel, "ptx_has_needle": ptx}
    print(f"[{results[name]['status']}] {name}: max_abs={max_abs:.4g} rel={rel:.4g} ptx({ptx_needle})={ptx}")


def _run_case(name, fn, results):
    try:
        fn()
    except Exception as e:  # noqa: BLE001 -- we want the full matrix even if one case dies
        results[name] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")
        traceback.print_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=256)
    args = ap.parse_args()
    M, N, K = args.M, args.N, args.K

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}  core_count={getattr(dev, 'core_count', '?')}")
    if "H100" not in dev.device_kind and "Hopper" not in dev.device_kind:
        print(f"WARNING: not an H100 ({dev.device_kind}); fp8 wgmma requires sm_90a")

    k0, k1 = jax.random.split(jax.random.key(0))
    results: dict = {}

    # ---- dense, both semantics, baseline (e4m3^2) + mixed (e4m3 x e5m2) ----
    sem_map = {"Lane": plgpu.LoweringSemantics.Lane, "Warpgroup": plgpu.LoweringSemantics.Warpgroup}
    for sem_name, sem in sem_map.items():
        for bdt_name in ("e4m3", "e5m2"):
            adt, bdt = E4M3, _DT[bdt_name]
            a = _rand_fp8(k0, (M, K), adt)
            b = _rand_fp8(k1, (K, N), bdt)
            bT = b.T  # K-major RHS for FP8 wgmma (presented (K,N) via transpose_ref in-kernel)
            ref = a.astype(jnp.float32) @ b.astype(jnp.float32)
            needle = f".f32.e4m3.{bdt_name}"
            tag = f"dense/{sem_name}/e4m3x{bdt_name}"

            def fn(a=a, bT=bT, ref=ref, sem=sem, needle=needle, tag=tag):
                out = jax.block_until_ready(_dense_mma(a, bT, semantics=sem))
                _check_numerics(tag, out, ref, results, ptx_needle=needle)

            _run_case(tag, fn, results)

    # ---- production path: haliax transposed_ragged_dot (Warpgroup), mixed E4M3 x E5M2 ----
    # lhs (M=hidden, K=tokens) E4M3 activations; rhs (N=out, K=tokens) E5M2 grad; single group.
    def wgrad_case():
        Kt = max(K, 256)
        lhs = _rand_fp8(k0, (M, Kt), E4M3)
        rhs = _rand_fp8(k1, (N, Kt), E5M2)
        gs = jnp.array([Kt], dtype=jnp.int32)
        out = jax.block_until_ready(transposed_ragged_dot(lhs, rhs, gs, out_dtype=jnp.float32))
        ref = (lhs.astype(jnp.float32) @ rhs.astype(jnp.float32).T)[None]  # (1, M, N)
        _check_numerics("wgrad/Warpgroup/e4m3xe5m2", out, ref, results, ptx_needle=".f32.e4m3.e5m2")

    _run_case("wgrad/Warpgroup/e4m3xe5m2", wgrad_case, results)

    # ---- negative control: e4m3 x bf16 must STILL be rejected ----
    def neg_case():
        a = _rand_fp8(k0, (M, K), E4M3)
        b = jax.random.normal(k1, (K, N), jnp.bfloat16)
        try:
            jax.block_until_ready(_dense_mma(a, b, semantics=plgpu.LoweringSemantics.Lane))
        except Exception as e:  # noqa: BLE001
            results["neg/e4m3xbf16"] = {"status": "OK (correctly rejected)", "error": f"{type(e).__name__}"}
            print(f"[OK] neg/e4m3xbf16 correctly rejected: {type(e).__name__}")
            return
        results["neg/e4m3xbf16"] = {"status": "FAIL (should have been rejected)"}
        print("[FAIL] neg/e4m3xbf16 was NOT rejected")

    _run_case("neg/e4m3xbf16", neg_case, results)

    # ---- summary ----
    print("\n===== SUMMARY =====")
    for name, r in results.items():
        print(f"  {name:34} {r}")
    print(f"\nptx dump dir: {_PTX_DUMP}")
    mixed_ptx = _scan_ptx_for(".f32.e4m3.e5m2", ".f16.e4m3.e5m2")
    print(f"mixed-fp8 PTX present anywhere: {mixed_ptx}")


if __name__ == "__main__":
    main()
