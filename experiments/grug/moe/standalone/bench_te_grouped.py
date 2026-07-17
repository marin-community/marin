# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-004b: NVIDIA TransformerEngine grouped_dense baseline at grug MoE shapes.

Benchmarks ``transformer_engine.jax.dense.grouped_dense`` (grouped_quantize ->
cuBLAS ``nvte_grouped_gemm``) on 1x GB200 at the row-13 per-device grouped
GEMMs (M = 262144 tokens, E = 64, uniform 4096 tokens/expert):

  w13   x[M, 2560] @ w[E, 2560, 1280] -> [M, 1280]
  w2    x[M, 1280] @ w[E, 1280, 2560] -> [M, 2560]

Arms per shape:
  te_bf16_fwd / te_bf16_fwdbwd      grouped_dense with the no-op quantizer set
  te_mxfp8_fwd / te_mxfp8_fwdbwd    MXFP8 1D block scaling (e4m3 fwd, e5m2 bwd);
                                    fwd INCLUDES TE's grouped_quantize of x and
                                    kernel -- it is a GEMM+producer number
  te_mxfp8_gemm_only                tex.grouped_gemm on pre-quantized tensors
                                    (comparable to the CuTeDSL GEMM-only numbers
                                    in bench_mxfp8_grouped.py)
  te_mxfp8_quantize_{x,w}           TE 2x2x grouped-quantize producer cost
  bf16_ragged                       jax.lax.ragged_dot XLA yardstick

fwd TF/s uses 2*M*K*N; fwd+bwd uses 6*M*K*N (fwd + dgrad + wgrad). Numerics:
rel-Frobenius vs a per-expert f32 reference (~4e-2 class expected for MXFP8),
and MXFP8 grads vs the TE bf16 grads.

TE constraints hit at these shapes (v2.16): kMaxGroups = 64 (E=64 fits), MXFP8
V2 grouped quantize/GEMM needs K, N and every group size to be multiples of
128 (2560/1280/4096 all fit), MXFP8 grouped GEMM needs cuBLAS >= 13.3.

Install on the aarch64 GB200 pod (no prebuilt aarch64 wheel for the jax glue;
transformer_engine_cu13 has a manylinux aarch64 wheel up to 2.16.0, and the
transformer_engine_jax sdist compiles a small pybind11 extension against the
pod's jax[cuda13] -- build with --no-build-isolation so it picks cu13, and
--no-deps so its static cu12 metadata is ignored):

  uv pip install "transformer-engine-cu13==2.16.0" "pybind11[global]" flax ninja cmake
  # + libnccl.so symlink for the -lnccl link; see run_te_bench.sh for the full recipe
  LIBRARY_PATH=<nccl lib dir> uv pip install --no-build-isolation --no-deps "transformer-engine-jax==2.16.0"

Usage: python bench_te_grouped.py [--tokens 262144] [--iters 50]
         [--recipes bf16,mxfp8] [--out out.json]
"""

import argparse
import importlib
import importlib.metadata
import json
import statistics
import time
import traceback

import jax
import jax.numpy as jnp
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.dense import grouped_dense
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScalingMode,
    TensorUsage,
    noop_quantizer_set,
)

E = 64
# (label, K, N): row-13 per-device grouped GEMMs (d2560, F1280).
SHAPES = [
    ("w13_k2560_n1280", 2560, 1280),
    ("w2_k1280_n2560", 1280, 2560),
]
CONTRACTING_DIMS = ((1,), (1,))
MXFP8_GROUP_ALIGN = 128  # TE V2 MXFP8 grouped quantize/GEMM group-size multiple
PROBE_PACKAGES = [
    "transformer-engine-cu13",
    "transformer-engine-jax",
    "jax",
    "jax-cuda13-plugin",
    "nvidia-cublas",
    "nvidia-cudnn-cu13",
    "nvidia-nccl-cu13",
    "nvidia-cutlass-dsl",
    "flax",
]


def probe_versions() -> dict:
    versions = {}
    for pkg in PROBE_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"
    # The pip dist installs no importable transformer_engine_jax module; TE's
    # framework loader registers it in sys.modules when transformer_engine.jax
    # is imported, so resolve it lazily rather than via a top-level import.
    te_bind = importlib.import_module("transformer_engine_jax")
    versions["runtime_cublasLt"] = te_bind.get_cublasLt_version()
    versions["runtime_cuda"] = te_bind.get_cuda_version()
    versions["runtime_cudnn"] = te_bind.get_cudnn_version()
    dev = jax.devices()[0]
    versions["device"] = f"{dev.device_kind} (cc {dev.compute_capability})"
    return versions


def timed(fn, args, iters, warmup):
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def rel_frob(a, ref):
    a = jnp.asarray(a, jnp.float32)
    ref = jnp.asarray(ref, jnp.float32)
    return float(jnp.linalg.norm(a - ref) / jnp.linalg.norm(ref))


def ragged_reference(x, w_ekn, group_size: int):
    """Per-expert f32 matmul reference (uniform groups; dense per-group dots)."""
    outs = []
    for ei in range(E):
        xg = x[ei * group_size : (ei + 1) * group_size].astype(jnp.float32)
        outs.append(xg @ w_ekn[ei].astype(jnp.float32))
    return jnp.concatenate(outs, axis=0)


def make_quantizer_set(n_groups: int):
    return QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e5m2,
        is_2x2x=True,
        n_groups=n_groups,
    )


def run_case(label: str, k: int, n: int, m: int, iters: int, warmup: int, recipes: set[str]) -> dict:
    group_size = m // E
    gs_dev = jnp.full((E,), group_size, dtype=jnp.int32)
    fwd_flops = 2 * m * k * n
    fwdbwd_flops = 3 * fwd_flops
    res: dict = {"m": m, "k": k, "n": n, "group_size": group_size, "arms": {}, "errors": {}}
    print(f"\n== {label} M={m} K={k} N={n} groups={E}x{group_size} ==", flush=True)

    key = jax.random.PRNGKey(0)
    kx, kw, kc = jax.random.split(key, 3)
    x = jax.random.normal(kx, (m, k), dtype=jnp.bfloat16)
    w = jax.random.normal(kw, (E, k, n), dtype=jnp.bfloat16) / (k**0.5)
    ct = jax.random.normal(kc, (m, n), dtype=jnp.bfloat16)

    def bench(arm: str, fn, args, flops):
        t = timed(fn, args, iters, warmup)
        res["arms"][arm] = {"ms": t * 1e3, "tfs": flops / t / 1e12}
        print(f"  {arm:22s} {t * 1e3:8.3f} ms  {flops / t / 1e12:7.1f} TF/s", flush=True)

    def make_fwd(qs):
        def f(xx, ww):
            return grouped_dense(xx, ww, gs_dev, CONTRACTING_DIMS, quantizer_set=qs)

        return f

    def make_fwdbwd(qs):
        def f(xx, ww, cc):
            out, pull = jax.vjp(make_fwd(qs), xx, ww)
            # Return out too so the fwd GEMM is not DCE'd out of the timed graph.
            return out, pull(cc)

        return f

    grads_bf16 = None
    if "bf16" in recipes:
        fwd_fn = jax.jit(make_fwd(noop_quantizer_set))
        out = jax.block_until_ready(fwd_fn(x, w))
        assert out.shape == (m, n), out.shape
        bench("te_bf16_fwd", fwd_fn, (x, w), fwd_flops)

        fwdbwd_fn = jax.jit(make_fwdbwd(noop_quantizer_set))
        _, grads_bf16 = jax.block_until_ready(fwdbwd_fn(x, w, ct))
        bench("te_bf16_fwdbwd", fwdbwd_fn, (x, w, ct), fwdbwd_flops)

        ragged_fn = jax.jit(lambda xx, ww, gg: jax.lax.ragged_dot(xx, ww, gg, preferred_element_type=jnp.bfloat16))
        bench("bf16_ragged", ragged_fn, (x, w, gs_dev), fwd_flops)

        ref = ragged_reference(x, w, group_size)
        res["errors"]["te_bf16_fwd"] = rel_frob(out, ref)
        print(f"  te_bf16_fwd err(f32 ref) {res['errors']['te_bf16_fwd']:.2e}", flush=True)
        assert res["errors"]["te_bf16_fwd"] < 1e-2
        del out, ref

    if "mxfp8" in recipes:
        assert k % 128 == 0 and n % 128 == 0, "TE MXFP8 grouped GEMM needs K, N % 128 == 0"
        assert group_size % MXFP8_GROUP_ALIGN == 0, "TE V2 MXFP8 needs 128-aligned group sizes"

        qs = make_quantizer_set(E)
        fwd_fn = jax.jit(make_fwd(qs))
        out = jax.block_until_ready(fwd_fn(x, w))
        assert out.shape == (m, n), out.shape
        bench("te_mxfp8_fwd", fwd_fn, (x, w), fwd_flops)

        fwdbwd_fn = jax.jit(make_fwdbwd(qs))
        _, grads_mx = jax.block_until_ready(fwdbwd_fn(x, w, ct))
        bench("te_mxfp8_fwdbwd", fwdbwd_fn, (x, w, ct), fwdbwd_flops)

        ref = ragged_reference(x, w, group_size)
        res["errors"]["te_mxfp8_fwd"] = rel_frob(out, ref)
        print(f"  te_mxfp8_fwd err(f32 ref) {res['errors']['te_mxfp8_fwd']:.2e}", flush=True)
        assert res["errors"]["te_mxfp8_fwd"] < 0.1, "MXFP8 fwd error above the ~4e-2 class"
        if grads_bf16 is not None:
            res["errors"]["te_mxfp8_dgrad_vs_bf16"] = rel_frob(grads_mx[0], grads_bf16[0])
            res["errors"]["te_mxfp8_wgrad_vs_bf16"] = rel_frob(grads_mx[1], grads_bf16[1])
            print(
                f"  te_mxfp8 dgrad err(vs bf16) {res['errors']['te_mxfp8_dgrad_vs_bf16']:.2e}  "
                f"wgrad err(vs bf16) {res['errors']['te_mxfp8_wgrad_vs_bf16']:.2e}",
                flush=True,
            )
        del out, ref, grads_mx

        # Optional low-level arms: GEMM on pre-quantized tensors (mirrors the
        # grouped_dense fwd rule) + producer costs. Guarded: these poke TE
        # internals (tex.*), and a signature drift must not lose the
        # grouped_dense numbers above on a remote time-boxed run.
        try:
            qs1 = QuantizerFactory.create_set(
                scaling_mode=ScalingMode.MXFP8_1D_SCALING,
                fwd_dtype=jnp.float8_e4m3fn,
                bwd_dtype=jnp.float8_e5m2,
                is_2x2x=False,
                n_groups=E,
            )
            lhs = tex.grouped_quantize(x, qs1.x, gs_dev, flatten_axis=-1).get_tensor(usage=TensorUsage.LHS)
            rhs = tex.grouped_quantize(w, qs1.kernel, flatten_axis=-1).get_tensor(usage=TensorUsage.RHS)
            gemm_fn = jax.jit(lambda a, b: tex.grouped_gemm(a, b, contracting_dims=CONTRACTING_DIMS))
            out = jax.block_until_ready(gemm_fn(lhs, rhs))
            assert out.shape == (m, n), out.shape
            bench("te_mxfp8_gemm_only", gemm_fn, (lhs, rhs), fwd_flops)
            del lhs, rhs, out

            qx_fn = jax.jit(lambda t: tex.grouped_quantize(t, qs.x, gs_dev, flatten_axis=-1))
            bench("te_mxfp8_quantize_x", qx_fn, (x,), 0)
            qw_fn = jax.jit(lambda t: tex.grouped_quantize(t, qs.kernel, flatten_axis=-1))
            bench("te_mxfp8_quantize_w", qw_fn, (w,), 0)
        except Exception:
            print("  low-level tex.* arms failed (grouped_dense numbers above still stand):", flush=True)
            traceback.print_exc()

    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=262144)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--recipes", default="bf16,mxfp8", help="comma-separated subset of bf16,mxfp8")
    p.add_argument("--out", default="bench_te_grouped.json")
    a = p.parse_args()
    recipes = set(a.recipes.split(","))
    assert recipes <= {"bf16", "mxfp8"}, recipes
    assert a.tokens % E == 0

    versions = probe_versions()
    print("versions:", json.dumps(versions, indent=2), flush=True)

    results = {"versions": versions, "tokens": a.tokens, "experts": E, "cases": {}}
    for label, k, n in SHAPES:
        results["cases"][label] = run_case(label, k, n, a.tokens, a.iters, a.warmup, recipes)

    # Job pods are ephemeral: print the JSON to stdout so the logs carry it.
    print("\nRESULTS_JSON " + json.dumps(results), flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
