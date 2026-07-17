# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-002 debug: pod-side trace of the grouped-kernel cutlass_call.

Runs the exact FunctionSpec + get_or_compile_kernel path the FFI would run,
at small shapes, with trace-time pointer diagnostics enabled in the vendored
kernel ([mxfp8-dbg] prints). Then attempts the full jit path on tiny device
arrays. The x86 CPU-only trace of the identical spec compiles clean; this
isolates what differs on the GB200 pod.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import numpy as np

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass.jax import compile as cjc

from mxfp8_grouped.adapter import (
    _build_launcher,
    _B200_SMS,
    _FP8_TMA_VEC,
    SF_VEC_SIZE,
    _ceil_div,
    _round_up,
    build_sfa,
    build_sfb,
    dequantize_mxfp8,
    ensure_blackwell_arch,
    mxfp8_grouped_mm,
    quantize_mxfp8,
)
from mxfp8_grouped.moe_utils import MoEScaledGroupedGemmTensormapConstructor

E, M, K, N = 8, 8192, 256, 256
group_sizes = [M // E] * E

print(f"device: {jax.devices()[0].device_kind}, CUTE_DSL_ARCH={os.environ.get('CUTE_DSL_ARCH')}")
if jax.default_backend() == "gpu":
    ensure_blackwell_arch()
else:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
print(f"after ensure: CUTE_DSL_ARCH={os.environ.get('CUTE_DSL_ARCH')}")

# ── Part A: direct get_or_compile_kernel (same spec the FFI builds) ──
rng = np.random.default_rng(0)
x_q = jnp.asarray(rng.integers(0, 100, (M, K), dtype=np.uint8)).view(jnp.float8_e4m3fn)
w_q = jnp.asarray(rng.integers(0, 100, (E, N, K), dtype=np.uint8)).view(jnp.float8_e4m3fn)
sfa_cols = _round_up(_ceil_div(K, SF_VEC_SIZE), 4)
sfa_rows = sum(_round_up(g, 128) for g in group_sizes)
sfb_cols = _round_up(N, 128) * sfa_cols
x_sf = jnp.asarray(rng.integers(120, 130, (sfa_rows, sfa_cols), dtype=np.uint8)).view(jnp.float8_e8m0fnu)
w_sf = jnp.asarray(rng.integers(120, 130, (E, sfb_cols), dtype=np.uint8)).view(jnp.float8_e8m0fnu)
offs = jnp.cumsum(jnp.asarray(group_sizes, dtype=jnp.int32)).astype(jnp.int32)

desc_bytes = MoEScaledGroupedGemmTensormapConstructor.get_workspace_size("2Dx3D", E)
ws_bytes = desc_bytes + E * 4

launcher = _build_launcher(
    e=E,
    mma_tiler_mnk=(128, 128, 128),
    cluster_shape_mnk=(1, 1, 1),
    max_active_clusters=_B200_SMS,
)

ts = cjax.TensorSpec
input_spec = (
    ts(mode=(0, 1), divisibility=(1, _FP8_TMA_VEC), static=True),
    ts(mode=(0, 2, 1), divisibility=(1, 1, _FP8_TMA_VEC), static=True),
    ts(mode=(0, 1), static=True),
    ts(mode=(0, 1), static=True),
    ts(mode=(0,), static=True),
)
output_spec = (
    ts(mode=(0, 1), static=True),
    ts(mode=(0,), static=True),
)
ins_flat, in_tree = jax.tree.flatten((x_q, w_q, x_sf, w_sf, offs))
outs_flat, out_tree = jax.tree.flatten(
    (jax.ShapeDtypeStruct((M, N), jnp.bfloat16), jax.ShapeDtypeStruct((ws_bytes,), jnp.uint8))
)
spec = cjc.build_function_spec(
    ins_flat, in_tree, outs_flat, out_tree, input_spec, output_spec, {}, None, True, {}
)

print("=== Part A: direct get_or_compile_kernel ===")
result = cjc.get_or_compile_kernel(launcher, spec)
print("PART A OK, module bytes:", len(result.module))

# ── Part B: real jit path with device arrays ──
print("=== Part B: jit path on device ===")
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)
w = jax.random.normal(key, (E, K, N), dtype=jnp.bfloat16) / (K**0.5)
xq, xs = quantize_mxfp8(x)
wq, wsc = quantize_mxfp8(jnp.swapaxes(w, 1, 2))
sfa = build_sfa(xs, group_sizes)
sfb = build_sfb(wsc)
fn = jax.jit(lambda a, b, c, d, o: mxfp8_grouped_mm(a, c, b, d, o))
out = jax.block_until_ready(fn(xq, wq, sfa, sfb, offs))
print("PART B ran:", out.shape, float(jnp.mean(jnp.abs(out.astype(jnp.float32)))))

# Correctness at small shape: dequantize the same quantized operands.
x_deq = dequantize_mxfp8(xq, xs)
w_deq = jnp.swapaxes(dequantize_mxfp8(wq, wsc), 1, 2)  # (E, K, N)
refs = []
start = 0
for ei, g in enumerate(group_sizes):
    refs.append(x_deq[start : start + g] @ w_deq[ei])
    start += g
ref = jnp.concatenate(refs, axis=0)
err = float(jnp.linalg.norm(out.astype(jnp.float32) - ref) / jnp.linalg.norm(ref))
err_bf = float(
    jnp.linalg.norm(out.astype(jnp.float32) - ref.astype(jnp.bfloat16).astype(jnp.float32)) / jnp.linalg.norm(ref)
)
print(f"PART B rel-fro err vs deq ref: f32 {err:.3e}, bf16-rounded {err_bf:.3e}")
assert err_bf < 1e-3, err_bf
print("PART B OK")
