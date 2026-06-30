# jaxlib C++ change for mixed E4M3×E5M2 wgmma on the Warpgroup (production) path

> **STATUS: BUILT + VERIFIED (GFP8-036).** This change was applied to `jax-v0.10.0`, built from
> source (clang-19, bazel 7.7.0) into a `jaxlib-0.10.0+selfbuilt` cp312 wheel, installed over stock
> jaxlib (stock `jax-cuda13-plugin` kept), and exercised on a real H100. The previously-blocked
> production Warpgroup path — `transposed_ragged_dot` with E4M3 activations × E5M2 grad — now
> **compiles** (verifier passes), emits the literal `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2`,
> and is **numerically exact** (max_abs = 0) vs an f32 reference. Build/verify recipe:
> `build_patched_jaxlib.sh` (CPU node) → `run_h100_verify.sh` (H100). Jobs `/matt/mixed-fp8-jaxlib-build3`,
> `/matt/mixed-fp8-h100-verify`.

The Python patch (`mixed_fp8_wgmma_patch.py` / `jax_mixed_fp8_wgmma.diff`) is sufficient for the
**Lane** lowering path (the pure-Python PTX emitter). It is **not** sufficient for the **Warpgroup**
path that our production kernels (`ragged_dot_mgpu`, `transposed_ragged_dot`) use, because that path
builds a `mosaic_gpu.wgmma` MLIR op whose verifier is **compiled into the jaxlib native extension**.

H100 evidence (`test_mixed_fp8_wgmma.py`, job `/matt/mixed-fp8-wgmma-run2`):

```
'mosaic_gpu.wgmma' op The `a` and `b` inputs must have the same element type.
  %422 = "mosaic_gpu.wgmma"(%arg15, %418, %421) :
    (vector<128x128xf32>,
     memref<128x128xf8E4M3FN, ...>,   # a (e4m3)
     memref<128x128xf8E5M2, ...>)     # b (e5m2)  <- mixed op constructed correctly; verifier rejects
    -> vector<128x128xf32>
```

The op is *constructed* correctly mixed (the Python patch removes the trace-time gate), and the
dialect→LLVM lowering rule (`dialect_lowering.py:_mgpu_wgmma_op_lowering_rule`) is **Python** and calls
the already-patched `wgmma.wgmma` emitter — so the **only** thing blocking the Warpgroup path is the
runtime verifier. The ODS already lists both FP8 types as valid A/B operand types
(`MosaicGPU_WGMMASupportedABType`); the same-type rule is a hand-written `verify()`, not an ODS trait.

## The change

File: `jaxlib/mosaic/dialect/gpu/mosaic_gpu.cc`, `WGMMAOp::verify()` (the same-element-type check,
introduced in jax commit `66f45d0`, 2024-12-11; never relaxed for FP8 since; no upstream tracking issue).

```diff
-  if (a_type.getElementType() != b_type.getElementType()) {
-    return error("The `a` and `b` inputs must have the same element type.");
-  }
+  auto a_el = a_type.getElementType();
+  auto b_el = b_type.getElementType();
+  // FP8 is the explicit PTX-ISA exception to .atype == .btype: wgmma.mma_async accepts independent
+  // .e4m3/.e5m2 operand types on sm_90a (see PTX ISA 9.7.16; CUTLASS emits ...f32.e4m3.e5m2).
+  bool both_fp8 =
+      llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(a_el) &&
+      llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(b_el);
+  if (a_el != b_el && !both_fp8) {
+    return error("The `a` and `b` inputs must have the same element type "
+                 "(except for the e4m3/e5m2 FP8 pair).");
+  }
```

`llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(...)` is the idiom already used in this
exact translation unit (e.g. the `MultimemLoad` verifier ~line 1001), so the headers are in scope —
this is the form that was **actually built** (logbook GFP8-036), not the `.isFloat8E4M3FN()` method
form. That is the entire functional change for the Warpgroup path. No change is needed to the dialect
lowering or the `.td` operand constraints.

## Cost

The verifier is compiled into the `jaxlib` wheel (`gentbl` + `cc_library mosaic_gpu` in
`jaxlib/mosaic/dialect/gpu/BUILD`); there is **no Python override**. Shipping this requires either:

1. **Build jaxlib from source** (bazel) with the one-line change and carry a custom wheel, or
2. **Upstream a PR** to jax-ml/jax relaxing the verifier for the FP8 pair (small, matches documented
   PTX-ISA/CUTLASS behavior, ODS already permits both operand types) — the cleanest long-term path.

Until one of those lands, the genuine-f8 wgrad stays all-E4M3 (`grad_dtype=float8_e4m3fn`); the
E5M2-grad hybrid on the fast Mosaic path is reachable only with the jaxlib change. The Lane path works
today with the Python patch alone but is not what the production kernels use.
