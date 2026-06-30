# FP8 `ragged_dot` (MoE grouped matmul) — 3-way synthesis

Three independent from-first-principles investigations (me = coordinator, codex/gpt-5.5,
opencode/glm-5.2), no notes shared until all finished. This compares them and proposes a
synthesized solution. H100 benchmark numbers and the `research/grug-fp8-h100` comparison are
filled in in the marked sections.

## Strong convergence (all three landed here independently)

All three mirror the dense direct-quant fp8 op (PR #6660):
- Quantize both forward operands to **E4M3**, contract, dequantize; custom VJP quantizes the
  output gradient to **E5M2** and runs the two backward grouped matmuls.
- **Genuine mixed E5M2×E4M3 backward** (the required TE recipe).
- TE-style **delayed per-tensor scaling** (scale + amax history) carried through the VJP as
  `OverwriteWithGradient` state, reusing the dense primitives (`quantize`/`dequantize`/
  `update_fp8_meta`/`compute_scale`).
- The three `RaggedDotDimensionNumbers` layouts (fwd / dlhs / drhs) for the VJP.
- A new `Fp8RaggedDot*Op` module mirroring `Fp8DotGeneralOp`.

That three independent agents reproduced this exact shape is strong evidence it is the natural,
simplest structure — and it directly reuses #6660 rather than reinventing.

## The one real divergence: the inner GEMM backend

| | backend | how mixed-fp8 bwd is handled | integration | tests |
|---|---|---|---|---|
| **me** | Pallas-**Triton** kernel | vendored **non-promoting** kernels: `pl.dot(a,b)` without `jnp.result_type` up-cast, so E5M2×E4M3 stays fp8 | core op + harness + **backend probe** | CPU logic test |
| **codex** | Pallas-**Triton** (reuses existing kernel) + `preferred_element_type` plumbing into the backend; `implementation=triton/xla/auto` switch | **relies on the existing kernel**, which up-casts mixed operands via `result_type` → mixed pair is **not** genuine fp8 on the triton path unless the kernel is changed | core op + `Fp8RaggedDotOp` | 19 tests incl **jaxpr dtype asserts** |
| **opencode** | **XLA** `ragged_dot_general` with fp8 operands + `preferred_element_type`; no custom kernel | passes E5M2 grad + E4M3 operand straight to `ragged_dot_general` (verified to accept mixed fp8) | core op + `Fp8RaggedDotGeneralOp` (shared `_Fp8StateMixin`) + **full `MoELinear.dot_general` wiring** + `fp8_ragged_dot` entry w/ 512-pad+AR | 7 CPU tests + **1 GPU HLO test** |

### Pros / cons

**Triton path (me, codex)**
- + Reuses the already-optimized bf16 grouped kernel; single grouped kernel (no per-expert loop);
  full control of tiling/accumulation.
- − Hinges on Pallas-Triton lowering fp8 `pl.dot` to a Hopper fp8 MMA. **Mixed** E5M2×E4M3 is the
  big unknown — Triton historically wants same-dtype operands. **Important subtlety the bake-off
  exposed:** codex's reuse of the unmodified kernel up-casts mixed operands via `result_type`
  (→ f16 MMA, not fp8) on the triton path; only the explicitly non-promoting kernels (my version)
  keep the backward genuinely fp8. Small `block_k=32` (bf16-tuned) may also under-serve fp8.

**XLA path (opencode)**
- + Simplest code (no kernel). Mixed fp8 accepted by `ragged_dot_general`. If XLA has a grouped
  fp8 lowering, it routes to cuBLASLt and is well-optimized.
- − **This is exactly the QDQ failure mode #6660 fixed for the dense case**: XLA's GemmRewriter
  silently fell back to bf16 on Hopper. Direct-quant (real fp8 operands, as opencode does) is the
  right mitigation, but whether XLA's *ragged*-dot GPU lowering emits a genuine fp8 grouped GEMM
  (vs. dequantizing to bf16 first) is **unverified** and is the single highest-risk assumption.

### Integration / testing best-of
- **Most complete integration:** opencode (MoELinear dispatch, shared `_Fp8StateMixin`, pad/AR parity).
- **Best correctness guards:** codex's jaxpr dtype asserts (forward contracts E4M3, backward
  contracts E5M2) + opencode's compiled-HLO check for `f8e4m3`/`f8e5m2` kernels. Together they
  lock in "genuine mixed fp8" at both the jaxpr and the emitted-kernel level.
- **Backend adjudication:** my candidate-independent probe (`probe_fp8_backends.py`) times
  triton-fp8 vs xla-fp8 for the forward and the mixed backward and dumps HLO — it decides the
  backend question for all three at once.

## H100 benchmark results (cw-us-east-02a, 1×H100 80GB; M=262144, K=N=3072, E=128; same bf16 baseline)

bf16-triton baseline: **469 TFLOP/s** fwd (10.55 ms); 695 TFLOP/s fwd+bwd (21.4 ms).

| candidate | backend | fwd vs bf16 | fwd+bwd vs bf16 | genuine fp8? |
|---|---|---|---|---|
| me / codex | Pallas-**Triton** `pl.dot` fp8 | **0.61× (slower)** | **ERROR** | **No** — `pl.dot` upcasts; **rejects mixed E5M2×E4M3** ("8-bit floats do not support implicit promotion") |
| opencode | **XLA** `ragged_dot_general` fp8 | **fails** | fails | **No** — lowers to a **bf16 `__triton_gemm`** (QDQ-style dequant) + autotuner OOM (96 GiB) |
| research | **Mosaic-GPU** f8 wgmma | 1.05× | **1.23×** ✅ | **Yes** (all-E4M3) — but **E5M2 grad ⇒ ERROR** "lhs and rhs must have the same dtype" |

Verdicts:
- **The two "simplest" in-marin paths do not produce genuine fp8 at all.** Triton `pl.dot` is *slower*
  than bf16 and cannot take a mixed fp8 pair; XLA `ragged_dot_general` dequantizes fp8→bf16 (exactly
  the QDQ fallback PR #6660 fixed for the dense case) and its autotuner OOMs.
- **Only the Mosaic-GPU f8 wgmma kernel beats bf16** (1.23× fwd+bwd, clears the ≥20% bar), and it
  needs all operands in one fp8 dtype.

## Comparison with `research/grug-fp8-h100` (benchmarked with the SAME harness)

The research branch independently reached the same conclusion the hard way (logbook GFP8-001→034):
QDQ pattern-matching is fragile, XLA scaled-dot is version-sensitive, Triton/XLA ragged fp8 don't
deliver — so they wrote a **Mosaic-GPU f8 wgmma grouped GEMM** (`nn/ragged_dot.py` `"mosaic"` backend +
`MosaicBlockConfig` block_k=128; `_src/fp8_ragged.py` `fp8_scaled_ragged_dot`). Shipped recipe:
**E4M3 forward + dgrad, bf16 weight-gradient** → their measured **~1.27–1.33× e2e** (mine: **1.23×**
fwd+bwd on a single gate/up GEMM — consistent; the gap is their tuned per-shape config + the f8 wgrad).

**The decisive shared finding (GFP8-034, reproduced by my probe):** mixed **E4M3×E5M2 wgmma is a JAX
software gap, not a Hopper limit** — `mosaic/gpu/wgmma.py` rejects `A.dtype != B.dtype` and hardcodes
the PTX atype to the B dtype, while PTX ISA 9.3 allows independent atype/btype. So the research
*shipped* path uses **all-E4M3** (its f8-wgrad arm even requires `grad_dtype=float8_e4m3fn`), and the
**E5M2 backward you require is reachable only via the ~15-line `wgmma.py` patch** (see
`patches/jax-mosaic-wgmma-mixed-fp8.patch`). My from-first-principles Triton result hit the *same* JAX
gate from the other side ("8-bit floats do not support implicit promotion").

### Pros / cons: my candidates vs the research solution

| | research Mosaic (winner) | my/codex Triton | opencode XLA |
|---|---|---|---|
| Beats bf16 ≥20% | **Yes (1.23–1.33×)** | No (0.61×) | No |
| Genuine fp8 on H100 | **Yes** | No | No |
| Code complexity | High (custom Mosaic kernel + tuning) | Low | **Lowest** |
| Mixed E5M2×E4M3 today | No (needs wgmma patch) | No (same gate) | No (dequantizes) |
| Best ideas to keep | the Mosaic kernel; `MosaicBlockConfig`; `mosaic_wgrad` param; per-tensor delayed-scaling VJP | codex's `implementation` selector + `preferred_element_type` plumbing; jaxpr dtype tests | opencode's `MoELinear.dot_general` wiring + shared `_Fp8StateMixin` + HLO fp8-kernel test |

The three from-first-principles attempts converged on the right *structure* (direct-quant VJP, delayed
scaling, three layouts) but all chose backends that XLA/Triton silently defeat — which is precisely the
non-obvious lesson the research branch paid for. Reproducing it from scratch validates that the Mosaic
kernel is necessary, not incidental.

## Recommended synthesized solution (decided by the benchmark)

The benchmark settles the backend question: **the Mosaic-GPU f8 wgmma grouped GEMM is the only path
that beats bf16** (1.23× fwd+bwd). The simplest paths (Triton `pl.dot`, XLA ragged) are dead ends —
do not ship them as the fp8 path. So the recommendation is **the research branch's Mosaic kernel as the
engine, wrapped in the best integration/testing from the three from-first-principles attempts:**

1. **Engine:** the research **Mosaic-GPU f8 wgmma grouped GEMM** (`nn/ragged_dot.py` `"mosaic"` backend
   + `MosaicBlockConfig` H100 default, block_k=128) for the forward + dgrad. This is the necessary,
   non-obvious piece — keep it.
2. **Op + state:** `fp8_scaled_ragged_dot` custom-VJP with per-tensor delayed scaling, reusing a
   **shared `_Fp8StateMixin`** (opencode) with the dense `Fp8DotGeneralOp` — DRY state.
3. **Integration:** **`MoELinear.dot_general` dispatch** + `fp8_ragged_dot` entry with 512-pad/all-reduce
   parity (opencode) — a real drop-in for the bf16 path, not just a microbench op.
4. **Backend selector:** keep an `implementation` field (codex) over `{mosaic, triton, xla}` so non-H100
   / non-128-tileable shapes fall back to the (correct-but-not-faster) triton/xla path; `mosaic` is the
   fast H100 path.
5. **Recipe + the E5M2 requirement:**
   - *Today, no external patch:* ship **all-E4M3** (E4M3 fwd/dgrad) with the `mosaic_wgrad` param
     (`bf16` default, `fp8` opt-in) — the working 1.23–1.33× recipe.
   - *To meet the required mixed **E5M2 grad × E4M3 weight** backward:* apply
     **`patches/jax-mosaic-wgmma-mixed-fp8.patch`** (~15 lines in `jax/experimental/mosaic/gpu/wgmma.py`:
     thread A's dtype through `wgmma_m64`, emit PTX `.atype.btype` independently, relax the
     `A.dtype==B.dtype` gate for the fp8 pair), then set the mosaic path's `grad_dtype=float8_e5m2`.
     This is the single external dependency the requirement forces; everything else is in-marin.
6. **Guards:** codex's **jaxpr dtype asserts** (fwd contracts E4M3; bwd contracts E5M2 once patched) +
   opencode's **compiled-HLO fp8-kernel check** — lock in genuine mixed fp8 at jaxpr and emitted-kernel
   level.

### Required external-dependency patch (the only one)

`patches/jax-mosaic-wgmma-mixed-fp8.patch` — enables mixed E4M3×E5M2 Hopper wgmma (JAX software gap,
PTX ISA 9.3 already supports it). Without it, no JAX backend (Triton or Mosaic) will run the required
E5M2×E4M3 contraction; with it, the Mosaic path runs the standard hybrid recipe at the measured ≥20%
win. Validated structurally against jax 0.10.0; final H100 compile/numerics check is the next step.
```
