# FP8 `ragged_dot` — handoff to the `research/grug-fp8-h100` agent

**From:** the `fp8-redux` branch (a from-first-principles re-investigation, deliberately blind to
your branch until the end).
**Purpose:** hand you everything the independent investigation found so you can refine the existing
Mosaic approach. This is written to be read standalone — no conversation context needed.
**Bottom line for you:** the independent work **converges on your conclusion** (Mosaic-GPU f8 wgmma
is the only backend that beats bf16) and **independently re-derives your decisive blocker** (mixed
E4M3×E5M2 wgmma is a JAX software gate, not Hopper). It also produced a **concrete ~15-line jax
patch** that should unblock the genuine E5M2 backward you currently have to forgo. Details below.

---

## 1. What this investigation was

Three agents (a coordinator + `codex`/gpt-5.5 + `opencode`/glm-5.2) each built an FP8 mixed-precision
`ragged_dot` (E4M3 fwd / E5M2 bwd) from scratch, **without looking at your branch**, then benchmarked
all candidates head-to-head on H100 (`cw-us-east-02a`, 1×H100 80GB, jax 0.10.0) and finally compared
against your shipped solution using the *same* harness. The point was to see whether an unbiased pass
reproduces your approach or finds something simpler. It reproduced your approach.

## 2. The convergent result (matches your branch)

At grug MoE scale (M=262144, K=N=3072, E=128), single gate/up GEMM, same bf16-triton baseline:

| candidate | backend | fwd vs bf16 | fwd+bwd vs bf16 | genuine fp8? |
|---|---|---|---|---|
| coordinator / codex | Pallas-**Triton** `pl.dot` fp8 | **0.61×** (slower) | **ERROR** | No — `pl.dot` upcasts; **rejects mixed E5M2×E4M3** |
| opencode | **XLA** `ragged_dot_general` fp8 | **fails** | fails | No — lowers to **bf16 `__triton_gemm`** (QDQ dequant) + autotuner OOM (96 GiB) |
| **your branch** | **Mosaic-GPU** f8 wgmma | **1.05×** | **1.23×** ✅ | **Yes** (all-E4M3); E5M2 grad ⇒ ERROR |

Raw logs: `bench_results/H100_RESULTS_firstprinciples.txt` (the three candidates + a backend probe)
and `bench_results/H100_RESULTS_research_mosaic.txt` (your branch, same harness).

Key independent confirmations of things your logbook already records (GFP8-001→034):
- **Triton `pl.dot` is slower than bf16 for fp8 here (0.61×)** and *cannot* take a mixed fp8 pair —
  it raises `TypePromotionError: ... 8-bit floats do not support implicit promotion`. So the
  "just reuse the bf16 Pallas kernel" path is a genuine dead end, not a tuning problem.
- **XLA `ragged_dot_general` with fp8 operands dequantizes to a bf16 `__triton_gemm`** — exactly the
  QDQ fallback PR #6660 fixed for the *dense* case, except XLA's *ragged* GPU lowering doesn't have
  the fp8 path, and its autotuner additionally **OOMs trying to alloc 96 GiB**. Also a dead end.
- **Mosaic f8 wgmma is the only thing that clears ≥20%.** Your shipped recipe (E4M3 fwd+dgrad,
  bf16 wgrad) measured **1.23× fwd+bwd** here (856.7 vs 694.7 TFLOP/s), consistent with your
  logbook's ~1.27–1.33× e2e (the gap = your per-shape tuned config + f8 wgrad).

## 3. The one thing that may help you refine: the mixed-FP8 wgmma patch

Your branch ships **all-E4M3** because requesting `grad_dtype=float8_e5m2` on the mosaic path raises:

```
lhs and rhs must have the same dtype, got float8_e5m2 and float8_e4m3fn
```

The independent investigation traced this to its root and wrote a fix:

- **Root cause:** `jax/experimental/mosaic/gpu/wgmma.py` (a) rejects `A.dtype != B.dtype`, and
  (b) hardcodes the emitted PTX operand type to the B-operand's dtype: `.{el_ty}.{el_ty}`.
- **Hopper actually supports mixed fp8:** PTX ISA 9.3 defines
  `wgmma.mma_async.sync.aligned.m64nNkK.dtype.atype.btype` with **independent** `atype`/`btype`.
  So E5M2(A)×E4M3(B) is a real hardware instruction; JAX just never emits it.
- **The probe confirmed it from the other side:** the Triton path's
  `"8-bit floats do not support implicit promotion"` is the *same* JAX-level gate showing up in a
  different lowering. So this single gate is what blocks the required E5M2 backward across *every*
  in-JAX backend, Mosaic included.

**The patch:** `patches/jax-mosaic-wgmma-mixed-fp8.patch` (~15 lines). It:
1. adds a helper `_wgmma_ptx_el_ty()` mapping each operand dtype → its PTX string (`e5m2`/`e4m3`/`s8`);
2. adds an `a_element_type` param to `wgmma_m64` (defaults to B's dtype → no behavior change for
   existing same-dtype callers);
3. emits `.{out_ty}.{a_el_ty}.{b_el_ty}` instead of `.{el_ty}.{el_ty}`;
4. relaxes the `element_type != element_type2` gate to allow the **fp8 pair** specifically;
5. threads `a_element_type=element_type2` at the `wgmma()` → `wgmma_m64()` call site.

Because both fp8 types are 1 byte, all stride/swizzle/k-group math (keyed on `bytewidth`) is
unchanged — only the emitted PTX `atype` differs. **Status: structured against jax 0.10.0, NOT yet
H100-compile/numerics-validated** (line numbers are approximate; match by context). Validating it is
the obvious next experiment for your branch.

**After the patch lands:** set the mosaic path's `grad_dtype=float8_e5m2` (instead of forcing
all-E4M3 / `MosaicWgradMode.FP8` requiring `grad_dtype=float8_e4m3fn`) to get the standard TE
E5M2-grad × E4M3-weight backward — the recipe the work trial actually requires. This is the single
external dependency; everything else stays in-marin.

> Note on the requirement: per the project memory `grug-fp8-ragged-mixed-required`, the backward
> **must** genuinely contract E5M2 grad × E4M3 weight/act — the same-dtype E5M2×E5M2 (or all-E4M3)
> shortcut is explicitly not acceptable as the final recipe. The all-E4M3 path is the
> works-today fallback; the patch is the path to the required recipe.

## 4. Best ideas from the three blind attempts worth grafting onto your branch

Your Mosaic engine is the right core. These are the integration/testing pieces the from-scratch
agents produced that your branch could absorb:

1. **`MoELinear.dot_general` dispatch + `fp8_ragged_dot` entry with 512-pad / all-reduce parity**
   (opencode) — makes the op a true drop-in for the bf16 path, not just a microbench op.
2. **Shared `_Fp8StateMixin`** between the dense `Fp8DotGeneralOp` and the new ragged op (opencode) —
   DRY for the delayed-scaling state (scale + amax history as `OverwriteWithGradient`).
3. **An `implementation` selector** over `{mosaic, triton, xla}` (codex) so non-H100 / non-128-tileable
   shapes fall back to the correct-but-slower path while `mosaic` is the fast H100 path. (You already
   have the `"mosaic"` backend; this just formalizes the fallback.)
4. **Correctness guards that lock in "genuine mixed fp8" at two levels** — codex's **jaxpr dtype
   asserts** (fwd contracts E4M3; bwd contracts E5M2 once patched) + opencode's **compiled-HLO check**
   for `f8e4m3`/`f8e5m2` kernels. Together these would catch a silent regression to bf16/QDQ.
5. **The candidate-independent backend probe** (`probe_fp8_backends.py`) — times triton-fp8 vs xla-fp8
   for fwd + mixed bwd and dumps HLO; a fast regression check that the fast path is still genuinely fp8.

## 5. Artifacts (all on the `fp8-redux` branch, commit `93ec810c5d`)

| path | what |
|---|---|
| `SYNTHESIS.md` | full 3-way comparison + pros/cons + your-branch comparison + recommendation |
| `patches/jax-mosaic-wgmma-mixed-fp8.patch` | the mixed-fp8 wgmma jax patch (the key deliverable for you) |
| `bench_results/H100_RESULTS_firstprinciples.txt` | raw H100 log: probe + 3 candidates |
| `bench_results/H100_RESULTS_research_mosaic.txt` | raw H100 log: your branch via the same harness |
| `bench_fp8_ragged_dot.py` | canonical harness (`--gemm gateup/down`, `--tokens`, `--bwd`, `--candidate module:attr`) |
| `probe_fp8_backends.py` | candidate-independent backend probe (triton vs xla, fwd + mixed bwd, `--dump-hlo`) |
| `lib/haliax/src/haliax/nn/fp8_ragged_dot.py` | coordinator prototype (non-promoting Triton kernels; instructive, not the winner) |
| `lib/haliax/src/haliax/_src/fp8.py` | reused #6660 dense primitives (`quantize`/`dequantize`/`update_fp8_meta`/`quantized_dot`) |
| `NOTES_coordinator.md` | coordinator approach notes |

The benchmark script used for your branch lives at
`/home/matt/.claude/jobs/5de6ee1c/tmp/agentwork/bench_research_mosaic.py` (calls your
`fp8_scaled_ragged_dot(..., implementation="mosaic", mosaic_wgrad=...)` with the shared baseline/shapes).

## 6. Suggested next steps for your branch (in priority order)

1. **Validate `patches/jax-mosaic-wgmma-mixed-fp8.patch` on H100** — apply it, set the mosaic path's
   `grad_dtype=float8_e5m2`, rerun the fwd+bwd harness, and confirm (a) it compiles, (b) the backward
   contracts genuine E5M2×E4M3, (c) it still clears ≥20%. This is the one thing standing between your
   shipped all-E4M3 path and the required TE recipe.
2. **Add the jaxpr + compiled-HLO dtype guards** so the required-mixed-fp8 invariant can't silently
   regress to bf16/QDQ or to all-E4M3.
3. **If you want the f8 wgrad arm to use E5M2 too** (currently `MosaicWgradMode.FP8` forces all-E4M3),
   the same patch is the enabler — re-measure the wgrad TFLOP/s after.
4. Consider upstreaming the wgmma patch to jax — it's a small, general capability fix (mixed fp8 wgmma
   is in the ISA), and it would remove marin's only external dependency for this feature.
