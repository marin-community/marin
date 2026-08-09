# Shuttle Grug training lowering

## Decision

Make Shuttle an out-of-tree HLO transformation plus runtime first. Do not build
a new PJRT GPU backend or production MLIR dialect yet.

JAX 0.11 exposes the experimental
`jax.extend.xla.register_hlo_module_transformation` API. A pre-scheduler
callback receives serialized `HloModuleProto` after SPMD partitioning and may
return a rewritten module before XLA layout assignment, fusion, scheduling,
and buffer assignment. That is sufficient for the first real path:

```text
ordinary Grug train step
→ JAX autodiff and SPMD lowering
→ Shuttle HLO analysis/rewrite
→ generic Shuttle region call
→ normal XLA GPU compilation and runtime
```

The runtime ABI should be model-agnostic, for example
`shuttle.execute_region_v1(plan_fingerprint, buffers, numerical_contract)`.
Named calls such as `flash_attention`, `mok_forward`, or `gdn_chunk` are not
acceptable generated boundaries.

Primary references:

- [JAX extension API](https://docs.jax.dev/en/latest/jax.extend.xla.html)
- [XLA custom calls and FFI](https://openxla.org/xla/custom_call)
- [XLA GPU architecture](https://openxla.org/xla/gpu_architecture)

## Current evidence

The checked-in probe lowers a natural one-layer Grug MoE train step through
`value_and_grad` and an optimizer update. With reference attention, scatter
MoE, and XLA ragged contractions, JAX 0.10.1 emits:

| Optimizer | StableHLO size | `dot_general` | `reduce` | `sort` | `scatter` | semantic custom calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SGD | 329,403 chars | 82 | 96 | 2 | 16 | 0 |
| AdamW | 466,782 chars | 82 | 96 | 2 | 16 | 0 |

No Grug source change is required to expose the math. The performance frontend
can continue to use expert implementations; the clean compiler fixture should
explicitly select reference attention, XLA ragged contraction, and a generic
relation path.

## Required implementation sequence

1. Create a pinned JAX/JAXLIB 0.11 probe environment without changing Marin's
   main lock. Register a no-op pre-scheduler transform and verify it observes a
   trivial H100 module. Disable the persistent compilation cache or include the
   Shuttle revision and plan fingerprint in an isolated cache namespace.

2. Compile the frozen one-layer Grug train step on H100. Save both frontend
   StableHLO and the exact post-SPMD `HloModuleProto` seen by Shuttle. Add an
   opcode/custom-call census and reject FA4, DeepEP, Sonic, or other semantic
   kernels in the clean fixture.

3. Recover one dense forward/backward contraction region from post-SPMD HLO
   but return the module unchanged. Compare recovered axes, layouts, saved
   values, and donation/aliasing against the frontend fixture. Stop and seek a
   pre-SPMD hook if SPMD erases the necessary structure.

4. Add a narrow C++ transform bridge capable of inserting a generic XLA FFI
   custom call. Python HLO bindings are sufficient for inspection but do not
   currently construct arbitrary custom calls. The handler should launch a
   Shuttle-generated artifact on XLA's stream and leave unrelated HLO alone.

5. Replace one contraction family in this order:

   - forward GEMM;
   - input gradient (`dY @ Wᵀ`);
   - weight gradient (`Xᵀ @ dY`);
   - surrounding generated Map/Fold bodies.

   Compare the generated contractions against the same expert mainloop family
   used by CODA/QuACK. The proof is complete when changing the Map/Fold body or
   transpose roles regenerates code without editing a workload kernel.

   The first concrete proof should be a linear SwiGLU projection:

   ```text
   gate_up = x @ W13
   z = SiLU(gate) * up

   dgate = dz * up * SiLU'(gate)
   dup   = dz * SiLU(gate)
   dx    = concat(dgate, dup) @ W13^T
   dW13  = x^T @ concat(dgate, dup)
   ```

   Benchmark save-preactivation and recompute-preactivation policies
   separately at `M=2048,K=4096,I=14336`. Mutate the scalar Map to
   `tanh(gate) * up`; the same reverse-mode and physical generators must emit
   the changed forward and backward bodies.

   The second proof should be RMSNorm followed by GEMM. Its backward adds the
   first nontrivial Fold adjoint:

   ```text
   dn     = dy @ W^T
   dW     = normalized^T @ dy
   dgamma = Fold_rows(dn * x * r)
   dot    = Fold_hidden(dn * gamma * x)
   dx     = r * gamma * dn - x * (r^3 / H) * dot
   ```

   Compare save-normalized, save-`x`/`r`, and recompute-statistic policies.
   Keep source-ordered consumer preparation and delayed
   real-algebra-equivalent scaling as distinct candidates.

6. Add streaming normalized-exponential backward. Treat saved LSE versus
   recomputation as a planner/materialization choice. Official FA3 backward is
   the oracle only; it is not an accepted generated path.

7. Add segmented MoE gradients: W13/W2 input and weight gradients, activation
   backward, deterministic inverse-relation merge, and router-weight gradient.
   Transport may remain a generic DeepEP or XLA collective primitive, but
   semantic accumulation must be Shuttle-generated.

8. Replace one complete Grug block's forward and backward while leaving loss,
   optimizer, and unrelated blocks in XLA. Then scale to two blocks and finally
   the complete tiny train step.

## Gradient acceptance boundaries

Every component comparison must use identical inputs, upstream gradients,
saved values, dtypes, shapes, and hardware. Record forward, dgrad, and wgrad
separately before timing a whole backward region.

- Dense Contract: QuACK/CUTLASS/CODA-derived generic mainloops as the physical
  oracle; generated preparation/finalization ASTs remain Shuttle-owned.
- Attention: official FA3 backward as the H100 oracle; compare saved-LSE and
  recompute policies separately.
- MoE: generic grouped/ragged contraction VJPs plus MoK/DeepEP only as complete
  performance or transport references. No complete MoK backward call.
- RMS/LayerNorm/SwiGLU/RoPE: independent JAX reference plus generated Map/Fold
  mutations. Preserve source-order and reassociation policies explicitly.

Use the existing clean-synthesis completion criterion of generated latency no
more than 1.20 times the matched expert comparison for at least one primary
shape. Keep full raw distributions and deterministic hashes.

The first generic reverse-mode slice is now implemented in
`tile_lifetime.autodiff`. It differentiates scalar Map ASTs, transposes generic
multilinear Contracts, broadcasts sum-Fold cotangents, and reduces broadcast
Map adjoints. A SwiGLU-to-tanh mutation changes the derived VJP without backend
source edits. An independently derived RMSNorm-GEMM adjoint matches the generic
program executor for `dx`, `dgamma`, and `dW`; the complete tile-lifetime suite
passes 220 tests. Physical generation and matched H100 timing remain open.

## MLIR dialect decision

An internal `shuttle.flow`/`shuttle.schedule` dialect may become useful once
post-SPMD recovery and one generated backward region work. It should represent
Shuttle's own generic algebra and bounded schedules, not replace StableHLO or
become a prerequisite for the first integration. The experiment should earn
the dialect by exposing a representation problem that the current Python IR
cannot handle.
