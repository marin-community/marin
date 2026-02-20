Triangular inversion focus for this session:

- Target the strict lower-triangular inversion bottleneck in GDN kernels with high-upside redesigns.
- Bold changes are explicitly allowed if they preserve end-to-end model behavior and improve throughput/MFU.
- Equivalent mathematical reformulations are in scope, including approaches that avoid explicit chunk-size-by-chunk-size matrix inversion.

Optimization direction:

- Prefer numerically stable and parallel TPU-friendly formulations over sequential forward-substitution style methods.
- Consider reformulating the recurrence or chunk update so it is solved via associative/blockwise composition or another equivalent decomposition.
- It is acceptable to replace explicit inversion with an equivalent solve/transform pipeline if correctness is preserved.
- Preserve deployability on target training configs; avoid designs that only work at tiny settings or that repeatedly fail with scoped-vmem OOM.

Concrete high-upside angles to explore (choose 1):

1) **Hierarchical block inversion (FLA-style) instead of global series.**
   - Invert base blocks (e.g. 16×16 or 32×32) with an exact, stable method.
   - Merge blocks via block-triangular identities using MXU matmuls.
   - Optionally compute the full inverse once and reuse it across multiple RHS solves and transpose solves in bwd.

2) **Preconditioned nilpotent-series solve (stability-first).**
   Strict lower-triangular matrices are nilpotent, so the Neumann series terminates exactly,
   but intermediate powers can explode.

   Stabilization tactics to consider:
   - diagonal scaling / balancing: choose per-row or per-column scales `D` and solve in the scaled basis
     (e.g. `A' = D^{-1} A D`), then unscale.
   - power-of-two rescaling: compute `(I - A)^{-1}` via repeated squaring with explicit rescaling
     to keep intermediate magnitudes bounded.

3) **Avoid explicit inverse: solve only what you need.**
   - Prefer solving against stacked RHS (V and K RHS together) rather than forming the inverse.
   - Use a block-recursive solve that keeps the critical path MXU-heavy.

4) **Turn long triangular work into a pipeline, not an unrolled loop.**
   - If any approach requires iterating over many blocks, implement the iteration using `pltpu.emit_pipeline`
     (sequential stage axis) rather than Python loops that unroll and blow VMEM.

Validation bar:

- Keep correctness tests green (`test_gdn_kernels.py`, `test_gdn_layer.py`).
- Verify end-to-end behavior remains aligned with reference implementation semantics.
- Require a completed profile run with measured throughput/MFU change before claiming success.

Numerics guardrails:

- Be careful with numerically unstable inversion shortcuts (for example naive high-power series without stabilization).
- If using approximation or preconditioning, explicitly validate numerical error and training stability impact.
