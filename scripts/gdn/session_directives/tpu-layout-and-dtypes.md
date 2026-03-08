TPU layout & dtype focus for this session:

Goal: remove “silent” TPU performance cliffs that come from vector-register layout, transposes, and dtype choices.

Key facts to internalize (TPU Pallas):

1) **The last two axes are the vector-register axes.**
   They map to `(sublanes=8, lanes=128)` and elementwise ops are padded to these tile sizes.
   A shape like `(..., 128, 1)` is far worse than `(..., 1, 128)`.

2) **Explicit transposes/reshapes of the last two axes are expensive.**
   Prefer `lax.dot_general` dimension numbers so the transpose is fused into the matmul.

3) **Use BF16 inputs + FP32 accumulation on TPU unless proven otherwise.**
   - Keep `q/k/v` in BF16 in VMEM.
   - Use `preferred_element_type=jnp.float32` for dot outputs.
   - Avoid `astype(f32)` on every block; it can add conversions without improving precision.

Concrete tasks (do at least one):

A) **Audit and fix pathological singleton placement.**
   - Find any `(..., Ct, 1)` / `(..., 128, 1)` arrays in the Pallas kernels.
   - Change to `(..., 1, Ct)` where broadcastability is needed.
   - Better: keep `g/b/dg/db` as rank-4 tensors `(..., Ct)` so there is no trailing singleton at all.
   - Ensure the in_specs/out_specs match the new layout (and keep the kernel indexing simple).

B) **Replace `jnp.matmul(..., x.T)` with a single `dot_general` helper.**
   - Implement a helper like `mxu_dot(a, b, transpose_b: bool, precision: lax.Precision, out_dtype=f32)`.
   - Use it for every matmul in fwd+bwd so transposes are fused and dtype handling is consistent.

C) **Make dtype/precision explicit and consistent.**
   - Decide on one kernel-wide policy: BF16 inputs, FP32 accumulation.
   - Only promote to FP32 for numerically sensitive elementwise ops (e.g. exp/log).
   - Keep the policy identical in fwd and bwd (avoid hidden dtype mismatches that force extra casts).

Measurement requirement:
- After the change, run TPU correctness tests and one profile run.
- In the trace, verify that the same hotspots are faster (not just moved).
