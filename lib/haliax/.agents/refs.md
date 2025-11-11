# NamedRef & SliceRef Plan

## Goals
- Provide a `NamedRef` abstraction mirroring `jax.Ref` while preserving Haliax axis metadata.
- Offer helper APIs (`hax.new_ref`, `hax.freeze`, `hax.ref.get/swap`) that operate on named refs and enforce shape/axis compatibility.
- Enable "slice refs" that partially apply named indexers so subsequent operations only supply the remaining indices, including composing multiple slice refs.

## Architectural Outline
1. **Core data model**
   - Introduce `NamedRef` in `src/haliax/core.py` (or a focused `src/haliax/ref.py` module) as a thin wrapper around `jax.Ref` plus axis metadata (likely reuse `NamedArray` named shape helpers).
   - Register `NamedRef` as a PyTree node: treat the underlying `jax.Ref` handle as a leaf to avoid unwanted tracing semantics.
   - Expose properties: `.axes`, `.named_shape`, `.shape`, `.dtype`, and `.value` (reads through `ref[...]`).

2. **Creation and conversion helpers**
   - Implement `haliax.new_ref(value, axes|named_shape)` returning `NamedRef`; allow existing `NamedArray` inputs to propagate their axis info.
   - Add `haliax.freeze(ref)` returning a `NamedArray` while invalidating the underlying `jax.Ref`, preserving metadata.
   - Provide `haliax.ref.get(ref, index)` / `haliax.ref.swap(ref, index, value)` that accept named indexers and return/require `NamedArray` values.

3. **Indexing semantics**
   - Reuse the existing NamedArray indexing utilities (`axis_spec_to_tuple`, `_normalize_indexers`, `_NamedIndexUpdateRef`) so `NamedRef.__getitem__` returns `NamedArray` views and `__setitem__` / `.set` accept either `NamedArray` or raw arrays (with shape checks).
   - Enforce error messaging consistent with NamedArray when axes mismatch or out-of-bounds indices occur.

4. **Slice refs**
   - Add a `NamedSliceRef` (or similar) that wraps a base `NamedRef` plus a frozen partial index mapping.
   - Support construction via `NamedRef.slice(selector)` where `selector` is any named index expression (dict, `Axis`, tuple, etc.).
   - On read/write, merge the stored selector with the new selector: the stored selector is applied first, and new selectors can further index the result. Implement logic to resolve nested slices (e.g. range then integer index) by normalizing everything to positional indices via helper utilities.
   - Ensure composition is associative: calling `.slice()` on a `NamedSliceRef` should produce another `NamedSliceRef` that collapses selectors appropriately.
   - Preserve dynamic slice selectors (Haliax `dslice` or `pallas.dslice`) so their static size information is retained during trace-time shape inference.
   - Provide an `unsliced()` helper to recover the original reference when staged selectors are no longer needed.
   - Handle ellipsis and implicit axes; ensure merging respects axis ordering and catches conflicting assignments.

5. **Integration & ergonomics**
   - Audit parts of the codebase that manipulate mutable state (e.g. caching layers, training loops) to expose the new API where appropriate. Initially, surface `NamedRef` under the `haliax.ref` namespace without retrofitting all call sites.
   - Provide a `NamedRef.unsafe_buffer_pointer()` passthrough for parity with `jax.Ref` diagnostics.

6. **Testing strategy**
   - Add `tests/test_named_ref.py` covering: creation from arrays, indexing/assignment (scalar, slices, advanced), `jit` usage, vmap plumbing, and ensuring restrictions (e.g. returning refs from jit) raise clear errors.
   - Include dedicated tests for slice refs verifying merged indexing (simple axis, slices-of-slices, dict updates, ellipsis) and interactions under `jit`.
   - Validate that gradients through pure functions using NamedRef internally behave like the array equivalents.

7. **Documentation & guidelines**
   - Document the new API in `docs/api.md` (table entry and short section) and, if needed, add a primer example in `docs/primer.md` showing plumbing state with NamedRef.
   - Update contributor notes that future ref-based utilities must keep numerical tolerances intact (no relaxed tolerances), aligning with the existing testing guidelines.
   - Call out any additional guidelines discovered during implementation (e.g., keeping dslice selectors intact for static shapes).
   - Consider a `.playbooks/` entry if ref plumbing becomes a common flow after initial implementation.

## Open Questions / Follow-ups
- Confirm whether `NamedRef` needs to support asynchronous dispatch or pytree flattening beyond leaf behavior.
- Decide if we want implicit conversion between `NamedRef` and `NamedArray` in certain helpers or require explicit `.value` reads for clarity.
- Explore caching strategies or wrappers for `foreach`-style utilities once the base `NamedRef` and slice refs land.
