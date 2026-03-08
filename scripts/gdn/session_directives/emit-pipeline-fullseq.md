Full-sequence GDN kernel via `pltpu.emit_pipeline`

Why this exists:
- TPU Pallas TC lowering does **not** support `lax.dynamic_slice` in-kernel, which blocks a single-kernel
  “for each chunk, load block from HBM” implementation.
- A Python loop (`for c in range(n_chunks)`) typically gets **unrolled** inside a Pallas kernel,
  which explodes VMEM usage for long sequences.

`pltpu.emit_pipeline` is the escape hatch: it lets you express a *sequential* stage axis inside the kernel,
with explicit buffering/prefetch, and it supports dynamic block indexing via `pl.ds(...)` / `pl.BoundedSlice`.

High-upside macro move:

Implement a **single forward kernel** that:
1) runs one program per (batch, head) (or per local `NH`),
2) keeps the recurrent state `S` in VMEM across the full sequence,
3) iterates over chunks as a **pipeline stage axis**, loading `q/k/v/g/beta` blocks from HBM for each stage,
4) writes `out` blocks (and optionally `chunk_starts`) to HBM each stage.

This can eliminate:
- the segment hierarchy,
- the host-level `scan` over segments,
- unnecessary HBM roundtrips for the recurrent carry.

Implementation sketch (forward)

1) Outer `pallas_call` grid:
   - grid: `(NH,)` (parallel)
   - compiler params: `dimension_semantics=("parallel",)`
   - allocate scratch: `S_ref` for the recurrent state (VMEM), plus any small scratch.

2) Inside the kernel:
   - build a pipeline with stage grid `(n_chunks,)` and `dimension_semantics=("parallel", "sequential")`.
   - stage `i` loads the i-th chunk block using index maps that use:
     - `pid_nh = pl.program_id(0)`
     - `pid_chunk = pl.program_id(1)` (inside the pipeline)
     - `pl.ds(pid_chunk * chunk_size, chunk_size)` for the sequence axis slice.

3) State handling:
   - stage 0: initialize `S_ref[...] = S0`
   - every stage: read `S_prev = S_ref[...]`, compute chunk outputs, write `S_ref[...] = S_next`
   - optionally store `S_prev` to a `chunk_starts` output (needed for bwd) at each stage.

4) Output handling:
   - write the chunk’s `out` slice to `out_ref` at the correct HBM offset.

Notes:
   - Keep per-stage VMEM footprint small: only the current chunk’s `q/k/v/g/beta` blocks + a few temporaries.
   - If sequence length is not divisible by chunk_size, use `pl.BoundedSlice` to handle the last partial chunk.

Suggested incremental plan

Step A (scaffold):
- Implement a forward-only pipeline kernel that does a **dummy** recurrence (e.g., `S_next = S_prev`)
  but exercises dynamic chunk loads and HBM writes. Verify it compiles and runs.

Step B (real recurrence, no solve):
- Implement the real per-chunk math but *keep* the current chunkwise “flash” path inside the stage.
  The key change is: stage loads come from HBM per chunk, rather than preloading a whole segment.

Step C (full integration):
- Replace the old segment scan with the single pipeline kernel.
- Keep correctness tests.

Step D (backward):
- Mirror the same idea for bwd: pipeline backwards over chunks (sequential), keep dS in VMEM.
  Use `chunk_starts` as needed, or recompute within the pipeline.

Hard requirements:
- No Python unrolled loops over `n_chunks` inside the Pallas TC kernel.
- No `lax.dynamic_slice` inside the kernel; use `pl.ds` / `pl.BoundedSlice` via pipeline indexing instead.
- Keep VMEM under the configured limit.

What to measure:
- The number of Pallas custom calls per step should drop.
- Each remaining custom call should have higher arithmetic intensity and higher MXU utilization.
