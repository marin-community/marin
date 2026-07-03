# Performance Workflow

Read this file before benchmarking, profiling, roofline analysis, or
autotuning. Use the execution-environment guidance and cadence from
`run-research` for long-running work. For profiling capture and comparison
details, read `reference/profiling.md`.

## Iteration Loop

Use this loop:

```text
profile -> hypothesis -> change -> tests -> microbench -> profile
```

Run one-axis sweeps first, then interaction sweeps. Keep comparisons
apples-to-apples: same shape, dtype, pass mode, backend, device count, and
environment unless that axis is under test. Only move the baseline after enough
repeated evidence, and note the baseline change explicitly.

Always report:

- Compile-including timing, such as time-to-first-step.
- Steady-state timing.
- Block sizes or tile sizes.
- Exact hardware type, device count, shape grid, and dtype grid.
- Relevant env vars and compiler flags.

## Roofline and Harnesses

For each relevant hardware type, estimate whether the kernel is compute-bound,
memory-bound, or limited by a known compiler/runtime bottleneck. Tie the
performance harness output to that estimate.

Performance harnesses should:

- Separate compile time from steady-state time.
- Warm up before measuring steady-state loops.
- Include representative small, medium, and target production shapes.
- Emit machine-readable rows with unique keys for
  `(implementation, shape, dtype, backend, device_count, block_sizes)`.
- Capture failures with enough context to distinguish unsupported shapes from
  correctness failures.

Required CSV/JSON fields:

- `kernel`
- `implementation`
- `shape`
- `dtype`
- `backend`
- `device_type`
- `device_count`
- `block_sizes`
- `compile_time`
- `steady_state_time`
- `error`
- `git_sha`
- `xla_flags`
- `backend_env`

## Autotuning

Keep tuning explicit and reviewable:

1. Define a bounded config space of block/tile candidates.
2. Define target shape/hardware buckets.
3. Benchmark every `(bucket, config)` pair and capture timing + failures.
4. Store raw results as artifacts, preferably CSV/JSON and a W&B artifact.
5. Derive a best-config table keyed by
   `(device_type, dtype, shape_bucket[, invariants])`.
6. Check in a Python tuned-table module with bucket definitions, best configs,
   an `infer_block_sizes(...)` helper, and default fallback to
   `BlockSizes.get_default()`.

Do not key tuned tables by every exact shape. Keep buckets stable and
reviewable.

## When to Stop Tuning

Stop tuning when one of these is true:

- Performance is within the agreed roofline target or another explicit target.
- The remaining roofline gap is explained by a documented compiler/runtime
  limitation.
- A bounded sweep over the planned config space shows no material improvement.
- Structural decomposition experiments point to the same bottleneck across
  repeated profiles.
- The user chooses to prioritize correctness/API landing over more tuning.

When stopping before reaching the target, document the best observed config,
the best observed timing, the limiting evidence, and the next experiment that
would be worth running.

## Fallback Autotuning

Support three fallback levels, similar to the fused softmax cross-entropy
kernel:

1. Static lookup fallback: infer block sizes from a checked-in tuned table by
   `(device, dtype, shape bucket)`, validate/sanitize for backend constraints,
   and fall back to default/safe entries when no exact tuned match exists.
2. Autotune-on-miss fallback: when tuned lookup misses and autotune is enabled,
   sweep a bounded candidate list, benchmark on the real implementation, select
   the best viable config, cache and persist the winner under a kernel-specific
   key that includes implementation + shape/device/dtype context.
3. Runtime failure fallback: if a candidate or implementation is unsupported by
   compile/runtime constraints, warn and try the next candidate/implementation
   in order when a sequence is available.

## Measurement Hygiene

Before claiming a regression:

- Check that no stale benchmark process is occupying the accelerator.
- Confirm lockfiles/state are clean.
- Confirm the comparison uses the same device count.
- Confirm command/config/env are identical except the tested axis.
- Validate machine-readable extraction: expected row counts, key uniqueness,
  and de-duplication.

Separate measurement code from the production path whenever possible. Prefer
persistent remote shells/scripts for long sweeps. For long remote runs, track a
monotonic progress signal and tail recent logs for context.

## Dump-Driven Diagnosis

When performance is unclear, run dump-first comparisons on one fixed shape:
XLA/reference path, full Pallas path, and decomposition variants through
temporary toggles. Use separate dump dirs per variant such as `hlo_*`, `llo_*`,
and `mosaic_*`.

Compare:

- Throughput.
- Fusion/custom-call placement.
- Schedule bundle counts.
- Pressure signals such as heavy `vrot`/`vsel`, spills, and vreg pressure.

Prefer structural fixes before broad tile sweeps when decomposition variants
indicate stage-structure issues. For the full LLO workflow, including flags,
artifact layout, comparison checklist, and replication loop, read
`reference/llo.md`.

## Performance Bad Patterns

- Moving the baseline after a single noisy win.
- Comparing across different device counts, dtype grids, or env flags.
- Publishing aggregate timings without row-level artifacts.
- Reporting only steady-state time when compile time materially affects usage.
- Treating unsupported candidate failures as successful slow timings.
