# Two-output row Fold H100 bootstrap failure

This artifact preserves the only authorized H100 attempt to compare one versus
two output columns per logical group in the generic row-axis Fold schedule. The
attempt failed during Iris dependency setup before the benchmark process
started. It contains no correctness result, handler count, timed sample, or
performance conclusion.

The candidate therefore remains unmeasured. The earlier accepted measurement
of `0.089681 ms` for the one-output coalesced pipeline remains the generated
baseline, and neither output schedule can be claimed to approach the `1.20x`
acceptance ratio from this attempt.

## Static audit

The measured source checkpoint is `e01a4638093556769d575418c0414a44ff8d0953`.
Both candidates are generated from the same ordinary JAX VJP and have identical
three-stage semantic fingerprints. Their generated CUDA source hashes differ:

| Outputs per group | Generated source SHA-256 |
| ---: | --- |
| 1 | `cda342a8445c934f30b8bb829b7aeb708016d587fa203e05358c1c7029cebe0f` |
| 2 | `129a8586de148ce8159cccff4e8ce7d6637f1944249a581cc6f50de13139edc3` |

Each output column retains the same eight-lane deterministic FP32 reduction
tree. The two-output schedule assigns two independent column accumulators to a
group and doubles the feature-gradient kernel's shared scratch from one to two
output lanes. It does not introduce reduction atomics; the only `std::atomic`
in the generated source is the typed-FFI invocation counter. The source is
Torch-free. Nineteen focused CPU/static tests and changed-file pre-commit passed
before allocation.

## Failed attempt

Iris job `/dlwh/shuttle-row-fold-output2-h100-20260809` requested one H100, one
CPU, 16 GB host memory, 50 GB disk, batch priority, no retries, and a one-hour
timeout. The 0.5 MB workspace bundle was admitted. Task setup then failed in
6.32 seconds because the narrow bootstrap project did not declare the `dev`
dependency group that Iris disables with `--no-group dev`:

```text
Resolved 32 packages in 628ms
error: Group `dev` is not defined in any project's `dependency-groups` table
```

No Python benchmark invocation, JAX GPU initialization, generated CUDA compile,
warmup, correctness check, or timed iteration occurred. Iris reports the job
terminal `failed`, with zero of one tasks completed and task exit code 2. The
exact task-label pod is absent. This direct job did not create a `dev_gpu`
holder session.

## Prepared correction

A corrected minimal bootstrap has been prepared but not submitted. It adds only
an explicit empty dependency group:

```toml
[dependency-groups]
dev = []
```

Linux-targeted `uv sync --all-packages --no-group dev --extra gpu --dry-run`
resolves 32 packages and would install JAX/JAXLIB/CUDA plugin/PJRT 0.10.1,
NVCC 13.3.73, and the local `marin-tile-lifetime` package. The corrected source
archive is 3.2 MB before compression and has SHA-256
`f73c03383676a05fc4e7dc5f477458de5bab4a1dce2c964fa1e4cb735772e9ba`.
The exact Iris workspace bundle is 521,060 bytes with SHA-256
`d4b0feed249d63fbda60db0fa286683374857aa9c723e662f4bb0fe404ab7f58`.

No corrected validation should be submitted without separate authorization.

