---
topic: shuttle-module-transform-plugin
description: Target-1 workload-free JAX module-transform plugin and shared acceptance harness
---

# Shuttle module-transform plugin: task logbook

## Scope

- Goal: inventory the current HLO rewrite/runtime and acceptance surfaces, then
  specify the smallest workload-free target-1 plugin and four-way harness.
- Primary metrics: lossless import, exact legality/coverage, no workload keys,
  one registration API, complete generated-handler provenance, and one shared
  JAX/oracle/source-ordered/fast-math record.
- Constraints: exact base `f3ce4aac3c`; read-only code audit; no GPU; do not
  scaffold `lib/shuttle` before the migration base lands.

## Entry log

### 2026-08-09 - Target-1 module-transform audit

- Hypothesis: current generic HLO recovery, local rewrite audits, and benchmark
  evidence can support one plugin without promoting a workload-specific
  composition harness.
- Commit Hash: `f3ce4aac3c` for the audited source; this entry's containing
  revision records the documentation checkpoint.
- Command: source inspection with `rg`, `sed`, and `nl` over
  `lib/tile_lifetime/src/tile_lifetime/xla_*.py`,
  `jax_hlo_rewrite_runtime.py`, the `xla_*` harnesses,
  `benchmark_boundary.py`, `command_buffer_capture.py`, and current design
  ledgers.
- Result: hypothesis supported. Generic mechanisms exist, but registration,
  coverage composition, handler manifests, and acceptance records are
  fragmented. The combined Grug harness is an evidence oracle, not a suitable
  public compiler API.
- Interpretation: implement typed module/region identity, a coverage ledger,
  provider-based candidate selection, a handler repository, and the shared
  four-way record around one Contract/Map forward/reverse slice.
- Next action: wait for the `lib/shuttle` migration base, then implement the CPU
  slice and stop for architecture review before GPU integration.
- Report:
  `.agents/projects/tile_lifetime_compiler/module_transform_plugin_acceptance_harness_audit_20260809.md`

### 2026-08-10 - Exact-pin native MLIR preflight

- Hypothesis: the reviewed native Shuttle dialect scaffold and XLA hook patch
  build against the XLA revision pinned by JAX 0.10.1, then pass the MLIR lit
  suite and four XLA hook tests on a CPU worker.
- Commit Hash: `0481d4ef2b9f7139b53784ea3d03790d72d1699c` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810/`, requesting 24 CPU,
  96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`; Debian 13; GCC 14.2.0;
  embedded OpenJDK 21.0.5.
- Result: hypothesis rejected at the first target. Both Shuttle operation
  TableGen actions rejected `ReturnLike` in `ShuttleOps.td:69`. Bazel loaded 86
  packages and configured 24,359 targets. No Shuttle C++ target completed, and
  the MLIR lit suite and all four XLA tests did not run.
- Interpretation: source-only parsing did not exercise the pinned native
  TableGen environment. The dialect trait definition must be corrected and
  statically re-reviewed before another native preflight.
- Next action: inspect the exact pinned MLIR operation traits, replace or import
  the intended terminator semantics, then run the same target matrix. This
  checkpoint does not authorize a relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810/README.md`

### 2026-08-10 - Native compile after ReturnLike correction

- Hypothesis: removing the unsupported `ReturnLike` trait allows the pinned
  native target matrix to build and test against XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`.
- Commit Hash: `bba4867bb9943c6c0553e0b910fdea8d6b83e14e` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-fixed/`, requesting 24
  CPU, 96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; Debian 13; GCC 14.2.0; embedded OpenJDK 21.0.5.
- Result: the narrow `@shuttle_mlir//:shuttle_ops_inc_gen` target passed. The
  following `@shuttle_mlir//:shuttle-opt` build failed compiling
  `ShuttleDialect.cc`. The first diagnostic reported that generated
  `ShuttleOps.h.inc` referenced an undeclared `mlir::BytecodeOpInterface`.
  The MLIR lit suite and all four XLA tests did not run.
- Interpretation: the ReturnLike TableGen blocker is closed. The generated
  operation declarations now expose a pinned-MLIR C++ include or interface
  compatibility blocker.
- Next action: inspect the exact pinned bytecode-interface include contract and
  generated operation header dependencies before any new native run. This
  checkpoint does not authorize a relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-fixed/README.md`
