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

### 2026-08-10 - Native dialect compile after bytecode include correction

- Hypothesis: declaring MLIR's bytecode operation interface in the public
  Shuttle operations header allows the ordered native target matrix to reach
  `shuttle-opt`, lit, and the four patched XLA tests.
- Commit Hash: `412928d5aee2325d178b6a0efd1eb8383e46c7c6` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-bytecode/`, requesting
  24 CPU, 96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`; LLVM
  `9a4faee1068c09efbf837cfb7b0f5693b24635f4`; Debian 13; GCC 14.2.0;
  embedded OpenJDK 21.0.5.
- Result: `@shuttle_mlir//:shuttle_ops_inc_gen` passed. The following
  `@shuttle_mlir//:ShuttleDialect` target failed compiling
  `ShuttleDialect.cc:18` because `llvm/ADT/SmallDenseSet.h` does not exist at
  the pinned LLVM revision. `shuttle-opt`, lit, and all four XLA tests did not
  run.
- Interpretation: the bytecode include correction advanced the ordered run to
  the explicit dialect compile gate. The pinned LLVM revision declares
  `llvm::SmallDenseSet` in `llvm/ADT/DenseSet.h` rather than a dedicated
  `SmallDenseSet.h` header.
- Next action: independently review an include-only correction against the
  exact pin before any new native run. This checkpoint does not authorize a
  relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-bytecode/README.md`

### 2026-08-10 - Native dialect compile after DenseSet correction

- Hypothesis: correcting the pinned LLVM DenseSet includes allows the ordered
  native target matrix to compile both Shuttle libraries, then reach
  `shuttle-opt`, lit, and the four patched XLA tests.
- Commit Hash: `d245bc23c181beb08c4b865044ab0d8aaf1279b0` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-denseset/`, requesting
  24 CPU, 96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`; LLVM
  `9a4faee1068c09efbf837cfb7b0f5693b24635f4`; Debian 13; GCC 14.2.0;
  embedded OpenJDK 21.0.5.
- Result: `@shuttle_mlir//:shuttle_ops_inc_gen` passed. The following
  `@shuttle_mlir//:ShuttleDialect` target failed compiling
  `ShuttleDialect.cc` because generated attribute and operation definitions
  instantiate `mlir::Builder`, `mlir::OpBuilder`, and
  `mlir::ImplicitLocOpBuilder` while only forward declarations were visible.
  `@shuttle_mlir//:ShuttlePasses`, `shuttle-opt`, lit, and all four XLA tests
  did not run.
- Interpretation: the DenseSet include blocker is closed. The ordered run now
  exposes an incomplete-type failure at the generated-definition include
  boundary.
- Next action: review the exact pinned MLIR generated-code include contract
  before any new native run. This checkpoint does not authorize a relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-denseset/README.md`

### 2026-08-10 - Native dialect compile after Builders correction

- Hypothesis: supplying complete MLIR builder types allows the ordered native
  target matrix to compile both Shuttle libraries, then reach `shuttle-opt`,
  lit, and the four patched XLA tests.
- Commit Hash: `cce8dbc849dea7d288308cf34e5b1baa957acfa6` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-builders/`, requesting
  24 CPU, 96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`; LLVM
  `9a4faee1068c09efbf837cfb7b0f5693b24635f4`; Debian 13; GCC 14.2.0;
  embedded OpenJDK 21.0.5.
- Result: `@shuttle_mlir//:shuttle_ops_inc_gen` passed. The following
  `@shuttle_mlir//:ShuttleDialect` target advanced past generated Builder
  diagnostics, then failed compiling four `mlir::ArrayAttr.front()` calls in
  `ShuttleDialect.cc`. `@shuttle_mlir//:ShuttlePasses`, `shuttle-opt`, lit, and
  all four XLA tests did not run.
- Interpretation: the generated Builder incomplete-type blocker is closed. The
  ordered run now exposes a handwritten container-API mismatch with the pinned
  MLIR revision.
- Next action: review all handwritten ArrayAttr element access against the exact
  pin before any new native run. This checkpoint does not authorize a relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-builders/README.md`

### 2026-08-10 - Native driver link after ArrayAttr correction

- Hypothesis: replacing the unsupported `ArrayAttr.front()` calls lets the
  ordered native target matrix compile both Shuttle libraries and proceed
  through the driver, lit, and four patched XLA tests.
- Commit Hash: `a7c10fa6941c7a53efcfb59d866fbdc827c29ff0` for
  the tested canonical source; this entry's containing revision seals the raw
  result.
- Command: `launch-command.txt` in
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-link/`, requesting 24
  CPU, 96GB memory, 250GB disk, no accelerator, and zero retries.
- Config: Bazel `7.7.0`; XLA
  `9b635916ecc6df6efee62d8e4b0c7ef87ef84d69`; LLVM
  `9a4faee1068c09efbf837cfb7b0f5693b24635f4`; Debian 13; GCC 14.2.0;
  embedded OpenJDK 21.0.5.
- Result: `@shuttle_mlir//:shuttle_ops_inc_gen`,
  `@shuttle_mlir//:ShuttleDialect`, and
  `@shuttle_mlir//:ShuttlePasses` passed. The following
  `@shuttle_mlir//:shuttle-opt` target compiled and failed at link time on the
  undefined Shuttle dialect constructor and type ID. The MLIR lit suite and all
  four XLA tests did not run.
- Interpretation: the `ArrayAttr` compatibility blocker is closed. The ordered
  run now exposes a missing generated dialect-definition boundary.
- Next action: independently review the generated dialect-definition include
  pattern against the exact pin before any new native run. This checkpoint does
  not authorize a relaunch.
- Artifact:
  `lib/shuttle/mlir/artifacts/native-preflight-20260810-link/README.md`
