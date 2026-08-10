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
