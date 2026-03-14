# GDN TPU Hillclimb Summary

## Current Baseline

- Fixed benchmark regime: `3/4` GDN
- Hybrid deployable baseline: roughly `6.05-6.09 MFU` at `~166-167 ms`
- Attention-only control: roughly `21.3 MFU` at `~57 ms`

## Current Gap Accounting

Matched hybrid vs attention-only comparison is currently explained by:

- `train_path_budget_ms`: about `38-39%` of the gap
- `hybrid_generic_shell_delta_budget_ms`: about `18%` of the gap
- `interaction_remainder_ms`: about `43%` of the gap

Within the shell/remainder story:

- `dispatch_shard_shell_delta_ms` is the largest actionable budget
- `ad_wrapper_shell_delta_ms` is the second budget
- xprof shows the remainder manifests mostly as `IDLE`, i.e. waiting / serialization / shell tax

## Current Mainline

The mainline is no longer same-boundary GDN kernel work.

Current mainline target order:
1. matched xprof `dispatch_shard_shell_delta_ms`
2. matched xprof `xprof_idle_attributed_ms`
3. summary-side `dispatch_shard_shell_delta_ms`
4. `ad_wrapper_shell_delta_ms`
5. `interaction_remainder_ms`

## Coverage Status

- `S3`: complete
  - shell-delta attribution established
  - diagnostic-only from now on unless attribution tooling changes
- `A3`: attempted and rejected
  - outward layer-level manual-backward boundary increased shell tax and slowed the full step
- outward `P3` block-boundary family: attempted and rejected
  - outward block custom-VJP / scan-switch / shard-map / custom-partitioning / no-checkpoint variants all failed
- broad `G1` branch-wrapper family: attempted and rejected
  - staged pre/kernel/post, whole-branch, and array-only wrappers all re-emitted shell under branch-local wrapper names and slowed the full step
- `D1`: completed as a partial diagnostic lead
  - narrower head-first branch layout handling slightly improved AD/layout behavior and step time
  - but it did not reduce `dispatch_shard_shell_delta_ms`, and xprof `IDLE` worsened
- `D2`: immediate kernel-entry, direct array-entry, and prepared-array leaf-call branch-core cuts attempted and rejected
  - immediate kernel-entry cut: summary-side `dispatch_shard_shell_delta_ms` dropped, but `ad_wrapper_shell_delta_ms`, `layout_shell_delta_ms`, total hybrid shell, and full-step time all worsened
  - direct array-entry cut: summary-side and xprof `dispatch_shard_shell_delta_ms` both improved, but `ad_wrapper_shell_delta_ms`, `layout_shell_delta_ms`, total hybrid shell, `interaction_remainder_ms`, and xprof `IDLE` still grew and the full step slowed
  - prepared-array leaf-call cut: summary-side `step_duration_ms`, `hybrid_generic_shell_delta_budget_ms`, and `dispatch_shard_shell_delta_ms` all improved, but `ad_wrapper_shell_delta_ms` and `interaction_remainder_ms` still grew, matched xprof `dispatch_shard_shell_delta_ms` stayed flat/up, and xprof `IDLE` worsened sharply
  - `D2` established the right subgraph but not the right ownership model

## W1 Definition

Use the prepared-array leaf-call subgraph from Iteration 109 as the optimization unit, but change the ownership model.

Required properties:

- one explicit sharding envelope at the prepared-array leaf-call boundary
- one explicit layout contract inside the island
- reuse current GDN leaf kernels initially
- keep AD/backward unchanged on the first `W1` pass
- avoid nested inner `shard_map` / `closed_call/shard_map` wrappers inside the island
- summary-side wins are only prefilter; matched xprof dispatch + idle are the real gate

## A2 Definition

Only after a positive `W1` on the same cut:

- keep the `W1` forward/sharding envelope fixed
- move AD/manual-backward ownership onto that already-proven cut
- reject immediately if matched xprof dispatch or idle regresses

## Anti-Goals

Do not spend mainline budget on:

- another `S3` refresh without tooling changes
- another outward `A3`/`P3` retry
- another broad `G1` wrapper that owns forward/backward/sharding all at once
- another pure cut-size `D2` chase without a scheduling/sharding-envelope hypothesis
- same-boundary GDN solver/tape/kernel tweaks
- CE side-arms unless attribution points back to CE again
- checkpoint/remat toggles as a mainline strategy

## Source Of Truth

- Historical evidence: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Current recipe: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/docs/recipes/optimize_gdn_pallas_tpu.md`
