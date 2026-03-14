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
1. `dispatch_shard_shell_delta_ms`
2. `ad_wrapper_shell_delta_ms`
3. `interaction_remainder_ms`
4. `xprof_idle_attributed_ms`

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
- `D2`: immediate kernel-entry and direct array-entry branch-core cuts attempted and rejected
  - immediate kernel-entry cut: summary-side `dispatch_shard_shell_delta_ms` dropped, but `ad_wrapper_shell_delta_ms`, `layout_shell_delta_ms`, total hybrid shell, and full-step time all worsened
  - direct array-entry cut: summary-side and xprof `dispatch_shard_shell_delta_ms` both improved, but `ad_wrapper_shell_delta_ms`, `layout_shell_delta_ms`, total hybrid shell, `interaction_remainder_ms`, and xprof `IDLE` still grew and the full step slowed
  - `D2` remains open only for an even smaller branch-core cut that reduces dispatch/shard ownership without recreating grouped-head AD/layout spill under generic decoder shell names

## D2 Definition

Use a smaller branch-core island inside the hybrid-specific GDN branch as the optimization unit.

Required properties:

- explicit branch-core sharding contract
- explicit branch-core layout contract
- reuse current GDN leaf kernels initially
- smaller cut than the rejected broad `G1` wrappers
- no new custom VJP on the first `D2` attempt
- the cut should begin below the generic branch wrapper and end before the generic decoder shell resumes

## A2 Definition

Only after a positive `D2` on the same cut:

- keep the `D2` forward/sharding cut fixed
- move AD/manual-backward ownership onto that already-proven cut
- reject immediately if dispatch/shard regresses

## Anti-Goals

Do not spend mainline budget on:

- another `S3` refresh without tooling changes
- another outward `A3`/`P3` retry
- another broad `G1` wrapper that owns forward/backward/sharding all at once
- same-boundary GDN solver/tape/kernel tweaks
- CE side-arms unless attribution points back to CE again
- checkpoint/remat toggles as a mainline strategy

## Source Of Truth

- Historical evidence: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Current recipe: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/docs/recipes/optimize_gdn_pallas_tpu.md`
