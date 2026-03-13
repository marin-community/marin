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
- `P3`: next required mainline prototype

## P3 Definition

Use a fixed `3 GDN + 1 attention` block as the optimization unit.

Required properties:

- bespoke backward/custom VJP at the block boundary
- explicit sharding contract
- explicit layout contract
- reuse current leaf kernels initially

## Anti-Goals

Do not spend mainline budget on:

- another `S3` refresh without tooling changes
- another near-identical `A3` retry
- same-boundary GDN solver/tape/kernel tweaks
- CE side-arms unless attribution points back to CE again

## Source Of Truth

- Historical evidence: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Current recipe: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/docs/recipes/optimize_gdn_pallas_tpu.md`
