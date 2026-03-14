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
- `G1`: first stronger branch-local ownership attempt attempted and rejected
  - staged pre/kernel/post branch ownership removed visible old train-path buckets but massively increased the real branch-local shell budgets and slowed the full step
  - the smaller retained whole-branch wrapper was also validated and rejected; it drives `dispatch_shard_shell_delta_ms` to about `65 ms`, keeps xprof `IDLE` above `39 ms`, and again makes `_gdn_branch_boundary_impl` the dominant shell source
  - future `G1` work needs a materially smaller, const-cleaner, and sharding-cleaner branch cut; do not spend another turn on a broad branch wrapper that just renames the shell tax

## G1 Definition

Use the hybrid-specific GDN branch inside a GDN-bearing decoder layer as the optimization unit.

Required properties:

- bespoke backward/custom VJP at the branch boundary
- explicit sharding contract
- explicit layout contract
- reuse current GDN leaf kernels initially
- the branch starts at normalized hidden state plus mask and ends at the branch contribution before the generic decoder shell resumes
- do not own the generic MLP / residual shell in the first prototype

## Anti-Goals

Do not spend mainline budget on:

- another `S3` refresh without tooling changes
- another near-identical `A3` retry
- another outward `P3` block wrapper on the same generic block/module shell
- same-boundary GDN solver/tape/kernel tweaks
- CE side-arms unless attribution points back to CE again

## Source Of Truth

- Historical evidence: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md`
- Current recipe: `/Users/calvinxu/Projects/Work/Marin/marin-gdn-pallas/docs/recipes/optimize_gdn_pallas_tpu.md`
