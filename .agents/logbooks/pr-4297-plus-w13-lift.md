# PR 4297 + 3821-Style W13 Follow-Up: Research Logbook

## Scope
- Goal: measure the additional H100x8 performance lift from layering the #3821-style expert-padded `w13` FC1 lowering on top of the #4297 Triton `ragged_dot` baseline.
- Primary metric(s): `throughput/examples_per_second`, `throughput/tokens_per_second`, `throughput/mfu`, `throughput/duration`, and loss parity on the fixed Grug MoE H100x8 repro harness.
- Constraints:
  - start from `research/pr-4297-followup` / commit `e55747ba2f1c41e1a341b54263079a858a153a30`
  - use the always-on CoreWeave `iris-ci` cluster (`cpu-erapids` + `h100-8x`)
  - keep the comparison apples-to-apples on the overlapping optimization surface
  - the `#4297` branch does not carry the full DeepEP module tree used by `#3821`, so benchmark the shared exact-cap-style `current` EP=8 kernel path rather than replaying the full DeepEP transport ladder
  - answer the incremental question, so the key pair is `4297 triton current` vs `4297 triton current + 3821-style w13 expert padding`
  - GitHub issue: `https://github.com/marin-community/marin/issues/4406`
- Relevant references:
  - #4297: Triton `ragged_dot` kernel for GPU MoE grouped matmul
  - #3821: expert-padded `w13` FC1 lowering on the exact-cap H100x8 path
- Stop criteria:
  - produce at least one clean paired measurement for `4297 triton` vs `4297 triton + w13 expert padded`
  - if the first pair moves materially, add replication until the incremental delta is directionally stable
  - if the integrated path is broken or numerically suspect, stop and report the blocker instead of forcing more runs

## Baseline
- Date: 2026-04-04
- Code refs:
  - `experiments/grug/moe/launch_h100_pr4297.py`
  - `experiments/grug/moe/summarize_h100_pr4297.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `experiments/grug/moe/model.py`
- Fixed baseline case:
  - hardware: CoreWeave H100x8 on `iris-ci`
  - benchmark: exact-cap-style `current` EP ring kernel (`ep_size=8`)
  - `tokens=262144`
  - `hidden=2048`
  - `mlp_dim=768`
  - `experts=128`
  - `topk=2`
  - `shared_expert_dim=0`
  - `distribution=random`
  - `warmup=5`
  - `iters=20`
- Prior PR 4297 training-harness result carried forward only as context:
  - `triton` beat `xla` by `+8.8%` tokens/s on the fixed ~256M Grug H100x8 training repro
- Working comparison baseline for this thread:
  - `RAGGED_DOT_IMPL=triton`
  - `kernel=current`
  - no `w13` expert-padded lowering enabled

## Experiment Log
### 2026-04-04 01:45 UTC - Kickoff
- Hypothesis:
  - the #3821 expert-padded `w13` FC1 lowering still yields a measurable incremental gain on top of the #4297 Triton `ragged_dot` baseline, but the delta will likely be smaller than the original #3821 isolated win because part of the local expert compute path is already faster under Triton.
- Command:
  - scaffolding only
- Config:
  - worktree: `/home/ubuntu/dev/marin-wt/pr-4297-plus-w13-lift`
  - branch: `research/pr-4297-plus-w13-lift`
  - base branch: `research/pr-4297-followup`
  - base commit: `e55747ba2f1c41e1a341b54263079a858a153a30`
- Result:
  - created a dedicated follow-up research branch/worktree rather than extending the old PR 4297 validation thread
  - confirmed the local repo already contains the PR 4297 Grug H100x8 repro harness and summarizer
  - confirmed the #3821 winning idea was the expert-padded `w13` FC1 lowering, not the earlier `w13-out-first` layout candidate
  - confirmed `lib/iris/examples/coreweave-ci.yaml` provides the always-on `iris-ci` CoreWeave lane the user requested
  - attempted to reuse the historical #3821 bench stack directly, but the PR 4297 branch does not include `levanter.kernels.deepep`, so the full DeepEP transport benchmark path does not import cleanly on this base
- Interpretation:
  - the right low-confound measurement on top of the PR 4297 base is the overlapping exact-cap-style `current` EP=8 kernel path, not the full historical DeepEP transport ladder
  - that still answers the user’s interaction question because both optimizations directly act on the same routed FC1 grouped-matmul surface in the `current` path
- Next action:
  - create a minimal exact-cap benchmark harness for `current` EP=8 on top of the PR 4297 base, then run the four-way matrix `xla`, `xla+padded`, `triton`, `triton+padded` on `iris-ci`
