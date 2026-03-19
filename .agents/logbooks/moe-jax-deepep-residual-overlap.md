# JAX DeepEP Residual Overlap: Research Logbook

## Scope
- Goal: reduce the remaining JAX DeepEP gap after the `w13` fix by attacking ring communication, synchronization, and overlap in the live exact-cap and `current` paths.
- Primary metric(s): `time_s` / `tokens_per_s` for `deepep_transport_capped_prewarmed` and `current`, plus matched fixed-shape `forward_backward` throughput against Megatron DeepEP on the `marin_3633_*` family where relevant.
- Constraints:
  - start from the sealed `w13` optimization result on branch `research/moe-jax-deepep-w13-optimization`
  - treat the `w13` FC1 fix as landed baseline, not an open root-cause branch
  - do not reopen `w2` unless new profiling evidence forces it
  - commit and push any benchmark-code change before launching remote pods
  - post to GitHub only for major milestones / discoveries
- GitHub issue: https://github.com/marin-community/marin/issues/3841
- Prior issue: https://github.com/marin-community/marin/issues/3821
- Prior sealed root-cause issue: https://github.com/marin-community/marin/issues/3752
- Experiment ID prefix: `OVLP-RES`

## Baseline
- Date: 2026-03-19
- Code refs:
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/scripts/deepep_jax_krt_bench.py`
  - `.agents/scripts/megatron_qwen_krt_bench.py`
- Fixed baseline case:
  - hardware: H100x8 on CoreWeave
  - `tokens=262144`
  - `topk=2`
  - `bench_pass=forward`
  - `ep=8`
  - `distribution=random`
  - `shared_expert_dim=0`
  - `warmup=5`
  - `iters=20`
- Inherited post-`w13` baseline from `cf16bcc29fbf5cf20d54b21b0cc61c1fa7ab9e83`:
  - `deepep_transport_capped_prewarmed`: `21,763,265.78 tok/s` (`12.045 ms`)
  - `current`: `17,120,042.64 tok/s` (`15.312 ms`)
  - `current - capped_prewarmed` residual: `3.267 ms`
- Trustworthy residual facts:
  - post-fix `w2_only` is effectively flat relative to the sealed baseline, so `w2` is not the next primary branch
  - the post-fix `current` trace is no longer dominated by the old `w13` kernel
  - communication accounts for `25.1%` of exclusive profiled duration
  - pre-op gaps are concentrated before `reduce-scatter` and `all-gather`
  - the live ring path in `grug_moe.py` still performs full `all_gather` dispatch and `psum_scatter` collect
- High-signal post-fix profile facts from `scratch/profiles/current-expertpadded-262144-report.md`:
  - top compute op: `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT` at `21.9%`
  - collectives:
    - `all-gather`: `67,140.606` exclusive across `48` calls
    - `reduce-scatter`: `66,014.115` exclusive across `24` calls
  - pre-op gaps:
    - before `reduce-scatter`: `229,021.318` total gap (`24` occurrences)
    - before `all-gather`: `175,005.335` total gap (`24` occurrences)
- Reduced same-global-token rerun conclusion:
  - JAX exact-cap DeepEP improved on every rerun row after the `w13` fix
  - JAX still does not broadly keep up with Megatron DeepEP on the high-token `forward_backward` rows from `#3717`

## Experiment Log
### 2026-03-19 01:14 UTC - Kickoff for the residual-overlap thread
- Experiment ID: `OVLP-RES-001`
- Hypothesis:
  - the next meaningful gap closure will come from reducing collective volume and/or hiding collective latency in the live ring path, not from reopening the already-successful `w13` micro-branch.
- Command:
  - admin/scaffolding only; no benchmark launched yet on this thread
- Config:
  - branch: `research/moe-jax-deepep-residual-overlap`
  - starting commit: `cf16bcc29fbf5cf20d54b21b0cc61c1fa7ab9e83`
  - predecessor branch: `research/moe-jax-deepep-w13-optimization`
  - 12-hour window marker: `.agents/projects/afk-start-time-20260319T011420Z.txt`
- Result:
  - forked a fresh residual-overlap branch from the completed `w13` thread
  - recorded the 12-hour working window locally so follow-up sessions can reason about remaining time
  - started new logbook rather than continuing the sealed `w13` branch logbook
  - opened follow-up GitHub issue `#3841`
- Interpretation:
  - the next work should be evidence-driven and residual-specific
  - the first enabling task is to add exact-cap `forward_backward` profiling support so the remaining high-token gap can be localized directly
- Next action:
  - create the new GitHub issue, backfill the issue URL here, then patch profiling support for exact-cap `forward_backward`.
