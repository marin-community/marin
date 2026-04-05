## TL;DR

This thread starts from PR #4297 as the new Grug MoE GPU baseline and asks a narrower question than the earlier DeepEP issues:

- after the Triton `ragged_dot` improvement lands, exactly how much H100x8 gap still remains versus the earlier Nemotron-or-Megatron MoE baseline
- and which parts of the stack still explain that residual

The working expectation is that #4297 reduces the old `w13` local-compute bottleneck, so the remaining gap should skew more toward communication, synchronization, and overlap than toward the first routed FC1 lowering itself.

## Scope

- Base branch: `research/pr-4297-vs-nemotron-gpu-gap`
- Base commit: `7c80b0e633c8d26517aa95db83902c250d2bce07`
- Hardware: CoreWeave `iris-ci` / `h100-8x`
- Local logbook path: `.agents/logbooks/pr-4297-vs-nemotron-gpu-gap.md`

## Description

This experiment follows several earlier GPU MoE threads without reopening all of them at once:

- #3752 localized the dominant pre-4297 exact-cap JAX-vs-Nemotron-or-Megatron gap to the first local expert compute block, especially `w13_ragged_dot` plus associated layout work.
- #3821 then found that expert-padded `w13` lowering materially improved that old path.
- #3841 continued from there and narrowed the remaining residual gap toward communication / synchronization / overlap work.
- #4406 tested whether the #3821-style padded `w13` trick still adds lift on top of PR #4297 Triton and found that it does not on the shared exact-cap `current` surface.

That leaves the next question:

> with PR #4297 as the starting point, where is Nemotron-or-Megatron still faster than current Grug MoE on H100x8, and by how much?

This thread is not a torch-port effort. Grug remains pure JAX. The output of this work should be a ranked residual-gap explanation that points to the next pure-JAX optimization targets.

## Hypothesis or Goal

Primary goals:

- establish fresh PR #4297 exact-cap and training-harness anchors on H100x8
- compare those anchors against the strongest prior Nemotron-or-Megatron baselines
- determine whether the residual gap still contains a meaningful local-compute component or is now mostly transport / overlap

Initial hypotheses:

- H1: PR #4297 materially reduces the old `w13` local-compute deficit, so the remaining exact-cap gap should be much smaller than the pre-4297 `262144` comparison from #3752 / #3677
- H2: the remaining gap should align better with the residual categories from #3841 than with the original `w13`-dominated explanation from #3752
- H3: the best next optimization target will be a transport / overlap / synchronization improvement unless new probe data shows a previously hidden Triton-era local-compute residual

## Decision Log

- 2026-04-05: start a new experiment issue rather than extending #4406, because this is a broader residual-gap thread rather than a narrow padded-`w13` interaction check.
- 2026-04-05: use PR #4297 commit `7c80b0e633c8d26517aa95db83902c250d2bce07` as the baseline rather than waiting for merge-to-main, because the user explicitly asked to assume #4297 is the starting point.
- 2026-04-05: start on the shared exact-cap H100x8 surface before broadening to full training-path comparisons, because it is the lowest-confound overlap with the earlier Nemotron-or-Megatron anchors.

## Negative Results Index

- Pending.

## Current Baseline

- Prior exact-cap anchor from #4406 on the shared `262144` cell:
  - `triton`: `26065183.53 tok/s`
  - `triton + w13 padded`: `25209636.97 tok/s`
- Prior matched Nemotron-or-Megatron exact-cap forward anchor at `262144` from #3752 / #3677:
  - `33094416.78 tok/s` (`7.921 ms`)
- Prior PR #4297 training-harness validation:
  - `xla`: `463640.70 tok/s`
  - `triton`: `504399.46 tok/s`

## Results

- Fresh PR #4297 exact-cap H100x8 re-anchor on the shared `262144` cell:
  - `xla`: `9975119.03 tok/s`
  - `triton`: `26121235.17 tok/s`
  - `triton` vs `xla`: `+161.86%`
  - matched Megatron forward anchor: `33094416.78 tok/s`
  - residual gap after PR #4297 Triton: Megatron still `1.267x` faster, or `21.07%` ahead in throughput
- Existing sealed PR #4297 H100 training pairs still support the original end-to-end story:
  - `xla`: `463640.70 tok/s`, `5.55% MFU`, `0.2827 s/step`
  - `triton`: `504399.46 tok/s`, `6.04% MFU`, `0.2599 s/step`
  - delta: `+8.8% tok/s`, `+0.49` MFU points, `-8.1%` duration
- A fresh exact-cap Triton trace on the current PR #4297 branch localizes the surviving residual away from the old `w13` explanation:
  - exclusive breakdown: `28.96%` compute, `35.11%` communication, `35.72%` host
  - top collective ops:
    - `all-gather`: `66791.946` exclusive across `48` calls
    - `reduce-scatter`: `65684.934` exclusive across `24` calls
  - largest pre-op gaps:
    - before `all-gather`: `131003.183` total
    - before `reduce-scatter`: `103636.604` total
- The first direct follow-up probe on that residual says gather-metadata cleanup is real but small:
  - `production_current`: `26144435.07 tok/s` (`10.027 ms`)
  - `harness_current`: `26099086.43 tok/s` (`10.044 ms`)
  - `packed_meta`: `26312299.37 tok/s` (`9.963 ms`)
  - `packed_meta` vs `harness_current`: `+0.82%`
- A collectives-only rung says the gather/scatter stack itself already dominates a large share of the forward:
  - `collectives_only_current`: `45871598.17 tok/s` (`5.715 ms`)
  - `collectives_only_packed_meta`: `45952027.56 tok/s` (`5.705 ms`)
  - metadata packing moves the collectives-only floor by only `+0.18%`
  - the collectives-only current rung is already `56.9%` of the full `harness_current` forward time
- Working conclusion:
  - PR #4297 has largely removed the old routed FC1 bottleneck on the exact-cap surface
  - the next pure-JAX optimization tranche should target ring communication / synchronization / overlap rather than replaying the old `w13` local-compute fix or continuing with gather-metadata cleanup
- Operational note:
  - one fresh live Triton training repro on `iris-ci` died before the first useful throughput sample because the child GPU job was killed while the parent ended `Pod not found`; it was not a usable training anchor, so the paired W&B runs remain the authoritative end-to-end reference for this thread
