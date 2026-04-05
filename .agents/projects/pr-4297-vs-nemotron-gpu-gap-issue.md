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

Pending. First milestone is a fresh PR #4297 H100x8 re-anchor followed by residual-gap localization.
