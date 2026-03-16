## TL;DR

This experiment follows the sealed `#3717` H100x8 same-global-token head-to-head and focuses on one question: why does JAX DeepEP still underperform Megatron DeepEP on the same fixed-shape MoE block regime, especially as global tokens scale up?

Known starting facts:
- JAX DeepEP is consistently better than JAX `current` on the completed same-shape cells.
- Megatron `deepep` is consistently better than Megatron `alltoall` on the same cells.
- The remaining gap is large at larger token counts. For example:
  - `tokens=131072, topk=2`: `jax_deepep=3.82M tok/s` vs `megatron_deepep=12.71M tok/s`
  - `tokens=262144, topk=2`: `jax_deepep=3.82M tok/s` vs `megatron_deepep=15.83M tok/s`

This thread is about quickly falsifying root-cause hypotheses and isolating which parts of the JAX path are still leaving performance on the table.

## Scope

- Primary question: what precise factors explain the remaining throughput gap between JAX DeepEP and Megatron DeepEP on the same H100x8 fixed-shape MoE block regime?
- In scope:
  - phase-level decomposition of the JAX DeepEP path
  - matched same-shape JAX vs Megatron comparisons
  - quick-turnaround sweeps that isolate layout, transport, local compute, and framework/runtime overhead
  - high-signal tuning knobs if they are cheap to test and directly related to the gap
- Out of scope unless evidence forces it:
  - broad new architecture changes
  - unrelated kernel bring-up work
  - large speculative sweeps with weak attribution value

## Description

The prior threads established the following sequence:

1. `#3677` narrowed the original JAX-vs-Megatron mismatch from a broad mystery to a transport-integration problem.
2. `#3711` fixed a major JAX reintegration mistake by using exact caps for DeepEP receive/local-assignment buffers, which made the JAX DeepEP path positive against JAX `current`.
3. `#3717` added JAX autodiff support, completed the `forward_backward` path, and produced a direct same-global-token head-to-head across JAX and Megatron implementations.

The remaining open question is no longer “does JAX DeepEP work?” It does. The open question is: which exact layers of the JAX path are responsible for the residual gap versus Megatron DeepEP, and which of those are large enough to matter?

This thread will investigate that with fast, falsifiable experiments and keep the public summary focused on major milestones only.

## Hypothesis or Goal

- Goal: attribute the JAX-vs-Megatron DeepEP gap to concrete phases/overheads rather than to an undifferentiated end-to-end number.
- Goal: identify which candidate causes are false quickly and avoid spending time on low-value branches.
- Goal: produce a stable evidence-backed explanation of the main contributors to the gap on the sealed same-shape H100x8 regime.

## Current baseline

Reference matrix from sealed `#3717` (`forward_backward`, fixed shape, H100x8, same global tokens, `distribution=random`):

```text
| tokens | topk | jax_current | jax_deepep | megatron_alltoall | megatron_deepep |
| 32768  | 2    |   2,957,986 |  3,312,032 |            537,099|         2,644,307|
| 32768  | 8    |     959,217 |  1,115,553 |            590,120|         1,965,951|
| 65536  | 2    |   3,156,598 |  3,752,354 |            785,280|         2,660,216|
| 65536  | 8    |     961,693 |  1,116,306 |          1,315,403|         3,088,961|
| 131072 | 2    |   3,228,500 |  3,824,437 |          3,454,152|        12,712,443|
| 131072 | 8    |     965,449 |  1,057,153 |          2,489,744|         6,577,851|
| 262144 | 2    |   3,247,008 |  3,821,071 |          6,661,309|        15,830,875|
| 262144 | 8    |         OOM |    985,350 |          6,426,325|         7,652,599|
```

## Decision log

- 2026-03-16: Start a new root-cause experiment rather than extending `#3717`, because the remaining work is explanatory/diagnostic rather than backward bring-up.

## Negative results index

- None yet in this thread.

## Conclusion

In progress.

### Links

* Prior fixed-shape GPU issue (`#3633`): https://github.com/marin-community/marin/issues/3633
* Prior Torch DeepEP / Hybrid-EP issue (`#3641`): https://github.com/marin-community/marin/issues/3641
* Prior layout-only JAX issue (`#3665`): https://github.com/marin-community/marin/issues/3665
* Prior Megatron scaling issue (`#3666`): https://github.com/marin-community/marin/issues/3666
* Prior JAX DeepEP root-cause issue (`#3677`): https://github.com/marin-community/marin/issues/3677
* Prior exact-cap reintegration issue (`#3711`): https://github.com/marin-community/marin/issues/3711
* Current experiment issue (`#3752`): https://github.com/marin-community/marin/issues/3752
* Prior JAX autodiff + scaling issue (`#3717`): https://github.com/marin-community/marin/issues/3717
* Research branch: https://github.com/marin-community/marin/tree/research/moe-jax-megatron-gap-root-cause
* Research logbook: https://github.com/marin-community/marin/tree/research/moe-jax-megatron-gap-root-cause/.agents/logbooks/moe-jax-megatron-gap-root-cause.md

## Results

Current state at kickoff:
- starting point is the sealed `#3717` matrix quoted above
- refreshed base: `origin/main` at `69d30f1c11fcb3ad3349f594f59766b7081370b0`
- new branch/worktree:
  - branch: `research/moe-jax-megatron-gap-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`
- immediate next step:
  - define the smallest quick-turnaround experiment set that can split the likely causes of the remaining gap into falsifiable buckets
