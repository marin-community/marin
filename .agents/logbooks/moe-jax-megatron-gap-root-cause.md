# JAX vs Megatron DeepEP Gap Root Cause: Research Logbook

## Scope
- Goal: determine the precise root causes of the remaining throughput gap between JAX DeepEP and Megatron DeepEP on the same fixed-shape H100x8 MoE block regime.
- Primary metric(s): `time_s`, `tokens/s`, and isolated phase timings for routing/layout, transport, local expert compute, and framework/runtime overhead.
- Constraints:
  - use the sealed `#3717` head-to-head matrix as the starting baseline
  - optimize for quick-turnaround experiments that eliminate false hypotheses early
  - only compare apples-to-apples shapes, passes, and token counts
  - update the public issue thread only for major milestones/discoveries
- GitHub issue: https://github.com/marin-community/marin/issues/3752

## Baseline
- Date: 2026-03-16
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/scripts/deepep_jax_krt_bench.py`
  - `.agents/scripts/megatron_qwen_krt_bench.py`
  - `.agents/scripts/moe_block_head_to_head.py`
- Baseline numbers:
  - Sealed `#3717` fixed-shape same-global-token head-to-head (`forward_backward`, H100x8, `distribution=random`):

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

## Experiment Log
### 2026-03-16 17:25 - Kickoff
- Hypothesis:
  - the remaining JAX-vs-Megatron DeepEP gap is attributable to a specific combination of benchmark-path overheads rather than a single unexplained transport failure.
- Command:
  - kickoff/scaffolding only; no benchmark command yet
- Config:
  - refreshed from `origin/main` at `69d30f1c11fcb3ad3349f594f59766b7081370b0`
  - branch: `research/moe-jax-megatron-gap-root-cause`
  - worktree: `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`
- Result:
  - created the new research worktree from refreshed `main`
  - initialized the local research logbook
- Interpretation:
  - this thread starts from a clean base and a sealed benchmark table rather than from an unresolved bring-up problem
- Next action:
  - create the experiment issue and write the public baseline/decision scaffolding before defining the first quick-turnaround experiment matrix
