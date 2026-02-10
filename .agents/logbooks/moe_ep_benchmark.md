# MoE EP Benchmark: Research Logbook

## Scope
- Goal: benchmark and improve Mixture-of-Experts throughput with expert parallelism (EP) in the existing hillclimb harness.
- Primary metrics: end-to-end `forward_backward` TF/s (per-device and aggregate), tokens/s, and delta vs fixed baseline.
- Constraints: keep comparisons apples-to-apples; isolate EP effects from unrelated kernel changes.

## Links
- Experiment issue: https://github.com/marin-community/marin/issues/2710
- Previous experiment issue: https://github.com/marin-community/marin/issues/2704
- Previous snapshot tag: https://github.com/marin-community/marin/tree/moe-hillclimb-harness-20260209
- Baseline report: `.agents/projects/moe_hillclimb_harness.md`
- Baseline harness: `lib/levanter/scripts/bench/bench_moe_hillclimb.py`

## Baseline
- Date: 2026-02-09
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `.agents/projects/moe_hillclimb_harness.md`
- Baseline numbers (from previous thread):
  - shared shape (`tokens=32768, hidden=2048, mlp_dim=1408, experts=60, shared=5632`) forward_backward full:
    - topk=1: 219.231 TF/s
    - topk=2: 192.449 TF/s
    - topk=4: 175.260 TF/s
    - topk=6: 164.920 TF/s
    - topk=8: 167.868 TF/s

## Stop Criteria
- EP benchmark path is implemented and reproducible.
- We have apples-to-apples EP vs baseline tables for key shapes (`topk=2` and `topk=8` at minimum).
- We can answer whether EP materially improves end-to-end training-path throughput.

## Experiment Matrix (Initial)
- Axis A: parallel mode
  - baseline (no EP)
  - EP with fixed shard counts
- Axis B: top-k
  - 2, 8
- Axis C: shared expert path
  - off, on (`shared_expert_dim=5632`)
- Axis D: pass mode
  - forward, forward_backward

## Experiment Log
### 2026-02-09 00:00 - Kickoff
- Hypothesis: EP will improve effective tokens/expert-shard and increase end-to-end training-path throughput at moderate/high top-k.
- Command: N/A (planning + setup)
- Config: N/A
- Result: Created new branch and logbook; next step is issue creation and cross-linking.
- Interpretation: Setup complete; ready for implementation and measurement.
- Next action: create experiment issue and add bidirectional links.

### 2026-02-09 13:40 - EP vs Non-EP (shared shape, forward_backward, random)
- Hypothesis: EP should improve end-to-end throughput by reducing per-device expert load.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --parallel-mode "$pm" --iters 3 --warmup 1; done; done'`
- Config:
  - tokens=32768, hidden=2048, mlp_dim=1408, experts=60, shared_expert_dim=5632
  - backend=gmm, impl=fused_w13, queue_mode=full, distribution=random, pass=forward_backward
  - devices=4 (single v5p host), commit=`5b6f53c5d`
- Result:
  - `none, topk=2`: 192.868 TF/s, 619304 tokens/s
  - `none, topk=8`: 167.512 TF/s, 268942 tokens/s
  - `ep, topk=2`: 143.985 TF/s, 462339 tokens/s
  - `ep, topk=8`: 116.472 TF/s, 186997 tokens/s
- Interpretation:
  - Current EP path is a negative result for this harness shape: `ep` is ~25% slower than `none` for topk=2 and ~30% slower for topk=8.
  - This points to EP overhead in the current implementation dominating any expert-shard benefit for this regime.
- Next action:
  - Profile `ep` path and isolate overhead sources (shard_map mask/nonzero/indexing, psum, data movement).
  - Test larger token counts / expert counts where EP has more chance to amortize overhead.

### 2026-02-09 13:44 - EP Crossover Sweep (larger tokens/experts)
- Hypothesis: EP overhead may amortize at larger workload (`tokens=65536`, `experts=120`).
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 65536 --hidden 2048 --mlp-dim 1408 --experts 120 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --parallel-mode "$pm" --iters 3 --warmup 1; done; done'`
- Result:
  - `none, topk=2`: 193.106 TF/s
  - `none, topk=8`: 167.836 TF/s
  - `ep, topk=2`: 146.336 TF/s
  - `ep, topk=8`: 122.254 TF/s
- Interpretation:
  - Negative result persists at larger shape; EP remains slower (`-24.2%` at topk=2, `-27.2%` at topk=8).
  - Current EP implementation overhead is not amortized by this scale increase.
- Next action:
  - Collect EP trace and break down cost in routing/dispatch vs expert matmul vs reduction.

### 2026-02-09 13:48 - Routing Pack Strategy A/B
- Hypothesis: replacing argsort-style pack ordering with tuple-sort might reduce routing overhead.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for strat in argsort tuple_sort; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy "$strat" --queue-mode full --bench-pass forward_backward --parallel-mode none --iters 3 --warmup 1; done; done'`
- Result:
  - `argsort, topk=2`: 192.912 TF/s
  - `tuple_sort, topk=2`: 191.980 TF/s (`-0.48%`)
  - `argsort, topk=8`: 167.906 TF/s
  - `tuple_sort, topk=8`: 165.552 TF/s (`-1.40%`)
- Interpretation:
  - No win from tuple-sort in current harness; argsort remains preferred.
- Next action:
  - Keep argsort as benchmark default and avoid additional pack-path churn for now.

### 2026-02-09 13:51 - Locality (`runs`) and Prequeue Sweep
- Hypothesis: stronger routing locality plus prequeue mode may provide measurable speedup.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do for qm in full prequeue; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode "$qm" --bench-pass forward_backward --parallel-mode none --iters 3 --warmup 1; done; done; done'`
- Result:
  - `random/full`: topk2 `193.011`, topk8 `167.604`
  - `random/prequeue`: topk2 `192.707`, topk8 `167.449`
  - `runs/full`: topk2 `192.446`, topk8 `167.561`
  - `runs/prequeue`: topk2 `192.632`, topk8 `167.822`
- Interpretation:
  - Prequeue impact is effectively neutral in this setup (all deltas within ~0.2%).
  - `runs` distribution does not materially move end-to-end throughput despite much longer run lengths.
- Next action:
  - Focus optimization effort on EP overhead and/or fusion opportunities, not pack strategy or prequeue toggles.

### 2026-02-09 13:58 - Stage Timing Breakdown (`topk=8`, forward)
- Hypothesis: if locality helps, `pack`/`combine` stage percentages should shrink under `runs`.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk 8 --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward --parallel-mode none --stage-timing --iters 3 --warmup 1; done'`
- Result:
  - `random`: forward `207.066 TF/s`; stage pct `pack=14.8`, `up=45.3`, `down=23.2`, `combine=16.8`
  - `runs`: forward `207.319 TF/s`; stage pct `pack=14.7`, `up=45.3`, `down=23.2`, `combine=16.8`
- Interpretation:
  - Stage mix is effectively identical between random and runs at this setting.
  - Locality signal in this harness does not translate to measurable pack/combine reduction.
- Next action:
  - Instrument EP path directly (no stage-timing support today) with profiler trace to pinpoint overhead site.

### 2026-02-09 14:10 - ICI-Aware Roofline Check
- Hypothesis: EP slowdown might be ICI-bandwidth bound rather than compute-path overhead.
- Inputs:
  - TPU v5p ICI spec: `1200 GB/s` per chip bidirectional (Cloud TPU docs).
  - EP path uses explicit `jax.lax.psum(y_local, "ep")` over output tensor `[tokens, hidden]` in `bench_moe_hillclimb.py`.
  - Tensor size at benchmark shape: `tokens * hidden * 2 bytes = 32768 * 2048 * 2 = 134,217,728 bytes` (`128 MiB`).
- Model:
  - Ring all-reduce bytes per device: `2 * (p - 1) / p * N`, with `p=4`.
  - Per all-reduce bytes/device: `2 * 3/4 * 128 MiB = 192 MiB`.
  - Forward+backward lower bound for explicit psum path (2 all-reduces): `~384 MiB`/device/step.
- Measured implied ICI bandwidth:
  - no-shared, EP, topk=2, `time_s=0.050523` => `~8.0 GB/s`
  - no-shared, EP, topk=8, `time_s=0.154982` => `~2.6 GB/s`
  - shared, EP, topk=2, `time_s=0.070874` => `~5.7 GB/s`
  - shared, EP, topk=8, `time_s=0.175233` => `~2.3 GB/s`
- Interpretation:
  - Observed EP throughput is far below ICI roofline from a bandwidth perspective (`<< 1200 GB/s`), so ICI bandwidth saturation is not the binding limit in current runs.
  - Current EP regression is more consistent with algorithmic overhead (extra sorting/masking/indexing, collectives synchronization, and current dense local dispatch shape) than raw ICI bandwidth limits.
- Next action:
  - Replace current EP local dispatch (full-length masked path) with variable-length/local-compacted expert assignment path to remove dummy-work before reevaluating ICI impact.

### 2026-02-09 14:22 - EP Compact Local Dispatch (Implemented)
- Change:
  - Updated `lib/levanter/scripts/bench/bench_moe_hillclimb.py` EP path to use compact per-shard dispatch buffers built once from fixed routing.
  - Removed dense global-length masked dispatch and dummy expert extension in favor of local-cap padding only.
  - Local cap is rounded to 512 to avoid current `gmm_sharded` pad dtype path.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for tk in 2 8; do .venv/bin/python lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --parallel-mode "$pm" --iters 3 --warmup 1; done; done'`
- Result (new):
  - `none, topk=2`: 192.599 TF/s
  - `none, topk=8`: 167.623 TF/s
  - `ep, topk=2`: 232.246 TF/s
  - `ep, topk=8`: 336.867 TF/s
- Before (same shape, pre-change):
  - `ep, topk=2`: 143.985 TF/s
  - `ep, topk=8`: 116.472 TF/s
- Delta:
  - EP absolute improvement: `+61.3%` (topk=2), `+189.2%` (topk=8)
  - EP vs non-EP: `+20.6%` (topk=2), `+101.0%` (topk=8)
- Interpretation:
  - Prior EP regression was primarily implementation overhead from dense per-shard dummy work, not ICI roofline.
  - Compact dispatch converts EP into a clear win on this benchmark regime.
- Next action:
  - Add dedicated correctness check harness (`none` vs `ep` forward diff) and run on TPU.
  - Re-run no-shared and larger-shape crossover with compact EP path.

### 2026-02-09 14:39 - No-Shared Ablation After Compact EP
- Hypothesis:
  - Previous headline TF/s included substantial shared-expert compute (`shared_expert_dim=5632`), so routed-only throughput needs to be measured explicitly.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 0 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode "$pm"; done; done'`
- Result:
  - `none, topk=2, shared=0`: `99.534 TF/s`
  - `none, topk=8, shared=0`: `132.117 TF/s`
  - `ep, topk=2, shared=0`: `144.861 TF/s`
  - `ep, topk=8, shared=0`: `338.854 TF/s`
- Interpretation:
  - Shared-expert path materially inflates headline total TF/s; routed-only baseline at this shape is much lower for `topk=2`.
  - Compact EP remains a clear win even without shared expert:
    - `topk=2`: `+45.5%` (`144.9` vs `99.5`)
    - `topk=8`: `+156.5%` (`338.9` vs `132.1`)
  - `topk=8` with compact EP now becomes strongly compute-favorable in this harness.
- Next action:
  - Re-run routing-pack strategy A/B under compact EP (not just non-EP).
  - Run remat leverage sweep (`none`, `expert_mlp`, `combine`) under compact EP.

### 2026-02-09 14:43 - Routing Pack Strategy A/B (Post-Compact-EP)
- Hypothesis:
  - Compact EP may change routing-pack tradeoffs; re-check `argsort` vs `tuple_sort`.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for strat in argsort tuple_sort; do for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 0 --backend gmm --impl fused_w13 --routing-pack-strategy "$strat" --queue-mode full --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode "$pm"; done; done; done'`
- Result (random distribution):
  - `none, topk=2`: argsort `99.569` vs tuple_sort `98.438` (`-1.14%`)
  - `none, topk=8`: argsort `132.224` vs tuple_sort `130.018` (`-1.67%`)
  - `ep, topk=2`: argsort `144.336` vs tuple_sort `144.798` (`+0.32%`)
  - `ep, topk=8`: argsort `339.429` vs tuple_sort `337.939` (`-0.44%`)
- Interpretation:
  - Non-EP still prefers `argsort`.
  - Under compact EP, routing-pack strategy is effectively neutral (within measurement noise).
  - Keep `argsort` as default for consistency and lower compile churn.
- Next action:
  - Quantify remat leverage now that EP is no longer overhead-bound.

### 2026-02-09 14:51 - Remat Leverage Sweep (Compact EP, No Shared)
- Hypothesis:
  - `combine` remat may be near-free; `expert_mlp` remat should incur the main recompute tax.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for rm in none expert_mlp combine; do for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 0 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode "$rm" --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done; done'`
- Result (random distribution):
  - `topk=2`: none `144.654`, expert_mlp `138.346` (`-4.36%`), combine `144.748` (`+0.06%`)
  - `topk=8`: none `339.391`, expert_mlp `313.676` (`-7.57%`), combine `338.917` (`-0.14%`)
- Interpretation:
  - `combine` remat is effectively free in this harness.
  - `expert_mlp` remat carries clear recompute cost (larger at higher topk).
  - If memory pressure requires remat, start with `combine`; only use `expert_mlp` when necessary.

### 2026-02-09 14:56 - Top-k Scaling Curve (Compact EP)
- Hypothesis:
  - Need explicit `topk` scaling curve for the current best EP path to quantify TF/s and tokens/s tradeoff.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for shared in 0 5632; do for tk in 1 2 4 6 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim "$shared" --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done; done'`
- Result (random distribution):
  - `shared=0`:
    - `topk=1`: `81.935 TF/s`, `1,578,564 tok/s`
    - `topk=2`: `145.044 TF/s`, `1,397,218 tok/s`
    - `topk=4`: `233.660 TF/s`, `1,125,431 tok/s`
    - `topk=6`: `294.641 TF/s`, `946,099 tok/s`
    - `topk=8`: `339.266 TF/s`, `817,043 tok/s`
  - `shared=5632`:
    - `topk=1`: `207.977 TF/s`, `801,382 tok/s`
    - `topk=2`: `233.910 TF/s`, `751,090 tok/s`
    - `topk=4`: `276.076 TF/s`, `664,865 tok/s`
    - `topk=6`: `310.619 TF/s`, `598,443 tok/s`
    - `topk=8`: `339.057 TF/s`, `544,360 tok/s`
- Interpretation:
  - TF/s increases monotonically with `topk` and plateaus near `~339 TF/s` by `topk=8` on this shape.
  - Tokens/s decreases as `topk` rises (expected), so `topk` remains a quality/perf modeling tradeoff.
  - Shared expert shifts absolute TF/s upward at low `topk`, but does not change the high-`topk` plateau.

### 2026-02-09 15:06 - Shared-Fused A/B (Compact EP, shared=5632)
- Hypothesis:
  - Fusing shared path (`w1|w3` concat matmul) should reduce overhead most at lower `topk`.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim 1408 --experts 60 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result (random distribution):
  - `topk=2`: non-fused `233.762 TF/s` vs shared-fused `239.150 TF/s` (`+2.30%`)
  - `topk=8`: non-fused `338.027 TF/s` vs shared-fused `344.375 TF/s` (`+1.88%`)
- Interpretation:
  - Shared fusion is a consistent positive delta at this shape.
  - It is now a good default choice in this harness when shared expert is enabled.

### 2026-02-09 15:11 - MLP Width Sweep (`topk=8`, routed-only)
- Hypothesis:
  - `mlp_dim=1408` may be tile-unfriendly; increasing width might improve achieved compute utilization.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -- /bin/bash -lc 'set -euo pipefail; for m in 1408 1536 1792 2048; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --tokens 32768 --hidden 2048 --mlp-dim "$m" --experts 60 --topk 8 --shared-expert-dim 0 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done'`
- Result (random distribution):
  - `mlp_dim=1408`: `339.738 TF/s`, `818,181 tok/s`
  - `mlp_dim=1536`: `363.281 TF/s`, `801,971 tok/s`
  - `mlp_dim=1792`: `405.068 TF/s`, `766,474 tok/s`
  - `mlp_dim=2048`: `448.627 TF/s`, `742,784 tok/s`
- Interpretation:
  - Runtime rises modestly as width grows, but achieved TF/s rises strongly because each step carries more useful compute.
  - On this harness, wider MLP dims are materially less “cursed” from a throughput perspective; `2048` gets very close to single-chip bf16 peak.

### 2026-02-09 15:40 - Qwen-like Family A Sweep (`hidden=2048`, `experts=64`, `topk=2`, EP)
- Hypothesis:
  - For a Qwen-like routed+shared shape, wider routed MLP widths should improve achieved TF/s with compact EP dispatch.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for m in 1408 1536 1792 2048; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim "$m" --experts 64 --topk 2 --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result (random):
  - `mlp_dim=1408`: `258.404 TF/s`, `829,742 tok/s`
  - `mlp_dim=1536`: `262.137 TF/s`, `816,973 tok/s`
  - `mlp_dim=1792`: `267.079 TF/s`, `786,131 tok/s`
  - `mlp_dim=2048`: `274.162 TF/s`, `764,506 tok/s`
- Interpretation:
  - Throughput in TF/s climbs monotonically with width at this shape; tokens/s decreases as expected.
  - This is a clean confirmation that the same width trend persists in a more Qwen-like expert/shared configuration.

### 2026-02-09 15:43 - Qwen-like Family B Sweep (`hidden=3072`, `experts=64`, `topk=2`, EP)
- Hypothesis:
  - A larger hidden/shared shape may have a different width sweet spot than family A.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for m in 2048 2304 2688 3072; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim "$m" --experts 64 --topk 2 --shared-expert-dim 8192 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result (random):
  - `mlp_dim=2048`: `285.808 TF/s`, `420,629 tok/s`
  - `mlp_dim=2304`: `277.258 TF/s`, `391,724 tok/s`
  - `mlp_dim=2688`: `283.538 TF/s`, `377,922 tok/s`
  - `mlp_dim=3072`: `290.630 TF/s`, `366,622 tok/s`
- Interpretation:
  - Non-monotonic middle widths; best endpoint here is `mlp_dim=3072`.
  - Shape-specific tuning remains important; broad “wider is always better” does not fully hold across all nearby widths.

### 2026-02-09 15:46 - Queue Mode Clarification (`forward_backward`)
- Observation:
  - In this harness, `bench_pass != "forward"` forces `modes = ["full"]` internally.
- Implication:
  - `queue_mode=prequeue` has no effect for `forward_backward` runs; any such A/B is intentionally identical and should not be interpreted as a real queue optimization result.

### 2026-02-09 15:52 - Real Queue A/B (`forward` only, EP, shared_fused)
- Hypothesis:
  - If prequeue is useful, the effect should appear in pure forward timing where `queue_mode=both` actually runs both branches.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode both --remat-mode none --bench-pass forward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done; for dist in random runs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 2 --shared-expert-dim 8192 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode both --remat-mode none --bench-pass forward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `2048/2048, random`: full `166.367`, prequeue `165.986` TF/s (`-0.23%`)
  - `2048/2048, runs`: full `166.042`, prequeue `165.628` TF/s (`-0.25%`)
  - `3072/3072, random`: full `177.592`, prequeue `177.757` TF/s (`+0.09%`)
  - `3072/3072, runs`: full `177.877`, prequeue `177.391` TF/s (`-0.27%`)
- Interpretation:
  - Prequeue remains effectively neutral (well within noise) on these EP shapes.
  - Routing locality (`runs`) still does not unlock a measurable forward throughput gain.

### 2026-02-09 15:55 - Routing Strategy A/B (EP, Qwen-like)
- Hypothesis:
  - `tuple_sort` (the explicit sorted tuple path) might outperform `argsort` under compact EP at these shapes.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for strat in argsort tuple_sort; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy "$strat" --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `2048/2048`: argsort `274.670` vs tuple_sort `275.212` TF/s (`+0.20%` tuple)
- Interpretation:
  - Difference is tiny; routing strategy is again effectively neutral at this regime.
  - `3072/3072` A/B remains pending due TPU host reachability loss (both codex1 and codex2 SSH timeouts).

### 2026-02-09 16:02 - Infra Incident Note
- Observation:
  - Mid-run, both dev TPU aliases (`dev-tpu-dlwh-codex1`, `dev-tpu-dlwh-codex2`) became unreachable (`ssh ... port 22: Operation timed out`), preventing completion of remaining queued sweeps.
- Impact:
  - Incomplete item: routing A/B at `hidden=3072, mlp_dim=3072`.
- Next step:
  - Resume that A/B immediately once either dev TPU is reachable again.

### 2026-02-09 16:21 - Routing Strategy A/B Completion (`3072/3072`, EP)
- Hypothesis:
  - Confirm whether routing strategy remains neutral on the larger Qwen-like shape.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for strat in argsort tuple_sort; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 2 --shared-expert-dim 8192 --backend gmm --impl fused_w13 --routing-pack-strategy "$strat" --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `3072/3072`: argsort `290.979` vs tuple_sort `290.965` TF/s (`-0.00%`)
- Interpretation:
  - Routing strategy is effectively identical at this shape.
  - Standardize on `argsort` for this experiment series to reduce branching and keep comparisons cleaner.

### 2026-02-09 16:23 - Remat Leverage (Qwen-like EP Shapes)
- Hypothesis:
  - `combine` remat remains near-free; `expert_mlp` remat should carry recompute tax on both tuned shapes.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for rm in none expert_mlp combine; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode "$rm" --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done; for rm in none expert_mlp combine; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 2 --shared-expert-dim 8192 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode "$rm" --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `2048/2048`: none `275.354`, expert_mlp `268.278` (`-2.57%`), combine `275.243` (`-0.04%`)
  - `3072/3072`: none `290.685`, expert_mlp `283.028` (`-2.63%`), combine `290.581` (`-0.04%`)
- Interpretation:
  - Pattern is stable across both shapes: `combine` remat is effectively free, `expert_mlp` remat costs about `2.6%`.

### 2026-02-09 16:27 - Top-k Scaling (Qwen-like EP Shapes)
- Hypothesis:
  - Confirm current top-k throughput trend on tuned Qwen-like shapes.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 1 2 4 6 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done; for tk in 1 2 4 6 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk "$tk" --shared-expert-dim 8192 --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `2048/2048` (`shared=5632`):
    - `topk=1`: `232.771 TF/s`, `822,178 tok/s`
    - `topk=2`: `275.583 TF/s`, `768,470 tok/s`
    - `topk=4`: `345.317 TF/s`, `677,614 tok/s`
    - `topk=6`: `400.419 TF/s`, `606,142 tok/s`
    - `topk=8`: `443.204 TF/s`, `546,088 tok/s`
  - `3072/3072` (`shared=8192`):
    - `topk=1`: `247.132 TF/s`, `396,773 tok/s`
    - `topk=2`: `290.871 TF/s`, `366,926 tok/s`
    - `topk=4`: `361.151 TF/s`, `318,907 tok/s`
    - `topk=6`: `414.946 TF/s`, `281,854 tok/s`
    - `topk=8`: `457.438 TF/s`, `252,458 tok/s`
- Interpretation:
  - Same trend as prior runs: TF/s rises with top-k while tokens/s decreases monotonically.
  - Larger hidden shape reaches higher absolute TF/s at high top-k.

### 2026-02-09 16:35 - Backend A/B (`gmm` vs `ragged_dot`, Qwen-like EP)
- Hypothesis:
  - `gmm` should still beat `ragged_dot`; quantify current gap on tuned shapes.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for be in gmm ragged_dot; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --backend "$be" --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done; for be in gmm ragged_dot; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 2 --shared-expert-dim 8192 --backend "$be" --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --shared-fused; done'`
- Result:
  - `2048/2048`: gmm `275.232` vs ragged_dot `268.852` TF/s (`-2.32%`)
  - `3072/3072`: gmm `290.988` vs ragged_dot `288.094` TF/s (`-0.99%`)
- Interpretation:
  - `gmm` remains best; gap is modest but consistent.
  - Use `gmm` as the default backend for ongoing runs unless testing kernel-specific behavior.

### 2026-02-09 17:05 - Overlap Harness Baseline (`bench_moe_mlp_profile.py`, topk=2)
- Context:
  - Shift from hillclimb harness to `bench_moe_mlp_profile.py` to compare:
    - `ref_dp`: reference DP MoE path.
    - `staged_dp`: staged DP path.
    - `fused_ep`: fused EP path (`fused_moe_fused_routing`).
  - These are not apples-to-apples with hillclimb absolute TF/s; this section is for relative overlap/path behavior within this harness.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --iters 3 --warmup 1 --staged-dp --staged-parallel; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk 2 --iters 3 --warmup 1 --staged-dp --staged-parallel'`
- Result:
  - `2048/2048, topk=2`:
    - `ref_dp`: `34.899 TF/s` (`8.725` per device)
    - `staged_dp`: `39.607 TF/s` (`9.902` per device)
    - `fused_ep`: `63.143 TF/s` (`15.786` per device)
  - `3072/3072, topk=2`:
    - `ref_dp`: `36.292 TF/s` (`9.073` per device)
    - `staged_dp`: `40.861 TF/s` (`10.215` per device)
    - `fused_ep`: `87.136 TF/s` (`21.784` per device)
- Interpretation:
  - In this harness, fused EP is substantially ahead at topk=2 on both shapes.
  - This indicates meaningful end-to-end overlap/path efficiency headroom versus staged DP in the same benchmark framing.

### 2026-02-09 17:25 - Overlap vs Top-k Sweep (`3072/3072`)
- Hypothesis:
  - If fused EP overlap advantage scales with routed compute density, gap vs DP paths should remain positive or widen with top-k.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 1 2 4 6 8; do echo "=== topk=${tk} ==="; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py --tokens 32768 --hidden 3072 --mlp-dim 3072 --experts 64 --topk "$tk" --iters 2 --warmup 1 --staged-dp --staged-parallel; done'`
- Result (TF/s per device in parentheses):
  - `topk=1`: ref `19.052` (`4.763`), staged `21.045` (`5.261`), fused_ep `48.700` (`12.175`)
  - `topk=2`: ref `36.304` (`9.076`), staged `40.809` (`10.202`), fused_ep `87.079` (`21.770`)
  - `topk=4`: ref `66.479` (`16.620`), staged `77.035` (`19.259`), fused_ep `131.548` (`32.887`)
  - `topk=6`: ref `92.423` (`23.106`), staged `109.129` (`27.282`), fused_ep `163.053` (`40.763`)
  - `topk=8`: ref `115.088` (`28.772`), staged `137.846` (`34.462`), fused_ep `159.369` (`39.842`)
- Interpretation:
  - Fused EP stays ahead across all tested top-k values.
  - Advantage is strongest around low/mid top-k and compresses at topk=8.

### 2026-02-09 17:58 - Overlap vs Top-k Sweep (`2048/2048`)
- Hypothesis:
  - Smaller shape may reach a different crossover point where fused EP advantage shrinks or reverses at high top-k.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex5 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 1 2 4 6 8; do echo "=== topk=${tk} ==="; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_mlp_profile.py --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --iters 2 --warmup 1 --staged-dp --staged-parallel; done'`
- Result (TF/s per device in parentheses):
  - `topk=1`: ref `18.438` (`4.609`), staged `20.459` (`5.115`), fused_ep `38.001` (`9.500`)
  - `topk=2`: ref `34.860` (`8.715`), staged `38.006` (`9.502`), fused_ep `63.007` (`15.752`)
  - `topk=4`: ref `63.212` (`15.803`), staged `74.062` (`18.515`), fused_ep `91.191` (`22.798`)
  - `topk=6`: ref `87.009` (`21.752`), staged `104.434` (`26.108`), fused_ep `98.888` (`24.722`)
  - `topk=8`: ref `107.521` (`26.880`), staged `130.640` (`32.660`), fused_ep `96.765` (`24.191`)
- Interpretation:
  - Fused EP leads up through topk=4, then loses to staged DP at topk>=6 on this smaller shape.
  - Crossover behavior suggests shape-dependent overheads in the fused EP path that become unfavorable at very high top-k for 2048/2048.

### 2026-02-09 18:00 - Immediate Direction from Overlap Sweeps
- Takeaways:
  - Overlap/path improvements in fused EP are real at low/mid top-k in both shapes.
  - High top-k behavior is shape-dependent: still positive at 3072/3072, negative at 2048/2048.
  - Next optimization passes should target high-top-k overheads on smaller shapes (where crossover occurs).
- Quick relative view (`fused_ep / staged_dp`):
  - `3072/3072`: `2.31x`, `2.13x`, `1.71x`, `1.49x`, `1.16x` for `topk=1,2,4,6,8`.
  - `2048/2048`: `1.86x`, `1.66x`, `1.23x`, `0.95x`, `0.74x` for `topk=1,2,4,6,8`.

### 2026-02-09 18:18 - Block-Tuning Attempt (Blocked by TPU Reachability)
- Goal:
  - Tune vendored fused EP block params on the crossover case (`2048/2048`, `topk=8`) by calling `fused_ep_moe(..., bt/bf/bd*/btc/bfc/...)` directly.
- Status:
  - Started a candidate sweep job on `dlwh-codex5`, but SSH reachability dropped before execution could proceed (`ssh ... port 22: Operation timed out`).
- Next step:
  - Re-run the direct block-parameter sweep first when TPU SSH recovers; if a better tuple is found, add key-specific entries to `tpu_inference_v1/tuned_block_sizes.py`.

### 2026-02-09 19:40 - Direct Block-Tuning Probe on `codex1` (`2048/2048`, `topk=8`)
- Goal:
  - Test whether explicit fused-kernel block params can recover the high-top-k crossover on the smaller shape.
- Method:
  - Called `fused_ep_moe` directly with fixed routing/logits and manual block tuples.
  - Shape: `tokens=32768`, `hidden=2048`, `mlp_dim=2048`, `experts=64`, `topk=8`, `ep_size=4`.
  - Key baseline from overlap harness for context:
    - `fused_ep`: `96.765 TF/s`
    - `staged_dp`: `130.640 TF/s`
- Results so far:
  - default (`bt=128,bf=1024,bd1=1024,bd2=1024,btc=64,bfc=1024,bd1c=1024,bd2c=1024` implied):
    - `97.455 TF/s` (`24.364/device`)
  - `bt=64,bf=1024,...`:
    - `92.729 TF/s` (`23.182/device`)
  - `bt=64,bf=2048,...`:
    - `100.938 TF/s` (`25.235/device`)
  - `bt=128,bf=2048,...`:
    - `101.305 TF/s` (`25.326/device`)  ← best observed
  - `bt=128,bf=1024,bd2=2048,...`:
    - `99.541 TF/s` (`24.885/device`)
- Interpretation:
  - Block tuning gives a small win (`~+4%` over default fused), but still far below `staged_dp` on this crossover point.
  - This suggests the top-k=8 small-shape gap is not primarily a simple block-size mismatch.
- Notes:
  - One remaining candidate (`bt=256,bf=1024,...`) was interrupted by a host alias/network resolution glitch before completion; rerun when SSH alias is stable.
