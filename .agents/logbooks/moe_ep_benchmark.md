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
### 2026-02-11 10:55 - Jaxformer-Inspired Routing Priority (Negative Result)
- Hypothesis:
  - Jaxformer-style score-priority routing order (sort by expert, then by token top-1 route score descending) might improve pack locality and improve end-to-end EP throughput.
- Change:
  - Added a new routing strategy `expert_score_sort` in:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - Implementation detail:
    - `sort((expert_id, -token_top1_score, flat_pos, tok_id))` over flattened assignments.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for strat in argsort expert_score_sort; do for dist in random runs; do for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy "$strat" --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done; done; done'`
- Result:
  - `argsort, random, topk=2`: `275.605 TF/s`
  - `expert_score_sort, random, topk=2`: `275.424 TF/s` (`-0.07%`)
  - `argsort, random, topk=8`: `444.654 TF/s`
  - `expert_score_sort, random, topk=8`: `443.716 TF/s` (`-0.21%`)
  - `argsort, runs, topk=2`: `276.074 TF/s`
  - `expert_score_sort, runs, topk=2`: `275.819 TF/s` (`-0.09%`)
  - `argsort, runs, topk=8`: `444.437 TF/s`
  - `expert_score_sort, runs, topk=8`: `444.483 TF/s` (`+0.01%`)
- Interpretation:
  - Strategy is performance-neutral within noise and slightly negative on random distribution.
  - Keep `argsort` as default; no evidence this routing reorder improves throughput for this harness.
- Next action:
  - Sweep capacity policies (`drop`/`pad`) to quantify whether bounded expert capacity can improve pack+compute efficiency at acceptable quality tradeoff.

### 2026-02-11 11:06 - Capacity Policy Sweep in EP (Top-k=2)
- Hypothesis:
  - Capacity clipping (`drop`) might reduce local expert work enough to improve EP throughput.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for policy in none drop pad; do for factor in 0.0 1.0 1.25; do if [ "$policy" = "none" ] && [ "$factor" != "0.0" ]; then continue; fi; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 2 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep --capacity-policy "$policy" --capacity-factor "$factor"; done; done'`
- Result:
  - `none, factor=0.0`: `275.019 TF/s`
  - `drop, factor=0.0`: `275.865 TF/s` (equivalent to no-cap path in current code)
  - `drop, factor=1.0`: `275.649 TF/s`, `drop_frac=0.0139`
  - `drop, factor=1.25`: `275.414 TF/s`, `drop_frac=0.0000`
  - `pad`: **unsupported** in EP path (`ValueError: parallel_mode='ep' currently does not support capacity_policy='pad'`)
- Interpretation:
  - `drop` is effectively neutral for throughput in this regime (all deltas within noise).
  - Capacity clipping does not currently unlock a meaningful EP speedup at this shape.
  - `pad` remains unavailable for EP in this harness.
- Next action:
  - Probe scaling with larger expert count (`experts=128`) and fixed `topk` to test whether EP throughput remains flat or regresses as shard-local expert tables grow.

### 2026-02-11 11:12 - Expert Count Scaling (`E=64/96/128`, topk=2/4/8)
- Hypothesis:
  - Increasing number of experts at fixed tokens should lower per-expert token density and may reduce MFU due to less favorable local GEMM shapes.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for e in 64 96 128; do for tk in 2 4 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts "$e" --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done; done'`
- Result:
  - `E=64`: topk2 `275.636`, topk4 `345.629`, topk8 `444.110` TF/s
  - `E=96`: topk2 `244.662`, topk4 `309.934`, topk8 `406.603` TF/s
  - `E=128`: topk2 `220.508`, topk4 `282.142`, topk8 `375.956` TF/s
- Interpretation:
  - Throughput declines monotonically as experts increase at fixed token budget.
  - This supports the practical rule that fewer/larger experts are better for accelerator efficiency at constant quality target and token budget.
  - The top-k trend still holds: larger top-k raises TF/s while lowering tokens/s.
- Next action:
  - Run one larger-token ablation (`tokens=65536`) for `E=128, topk=2/8` to test whether increased token budget can recover part of the lost throughput.

### 2026-02-11 11:24 - Larger Token Budget Ablation (`tokens=65536`, `E=128`)
- Hypothesis:
  - Increasing token budget should raise tokens/expert and recover compute efficiency, especially for low `topk`.
- Command(s):
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 65536 --hidden 2048 --mlp-dim 2048 --experts 128 --topk 2 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep'`
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --tokens 65536 --hidden 2048 --mlp-dim 2048 --experts 128 --topk 8 --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep'`
- Result:
  - `tokens=65536, E=128, topk=2`: `284.255 TF/s`, `792,650 tok/s`
  - `tokens=65536, E=128, topk=8`: `454.414 TF/s`, `559,900 tok/s`
  - Reference at `tokens=32768, E=128`:
    - `topk=2`: `220.508 TF/s`
    - `topk=8`: `375.956 TF/s`
- Interpretation:
  - Larger token budget materially improves throughput for high-expert regime:
    - `topk=2`: `+28.9%`
    - `topk=8`: `+20.9%`
  - This confirms token density per expert is a key lever; EP pays off better at larger effective tokens/expert.
- Next action:
  - Sweep `distribution=runs` for the same large-token shape to verify locality sensitivity stays weak even when token density increases.

### 2026-02-11 11:28 - Locality Check at Larger Token Budget (`tokens=65536`, `E=128`)
- Hypothesis:
  - With higher token density per expert, `runs` locality might improve throughput relative to random.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do for tk in 2 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --tokens 65536 --hidden 2048 --mlp-dim 2048 --experts 128 --topk "$tk" --shared-expert-dim 5632 --shared-fused --backend gmm --impl fused_w13 --routing-pack-strategy argsort --queue-mode full --remat-mode none --bench-pass forward_backward --iters 3 --warmup 1 --parallel-mode ep; done; done'`
- Result:
  - `random, topk=2`: `284.428 TF/s`
  - `runs,   topk=2`: `284.000 TF/s` (`-0.15%`)
  - `random, topk=8`: `454.303 TF/s`
  - `runs,   topk=8`: `453.553 TF/s` (`-0.17%`)
- Interpretation:
  - Even at larger token budget, routing locality as modeled by `runs` remains throughput-neutral in this harness.
  - Current bottlenecks are not meaningfully improved by this synthetic locality signal.
- Next action:
  - If we continue this branch: prioritize experiments that increase effective tokens/expert (batch/sequence/EP scale) and keep top-k low for training-path efficiency.

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

### 2026-02-11 12:42 - MaxText `use_custom_sort_vjp` Pattern Port (Prepared, TPU-Blocked)
- Goal:
  - Test whether MaxText-style permutation VJP helps our backward path (especially sort/unsort portions).
- Code changes:
  - Added a MaxText-style custom VJP for permutation sorts in:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - New knobs:
    - `--dispatch-permute-mode {direct_take,repeat_sort}`
      - `direct_take`: existing `x_repeat_sort = x[token_idx_sort]`.
      - `repeat_sort`: MaxText-style `x_repeat = repeat(x, topk)` then permutation-sort by `sort_idx`.
    - `--use-custom-sort-vjp`
      - Uses explicit backward via inverse permutation (`argsort(sort_indices)`).
  - The non-scatter combine unsort path now optionally routes through the same custom permutation VJP helper.
- Local validation:
  - CPU smoke tests passed for:
    - forward `direct_take`
    - forward `repeat_sort + custom_vjp`
    - forward_backward `repeat_sort + custom_vjp`
- TPU status:
  - Bench collection blocked by infrastructure reachability during this window:
    - `dev-tpu-dlwh-codex1`: timeout/refused on port 22.
    - `dev-tpu-dlwh-codex2`: timeout on port 22.
  - Also observed pre-existing TPU occupancy on codex1 before it became unreachable (`/dev/vfio/* busy`).
- Ready-to-run A/B matrix (once TPU SSH is back):
  - Baseline:
    - `... --dispatch-permute-mode direct_take`
  - Baseline + custom VJP:
    - `... --dispatch-permute-mode direct_take --use-custom-sort-vjp`
  - MaxText-style permutation:
    - `... --dispatch-permute-mode repeat_sort`
  - MaxText-style + custom VJP:
    - `... --dispatch-permute-mode repeat_sort --use-custom-sort-vjp`
  - Suggested fixed args for apples-to-apples:
    - `--distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode none --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 4 --iters 4 --warmup 2 --routing-pack-strategy argsort`

### 2026-02-11 12:55 - MaxText Ring-of-Experts Comm Path Port (Prepared, TPU-Blocked)
- Goal:
  - Add a communication-path A/B for EP high-top-k behavior:
    - Current compact EP path (`psum` over full token outputs).
    - MaxText-style ring path (`all_gather` dispatch + `psum_scatter` collect).
- Code changes:
  - Added `--ep-comm-path {compact_psum,ring_ag_rs}` in:
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `compact_psum`:
    - Existing behavior unchanged.
  - `ring_ag_rs`:
    - Shards token inputs across EP axis.
    - Performs dispatch communication via `lax.all_gather(..., axis_name="ep", tiled=True)`.
    - Runs local-expert compute on gathered tokens.
    - Collects outputs via `lax.psum_scatter(..., axis_name="ep", scatter_dimension=0, tiled=True)`.
    - Applies shared expert on local token shard.
  - Guard:
    - `ring_ag_rs` requires `tokens % ep_size == 0`.
- Local validation:
  - CPU smoke checks pass for:
    - `parallel_mode=ep, ep_comm_path=compact_psum` (forward)
    - `parallel_mode=ep, ep_comm_path=ring_ag_rs` (forward)
    - `parallel_mode=ep, ep_comm_path=ring_ag_rs` (forward_backward)
- TPU status:
  - Could not collect TPU numbers in this window due VM alias reachability timeouts/refusals:
    - `dev-tpu-dlwh-codex1`
    - `dev-tpu-dlwh-codex2`
- Ready-to-run TPU A/B:
  - Baseline:
    - `... --parallel-mode ep --ep-comm-path compact_psum`
  - Ring path:
    - `... --parallel-mode ep --ep-comm-path ring_ag_rs`
  - Suggested high-top-k comparison:
    - `--distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --iters 4 --warmup 2 --routing-pack-strategy argsort`

### 2026-02-11 13:34 - EP Comm Path A/B (High Top-k): `compact_psum` vs `ring_ag_rs`
- Goal:
  - Complete the pending MaxText-style ring communication comparison at high top-k.
- Command(s):
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --iters 4 --warmup 2 --routing-pack-strategy argsort --ep-comm-path compact_psum`
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --iters 4 --warmup 2 --routing-pack-strategy argsort --ep-comm-path ring_ag_rs`
- Result:
  - `compact_psum`: `472.032 TF/s`
  - `ring_ag_rs`: `503.310 TF/s`
  - Delta: `+6.63%` for `ring_ag_rs`.
- Interpretation:
  - Ring-style dispatch/collect path is a real improvement for this high-top-k EP configuration.

### 2026-02-11 13:36 - MaxText Sort-VJP Pattern A/B (Completed)
- Goal:
  - Evaluate `dispatch_permute_mode` and `use_custom_sort_vjp` variants in both non-EP and EP paths.
- Command(s):
  - `parallel_mode=none` matrix on `codex1` and `parallel_mode=ep --ep-comm-path ring_ag_rs` matrix on `codex2`:
    - `dispatch_permute_mode in {direct_take, repeat_sort}`
    - `use_custom_sort_vjp in {off, on}`
  - Fixed args:
    - `--distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --routing-pack-strategy argsort`
- EP result (`ring_ag_rs`):
  - `direct_take, no_vjp`: `504.948 TF/s`
  - `direct_take, custom_vjp`: `503.916 TF/s`
  - `repeat_sort, no_vjp`: `505.079 TF/s`
  - `repeat_sort, custom_vjp`: `504.620 TF/s`
- EP interpretation:
  - All four variants are effectively identical (within noise, <0.3% spread).
- Non-EP result:
  - Initial 4-point matrix showed high variance. Follow-up controlled paired repeats (`iters=5, warmup=2`, `dispatch_permute_mode=direct_take`) were stable:
    - no_vjp: `191.837`, `191.861`, `191.897` TF/s (mean `191.865`)
    - custom_vjp: `216.859`, `216.948`, `216.771` TF/s (mean `216.859`)
  - Delta: `+13.03%` for `custom_vjp` in non-EP backward-inclusive path.
- Non-EP interpretation:
  - `use_custom_sort_vjp` is a strong win in this non-EP forward_backward configuration.
  - For EP high-top-k with ring comm, sort-VJP changes are neutral.

### 2026-02-11 13:49 - True Prequeue A/B (`bench_pass=forward`, `queue_mode=both`)
- Goal:
  - Finish queued prequeue experiment on the real prequeue path (forward-only), across `random` and `runs`.
- Command(s):
  - Non-EP (`codex1`):
    - `... --bench-pass forward --queue-mode both --parallel-mode none --dispatch-permute-mode direct_take`
  - EP ring (`codex2`):
    - `... --bench-pass forward --queue-mode both --parallel-mode ep --ep-comm-path ring_ag_rs --dispatch-permute-mode direct_take`
  - Fixed shape: `tokens=32768 hidden=2048 mlp_dim=2048 experts=64 topk=8`.
- Result (non-EP):
  - `random`: full `210.201`, prequeue `277.222` TF/s (`+31.88%`)
  - `runs`: full `208.477`, prequeue `277.212` TF/s (`+32.97%`)
- Result (EP ring):
  - `random`: full `302.337`, prequeue `303.238` TF/s (`+0.30%`)
  - `runs`: full `302.644`, prequeue `303.643` TF/s (`+0.33%`)
- Interpretation:
  - Prequeue is a major forward-path lever for non-EP at this shape.
  - For EP ring path, prequeue is effectively neutral.

### 2026-02-11 13:56 - EP Comm Path by Top-k (`compact_psum` vs `ring_ag_rs`)
- Goal:
  - Determine whether ring comm advantage is only high-topk or broad across `topk`.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 2 4 8; do for path in compact_psum ring_ag_rs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take --ep-comm-path "$path"; done; done'`
- Result:
  - `topk=2`: compact `198.277`, ring `224.542` TF/s (`+13.25%`)
  - `topk=4`: compact `321.985`, ring `355.305` TF/s (`+10.35%`)
  - `topk=8`: compact `470.598`, ring `503.816` TF/s (`+7.06%`)
- Interpretation:
  - Ring comm path wins across the full tested top-k range, not just at `topk=8`.
  - Relative gain decreases as top-k increases, but remains materially positive.
  - Practical default for EP in this harness should be `ep_comm_path=ring_ag_rs`.

### 2026-02-11 14:00 - Trace Capture: EP `compact_psum` vs `ring_ag_rs` (`topk=8`)
- Goal:
  - Capture Perfetto traces for both EP comm paths on identical config and quantify timeline deltas.
- Harness update:
  - Added `--trace-dir` option to `lib/levanter/scripts/bench/bench_moe_hillclimb.py`.
  - Each benchmark run now optionally wraps `_bench_one_distribution(...)` with `jax.profiler.trace(...)` into a unique subdirectory.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; trace_root=/tmp/moe_ep_comm_traces_20260211_1400; mkdir -p "$trace_root"; for path in compact_psum ring_ag_rs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take --ep-comm-path "$path" --trace-dir "$trace_root/$path"; done'`
- Measured run result (same traced runs):
  - compact: `468.496 TF/s`
  - ring: `502.845 TF/s`
  - delta: `+7.33%` (ring)
- Trace artifacts (local):
  - `/tmp/marin-moe-ep-benchmark/.profiles/moe_ep_comm_traces_20260211_1400/compact_psum/1770847606631578491_random_topk8_fused_w13_preweight_ep_compact_psum_forward_backward_full/plugins/profile/2026_02_11_22_07_23/perfetto_trace.json.gz`
  - `/tmp/marin-moe-ep-benchmark/.profiles/moe_ep_comm_traces_20260211_1400/ring_ag_rs/1770847715481704075_random_topk8_fused_w13_preweight_ep_ring_ag_rs_forward_backward_full/plugins/profile/2026_02_11_22_09_01/perfetto_trace.json.gz`
- Quick timeline comparison:
  - `PjitFunction(loss_fn)` total duration in trace:
    - compact: `22.284 ms`
    - ring: `19.386 ms`
    - savings: `2.898 ms` (`~13.0%`)
- Interpretation:
  - Ring path reduces end-to-end loss step time in this config; trace-level `loss_fn` timing is consistent with benchmark TF/s gain.

### 2026-02-11 14:07 - EP Comm Path by Top-k (No Shared)
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 2 4 8; do for path in compact_psum ring_ag_rs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take --ep-comm-path "$path"; done; done'`
- Result:
  - `topk=2`: compact `198.277`, ring `224.542` TF/s (`+13.25%`)
  - `topk=4`: compact `321.985`, ring `355.305` TF/s (`+10.35%`)
  - `topk=8`: compact `470.598`, ring `503.816` TF/s (`+7.06%`)
- Interpretation:
  - Ring comm wins across `topk=2..8`; advantage shrinks with larger top-k.

### 2026-02-11 14:11 - EP Comm Path by Top-k (Shared Expert Enabled)
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 2 4 8; do for path in compact_psum ring_ag_rs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take --ep-comm-path "$path"; done; done'`
- Result:
  - `topk=2`: compact `276.140`, ring `431.147` TF/s (`+56.14%`)
  - `topk=4`: compact `346.225`, ring `506.276` TF/s (`+46.23%`)
  - `topk=8`: compact `444.874`, ring `599.571` TF/s (`+34.77%`)
- Interpretation:
  - Ring comm has even larger benefit when shared expert is enabled at this shape.

### 2026-02-11 14:16 - Structural Cause Identified in Current Harness
- Code observation:
  - `ep_routed_only` (`compact_psum` path) uses `jax.shard_map` with token input spec `P(None, None)` (replicated token activations across EP devices):
    - `lib/levanter/scripts/bench/bench_moe_hillclimb.py` around EP setup and `ep_routed_only`.
  - `ep_routed_ring` (`ring_ag_rs`) uses token input spec `P("ep", None)` plus explicit `all_gather` dispatch and `psum_scatter` collect.
- Implication:
  - `compact_psum` currently pays for a replicated-token EP path, while ring path is token-sharded and collective-structured.
  - This explains why ring wins broadly and why the gain is especially large with shared expert enabled in this harness.

### 2026-02-11 14:20 - Prequeue Check on Best EP Ring + Shared Path
- Goal:
  - Verify whether prequeue matters on the strongest current path (`ep_comm_path=ring_ag_rs`, shared expert enabled).
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --impl fused_w13_preweight --backend gmm --bench-pass forward --queue-mode both --parallel-mode ep --ep-comm-path ring_ag_rs --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done'`
- Result:
  - `random`: full `359.184`, prequeue `360.864` TF/s (`+0.47%`)
  - `runs`: full `360.481`, prequeue `360.556` TF/s (`+0.02%`)
- Interpretation:
  - Even on the strongest EP ring + shared path, prequeue remains effectively neutral.

### 2026-02-11 14:23 - Non-EP Baseline vs EP Ring (Shared Shape)
- Goal:
  - Add direct non-EP baseline points on the same TPU for the shared shape, to quantify EP ring speedup factors.
- Command (non-EP baseline):
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for tk in 2 4 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode none --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done'`
- Non-EP result:
  - `topk=2`: `234.975 TF/s`
  - `topk=4`: `224.058 TF/s`
  - `topk=8`: `222.883 TF/s`
- EP ring result (from previous section):
  - `topk=2`: `431.147 TF/s`
  - `topk=4`: `506.276 TF/s`
  - `topk=8`: `599.571 TF/s`
- Speedup (EP ring / non-EP):
  - `topk=2`: `1.83x`
  - `topk=4`: `2.26x`
  - `topk=8`: `2.69x`
- Interpretation:
  - On this shared configuration, EP ring is decisively ahead and its relative advantage grows with top-k.

### 2026-02-11 14:27 - `use_custom_sort_vjp` Sanity on Best EP Ring + Shared Path
- Goal:
  - Re-check `use_custom_sort_vjp` on best-performing path after one anomalous low reading.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for rep in 1 2; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --ep-comm-path ring_ag_rs --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --iters 5 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --ep-comm-path ring_ag_rs --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --iters 5 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take --use-custom-sort-vjp; done'`
- Result:
  - no_vjp: `598.585`, `598.083` TF/s
  - custom_vjp: `598.534`, `596.475` TF/s
- Interpretation:
  - Effect is neutral within normal run variance.
  - The earlier `482 TF/s` custom-vjp reading was a bad outlier.

### 2026-02-11 14:35 - Expert Count Scaling at High Top-k (Shared Shape)
- Goal:
  - Evaluate whether EP ring speedup persists as expert count rises.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for e in 64 96 128; do for pm in none ep; do extra=""; if [ "$pm" = ep ]; then extra="--ep-comm-path ring_ag_rs"; fi; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode "$pm" $extra --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts "$e" --topk 8 --shared-expert-dim 5632 --shared-fused --iters 3 --warmup 1 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done; done'`
- Result:
  - `E=64`: non-EP `222.475`, EP ring `599.976` TF/s (`2.70x`)
  - `E=96`: non-EP `215.872`, EP ring `531.705` TF/s (`2.46x`)
  - `E=128`: non-EP `208.583`, EP ring `480.872` TF/s (`2.31x`)
- Interpretation:
  - EP ring remains strongly favorable as experts increase.
  - Absolute TF/s declines with expert count for both paths, but EP ring keeps a large multiplicative advantage.

### 2026-02-11 14:48 - Routing Pack Strategy Sweep on Best Shared Shape
- Goal:
  - Verify whether Jaxformer/MaxText-inspired routing pack ordering variants matter in the current best configuration.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-east5-a.yaml --tpu-name dlwh-codex2 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do for strat in argsort tuple_sort expert_score_sort; do extra=""; if [ "$pm" = ep ]; then extra="--ep-comm-path ring_ag_rs"; fi; uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode "$pm" $extra --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy "$strat" --dispatch-permute-mode direct_take; done; done'`
- Result:
  - non-EP:
    - `argsort`: `222.379 TF/s`
    - `tuple_sort`: `222.097 TF/s`
    - `expert_score_sort`: `222.040 TF/s`
  - EP ring:
    - `argsort`: `595.189 TF/s`
    - `tuple_sort`: `597.717 TF/s`
    - `expert_score_sort`: `598.512 TF/s`
- Interpretation:
  - non-EP is effectively identical across routing pack strategy variants.
  - EP ring shows only a small spread (`~0.6%` from lowest to highest). No major routing-pack win here.

### 2026-02-11 15:03 - Random vs Runs Distribution on Same Hardware (codex1)
- Goal:
  - Confirm whether run-locality in assignments materially changes throughput for the current best shared shape.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for pm in none ep; do extra=""; if [ "$pm" = ep ]; then extra="--ep-comm-path ring_ag_rs"; fi; for dist in random runs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode "$pm" $extra --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk 8 --shared-expert-dim 5632 --shared-fused --iters 5 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done; done'`
- Result:
  - non-EP:
    - `random`: `223.113 TF/s`
    - `runs`: `222.789 TF/s` (`-0.15%`)
  - EP ring:
    - `random`: `596.961 TF/s`
    - `runs`: `600.937 TF/s` (`+0.67%`)
- Interpretation:
  - Throughput is effectively insensitive to this random-vs-runs switch at this shape in both non-EP and EP ring paths.
  - The synthetic run-locality here mainly changes run statistics, not end-to-end step throughput.

### 2026-02-11 15:16 - EP Ring + Shared Top-k Sweep (Random vs Runs)
- Goal:
  - Re-check top-k scaling and locality interaction under the best current EP path (`ring_ag_rs`) with shared expert enabled.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for dist in random runs; do for tk in 1 2 4 6 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution "$dist" --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --ep-comm-path ring_ag_rs --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done; done'`
- Result (EP ring + shared):
  - random:
    - `topk=1`: `379.192 TF/s`
    - `topk=2`: `428.870 TF/s`
    - `topk=4`: `505.270 TF/s`
    - `topk=6`: `560.017 TF/s`
    - `topk=8`: `598.993 TF/s`
  - runs:
    - `topk=1`: `378.953 TF/s`
    - `topk=2`: `428.919 TF/s`
    - `topk=4`: `504.782 TF/s`
    - `topk=6`: `559.219 TF/s`
    - `topk=8`: `598.334 TF/s`
- Interpretation:
  - Top-k scaling remains monotonic in this harness at fixed shape (higher top-k => higher measured TF/s, due larger counted MoE FLOP volume).
  - Random vs runs is again effectively neutral across top-k on this path.

### 2026-02-11 15:18 - EP Comm Path Sweep Expanded to `topk={1,2,4,6,8}` (Shared)
- Goal:
  - Extend the EP comm-path comparison to include `topk=1` and `topk=6` on the same host/hardware.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for path in compact_psum ring_ag_rs; do for tk in 1 2 4 6 8; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --ep-comm-path "$path" --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts 64 --topk "$tk" --shared-expert-dim 5632 --shared-fused --iters 4 --warmup 2 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done; done'`
- Result (random, shared):
  - `topk=1`: compact `233.426`, ring `379.055` TF/s (`+62.4%`)
  - `topk=2`: compact `276.195`, ring `429.483` TF/s (`+55.5%`)
  - `topk=4`: compact `346.406`, ring `505.240` TF/s (`+45.9%`)
  - `topk=6`: compact `401.536`, ring `559.235` TF/s (`+39.3%`)
  - `topk=8`: compact `446.163`, ring `598.411` TF/s (`+34.1%`)
- Interpretation:
  - Ring path wins for every tested top-k, including low-top-k where the relative gain is largest.
  - Relative ring advantage shrinks as top-k grows, but remains material throughout the tested range.

### 2026-02-11 15:37 - Expert Count Sweep for EP Comm Paths (Shared, `topk=8`)
- Goal:
  - Check whether ring-vs-compact advantage persists as expert count increases.
- Command:
  - `RAY_AUTH_MODE=token uv run scripts/ray/dev_tpu.py --config infra/marin-us-central1.yaml --tpu-name dlwh-codex1 execute -e LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=50000" -- /bin/bash -lc 'set -euo pipefail; for e in 64 96 128; do for path in compact_psum ring_ag_rs; do uv run --package levanter --extra tpu lib/levanter/scripts/bench/bench_moe_hillclimb.py --distribution random --impl fused_w13_preweight --backend gmm --bench-pass forward_backward --queue-mode full --parallel-mode ep --ep-comm-path "$path" --tokens 32768 --hidden 2048 --mlp-dim 2048 --experts "$e" --topk 8 --shared-expert-dim 5632 --shared-fused --iters 3 --warmup 1 --routing-pack-strategy argsort --dispatch-permute-mode direct_take; done; done'`
- Result:
  - `E=64`: compact `446.015`, ring `598.960` TF/s (`+34.3%`)
  - `E=96`: compact `408.888`, ring `534.409` TF/s (`+30.7%`)
  - `E=128`: compact `377.118`, ring `480.561` TF/s (`+27.4%`)
- Interpretation:
  - Ring remains clearly better as experts increase.
  - Absolute throughput drops as `E` increases for both paths, and ring’s relative margin narrows somewhat, but remains large.

### 2026-02-11 14:58 - TPU Availability Note (codex2)
- During a `random/runs` comparison run on `dlwh-codex2`, SSH to `34.42.114.33` was closed/refused mid-experiment.
- Recovery:
  - Re-ran the full matrix on `dlwh-codex1` (`infra/marin-us-central1.yaml`) and recorded final numbers above.

### 2026-02-11 16:48 - Mixtral-like Block Push Toward 1101 TF/s (Forward+Backward)
- Goal:
  - Push Mixtral-like MoE block throughput to `>=1101 TF/s`.
- Shape target:
  - `hidden=4096, mlp_dim=14336, experts=8, topk=2`, `shared_expert_dim=0`, backend `gmm`.
- Initial comm-path baseline (`tokens=32768`, `impl=fused_w13_preweight`, random):
  - non-EP: `316.611 TF/s`
  - EP compact: `801.379 TF/s`
  - EP ring: `859.113 TF/s`
- Token scaling (EP ring, `impl=fused_w13_preweight`):
  - `32768`: `859.003 TF/s`
  - `49152`: `941.925 TF/s`
  - `65536`: `980.645 TF/s`
  - `98304`: `1034.277 TF/s`
  - `131072`: `1054.411 TF/s`
- Impl sweep at `tokens=131072` (EP ring):
  - `baseline`: `1070.775 TF/s`
  - `fused_w13`: `1057.049 TF/s`
  - `preweight`: `1071.492 TF/s`
  - `fused_w13_preweight`: `1059.256 TF/s`
  - `scatter`: `1074.243 TF/s`  ← best
  - `fast`: `1050.897 TF/s`
- High-token sweep using best impl (`scatter`, EP ring):
  - `131072`: `1072.849 TF/s`
  - `163840`: `1084.843 TF/s`
  - `196608`: `1097.746 TF/s`
  - `212992`: `1094.071 TF/s`
  - `229376`: `1099.646 TF/s`
  - `245760`: `1102.002 TF/s`  ✅ target met
  - `262144`: `1109.714 TF/s`  ✅ target exceeded
- Routing-pack strategy check at `tokens=196608` (`scatter`, EP ring):
  - `argsort`: `1093.089 TF/s`
  - `tuple_sort`: `1095.387 TF/s`
  - `expert_score_sort`: `1094.950 TF/s`
  - Interpretation: negligible spread (wash).
