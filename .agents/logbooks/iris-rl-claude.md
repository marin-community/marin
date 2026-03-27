# Iris RL Performance Analysis — Claude Session

Source: conversation with Ahmed, 2026-03-25
Context: continuing from `.agents/logbooks/iris-rl-codex.md` (Codex agent's Iris RL migration work)

## Run Under Investigation

- **Root job**: `/ahmed/irl-e4p-100r4-0325-1023`
- **Branch**: `iris_rl`
- **Experiment script**: `experiments/exp_iris_rl_regression_direct_gcs_prod.py`
- **W&B project**: `marin_iris_rl_debug`
- **Train W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-172718-train
- **Rollout W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-172718-rollout-0

## R4 Status Check (2026-03-25 ~18:25 UTC)

All three components healthy and running:

| Component | State | Details |
|---|---|---|
| Root | RUNNING | 0 failures, 0 preemptions |
| Trainer | RUNNING | Step 57/100, attempt 0, no retries |
| Rollout | RUNNING | 1 preemption (heartbeat timeout, recovered), attempt 1 |

Trainer memory: 204 GiB current, **328 GiB peak** — under the 400 GiB cap (bumped from 300 GiB after r3 OOMs).

Training metrics at step 56:
- Loss (REINFORCE): -0.000568
- Eval pass@1 (MATH-500): 47.8%
- Train reward mean: 0.695
- MFU median: 90.4%
- Weight transfers: 58/58 successful

## Performance Comparison: Iris r4 vs On-Demand nb-inflight2

### Wall-clock per step

| Metric | **nb-inflight2** (on-demand) | **Iris r4** (current) | Ratio |
|---|---|---|---|
| Wall-clock/step (median) | **2.3 min** | **7.9 min** | 3.4x slower |
| Inter-step idle (median) | 1.1 min | 6.7 min | 6.1x slower |
| Train step duration | 67s | 61s | Iris is faster |
| fwd/bwd | 54s | 37s | Iris is faster |
| batch_prep | 21s | 31s | |
| weight_transfer | 15s | 16s | ~same |

The train-side compute is actually *faster* on Iris. The entire 3.4x wall-clock gap is rollout-side.

### Rollout configuration comparison

| Setting | **nb-inflight2** (exp2039) | **Iris r4** (exp_iris_rl_regression_direct_gcs_prod) |
|---|---|---|
| `n_prompts` | **64** | **16** |
| `n_generations_per_prompt` | 16 | 16 |
| vLLM `n` | 8 | 8 |
| Responses per rollout batch | **1024** (64 x 16) | **256** (16 x 16) |
| `train_batch_size` | 1024 | 1024 |
| Rollout batches per train step | **1** | **4** |
| `eval_frequency` | 1 | 1 |
| Eval set | 500 (MATH-500) | 500 (MATH-500) |
| `max_rollout_step_delay` | 1 | 1 |
| `replay_buffer.max_samples` | 1 | 1 |
| Rollout throughput (tok/s) | 3,602 | 3,249 |

Both configs were found in:
- nb-inflight2: `experiments/exp2039_rl_math500.py` (launched with `--rollout-shape exp2039`), config via `RLExperimentConfig` in `lib/marin/src/marin/rl/rl_experiment_utils.py` — `n_prompts=64`, `inference_n=8`
- Iris r4: `experiments/exp_iris_rl_regression_direct_gcs_prod.py` — `n_prompts=16`, vLLM `n=8`

### What the trainer sees per step (identical in both runs)

| | nb-inflight2 | Iris r4 |
|---|---|---|
| Total samples | 1024 | 1024 |
| Unique questions | 64 | 64 (16 x 4 batches) |
| Group size | 16 | 16 |
| IS ratio mean | 0.965 | 0.910 |

**Both runs are equally on-policy.** With `max_rollout_step_delay=1` and `max_samples=1`, the replay buffer rejects rollouts more than 1 trainer step stale. The 4 small Iris batches are all generated against the same weight checkpoint (the rollout worker fetches weights once, then produces batches before the trainer advances). The slight IS ratio difference is variance, not staleness.

## Root Cause: eval_frequency interacts badly with small n_prompts

**Identified at `lib/marin/src/marin/rl/rollout_worker.py:841`:**

```python
if step % self.config.curriculum_config.eval_frequency == 0:
```

`step` here is the rollout worker's internal counter, which increments once per rollout batch — NOT once per trainer step.

With `eval_frequency=1`:
- **nb-inflight2**: 1 rollout batch per train step → 1 eval per train step (correct)
- **Iris r4**: 4 rollout batches per train step → **4 evals per train step** (4x too many)

Each MATH-500 eval takes ~67s through vLLM. So Iris r4 spends ~4.5 min per train step on redundant evals. This is the dominant cause of the 5.6 min inter-step idle gap.

**Secondary overhead**: 4 small batches also means 4x the GCS write/read round-trips, 4x the weight transfer polling, and 4x the vLLM inference pass startup overhead.

## W&B Evidence

**Iris r4 rollout** logged 66 steps over 57 train steps, each with a 500-problem eval. `cumulative_batch_count` incremented by ~3-4 between each logged step, confirming ~3-4 training rollout batches per eval.

**nb-inflight2 rollout** logged only 15 steps over 185 train steps. `cumulative_batch_count` jumped by ~10 between logs. Far fewer evals relative to training work.

## Recommendations

### High impact, simple change
1. **Set `n_prompts=64`** in `exp_iris_rl_regression_direct_gcs_prod.py` (match nb-inflight2). This makes 1 rollout batch = 1024 samples = 1 train step, so `eval_frequency=1` naturally aligns to 1 eval per train step. **Expected speedup: ~3x.**

### Medium impact
2. **Or set `eval_frequency=4`** as a quick fix to keep small rollout batches but restore 1 eval per train step.

### Code-level fix (longer-term)
3. **Make `eval_frequency` count trainer steps, not rollout steps** in `rollout_worker.py:841`. The current semantics are a footgun — the rollout worker's internal step counter is an implementation detail that shouldn't determine eval cadence. This prevents the problem from recurring whenever someone changes `n_prompts`.

### Nice-to-have
4. Consider bumping vLLM `n` from 8 to 16 to match `n_generations_per_prompt=16`, eliminating the double vLLM call per prompt. (Needs memory testing.)

## Reference Runs in marin_post_training

| Run | W&B Links | State | Steps | Wall-clock/step | Notes |
|---|---|---|---|---|---|
| `exp2039nb-20260319-032500` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500-train) / [rollout](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500) | finished | 500 | 3.4 min (mean) | 28.5h total, successful |
| `exp2039-nb-inflight2` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2-train) / [rollout](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2) | crashed | 185 | 2.3 min (median) | fastest, inflight weight updates |
| `exp2039-20260318-040100` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-040100) | finished | ? | ? | memory profiling baseline |

## Validation: Iris e4par run (n_prompts=64, inflight updates)

A new Iris run was launched with `n_prompts=64` and inflight weight updates, matching nb-inflight2's config:

- **Train W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-train
- **Rollout W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-rollout-0
- **State**: running (as of 2026-03-26 ~06:00 UTC)
- **Steps completed**: 21/100 in 1.0 hours

### Timing confirms the fix

| Metric | **Iris r4** (n_prompts=16) | **Iris e4par** (n_prompts=64) | **nb-inflight2** (on-demand) |
|---|---|---|---|
| Wall-clock/step (median) | 7.9 min | **2.3 min** | 2.3 min |
| Inter-step idle (median) | 6.7 min | **1.3 min** | 1.1 min |
| Train step duration (mean) | 61s | 75s | 67s |
| batch_prep | 31s | 23s | 21s |
| fwd_bwd | 37s | 53s | 54s |
| weight_transfer | 16s | 15s | 15s |

**3.4x speedup confirmed.** The Iris e4par run matches nb-inflight2's wall-clock pace exactly (2.3 min/step median). The eval overhead is eliminated — with 1 rollout batch per train step, `eval_frequency=1` now correctly means 1 eval per train step.

### Rollout metrics

- Throughput: 3,636 tok/s, 8.3 req/s (faster than r4's 3,249 tok/s — larger batches amortize overhead)
- Rollout batch count: 23 batches for 21 train steps (~1:1 as expected)
- Train responses per batch: 1024 (64 prompts x 16 gens)
- Eval pass@1: **49.4%** at step 21 (tracking well)
- Train reward mean: 0.661
- IS ratio: 0.977 (very on-policy)

### Key config differences from r4

| Setting | r4 | e4par |
|---|---|---|
| `n_prompts` | 16 | **64** |
| Responses per rollout batch | 256 | **1024** |
| Rollout batches per train step | 4 | **1** |
| Evals per train step | 4 | **1** |

### ETA

At 2.3 min/step with 79 steps remaining: **~3 hours** (vs ~5.7 hours for r4).

## Multi-Sampler Run: e4ms2 (1 trainer + 2 rollout workers)

Launched 2026-03-26 with `n_prompts=64`, inflight weight updates, and **2 rollout workers** instead of 1.

- **Train W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-train
- **Rollout-0 W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-0
- **Rollout-1 W&B**: https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-1
- **State**: running, step 125/200 after 4.3 hours
- **Config**: `train_batch_size=1024`, `n_prompts=64`, `n_generations_per_prompt=16`, vLLM `n=8`

### Timing: fastest Iris run yet

| Metric | **Iris r4** (1 rollout, n=16) | **Iris e4par** (1 rollout, n=64) | **Iris e4ms2** (2 rollout, n=64) | **nb-inflight2** (on-demand) |
|---|---|---|---|---|
| Wall-clock/step (median) | 7.9 min | 2.3 min | **1.7 min** | 2.3 min |
| Inter-step idle (median) | 6.7 min | 1.3 min | **0.3 min** | 1.1 min |
| Train step duration (mean) | 61s | 75s | 71s | 67s |
| batch_prep | 31s | 23s | 11s | 21s |
| fwd_bwd | 37s | 53s | 60s | 54s |
| weight_transfer | 16s | 15s | 16s | 15s |

**4.6x faster than r4, 1.4x faster than the best on-demand run.** The second rollout worker nearly eliminates idle time — the trainer only waits 20s (median) between steps. Batch prep dropped from 23s to 11s since rollout data is always ready by the time the trainer needs it.

### Rollout worker metrics

| | rollout-0 | rollout-1 |
|---|---|---|
| Batches produced | 75 | 106 |
| Weight receives | 126 | 125 |
| Throughput (tok/s) | 3,339 | 3,262 |
| Throughput (req/s) | 6.5 | 6.4 |
| Batch time (s) | 157s | 159s |
| Train correct accuracy | 64.6% | 58.6% |
| Train mean reward | 0.630 | 0.570 |

Both workers running at similar throughput. rollout-1 has produced more training batches (106 vs 75) — rollout-0 also runs eval (pass@1: **46.8%** at step 124).

### Training metrics at step 125

- Loss (REINFORCE): -0.000513
- Policy entropy: 1.23
- IS ratio: 0.931
- MFU median: 90.3%
- Weight transfers: 127/127 successful (0 failures)

### ETA

At 1.7 min/step with 75 steps remaining: **~2.1 hours**.

## Replay Buffer Mechanics and Rollout Overproduction Analysis

Source: conversation between Ahmed and Codex agent, 2026-03-26.

### Replay buffer config (e4ms2 run)

| Setting | Value | Meaning |
|---|---|---|
| `max_samples` | 1 | Each rollout can be used in at most 1 trainer batch, then retired |
| `max_rollout_step_delay` | 1 | Rollouts only eligible while within 1 trainer step of freshness (step K data survives through trainer steps K+1 and K+2) |
| `capacity` | 4096 | Buffer holds at most 4 full minibatches |
| `train_batch_size` | 1024 | Trainer samples 1024 rollouts per step |

Code locations:
- `max_samples` retirement: `lib/marin/src/marin/rl/replay_buffer.py:170`
- Freshness check: `lib/marin/src/marin/rl/replay_buffer.py:125`
- Recency-biased sampling: `lib/marin/src/marin/rl/replay_buffer.py:279`

### Two-sampler prompt behavior

The two rollout workers do **not** coordinate or partition prompts. Each independently:
1. Has its own seed (`orchestration.py:176`)
2. Samples a lesson from the shared curriculum actor (`rollout_worker.py:858`)
3. Independently samples `n_prompts` examples from that lesson (`math_env.py:149`)

So both workers draw from the same distribution with different random seeds. Prompt overlap is possible by chance, but there is no explicit sharding.

### Overproduction quantification (at step 167)

| Metric | Value |
|---|---|
| Train steps completed | 167 |
| Rollout batches produced (rollout-0) | 97 |
| Rollout batches produced (rollout-1) | 141 |
| Total rollout batches produced | 238 |
| Production rate | **1.43 batches/trainer step** |
| Total rollouts produced | 238 × 1024 = **243,712** |
| Total rollouts consumed | 167 × 1024 = **171,008** |
| Excess produced | **72,704** rollouts (70 extra minibatches) |
| Min discarded (accounting for buffer residue) | 72,704 − 4,096 = **68,608** |
| Max discarded | **72,704** |
| **Waste percentage** | **~29-30%** of produced rollout data |

This waste is **not a bug** — it is the expected cost of overproducing to keep the trainer near saturation with `max_samples=1` and `max_rollout_step_delay=1`.

### Trainer bottleneck decomposition (at step 167)

| Metric | Median | Mean | p90 | Max |
|---|---|---|---|---|
| Wall-clock/step | 101.5s | — | 205.8s | 273.8s |
| Step duration (compute) | 61.1s | — | — | — |
| Hook time (weight transfer) | 15.4s | — | 18.7s | 36.6s |
| Batch prep (rollout wait) | 4.0s | 12.3s | 47.4s | — |
| MFU | 90.3% | — | — | — |

Key insight: **median batch prep is only 4.0s** — on typical steps the trainer is already well-fed. The problem is the **fat tail**: 38.5% of steps have residual >10s, 20% >40s, 13% >60s. Tail spikes correlate with checkpoint saves (steps 83, 89, 95, 101, 107, 114, 120, 126).

**Trainer-side floor** (compute + hooks): ~76.5s/step. Maximum possible speedup from better rollout supply alone: **101.5 / 76.5 = 1.33x**.

### Critical finding: `max_samples=2` with `delay=1` does NOT help

This was the key non-obvious result from the analysis:

**With current `delay=1`, step-K rollouts are already eligible for two trainer batch constructions** (steps K+1 and K+2). That gives 2 × 1024 = 2048 sample slots — exactly matching the 2048 unique rollouts produced by 2 samplers at step K.

So with `max_samples=1`, the system can **already** consume all unique step-K data across those two updates. The second sampler's batch is not being wasted due to `max_samples=1` — it's being wasted due to **sustained overproduction** (1.43 batches/step on average) outpacing consumption.

Changing to `max_samples=2, delay=1` would:
- **Not** create extra usable capacity (still only 2 trainer updates before K ages out)
- Allow **resampling already-used rollouts** before exhausting unused leftovers
- Actually **worsen** unique-sample utilization
- Increase off-policy-ness for no throughput gain

`max_samples=2` only becomes meaningful with `delay=2+`, which creates 3+ trainer updates (3072+ slots for 2048 unique rollouts), enabling real reuse. But even then, the gain is small because average supply already exceeds demand.

### Recommended knobs for reducing waste / improving throughput (ordered by impact)

1. **Move eval off the critical sampler path** — rollout-0 averages 213.5s/batch vs rollout-1's 149.1s/batch because rollout-0 carries eval. Freeing it would bring combined supply to ~74.6s/batch, nearly matching the 76.5s trainer floor.
2. **Reduce checkpoint frequency/cost** — checkpoint saves dominate the worst wall-clock spikes (p90 = 205.8s vs median = 101.5s).
3. **Reduce per-sampler batch size** (64→48 or 32 prompts) — directly reduces overproduction without losing trainer saturation.
4. **Add a third sampler** — would saturate the trainer even with current eval placement, but upside is bounded by the 76.5s floor.

### The throughput-vs-waste tradeoff

Throughput and waste are fundamentally at odds in this architecture:

- **To maximize throughput**: the trainer must never wait for data. That requires rollouts to be ready *before* the trainer asks, which means samplers must produce faster than the trainer consumes — i.e., overproduce. Overproduction = waste.
- **To minimize waste**: production should exactly match consumption (1:1 ratio). But samplers have variable latency (vLLM jitter, eval runs, network), so producing at exactly 1:1 means the trainer stalls on bad-luck steps.

The 2-sampler setup chose throughput: median batch-prep wait dropped from 23s (1 sampler) to 4s (2 samplers), but ~30% of rollout inference compute is discarded.

**The tradeoff curve** (estimated from e4ms2 data, `n_prompts` as the tuning knob):

| Strategy | Wall-clock/step | Waste | Notes |
|---|---|---|---|
| 1 sampler, n_prompts=64 | 2.3 min | ~0% | No overproduction, trainer waits on rollout |
| 2 samplers, n_prompts=32 | ~2.0 min (est.) | ~5% (est.) | Minimal overproduction |
| 2 samplers, n_prompts=48 | ~1.8 min (est.) | ~15% (est.) | Moderate tradeoff |
| 2 samplers, n_prompts=64 | 1.7 min | ~30% | Current config, trainer near saturation |

Reducing per-sampler `n_prompts` (64→48 or 32) is the knob that lets you slide along this curve — less overproduction, slightly more trainer wait, a middle ground on both axes. The right point depends on whether you're optimizing for TPU-hour cost (sample efficiency) or wall-clock time (getting results faster).

### Remaining theoretical headroom

- Current median: 101.5s/step
- Trainer floor: 76.5s/step
- Max additional speedup from supply-side saturation: **~1.33x**
- Next gains are **incremental**, not another 4.6x jump like the original n_prompts fix

## Reference Runs

### marin_iris_rl_debug (Iris runs)

| Run | W&B Links | State | Steps | Wall-clock/step | Notes |
|---|---|---|---|---|---|
| `e4ms2-20260326-121919` | [train](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-train) / [rollout-0](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-0) / [rollout-1](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-1) | running | 125 | **1.7 min** (median) | 2 rollout workers, n_prompts=64, **fastest** |
| `e4par-20260326-044831` | [train](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-train) / [rollout](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4par-20260326-044831-rollout-0) | running | 21 | **2.3 min** (median) | n_prompts=64, inflight updates, validates the batch size fix |
| `e4p-20260325-172718` (r4) | [train](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-172718-train) / [rollout](https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4p-20260325-172718-rollout-0) | running | 57 | 7.9 min (median) | n_prompts=16, 4x eval overhead |

### marin_post_training (on-demand Ray runs)

| Run | W&B Links | State | Steps | Wall-clock/step | Notes |
|---|---|---|---|---|---|
| `exp2039nb-20260319-032500` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500-train) / [rollout](https://wandb.ai/marin-community/marin_post_training/runs/exp2039nb-20260319-032500) | finished | 500 | 3.4 min (mean) | 28.5h total, successful |
| `exp2039-nb-inflight2` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2-train) / [rollout](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-nb-inflight2) | crashed | 185 | 2.3 min (median) | fastest on-demand, inflight weight updates |
| `exp2039-20260318-040100` | [train](https://wandb.ai/marin-community/marin_post_training/runs/exp2039-20260318-040100) | finished | ? | ? | memory profiling baseline |

## Recommended Pure-Inference Experiment Ladder (2026-03-26)

Source: Codex agent analysis (`.agents/logbooks/iris-rl-codex.md`)

For the TP-vs-packed-DP question, the next experiments should be **pure inference first**, not full RL:
- RL end-to-end step time has ~0.42 CV from trainer hooks, checkpoints, replay timing, and eval cadence
- Rollout-storage write time is only ~0.31s median per batch — negligible next to 163-228s rollout batch times
- 20-30 rollout-equivalent batches per condition is enough for the first pass

### INF-001: TP=4 baseline, current sampler shape

Purpose: establish the real single-engine inference baseline under current RL-like prompt/response shape.

Config:
- `TP=4`, current `max_model_len=2048`
- Current batch shape equivalent to RL sampler load
- Enable `kv_cache_metrics`
- Log prompt-token counts and response-token counts
- Run 20-30 rollout-equivalent batches after warmup

Measure: batch wall time, tokens/sec, HBM per chip, KV cache metrics, prompt/response token counts, total live-sequence pressure.

### INF-002: TP=4 cache-budget sweep

Purpose: determine whether current KV cache reservation is oversized and how much it can be reduced before throughput degrades.

Config sweep:
- Hold topology fixed at `TP=4`
- Vary `gpu_memory_utilization`: 0.90, 0.80, 0.70
- Optionally lower `max_model_len` only if cache metrics imply large unused slack at 2048

Measure: same as INF-001, especially whether throughput stays flat while cache reservation falls.

**Decision point**: if reducing cache budget does not hurt throughput much, that is evidence current TP=4 reservation is oversized and packed layouts become more plausible.

### INF-003: TP=2 single-engine feasibility run

Purpose: measure the actual cost of moving from TP=4 to TP=2 before attempting any co-location.

Config: one engine only, `TP=2`, same prompt/response shape, `kv_cache_metrics` enabled.

Measure: batch wall time, tokens/sec, HBM per chip, cache behavior.

**Decision point**: if single-engine TP=2 already looks bad on throughput or HBM margin, stop — no reason to attempt packed TP=2x2.

### INF-004: Packed TP=2 / DP=2-style co-location

Purpose: test the real hypothesis — two smaller independent inference replicas on one v5p-8 may beat one TP=4 engine.

Config:
- Two independent `TP=2` engines on one `v5p-8`
- Same total request load as INF-001
- Route requests across both replicas

Measure: aggregate tokens/sec, per-replica tokens/sec, HBM headroom per chip, cache behavior per replica, stability / OOM margin.

**Decision point**: if aggregate throughput improves and HBM remains safe, packed TP=2x2 is the better inference topology. If not, keep TP=4.

### INF-005: Short RL validation (only after an inference winner exists)

Purpose: confirm the inference winner actually improves trainer feed rate in the real loop.

Config: short RL run (~30-40 train steps), push checkpoints out of measurement window, minimize eval noise.

Measure: trainer batch-prep time, train step wall time, end-to-end throughput.

### Recommended order

1. INF-001 — **COMPLETED** (see results below)
2. INF-002
3. INF-003
4. INF-004 — only if INF-003 is promising
5. INF-005 — only after an inference winner exists

## INF-001 Results: TP=4 Baseline (2026-03-26)

- **Iris job**: `/ahmed/inf-001-tp4-0326-r4` (child: `/ahmed/inf-001-tp4-0326-r4/inf-001-inf001-tp4-20260326-195258`)
- **W&B run**: `inf001-tp4-20260326-195258` in `marin_iris_rl_debug`
- **Script**: `experiments/exp_inf_001_tp4_baseline.py`
- **Status**: SUCCEEDED

### Config

- Model: Llama 3.1 8B Instruct (`gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`)
- TPU: v5p-8, TP=4, `max_model_len=2048`, `gpu_memory_utilization=0.90`
- Batch shape: 64 prompts × 16 generations = 1024 completions/batch
- SamplingParams: `temperature=1.0, max_tokens=1024, top_k=4096, stop=["<|eot_id|>"], logprobs=1`
- Batches: 25 total (3 warmup + 22 measured)
- Prompts: MATH training set with standard few-shot prefix

### Summary

| Metric | Median | Mean | p10 | p90 |
|---|---|---|---|---|
| **Output tok/s** | **3,345** | 3,340 | 3,292 | 3,405 |
| **Batch time (s)** | **108.4** | 108.9 | 99.8 | 122.5 |

- **HBM after model load**: 86.2 GiB (max chip)
- **KV cache metrics**: enabled (`kv_cache_metrics=True`)
- **Throughput CV**: ~3.4% — very stable across batches

### Per-batch detail

| Batch | Time (s) | Output tokens | tok/s |
|---|---|---|---|
| 1 (warmup) | 239.5 | 382,647 | 1,598 |
| 2 (warmup) | 116.6 | 394,275 | 3,381 |
| 3 (warmup) | 104.7 | 350,581 | 3,349 |
| 4 | 117.7 | 366,087 | 3,110 |
| 5 | 131.9 | 419,002 | 3,177 |
| 6 | 118.0 | 391,893 | 3,321 |
| 7 | 110.0 | 372,300 | 3,385 |
| 8 | 100.1 | 329,415 | 3,291 |
| 9 | 99.3 | 334,471 | 3,367 |
| 10 | 100.7 | 333,287 | 3,308 |
| 11 | 99.9 | 332,636 | 3,329 |
| 12 | 119.3 | 406,226 | 3,406 |
| 13 | 113.9 | 386,635 | 3,394 |
| 14 | 113.1 | 375,479 | 3,319 |
| 15 | 112.9 | 379,509 | 3,362 |
| 16 | 102.9 | 342,610 | 3,328 |
| 17 | 110.3 | 382,226 | 3,465 |
| 18 | 106.9 | 352,780 | 3,301 |
| 19 | 103.1 | 354,102 | 3,434 |
| 20 | 124.2 | 409,888 | 3,300 |
| 21 | 100.0 | 339,751 | 3,398 |
| 22 | 106.4 | 362,189 | 3,403 |
| 23 | 101.2 | 342,888 | 3,388 |
| 24 | 90.0 | 299,515 | 3,329 |
| 25 | 112.0 | 378,720 | 3,383 |

### Key observations

1. **Throughput matches RL rollout workers**: e4ms2 rollout workers measured ~3,300 tok/s; pure inference gets 3,345 tok/s. The <2% difference confirms rollout overhead (grading, serialization) is negligible.
2. **Batch time variance driven by response length**: batch times range 90-132s because MATH problems produce variable-length responses (299k-419k output tokens per batch).
3. **HBM headroom**: 86.2 GiB per chip out of ~95 GiB available on v5p — only ~9 GiB free. This is relevant for INF-002 (cache budget sweep) and INF-003/INF-004 (TP=2 feasibility).
4. **Warmup**: first batch includes JIT compilation (239.5s, 1,598 tok/s). By batch 2, throughput is at steady state.

## INF-002: TP=4 Cache-Budget Sweep (2026-03-26)

Purpose: determine whether current KV cache reservation is oversized by sweeping `gpu_memory_utilization` (0.80, 0.70, 0.60) while holding TP=4 fixed. INF-001 baseline used 0.90.

### Jobs launched (in parallel)

| `gpu_memory_utilization` | Iris root job | Status |
|---|---|---|
| 0.80 | `/ahmed/inf-002-gpu80-0326` | RUNNING |
| 0.70 | `/ahmed/inf-002-gpu70-0326` | RUNNING |
| 0.60 | `/ahmed/inf-002-gpu60-0326` | RUNNING |

Config identical to INF-001 except `gpu_memory_utilization`:
- Model: Llama 3.1 8B Instruct, TP=4, v5p-8, `max_model_len=2048`
- Batch shape: 64 prompts × 16 generations = 1024 completions/batch
- 25 batches (3 warmup + 22 measured)
- Script: `experiments/exp_inf_001_tp4_baseline.py --gpu-memory-utilization <value>`

### Decision criteria

- If throughput stays flat as `gpu_memory_utilization` drops (0.90→0.80→0.70→0.60), KV cache is oversized → packed TP=2 layouts become plausible
- If throughput degrades or jobs OOM, that sets the floor for cache budget
- Compare HBM per chip across all four points to quantify freed memory

### Results

All three jobs **SUCCEEDED**.

| `gpu_memory_utilization` | HBM per chip (GiB) | Median tok/s | p10-p90 tok/s | Median batch time (s) | W&B run |
|---|---|---|---|---|---|
| 0.90 (INF-001 baseline) | 86.2 | 3,345 | 3,292-3,405 | 108.4 | `inf001-tp4-20260326-195258` |
| **0.80** | **76.6** | **3,381** | 3,238-3,446 | 107.4 | `inf002-gpu80-20260326-205559` |
| **0.70** | **67.0** | **3,357** | 3,241-3,402 | 110.4 | `inf002-gpu70-20260326-205600` |
| **0.60** | **57.4** | **3,385** | 3,275-3,462 | 107.0 | `inf002-gpu60-20260326-205600` |

### vLLM engine details (from logs — preserved here since Iris logs are ephemeral)

All runs: vLLM v0.13.2.post6, V1 engine, dtype=bfloat16, max_seq_len=2048, TP=4, enable_prefix_caching=True, enable_chunked_prefill=True, enforce_eager=True, kv_cache_metrics=True (sample=0.01), backend=openxla, v5p-8 (4 chips, 95.74 GiB each).

| `gpu_mem_util` | KV cache tokens | Max concurrency (2048 tok/req) | KV blocks/layer | HBM used/total per chip (GiB) |
|---|---|---|---|---|
| 0.90 | 2,700,928 | 1,318.8x | 21,101 | 86.17 / 95.74 |
| 0.80 | 2,387,200 | 1,165.6x | 18,650 | 76.60 / 95.74 |
| 0.70 | 2,073,472 | 1,012.4x | 16,199 | 67.02 / 95.74 |
| 0.60 | 1,759,744 | 859.3x | 13,748 | 57.45 / 95.74 |

KV cache block shape: `(128, 8, 2, 128)` per layer, sharded as `PartitionSpec('data', None, 'model')` across the 4-chip mesh.

**Our workload uses 64 prompts × 16 gens = 1,024 concurrent sequences × 2,048 max tokens = ~2.1M tokens worst case.** Even at 0.60 (1.76M cache tokens), this fits because most responses are <1024 tokens (mean ~350 tokens from INF-001 data). The cache is sized for worst-case max_model_len, but actual utilization is much lower.

### Key findings

1. **Throughput is completely flat across all cache budgets.** Median tok/s varies by <1.2% (3,345-3,385) — well within noise. Reducing `gpu_memory_utilization` from 0.90 to 0.60 has **zero throughput cost**.
2. **HBM scales linearly**: 86.2 → 76.6 → 67.0 → 57.4 GiB. Dropping to 0.60 frees **28.8 GiB per chip** vs the 0.90 baseline.
3. **KV cache is massively oversized at 0.90** for Llama 3.1 8B with `max_model_len=2048` and this batch shape (64 prompts × 16 gens). The model weights + KV cache fit comfortably at 0.60.
4. **Packed TP=2 is feasible on HBM**: at 0.60, each chip uses 57.4 GiB. Two TP=2 engines sharing 4 chips would need ~57.4 GiB per chip (each engine uses 2 chips). v5p has ~95 GiB per chip, so there's ~37 GiB headroom. Even at 0.70 (67 GiB), two engines would need ~67 GiB per chip — still fits.
5. **Early batches (4-5) showed ~2,800 tok/s** at reduced cache levels before recovering to ~3,350+ by batch 8+. This was likely additional JIT compilation paths for the smaller cache configuration, not a sustained throughput hit.

### Decision

**Proceed to INF-003** (TP=2 single-engine feasibility). The cache budget is not a blocker — use `gpu_memory_utilization=0.70` or `0.60` for TP=2 experiments to leave headroom for co-location.

## INF-003: TP=2 Single-Engine Feasibility (2026-03-26)

Purpose: measure the throughput cost of moving from TP=4 to TP=2 on a single engine before attempting packed TP=2×2. Decision gate: if TP=2 is already bad, stop.

Script: `experiments/exp_inf_003_tp2_feasibility.py`

### Jobs launched

| Variant | TP | `gpu_mem_util` | Iris root job | Rationale | Status |
|---|---|---|---|---|---|
| INF-003a | 2 | 0.90 | `/ahmed/inf-003-tp2-0326` | Fair comparison vs INF-001 baseline | RUNNING |
| INF-003b | 2 | 0.45 | `/ahmed/inf-003-tp2-gpu45-0326` | Simulates per-engine budget in packed TP=2×2 | RUNNING |

### Config (both jobs)

- Model: Llama 3.1 8B Instruct, **TP=2**, v5p-8, `max_model_len=2048`
- Batch shape: 64 prompts × 16 generations = 1024 completions/batch
- 25 batches (3 warmup + 22 measured)

### Why INF-003b (gpu_mem=0.45) — CORRECTION: flawed premise

**Original rationale (wrong)**: "If we pack two TP=2 engines on one v5p-8, each engine gets roughly half the memory budget."

**Correction**: This was based on a misunderstanding. In packed TP=2×2, each engine gets its **own dedicated 2 chips** via `TPU_VISIBLE_CHIPS` partitioning — there is no chip sharing between engines. Engine 1 uses chips 0,1 and engine 2 uses chips 2,3. Each v5p chip has 95 GiB independently. So `gpu_memory_utilization=0.90` is correct for each engine in the packed setup.

The INF-003b (gpu_mem=0.45) experiment is still useful data — it tells us how throughput responds to reduced cache at TP=2 (extending the INF-002 cache sweep to a different TP degree). But it does NOT simulate the packed TP=2×2 memory budget as originally claimed.

**The real comparison for packed TP=2×2 is**: INF-003a throughput (TP=2, gpu_mem=0.90) × 2 vs INF-001 throughput (TP=4, gpu_mem=0.90).

### Results

*(pending — will fill in when jobs complete)*

## INF-004 Plan: Data-Parallel TP=2×2 on Single v5p-8

Source: analysis of `/Users/ahmed/code/vllm_tpu_multi/.agents/logbooks/no_ray_multihost_vllm.md` (multi-host vLLM experiments by the same team).

### Background from vllm_tpu_multi

The multi-host vLLM work proved that independent replicas scale nearly linearly:
- v6e-16 (4 hosts × 4 chips): 4×TP=4 replicas → 11,870 tok/s (3.4x single-host)
- v5p-16 (2 hosts × 4 chips): 2×TP=4 replicas → 8,681 tok/s (1.8x single-host)
- Optimal concurrency: 256/replica across all hardware
- Architecture: separate `vllm serve` processes, one per host, HTTP round-robin routing via `benchmark_grpo_stress.py`

### How chip partitioning works on a single host

Each subprocess sets `TPU_VISIBLE_CHIPS` before JAX/vLLM init. This is a standard JAX env var — each process sees only its assigned chips. Combined with `VLLM_ENABLE_V1_MULTIPROCESSING=0`, each engine runs in its own process with its own chip subset.

```
Engine 1: TPU_VISIBLE_CHIPS=0,1  →  TP=2 on first 2 chips
Engine 2: TPU_VISIBLE_CHIPS=2,3  →  TP=2 on last 2 chips
```

No chip sharing, no memory contention. Each engine gets full 95 GiB per chip.

### Proposed architecture

```
Iris TPU job (v5p-8, 4 chips)
  └─ Parent process (coordinator)
       ├─ subprocess.Popen: TPU_VISIBLE_CHIPS=0,1, TP=2, prompts[0:32]
       ├─ subprocess.Popen: TPU_VISIBLE_CHIPS=2,3, TP=2, prompts[32:64]
       └─ Aggregates results from both, logs to W&B
```

Using `subprocess.Popen` (not `multiprocessing`) because JAX initialization is global — we need completely separate Python interpreters with different `TPU_VISIBLE_CHIPS` set before any import.

### Script: `exp_inf_004_tp2x2_packed.py`

Parent process:
1. Load prompts from MathEnv
2. Split prompts into 2 equal shares, write to temp files
3. Spawn 2 child processes, each running a worker function that:
   - Sets `TPU_VISIBLE_CHIPS` in env
   - Creates `vLLMInferenceContext(TP=2, gpu_mem=0.90)`
   - Runs batches on its prompt share
   - Writes per-batch results to a JSON file
4. Wait for both children
5. Aggregate: sum throughput, merge timing stats
6. Log to W&B

### Decision gate

If aggregate throughput of 2×TP=2 > INF-001 baseline (3,345 tok/s), packed DP wins.

From early INF-003a data (TP=2, gpu_mem=0.90): ~2,900-3,060 tok/s per engine. If that holds, 2×TP=2 ≈ 5,800-6,120 tok/s → **~1.7-1.8x improvement** over TP=4.

### Note on RL rollout worker API

The RL rollout worker uses `LLM.generate()` (sync offline batch API), not `vllm serve` (HTTP). The vllm_tpu_multi logbook found that `LLM.generate()` with a full batch gets ~80% of `vllm serve` throughput (no continuous batching). For INF-004 we'll use `LLM.generate()` to match the RL rollout path. Switching to `vllm serve` + HTTP would be a separate optimization.

### W&B logging improvement

As of this session, both benchmark scripts (`exp_inf_001_tp4_baseline.py`, `exp_inf_003_tp2_feasibility.py`) now log vLLM KV cache init details to W&B under the `kv_cache/` prefix:
- `kv_cache/num_gpu_blocks` — number of KV cache blocks allocated
- `kv_cache/block_size` — tokens per block (typically 128)
- `kv_cache/total_tokens` — total cache capacity in tokens
- `kv_cache/max_concurrency_2048` — max concurrent 2048-token requests
- `kv_cache/gpu_memory_utilization` — the actual value used

This data was previously only in ephemeral Iris job logs (preserved manually in the INF-002 results section above). Future runs will have it in W&B automatically. The currently running INF-003a and INF-003b jobs do NOT have this change — they launched before the update.
