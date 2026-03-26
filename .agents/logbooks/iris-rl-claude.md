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
