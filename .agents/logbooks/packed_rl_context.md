# Packed RL Context

Branch: `packed_rl`
Issue: `#4286` `Experiment: Packed vLLM inference on Iris RL`
Base: `origin/iris_rl` at `59601ab7660013797b6ae7f095d5b9c7e9615151`

## Purpose

This note is the branch-local entry point for the packed-rollout follow-up
thread. The goal is to avoid re-reading the entire Iris RL migration history
before touching packed vLLM inference again.

## Current Read

- Packed RL on Iris is real and functionally correct.
- One `v5p-8` rollout worker can host the packed `2 x TP=2` topology end to
  end.
- Packed rollout is still not the promoted default because normal
  `eval_frequency=1` reintroduces enough rollout-side contention that trainer
  cadence does not clearly beat the direct `e4ms2` baseline.
- The direct `e4ms2` topology remains the production baseline for now:
  `1` trainer plus `2` rollout workers, each `v5p-8`, `n_prompts=64`.

## What Has Already Been Proven

### Baseline to beat

Direct `e4ms2` is the relevant throughput target:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-train`
- rollout-0:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-0`
- rollout-1:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-1`

The migration logbooks treat this as the strongest non-packed Iris baseline.

### Packed RL validation ladder

`PKR-001`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-rollout-0`
- result:
  correctness success, throughput miss
- key takeaway:
  packed inference worked, but one packed rollout job still owned both train
  rollout generation and eval, so cadence landed closer to `e4par` than
  `e4ms2`

`PKR-003`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr003-20260327-072722-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr003-20260327-072722-rollout-0`
- result:
  succeeded with sparse eval
- key takeaway:
  when eval pressure is mostly removed, the packed sampler path itself looks
  materially healthier

`PKR-004`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr004-20260327-094215-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr004-20260327-094215-rollout-0`
- result:
  succeeded under normal eval cadence too
- key takeaway:
  the same eval-heavy contention pattern returned, so packed rollout is still
  not an obvious drop-in replacement for direct `e4ms2`

## Main Conclusion So Far

The packed rollout problem is no longer sampler correctness. The open problem
is trainer-facing cadence under realistic eval ownership and scheduling.

The branch should therefore focus on:

1. getting more useful train-serving work out of each packed `v5p-8`
2. reducing or isolating eval interference
3. deciding whether packed rollout can actually replace a two-rollout direct
   topology, not just run successfully

## 2026-04-10: W&B Data Pull and Root-Cause Revision

All nine run histories were pulled from W&B
(`marin-community/marin_iris_rl_debug`) and saved under
`.agents/wandb_snapshots/packed_rl_analysis/`. The raw data **contradicts
parts of the earlier "eval contention" narrative** and exposes a different
bottleneck.

### Actual wall-clock cadence (computed from `_timestamp` diffs)

| Run | Steps | Wall-clock median | Mean | Notes |
|-----|-------|-------------------|------|-------|
| `e4ms2` baseline | 200 | **100.5s/step** | 122.6s | stable steady state |
| `PKR-001` | 5 | **180.4s/step** | 199.6s | packed, normal eval |
| `PKR-003` | 10 | **182.5s/step** | 169.3s | packed, sparse eval |
| `PKR-004` | 10 | **176.5s/step** | 424.9s | packed, normal eval (one 2406s outlier) |

So PKR runs are ~80s/step slower than the baseline. The ~101s vs ~171s
numbers in the original issue are confirmed in the median.

### Trainer phase breakdown (medians, rows after warmup)

| Phase (sec) | e4ms2 | PKR-001 | PKR-003 | PKR-004 |
|---|---|---|---|---|
| `step_duration_seconds` (trainer compute) | **61.1** | **61.0** | 82.0 | 81.7 |
| `forward_backward_duration_seconds` | 57.2 | 25.6 | 44.1 | 39.3 |
| `batch_prep_duration_seconds` | 4.0 | 35.4 | 32.8 | 37.4 |
| `loading_time` | 4.0 | **95.4** | 44.3 | 42.4 |
| `hook_time` | 16.0 | 9.7 | 12.4 | 11.4 |
| `weight_transfer/serve_time_seconds` | 15.6 | 9.6 | 12.2 | 11.3 |

**Key observation**: `step_duration_seconds` (the actual trainer compute
step) is **identical between `e4ms2` and `PKR-001` at 61s**. The trainer
compute path is not slower. The entire ~80s/step gap lives in
`loading_time` / `batch_prep_duration_seconds` — i.e. the trainer sitting
idle waiting for the next rollout batch.

### Rollout-side timing (medians)

| Metric | e4ms2 ro-0 | e4ms2 ro-1 | PKR-001 | PKR-003 | PKR-004 |
|---|---|---|---|---|---|
| rollout wall-clock cadence (sec/batch) | 121.6 | 132.3 | **185.7** | 139.2 | 200.3 |
| `inference.throughput/batch_time_seconds` | 145.3 | 147.2 | **70.1** | 132.5 | 124.1 |
| `inference.throughput/tokens_per_second` | 3,336 | 3,281 | **6,223** | 3,381 | 3,248 |
| `inference.throughput/requests_per_second` | 7.05 | 6.96 | **14.61** | 7.73 | 8.25 |
| rollout storage `last_total_time` | 0.31s | — | 0.26s | 0.24s | 0.22s |

**Key observation**: PKR-001's packed inference is **nearly 2× faster per
batch than a single `e4ms2` worker** (70s vs 145s, 6,223 tok/s vs 3,336
tok/s). The packed sampler path is not slow — it's actually exploiting the
2×TP=2 parallelism as designed.

PKR-003 and PKR-004 drop back to ~3,300 tok/s, i.e. single-replica speed.
Their packed replicas are no longer running train in parallel — train is
falling back to replica 0 only because eval is holding replica 1.

### Packed replica activity (final counters)

| Run | r0 requests | r1 requests | r0 activations | r1 activations |
|---|---|---|---|---|
| PKR-001 | 15 | 15 | 6 | 6 |
| PKR-003 | 17 | 17 | 10 | 10 |
| PKR-004 | 30 | 33 | 9 | 9 |

PKR-004 has the only run where replica 1 did more requests than replica 0,
consistent with extra eval batches routed to replica 1.

### Rollout storage is not the bottleneck

Rollout storage write+serialize is <0.4s per batch across all runs. Storage
is negligible relative to the ~80s/step gap.

### Revised root-cause analysis

The earlier narrative was: "packed is slow because eval contention."

The data says: **packed with 1 rollout worker is slow because there is no
pipeline parallelism between the trainer and the rollout worker**.

Cycle reconstruction for PKR-001:

- Trainer produces weight `N` (≈61s compute + publish)
- Rollout worker fetches weight `N` (~14s, per `transfer/fetch_time`)
- Rollout worker activates weight `N` on both replicas
- Rollout worker generates the next batch (~70s)
- Rollout worker writes the batch (~0.3s)
- Trainer ingests the batch, computes weight `N+1`, publishes
- …loop…

`61 + 14 + ~5 + 70 + 10 + 16 ≈ 176s` — matches PKR-001's measured
180s/step almost exactly.

The `e4ms2` baseline breaks this chain by running **two independent
rollout workers**. One worker is always one step ahead of the other, so
the trainer can consume from an always-full pipeline. Each e4ms2 worker's
own batch_time is 145s (slower than packed!), but with two workers the
effective rollout arrival rate is ~72.5s — faster than trainer compute, so
the trainer never waits.

**Packed inference on `v5p-8` is not the problem. The problem is running
only one rollout worker, which eliminates the pipeline depth that two
independent workers provide for free.**

Eval contention (the earlier story) is a *secondary* degradation: in
PKR-003/PKR-004 it knocks packed throughput from ~6,200 tok/s back down
to ~3,300 tok/s by preventing the 2-replica parallel mode. But even if
eval were completely isolated, a single packed worker would still be
~80s/step slower than `e4ms2` because of the missing pipeline depth.

### Open corrections to earlier logbook claims

1. **"Eval ownership is the major remaining gap"** — **was actually closer
   to the truth than the intermediate "pipeline depth" framing below
   suggested**. The real story (after re-reading the code) is that eval
   and train compete for the same rollout-worker generate capacity, and
   with 1 worker there's less spare capacity to absorb eval than with 2.

2. **"Packed inference is slower than baseline"** — false. PKR-001's
   packed inference at 6,223 tok/s beats `e4ms2`'s per-worker 3,336 tok/s
   by ~1.9× — exactly the 2×TP=2 speedup the design predicted.

## 2026-04-10 (third correction): the rollout main loop is single-threaded

Everything below about work division and eval gating is still true, but
the *mechanical* reason packed can't imitate the 2-worker setup — even
though it has two physically idle replicas — is one more layer deeper.

### The rollout worker main loop is single-threaded Python

Look at `rollout_worker.py:943-1080`:

```python
while self._running:
    ...
    if _should_run_micro_eval(...):          # (worker 0 only)
        self._evaluate_lesson(...)           # BLOCKS the main thread
    if _should_run_curriculum_eval(...):     # (worker 0 only)
        self._evaluate_curriculum(...)       # BLOCKS the main thread
    ...
    rollout_batch, env_metrics = self._sample_batch(
        lesson_id=lesson_id, ..., mode="train", ...
    )                                        # BLOCKS the main thread
    self._rollout_writer.write_batch(rollout_batch)
    self._curriculum_actor.update_lesson_stats.remote(...).result()
    step += 1
```

Everything in this loop is sequential. Each `env.sample()` call goes all
the way down to `inference_ctx.batch_completions(...)`, which (for the
packed context) reserves replicas, dispatches, does `future.result()` on
every submitted replica, and only then returns to the caller. The main
thread is stuck inside one call at a time.

That means the path I drew in the previous correction — "r0 does train
while r1 does eval in parallel" — **never actually happens in this
architecture**, even though the packed parent has the plumbing for it.

A single iteration of the packed loop looks like:

```
t=0     main thread calls env.sample(mode="eval")
        ├─ _reserve_replicas(EVAL) → reserves r1 only
        ├─ dispatches 1 generate request to r1
        ├─ future.result() BLOCKS main thread
        │
        │       r0 is physically idle, nothing dispatched to it,
        │       because no other Python thread is going to call
        │       batch_completions() — main thread is stuck here
        │
t≈45s   eval on r1 finishes
        └─ _release_replicas → returns

        main thread calls env.sample(mode="train")
        ├─ _reserve_replicas(TRAIN) → reserves r0 AND r1 (both free)
        ├─ dispatches 2 generate requests in parallel via _executor
        ├─ future.result() on both
        │   (these do run in parallel — that's the 2xTP=2 win)
        │
t≈45+70 both replicas finish
        └─ returns
```

So the packed context's parallelism is at the **inference layer** (two
TP=2 vLLM engines can run simultaneously when both are handed work),
but the **orchestration layer** above it (the rollout worker's Python
main loop) still serializes eval and train. The "dedicate r0 to train,
r1 to eval" split that the dispatch table in
`choose_packed_target_replica_indices` *hints at* only works if two
different threads call `batch_completions` concurrently — which this
codebase never does.

The `eval_waiters` counter and the condition-variable dance in
`_reserve_replicas` were clearly built for a world where eval and
train would be fired from different threads (that's the only reason
any of that machinery exists). The actual calling code doesn't fire
them concurrently.

### Why `e4ms2` naturally does what packed can't

`e4ms2` doesn't need any intra-process concurrency. It runs **two
separate OS processes**, each with its own single-threaded main loop:

- Worker 0 process: `[eval 45s] → [train 70s] → [eval 45s] → [train 70s] → ...`
- Worker 1 process: `[train 70s] → [train 70s] → [train 70s] → [train 70s] → ...`

The OS scheduler runs both processes concurrently on different TPU
pods. Worker 1 just keeps producing train batches; it never stops for
eval because `worker_index != 0` short-circuits both `_should_run_*`
gates. Worker 0 sits out every time it's on an eval step, but the
trainer doesn't care because worker 1 is feeding the replay buffer.

So "two OS processes running sequential main loops" gives you the
concurrency for free, without any threading inside the rollout worker.

### Why packed fails this test on one `v5p-8`

Packed puts two replicas in one process to save a TPU. The inference
parallelism works (PKR-001's 6,223 tok/s proves it). What doesn't
carry over is the *orchestration* parallelism: one rollout worker
process = one main loop = one env.sample() in flight at a time.

- While the loop is in the eval `env.sample()`, r0 is idle for the
  full ~45-60 s of eval.
- While the loop is in the train `env.sample()`, both replicas are
  busy, but only because there is a single caller waiting on both
  futures at once — that caller cannot simultaneously fire another
  request into r0.

So the packed `v5p-8` effectively runs eval and train **back to
back**, exactly as a single worker would. The 2× replica parallelism
only kicks in *within* a train batch (splitting 64 prompts into 32+32
for ~2× speedup), not *across* train-vs-eval.

### What this means for "why is a replica ever waiting"

**Replica 0 is waiting during every eval run.** Not because the
dispatcher chose to exclude it — it would happily accept work — but
because the single Python main thread is stuck inside
`batch_completions(EVAL)` waiting on replica 1's future. No other
Python code exists to dispatch a train request to r0 while that
future is pending.

### The real minimal fix

To make packed behave like two-worker `e4ms2`, the packed setup needs
to fire train and eval `env.sample()` calls **concurrently** from the
same process. Options:

1. **Run eval in a background thread inside the rollout worker.**
   Spawn a separate thread whose loop does nothing but eval; the main
   loop keeps doing train. Both threads call `batch_completions()`
   against the same packed context. The existing `_reserve_replicas`
   /`eval_waiters` machinery already supports this — that's why it's
   there. This is probably a small code change (~30–60 lines) with a
   big payoff.

2. **Move eval to a separate Fray actor.** More invasive but cleaner.
   Eliminates the question entirely by taking eval off the rollout
   worker's timeline.

3. **Do not run full curriculum eval on this worker at all.** Run it
   from the trainer between steps, or on a separate small TPU, so the
   packed worker's main loop is effectively train-only and matches
   `e4ms2` worker-1's cadence across both replicas.

Any of these three would let r0 actually serve train while r1 serves
eval, which is the whole point of the 2-replica packed design.

The first option is the cheapest test and directly validates whether
the `_reserve_replicas` design choices were worth making.

## 2026-04-10 (second correction): work division, eval gating, and the real math

My earlier "corrected" story was still incomplete. After re-reading
`rollout_worker.py`, `curriculum.py`, and `replay_buffer.py` and going
back to the data, here is how it actually works.

### How work is divided between rollout workers

**It isn't.** Each rollout worker independently runs the same loop with
the same `n_prompts` (set to 64 in these runs) and the same
`n_generations_per_prompt` (16). Each `env.sample()` call produces 64
unique questions × 16 responses each = 1024 samples, which is exactly
one `train_batch_size` (1024 in `RLJobConfig`).

So each worker independently produces full trainer batches. They do not
split the 64 questions between them. The only coordination is a single
curriculum actor that they both call via RPC, which returns a lesson id
for a given seed (`sample_lesson(seed: int)`); with a single-lesson
curriculum like `math_full` both workers pick the same lesson every
time, then each independently samples questions from the environment
with its own RNG.

Rollouts are pooled through a shared `ReplayBuffer`:

- Both workers write `RolloutBatch`es into `rollout_storage` (GCS
  queue). Each batch has a `weight_step` equal to the active weight
  when it was generated.
- `ReplayBufferConfig` here is
  `capacity=4096, max_samples=1, max_rollout_step_delay=1`.
- The trainer reads the queue, drops any batch older than
  `current_step - 1`, computes RLOO advantages, and samples
  `local_batch_size` individual rollouts per step.
- Because `max_samples=1`, each rollout is consumed at most once. Any
  extras that can't be used within the 1-step staleness window are
  dropped.

So with two workers both producing full batches, **some rollouts get
dropped** — this IS wasteful, but it's the simple-and-robust design
choice. In the `e4ms2` baseline: 103 + 137 = 240 rollout batches
produced over 200 trainer steps → ~17% of rollout compute discarded.

### Eval is pinned to worker_index == 0

The key gate I missed on the first read is in
`lib/marin/src/marin/rl/rollout_worker.py:365-399`:

```python
def _should_run_curriculum_eval(*, worker_index, ...):
    if worker_index != 0:
        return False
    ...

def _should_run_micro_eval(*, worker_index, ...):
    if worker_index != 0:
        return False
    ...
```

**Only worker 0 runs curriculum eval and micro eval.** All other rollout
workers are train-only. This is why `e4ms2`'s rollout-1 shows 137
logged train batches vs rollout-0's 103 over the same 24,326 s: worker
1 spends all its time on train while worker 0 is also running full
eval (500 problems on MATH-500) every `eval_frequency=1` steps.

In the packed single-worker runs, that worker IS worker 0, so it
inherits all the eval load with no second worker to compensate.

### The actual per-worker cadence numbers (from wandb)

From `_timestamp` diffs in the per-run JSONs (full 200-step `e4ms2`):

| Worker | logged train batches | wall span (s) | per-batch cadence (s) |
|---|---|---|---|
| `e4ms2` rollout-0 (train + eval) | 103 | 24,326 | **236** |
| `e4ms2` rollout-1 (train only)   | 137 | 24,350 | **178** |
| `e4ms2` combined                 | 240 | ~24,400 | **102** |
| `e4ms2` trainer                  | 200 steps | 24,405 | **122 / step** |

Worker 0 is ~33% slower than worker 1 because of eval work. But more
importantly: **each individual worker's per-batch cadence (178 s or
236 s) is significantly slower than the trainer's step cadence
(122 s).** One worker alone cannot keep up.

The magic of two workers is just the arithmetic: two workers running
independently bring the combined production cadence to ~102 s per
batch, slightly *faster* than the trainer's 122 s step cadence, so the
trainer almost never has to wait (`loading_time` median = 4 s).

### Why `PKR-001` is starved

PKR-001 has one rollout worker, which is `worker_index=0`, so it's
doing train + full eval at eval_frequency=1. It's the exact
equivalent of `e4ms2`'s rollout-0 running alone, with no rollout-1 to
compensate.

- Trainer compute cost: still 61 s per step (identical to `e4ms2`).
- Trainer needs one rollout per step to proceed.
- Single worker per-batch cadence: ≥178 s (what rollout-1 alone would
  be without eval), and much worse with eval on top — probably in the
  200 s+ range, matching the 185 s rollout wall-clock cadence measured
  for PKR-001.
- `max_rollout_step_delay=1` means the trainer can only use rollouts
  from the current weight or the one before; it can't stockpile older
  rollouts to hide the gap.
- So the trainer's `loading_time` jumps from 4 s to 95 s per step — it's
  waiting on the sole rollout worker to finish its next batch.

Total packed cadence ≈ 61 s (compute) + 95 s (wait for next batch) +
~24 s (hook + weight serve) ≈ 180 s/step, matching the measured
180.4 s median.

### The actual diagrams

Baseline (`e4ms2`), the way it really is:

```
                            ┌──────────────────┐
                            │ Trainer           │
                            │ publishes wt N    │
                            │ every ~122 s      │
                            └─────────┬─────────┘
                                      │ Arrow Flight stream
                 ┌────────────────────┴────────────────────┐
                 │                                         │
                 ▼                                         ▼
    ┌──────────────────────────┐             ┌──────────────────────────┐
    │ Rollout-0 (worker_index=0)│             │ Rollout-1 (worker_index=1)│
    │ v5p-8 TP=4               │             │ v5p-8 TP=4               │
    │                          │             │                          │
    │ bg wt-sync thread (1 Hz) │             │ bg wt-sync thread (1 Hz) │
    │                          │             │                          │
    │ main loop:               │             │ main loop:               │
    │  sample lesson           │             │  sample lesson           │
    │  [micro_eval]  ←── eval  │             │  (skipped, idx != 0)     │
    │  [full eval 500 probs]   │             │  (skipped, idx != 0)     │
    │  env.sample(n_prompts=64,│             │  env.sample(n_prompts=64,│
    │             n_gens=16)   │             │             n_gens=16)   │
    │  → 1024 samples          │             │  → 1024 samples          │
    │  write batch to storage  │             │  write batch to storage  │
    │                          │             │                          │
    │  cadence: 1 batch/236 s  │             │  cadence: 1 batch/178 s  │
    │  (eval load slows it)    │             │                          │
    └────────────┬─────────────┘             └────────────┬─────────────┘
                 │                                        │
                 └───────────────┬────────────────────────┘
                                 ▼
                        ┌─────────────────┐
                        │ Rollout storage │
                        │ (GCS queue)     │
                        └────────┬────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │ Trainer replay buffer │
                     │  max_rollout_step_    │
                     │    delay = 1          │
                     │  max_samples = 1      │
                     │                       │
                     │ pulls 1024 samples/   │
                     │ trainer step          │
                     │                       │
                     │ combined arrival rate:│
                     │   1 batch / 102 s     │
                     │ consumption rate:     │
                     │   1 step / 122 s      │
                     │ → trainer never waits │
                     │   (loading_time ≈ 4 s)│
                     └───────────────────────┘
```

Packed (`PKR-001`), the way it really is:

```
                            ┌──────────────────┐
                            │ Trainer           │
                            │ publishes wt N    │
                            │ every ~180 s      │
                            │ (starved)         │
                            └─────────┬─────────┘
                                      │
                                      ▼
    ┌───────────────────────────────────────────────────────┐
    │ Packed parent process, worker_index=0                  │
    │ v5p-8, one process, two local vLLM children            │
    │                                                        │
    │ main loop:                                             │
    │   sample lesson                                        │
    │   [micro_eval]     ← yes (worker_index == 0)           │
    │   [full eval 500]  ← yes (eval_frequency=1 → every step)│
    │   env.sample(n_prompts=64, n_gens=16)                  │
    │                                                        │
    │   batch_completions(request_kind=...)                  │
    │     _reserve_replicas(kind)  ← parent-side mutex       │
    │        train: prefer (0,1), fall back to (0,) if r1    │
    │               is reserved by eval                      │
    │        eval: always (1,)                               │
    │     _resolve_dispatch_weight()  ← activate pending wt  │
    │                                   if both idle         │
    │     split prompts across reserved replicas             │
    │                                                        │
    │   ┌─────────────────────┐  ┌─────────────────────┐    │
    │   │ Replica 0 (TP=2)    │  │ Replica 1 (TP=2)    │    │
    │   │ chips 0,1           │  │ chips 2,3           │    │
    │   │                     │  │                     │    │
    │   │ own bg wt-sync      │  │ own bg wt-sync      │    │
    │   │ fetches weights     │  │ fetches weights     │    │
    │   │ independently       │  │ independently       │    │
    │   │                     │  │                     │    │
    │   │ serves generates    │  │ serves generates    │    │
    │   │ via socket IPC from │  │ via socket IPC from │    │
    │   │ parent              │  │ parent              │    │
    │   └─────────────────────┘  └─────────────────────┘    │
    │                                                        │
    │   cadence: ~180 s per train batch                      │
    │   (train and eval time-share the same 4 chips)         │
    └────────────────────┬───────────────────────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Rollout storage │
                └────────┬────────┘
                         │
                         ▼
            ┌───────────────────────────┐
            │ Trainer replay buffer     │
            │ arrival: 1 batch / ~180 s │
            │ consumption (desired):    │
            │   1 step / ~90 s          │
            │ → trainer WAITS           │
            │   (loading_time ≈ 95 s)   │
            └───────────────────────────┘
```

### Takeaways

1. **Rollout workers do not coordinate work.** Each one independently
   runs the full `n_prompts=64` × `n_gens=16` = 1024-sample batch every
   iteration. Some of this output is dropped by the replay buffer due
   to the 1-step staleness window — that's ~17 % waste in `e4ms2`, and
   it's the price of the zero-coordination design.

2. **Eval is pinned to worker 0.** In 2-worker setups this quietly
   does a lot of work for you: worker 1 is effectively a train-only
   machine, so even when worker 0 is spending a long time on full eval
   the trainer still gets fresh batches from worker 1. In a 1-worker
   (or packed) setup, there is no such fallback — worker 0 has to do
   everything.

3. **The 2 workers beat 1 because of arrival-rate arithmetic, not
   coordination.** One worker cadences at 178–236 s per batch, slower
   than the trainer's 122 s step cadence. Two workers independent of
   each other halve the combined arrival time to 102 s, which just
   barely stays ahead of the trainer. In the single-worker packed
   case, there's no halving, so the trainer waits.

4. **`max_rollout_step_delay=1` is a hard constraint.** The trainer
   can't use stockpiled rollouts from earlier weights. So the relevant
   rate is not "total rollouts over run time", it's "rollouts that
   arrive within 1 weight step of each trainer step". With one packed
   worker, the effective arrival rate within that window is not enough.

### Reasonable next steps (again, updated)

1. **Move full curriculum eval off the rollout worker entirely.** Run
   it on the trainer itself between steps, or on a separate small TPU.
   This turns the packed single-worker setup into a "packed train-only
   worker", and the effective per-batch cadence should drop toward the
   ~70-80 s range the 2-replica packed generator can achieve. At that
   rate one packed `v5p-8` could actually keep up with a trainer
   running at ~90 s/step. **This is the cheapest test of the original
   cost-halving pitch.**

2. **Relax `max_rollout_step_delay`.** Letting the trainer accept
   slightly older rollouts (e.g. delay=2 or 3) lets the single packed
   worker front-load production, at the cost of some off-policy drift.
   Quantify the off-policy hit first.

3. **Lower `eval_frequency` from 1 to e.g. 5.** Trades eval-metric
   resolution for cadence. PKR-003 already hinted that this helps but
   isn't a full fix on its own.

4. **If none of the above gets it cheap, go back to 2 packed workers
   on 2 × `v5p-8`.** Same TPU budget as `e4ms2` but with 4 replicas
   serving rollouts, which clearly wins on throughput but loses the
   cost-halving pitch.

## 2026-04-10: Correction — "pipeline depth" was the wrong model

After re-reading the rollout worker and weight-transfer code I have to
retract the "pipeline depth 2" framing from my earlier note. It was
wrong in two ways:

1. "Pipeline parallelism" normally means splitting model *layers* across
   devices, which is not what's happening here at all. I was abusing the
   term.

2. More importantly, I speculated that the two `e4ms2` workers
   coordinate so one runs against weight `N` while the other runs against
   weight `N+1`. **They do not.** I should not have said it without
   checking.

### How weight sync actually works (from the code)

Every rollout worker — both the `e4ms2` workers and the packed parent's
child replicas — does the same thing:

- One background thread (`_sync_weights_loop` in `rollout_worker.py`, or
  `_weight_sync_loop` in `packed_vllm_worker.py`) polls
  `transfer_client.receive_weights(...)` once a second.
- When a new weight arrives, the thread stores it as a **pending**
  weight. The currently loaded weight is the **active** weight.
- The main generate loop never waits for a new weight. It calls
  `batch_completions()` against whatever weight is active.
- Activation happens opportunistically: in the packed path, on the next
  `batch_completions()` call, if both replicas have the same pending
  weight and are idle, `_resolve_dispatch_weight` activates it before
  dispatching. In the non-packed path, the background thread installs
  the weight directly.

So both `e4ms2` workers are effectively always generating against the
latest weight they've received. They do **not** intentionally stagger —
they both pick up weight `N+1` about 1 second after the trainer
publishes it, at roughly the same wall-clock moment. At any instant
both workers are almost always on the same weight.

The fact that `transfer/total_polls` is ~1,150–2,600 (≈ runtime in
seconds) per run confirms the 1-second poll cadence.

### The actual architecture (diagrams)

Two-rollout-worker baseline (`e4ms2`):

```
                 ┌──────────────────┐
                 │  Trainer          │     step N → publish wt N
                 │  (1 x v5p-8)      │     step N+1 → publish wt N+1
                 │  compute step N   │     ...
                 │  publish wt N+1   │
                 └────────┬─────────┘
                          │ Arrow Flight weight stream
            ┌─────────────┴─────────────┐
            │                           │
            ▼                           ▼
    ┌────────────────┐          ┌────────────────┐
    │ Rollout-0      │          │ Rollout-1      │
    │ (v5p-8, TP=4)  │          │ (v5p-8, TP=4)  │
    │                │          │                │
    │ bg thread      │          │ bg thread      │
    │  polls every 1s│          │  polls every 1s│
    │  picks up wt   │          │  picks up wt   │
    │  asynchronously│          │  asynchronously│
    │                │          │                │
    │ main loop:     │          │ main loop:     │
    │  sample lesson │          │  sample lesson │
    │  [maybe eval]  │          │  [maybe eval]  │
    │  generate train│          │  generate train│
    │  write batch   │          │  write batch   │
    └────────┬───────┘          └────────┬───────┘
             │                           │
             └────────────┬──────────────┘
                          ▼
                   ┌───────────────┐
                   │ Rollout store │
                   └───────┬───────┘
                           │ trainer reads next batch
                           ▼
```

Key points:
- **The two workers are independent processes** running the exact same
  code, one per TPU pod.
- They do NOT talk to each other or coordinate weight versions.
- Both are on the latest active weight (≈ 1s after trainer publishes).
- Their generate loops run in parallel, so the combined throughput to
  the rollout store is ≈ 2× one worker's rate.

Packed worker:

```
                 ┌──────────────────┐
                 │  Trainer          │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────────────────┐
                 │ Packed parent process         │
                 │ (1 x v5p-8)                   │
                 │                               │
                 │ main loop (Python):           │
                 │  env.sample() →               │
                 │   batch_completions():        │
                 │    _reserve_replicas(kind)    │
                 │      (mutex / CV)             │
                 │    _resolve_dispatch_weight() │
                 │    split prompts across reps  │
                 │    submit parallel generate   │
                 │                               │
                 │  ┌──────────┐  ┌──────────┐   │
                 │  │ Replica0 │  │ Replica1 │   │
                 │  │ TP=2     │  │ TP=2     │   │
                 │  │ chips 0,1│  │ chips 2,3│   │
                 │  │          │  │          │   │
                 │  │ own bg   │  │ own bg   │   │
                 │  │ sync     │  │ sync     │   │
                 │  │ thread   │  │ thread   │   │
                 │  │ (polls   │  │ (polls   │   │
                 │  │  every 1s)  │  every 1s)│  │
                 │  └──────────┘  └──────────┘   │
                 └───────────────┬──────────────┘
                                 │
                                 ▼
                          ┌───────────────┐
                          │ Rollout store │
                          └───────────────┘
```

Key points:
- **Exactly one parent process on one `v5p-8`.** Two TP=2 vLLM children
  live inside it, pinned to chips (0,1) and (2,3).
- Train requests prefer (0,1) — if replica 1 is free and no eval is
  waiting, train runs on both and gets ~2× throughput. Otherwise train
  falls back to replica 0 only (half throughput).
- Eval requests *always* go to replica 1 exclusively.
- Reservations are a single parent-side mutex + condition variable. While
  eval is running on replica 1, train either waits or runs single-rep.

### Why `e4ms2` beats `PKR-001` — the simple story

It isn't pipeline depth, it's **aggregate capacity under an eval load**.

Each rollout worker (packed or not) does a *mix* of train rollouts and
eval rollouts in its main loop, gated by `eval_frequency`. With
`eval_frequency=1` both runs spend a meaningful fraction of time on eval.

- `e4ms2`: 2 independent workers × full TP=4 capacity each. While one
  worker is busy on an eval batch, the other worker is still producing
  train batches for the trainer. The trainer's batch appetite (1 batch
  per ~61s of compute) is comfortably covered, so
  `batch_prep_duration_seconds` stays ~4s (no waiting).

- `PKR-001`: 1 worker × (2 × TP=2 replicas). When eval runs on replica
  1, train either waits for replica 1 or drops to single-rep on replica
  0. When eval is NOT running, train uses both replicas at ~6,200
  tok/s. Averaged over the run, the effective train-batch rate per
  worker is lower than one full-capacity `e4ms2` worker's rate,
  *and* there's only one of it. So when the trainer asks for the next
  batch, the rollout worker is often mid-generation and the trainer
  waits (`loading_time` ≈ 95s).

The packed replicas' per-kind counters from PKR-003 / PKR-004 confirm
this: PKR-004 (10 steps, eval_frequency=1) has r1 doing 6 eval gens and
r0/r1 asymmetry (30 vs 33), meaning 3 train batches on PKR-004 fell back
to replica-0-only mode because replica 1 was tied up with eval.

### Why this still matters

- Packed inference itself is still ~2× faster per batch when both
  replicas run in parallel (PKR-001: 6,223 tok/s vs `e4ms2` per-worker
  3,336 tok/s). The design works.
- The problem is that a single packed `v5p-8` has no slack: everything
  (train, full curriculum eval, micro-eval) has to time-share the same
  4 chips, and with normal `eval_frequency=1` eval steals enough time
  that the trainer starves.
- `e4ms2`'s advantage is simply having 2× the rollout capacity to
  absorb eval without starving train. It doesn't need any clever
  coordination — two independent loops just happen to add up.

### What this implies for next steps

Unchanged direction, corrected framing: packed is only interesting as a
single-`v5p-8` drop-in replacement for `e4ms2` if the packed worker has
enough spare capacity to serve train rollouts at or above the trainer's
consumption rate *while* also covering eval.

Candidate approaches:

1. **Push eval off the packed worker.** Run full curriculum eval on a
   separate small TPU (or on the trainer itself between steps). This
   would leave both packed replicas dedicated to train. If train-only
   packed hits ~6,200 tok/s sustained and the trainer consumes at
   ~3,300 tok/s × effective batch size, the packed worker would sit at
   ~50% utilization and the trainer would stop waiting. This is the
   cleanest test of the remaining "packed = half TPU cost" pitch.

2. **Reduce `eval_frequency`.** PKR-003 (sparse eval) already hinted at
   this. Running full eval every N steps instead of every step lowers
   average eval load proportionally. Tradeoff is worse eval-metric
   resolution.

3. **Two packed workers on 2 × `v5p-8`.** Same TPU budget as `e4ms2`
   but with 4 total replicas. Would clearly win on throughput but
   abandons the cost-halving pitch — only worth trying if option (1)
   fails.

4. **Allow stale-weight train rollouts.** Decouple rollout generation
   from trainer cadence so the packed worker can generate continuously
   even while the trainer is still computing the next weight. Off-
   policy effects need measurement before trying this.

### Data location

- `.agents/wandb_snapshots/packed_rl_analysis/` — per-run JSONs with full
  history and config (one file per run). Use these instead of re-pulling
  from W&B on the next session.


## Read These First

### Primary narrative

- `.agents/logbooks/iris-rl-codex.md`
  - packed rollout planning, implementation, and `PKR-001` through `PKR-004`
    outcome notes
- `.agents/logbooks/iris-rl-claude.md`
  - baseline `e4ms2` throughput numbers and rollout-side inference analysis

### Packed-RL launchers and code

- `experiments/exp_iris_rl_regression_direct_gcs_packed.py`
- `experiments/exp_iris_rl_regression_direct_gcs_packed_candidate.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_protocol.py`
- `lib/marin/src/marin/rl/rollout_worker.py`

### Tests

- `tests/rl/test_packed_vllm_inference_ctx.py`
- `tests/rl/test_rollout_worker.py`

### Nearby artifacts

- `experiments/exp_inf_004_tp2x2_packed.py`
  - earlier packed inference context relevant to the RL-side packed rollout
    work
- `scratch/babysit_packed_rl_queue.sh`
  - historical queue / babysit script used during `PKR-003` and `PKR-004`

## Specific Hypotheses Worth Rechecking

1. The major remaining gap is eval ownership, not train sampling.
2. One packed rollout worker only becomes interesting if it preserves enough
   trainer-facing cadence to replace two independent rollout workers.
3. If packed rollout still misses after stronger eval isolation, then the real
   answer may remain "prefer more independent sampler replicas" on `v5p`.

## Things Not To Rediscover

- The direct `e4ms2` path already works and is the current baseline.
- Checkpointing is expensive but stable; that is a separate thread in
  `.agents/projects/ckpt_rl.md`.
- Retry / resume identity bugs and rollout W&B step bugs were already fixed on
  `iris_rl` before this branch split.

## Merge Log

### 2026-04-10: Merge origin/main into packed_rl

The `iris_rl` branch was merged into main upstream, causing 30 conflicts
with this branch. Resolved as follows:

**Strategy**: Take main as canonical base for all shared RL infrastructure,
then layer packed-RL-specific additions back on top.

**Conflicts resolved (30 files)**:
- 5 infrastructure files (levanter checkpoint, tensorstore, iris bug_report, test_checkpoint) → took main
- `exp2039_rl_math500.py` → accepted main's deletion
- `experiments/llama_3_8b_rl_math500.py` → took main
- 12 core RL files (environments, inference_ctx, orchestration, rl_job, rollout_worker, etc.) → took main
- 8 test files → took main

**Packed additions layered back**:
- `InferenceRequestKind` StrEnum added to `inference_ctx/base.py`
- `request_kind` parameter threaded through `batch_completions()` (base, levanter, vllm) and `MarinEnv.sample()` (base, prime_intellect_env)
- `PackedvLLM` imports/exports added to `inference_ctx/__init__.py`
- `PackedvLLMInferenceContextConfig` added to `rl_job.py` type union
- Stale `ArrowFlightExportStrategy` reference removed from `weight_transfer/__init__.py` (main removed this enum; auto-merge left a dangling reference)

**Packed files preserved without conflict**:
- `packed_vllm.py`, `packed_vllm_worker.py`, `packed_vllm_protocol.py`
- `exp_iris_rl_regression_direct_gcs_packed.py`, `exp_iris_rl_regression_direct_gcs_packed_candidate.py`
- `test_packed_vllm_inference_ctx.py`

**Known residual**: `xp_iris_rl_regression_direct_gcs_prod.py` (packed_rl-only experiment probe) still references the removed `ArrowFlightExportStrategy`. This file is not on main and wasn't part of the conflict resolution; it will need updating if used.

**Commit**: `e655aa35b`

## Suggested Immediate Next Step

Use this branch to answer one question cleanly:

"Can packed vLLM inference on `v5p-8` preserve enough trainer-facing cadence
under realistic eval settings to justify replacing the direct `e4ms2`
topology?"
