# Packed RL Context

Branch: `packed_rl`
Issue: `#4286` `Experiment: Packed vLLM inference on Iris RL`
Base: `origin/iris_rl` at `59601ab7660013797b6ae7f095d5b9c7e9615151`
Last updated: 2026-04-10

## Purpose

This note is the branch-local entry point for the packed-rollout follow-up
thread. The goal is to avoid re-reading the entire Iris RL migration history
before touching packed vLLM inference again.

## One-paragraph summary

Packed vLLM inference on Iris is correct and the packed sampler itself is
roughly 2× faster per batch than the baseline's per-worker rate (PKR-001 hit
6,223 tok/s vs `e4ms2` per-worker 3,336 tok/s). Despite that, a single
packed `v5p-8` rollout worker produces a trainer cadence of ~180 s/step vs
the two-rollout `e4ms2` baseline at ~122 s/step. The root cause is *not*
inference throughput, eval contention, or weight-sync overhead — it is that
the rollout worker's Python main loop is single-threaded and serializes
eval and train `env.sample()` calls, so while the eval call is blocked
inside `future.result()` on replica 1, replica 0 sits physically idle with
no concurrent Python caller to dispatch train work to it. `e4ms2` gets
around this by running two OS processes with independent sequential main
loops; the packed design collapses this into one process with two
replicas, which parallelized the GPU layer but left the orchestration layer
serial. The minimal fix is to add a background eval thread to the rollout
worker (the packed context's `_reserve_replicas` / `eval_waiters` lock
machinery already supports it).

## Hardware and config context

- Trainer: 1 × `v5p-8`, `train_batch_size=1024`
- Rollout (baseline `e4ms2`): 2 × `v5p-8`, each TP=4
- Rollout (packed `PKR-*`): 1 × `v5p-8`, split into 2 × TP=2 replicas
  on chips (0,1) and (2,3)
- Curriculum: single lesson `math_full`,
  `n_prompts=64, n_generations_per_prompt=16, max_output_tokens=1024`
- Eval: full curriculum eval (`eval_n_examples=500`),
  `eval_frequency=1` in the normal runs (`PKR-001`, `PKR-004`, `e4ms2`);
  sparse in `PKR-003`
- Replay buffer:
  `capacity=4096, max_samples=1, max_rollout_step_delay=1, alpha=3.0`
- Weight transfer: Arrow Flight, inflight updates enabled
- Eval gate: in `rollout_worker.py:365-399`, both
  `_should_run_curriculum_eval` and `_should_run_micro_eval` short-circuit
  to `False` when `worker_index != 0`. Only worker 0 runs eval; all other
  workers are train-only.

## W&B run pointers

Baseline `e4ms2`:

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-train`
- rollout-0:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-0`
- rollout-1:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/e4ms2-20260326-121919-rollout-1`

`PKR-001` (packed, normal eval):

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr001-20260327-012712-rollout-0`

`PKR-003` (packed, sparse eval):

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr003-20260327-072722-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr003-20260327-072722-rollout-0`

`PKR-004` (packed, normal eval, later config):

- trainer:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr004-20260327-094215-train`
- rollout:
  `https://wandb.ai/marin-community/marin_iris_rl_debug/runs/pkr004-20260327-094215-rollout-0`

Full unsampled history (train metrics, rollout metrics, rollout storage
timing) for every run above is saved locally under
`.agents/wandb_snapshots/packed_rl_analysis/` so future sessions don't
need to re-query W&B.

## Key measured numbers (from the saved W&B snapshots)

Wall-clock cadence (sec/step, computed from `_timestamp` diffs after
warmup):

| Run | Steps | Median | Mean |
|---|---|---|---|
| `e4ms2` trainer     | 200 | **100.5** | 122.6 |
| `PKR-001` trainer   | 5   | 180.4 | 199.6 |
| `PKR-003` trainer   | 10  | 182.5 | 169.3 |
| `PKR-004` trainer   | 10  | 176.5 | 424.9 (one 2406 s outlier) |

Trainer phase breakdown (medians, seconds, rows after warmup):

| Phase                                  | `e4ms2` | `PKR-001` | `PKR-003` | `PKR-004` |
|---|---|---|---|---|
| `step_duration_seconds` (trainer compute) | **61.1** | **61.0** | 82.0 | 81.7 |
| `forward_backward_duration_seconds`    | 57.2 | 25.6 | 44.1 | 39.3 |
| `batch_prep_duration_seconds`          | 4.0 | 35.4 | 32.8 | 37.4 |
| `loading_time`                         | 4.0 | **95.4** | 44.3 | 42.4 |
| `hook_time`                            | 16.0 | 9.7 | 12.4 | 11.4 |
| `weight_transfer/serve_time_seconds`   | 15.6 | 9.6 | 12.2 | 11.3 |

Note that `step_duration_seconds` (pure trainer compute) is **identical**
between `e4ms2` and `PKR-001` at ~61 s. The entire gap lives in the
trainer's `loading_time` / `batch_prep_duration_seconds` — i.e. the
trainer sitting idle waiting for a rollout batch to arrive.

Rollout-side timing (medians):

| Metric | e4ms2 ro-0 | e4ms2 ro-1 | PKR-001 | PKR-003 | PKR-004 |
|---|---|---|---|---|---|
| rollout wall-clock cadence (sec/batch) | 121.6 | 132.3 | **185.7** | 139.2 | 200.3 |
| `batch_time_seconds` (single env.sample) | 145.3 | 147.2 | **70.1** | 132.5 | 124.1 |
| `tokens_per_second` | 3,336 | 3,281 | **6,223** | 3,381 | 3,248 |
| rollout storage `last_total_time` | 0.31 | — | 0.26 | 0.24 | 0.22 |

Rollout worker productivity for the 200-step `e4ms2` run (computed from
row counts over wall span):

| Worker | train batches logged | wall span | per-batch cadence |
|---|---|---|---|
| rollout-0 (train + eval) | 103 | 24,326 s | 236 s |
| rollout-1 (train only)   | 137 | 24,350 s | 178 s |
| combined arrival         | 240 | ≈24,400 s | **102 s** |

Packed per-kind generate counts (from summary counters):

| Run | r0 train | r1 train | r0 eval | r1 eval | Notes |
|---|---|---|---|---|---|
| `PKR-001` | 15 total | 15 total | — | — | per-kind counters not present in this older run |
| `PKR-003` | 17 | 16 | 0 | 1 | sparse eval, 1 full eval in ~2030 s |
| `PKR-004` | 30 | 27 | 0 | 6 | normal eval; r0-vs-r1 asymmetry shows 3 train batches fell back to r0-only because r1 was on eval |

The PKR-004 train asymmetry (r0=30, r1=27) is the only evidence for
train-vs-eval overlap in the whole data set, and it's small — 3 train
batches out of 30 fell back to single-replica mode.

## The root cause (with code citations)

### The rollout worker main loop is single-threaded

`lib/marin/src/marin/rl/rollout_worker.py:943-1080`, simplified:

```python
while self._running:
    ...
    if _should_run_micro_eval(worker_index=self.config.worker_index, ...):
        self._evaluate_lesson(...)      # blocks on env.sample(mode="eval")
    if _should_run_curriculum_eval(worker_index=self.config.worker_index, ...):
        self._evaluate_curriculum(...)  # blocks on env.sample(mode="eval")
    ...
    rollout_batch, env_metrics = self._sample_batch(..., mode="train")  # blocks
    self._rollout_writer.write_batch(rollout_batch)
    self._curriculum_actor.update_lesson_stats.remote(...).result()
    step += 1
```

Every `env.sample()` call lands in
`inference_ctx.batch_completions(...)` on the packed parent
(`lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py:360-428`):

```python
replica_indices = self._reserve_replicas(request_kind)
dispatch_weight_id = self._resolve_dispatch_weight(...)
futures = [
    self._executor.submit(replica.generate, ...)
    for replica in selected_replicas
]
for (indices, _), (worker_index, future) in zip(...):
    response = future.result()   # main thread blocks here
```

The main thread is synchronous from top to bottom. Only one
`env.sample()` call is in flight at a time.

### The eval gate

`rollout_worker.py:365-399`:

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

Only `worker_index == 0` ever runs eval. Every other rollout worker is
train-only. In the single-worker packed runs, the only worker IS
worker 0, so it inherits 100 % of the eval load.

### What the dispatch table *would* do if called concurrently

`packed_vllm.py:574-596`:

```python
def choose_packed_target_replica_indices(*, request_kind, reserved_request_kinds, eval_waiters):
    replica_0_request = reserved_request_kinds[0]
    replica_1_request = reserved_request_kinds[1]

    if request_kind == InferenceRequestKind.TRAIN:
        if replica_0_request is not None:
            return None
        if replica_1_request is None and eval_waiters == 0:
            return (0, 1)
        return (0,)

    if request_kind in {InferenceRequestKind.EVAL, InferenceRequestKind.MICRO_EVAL}:
        if replica_1_request is not None:
            return None
        return (1,)
```

And `_reserve_replicas` (lines 473-492) uses a condition variable with
`eval_waiters += 1` when an eval caller enters. This is the machinery
for "r0 serves train, r1 serves eval, concurrently". **It only
functions if two different Python threads call `batch_completions` at
the same time.** Nothing in the current codebase does.

### What this means in a single packed iteration

```
t=0      main thread calls env.sample(mode="eval")
         └─ batch_completions(EVAL)
            ├─ _reserve_replicas(EVAL) → reserves replica 1
            ├─ dispatches 1 generate to replica 1
            └─ future.result()  ←── main thread BLOCKS
                  │
                  │  Replica 0 is physically idle the entire time.
                  │  The dispatch table would happily give it train
                  │  work — but nothing in Python is calling
                  │  batch_completions(TRAIN) right now. The sole
                  │  Python caller (main thread) is stuck here.
                  │
t≈50 s   eval future resolves on r1
         └─ _release_replicas → returns

         main thread calls env.sample(mode="train")
         └─ batch_completions(TRAIN)
            ├─ _reserve_replicas(TRAIN) → reserves (0, 1), both free
            ├─ dispatches 2 generates in parallel via _executor
            └─ future.result() on both  ← main thread BLOCKS

t≈50+70  both replicas finish (parallel TP=2 split)
         → one train batch produced; back to top of loop
```

So one iteration takes roughly `eval_time + train_time`, not
`max(eval_time, train_time)`. That's the entire gap.

### Why `e4ms2` doesn't need any of this

`e4ms2` is **two separate OS processes** running the identical
rollout-worker Python, one per `v5p-8`. Each process has its own
single-threaded main loop:

- Worker 0 (worker_index=0): `[eval 50 s] → [train 70 s] → [eval 50 s] → [train 70 s] → ...`
- Worker 1 (worker_index=1): `[train 70 s] → [train 70 s] → [train 70 s] → ...`
  (eval short-circuits on `worker_index != 0`, so worker 1 never stops)

The OS schedules the two processes on two different TPU pods, so they
run concurrently for free. Worker 1 is a steady train-batch pump;
worker 0's eval stalls are absorbed by worker 1's uninterrupted
output, and the replay buffer stays full.

Quantitatively: worker 0 produces 1 train batch per 236 s (slowed by
eval), worker 1 produces 1 train batch per 178 s, combined arrival
rate ≈ 1 batch per 102 s. The trainer needs 1 batch per ~122 s. The
combined arrival rate is faster than consumption, so
`loading_time ≈ 4 s` and the trainer never waits.

### Why one packed `v5p-8` can't imitate this

The packed design puts two TP=2 replicas in **one process** to save
a TPU. The inference layer parallelizes correctly (PKR-001's 6,223
tok/s during dual-replica train batches is proof). But the
orchestration layer — the rollout worker's Python main loop —
stays serial. With one process, eval and train share one sequential
main loop, and `[eval 50 s] + [train 70 s]` = 120+ s per iteration.
With the 90 s of trainer compute + hooks that wants a new batch, the
trainer starves.

The gap in one line: **packed parallelized the GPU layer but left
the Python orchestration layer serial, and the serial orchestration
layer is what the `e4ms2` baseline was secretly relying on for its
train/eval concurrency.**

## Rollout work division (no division, actually)

Each rollout worker — every `e4ms2` worker *and* the packed worker —
independently runs the same `env.sample(n_prompts=64, n_generations=16)`
call every iteration, producing 1024 samples = one full
`train_batch_size`. There is no coordination that splits the 64
questions across workers.

The `ReplayBuffer` (`max_rollout_step_delay=1`, `max_samples=1`)
pools rollouts from all workers, computes RLOO advantages, filters
anything older than `current_step - 1`, and each rollout is consumed
at most once. With two workers each producing full batches and the
trainer consuming one batch per step, some rollouts get dropped as
stale: `e4ms2` produced 240 rollout batches over 200 trainer steps,
so ~17 % of rollout compute is discarded. That's the price of the
zero-coordination design.

## The minimal fix

Three options, cheapest first:

1. **Add a background eval thread to the rollout worker.** Spawn
   one thread whose only job is running `_evaluate_curriculum` /
   `_evaluate_lesson` in a loop, calling `batch_completions(EVAL)`
   against the same packed context. The main thread keeps doing
   train. The existing `_reserve_replicas` / `eval_waiters` locking
   in `packed_vllm.py` is already correct for this — that is
   literally why the author put it there. Probably 30–60 LoC in
   `rollout_worker.py`. This is the direct test of whether the
   packed design's 2-replica split pays off.

2. **Move eval into a separate Fray actor / process.** More
   invasive but cleaner. Eliminates eval from the rollout worker's
   timeline entirely.

3. **Don't run full curriculum eval on the rollout worker at all.**
   Run it from the trainer between steps, or on a cheap CPU job. If
   the packed worker's main loop becomes effectively train-only, it
   should hit a per-batch cadence close to the ~70 s the 2-replica
   generator can sustain, which comfortably feeds the trainer.

**Option 1 is the right next experiment**: small code change, direct
validation of the packed design, no architectural churn, and if it
works the cost-halving pitch for packed is intact.

## Key code locations

- `lib/marin/src/marin/rl/rollout_worker.py`
  - main loop: lines 943–1080
  - eval gate: lines 365–399
  - `_sample_batch`, `_evaluate_lesson`, `_evaluate_curriculum`: 502, 794, 840
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm.py`
  - `PackedvLLMInferenceContext.batch_completions`: lines 360–428
  - `_reserve_replicas` / `_release_replicas`: 473–498
  - `choose_packed_target_replica_indices`: 574–596
  - `_resolve_dispatch_weight`: 444–471
- `lib/marin/src/marin/rl/environments/inference_ctx/packed_vllm_worker.py`
  - child process entry point, weight sync loop, generate handler
- `lib/marin/src/marin/rl/replay_buffer.py`
  - `add_batches`, `sample_rollouts`, `_is_rollout_fresh`,
    `_retire_overused_rollouts`
- `lib/marin/src/marin/rl/curriculum.py`
  - `SamplingParams` (n_prompts, n_generations_per_prompt): 66–85
  - `sample_lesson(seed)`: 383–403
- `experiments/exp_iris_rl_regression_direct_gcs_packed.py`
  - PKR-001/003/004 launcher, `DEFAULT_N_PROMPTS=64`,
    `DEFAULT_EVAL_FREQUENCY=1`, `num_rollout_workers=1`
- `experiments/exp_iris_rl_regression_direct_gcs_packed_candidate.py`
  - alternative packed launcher with the same curriculum and replay
    buffer settings

## Data location

- `.agents/wandb_snapshots/packed_rl_analysis/` — per-run JSONs with
  full unsampled history and config for every W&B run above. Reuse
  these instead of re-pulling from the W&B API on the next session.

## Things already settled — do not re-investigate

- **Packed sampler correctness.** PKR-001 passed. The 2 × TP=2
  topology works end to end on one `v5p-8`.
- **Packed sampler throughput.** PKR-001 hits 6,223 tok/s in
  dual-replica train batches, ~1.9 × one `e4ms2` worker's per-worker
  3,336 tok/s. The inference layer is not the bottleneck.
- **Rollout storage / GCS writes.** <0.4 s per batch across all runs.
  Not the bottleneck.
- **Weight transfer.** Fetch time ~14 s, comparable between packed
  and baseline. Not the bottleneck.
- **Trainer compute.** `step_duration_seconds` is identical (~61 s)
  between packed and baseline. Not the bottleneck.
- **Checkpointing.** Expensive but stable. Separate thread in
  `.agents/projects/ckpt_rl.md`.
- **Iris RL retry / resume identity bugs and rollout W&B step bugs.**
  Already fixed on `iris_rl` before this branch split.

## Merge log

### 2026-04-10: Merge origin/main into packed_rl

The `iris_rl` branch was merged into main upstream, causing 30
conflicts with this branch. Resolved by taking main as the canonical
base for all shared RL infrastructure, then layering back packed-
specific additions:

- `InferenceRequestKind` StrEnum in `inference_ctx/base.py`
- `request_kind` parameter threaded through `batch_completions()`
  (base, levanter, vllm) and `MarinEnv.sample()` (base,
  prime_intellect_env)
- `PackedvLLM` imports/exports in `inference_ctx/__init__.py`
- `PackedvLLMInferenceContextConfig` added to `rl_job.py` type union
- Stale `ArrowFlightExportStrategy` reference removed from
  `weight_transfer/__init__.py` (main removed the enum; auto-merge
  left a dangling reference)
- `experiments/exp2039_rl_math500.py` accepted main's deletion
- `experiments/xp_iris_rl_regression_direct_gcs_prod.py` deleted
  (branch-only experiment probe referencing the removed enum)

Packed-specific files survived without conflict: `packed_vllm.py`,
`packed_vllm_worker.py`, `packed_vllm_protocol.py`, the packed
experiment launchers, and `test_packed_vllm_inference_ctx.py`.

Merge commit: `e655aa35b`. Follow-up cleanup: `c3ced0e55`.
