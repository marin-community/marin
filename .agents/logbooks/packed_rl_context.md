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

1. **"Eval ownership is the major remaining gap"** — only partially true.
   The primary gap is rollout↔trainer serialization. Eval is a second-order
   effect that makes it worse by collapsing 2×TP parallelism to single-TP.

2. **"Reduce or isolate eval interference"** — necessary but not
   sufficient. Even with perfect eval isolation, one packed worker cannot
   match `e4ms2`'s effective pipeline depth of 2.

3. **"Packed inference is slower than baseline"** — false. PKR-001's
   packed inference at 6,223 tok/s beats `e4ms2`'s per-worker 3,336 tok/s
   by ~1.9× — exactly the 2×TP=2 speedup the design predicted.

### What this implies for next steps

The question to test is no longer "can we isolate eval?" — it's **"can we
break the trainer↔rollout serialization with a single packed worker?"**

Candidate approaches (need design/experiment):

1. **Two packed rollout workers on two `v5p-8` pods.** Same TPU budget as
   the baseline but 4 total replicas feeding the trainer, giving pipeline
   depth 2 *and* 2× per-worker packed throughput. Likely the cleanest win
   on paper but loses the "halve TPU cost" pitch.

2. **Stale-weight rollouts.** Let the single packed worker start the
   next batch against weight `N` while the trainer is still producing
   weight `N+1`. Breaks the serialization at the cost of on-policy
   freshness — need to quantify the off-policy hit.

3. **Rollout buffering / pre-fetch.** Have the packed worker generate
   several batches ahead of the trainer's consumption, buffering in the
   rollout storage. Requires careful bookkeeping of which weight each
   buffered batch was generated against.

4. **Overlap weight fetch/activate with generation.** The current flow
   is `fetch → activate → generate` sequentially per replica. If replica
   0 generated while replica 1 was fetching/activating the next weight
   (ping-pong style), the serial fetch cost (~14s) could be hidden.

The cost pitch for packed only holds if one of these approaches makes a
single packed `v5p-8` genuinely competitive with two independent rollout
workers in cadence, not just in raw tokens/sec.

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
