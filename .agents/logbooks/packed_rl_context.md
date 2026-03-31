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

## Suggested Immediate Next Step

Use this branch to answer one question cleanly:

"Can packed vLLM inference on `v5p-8` preserve enough trainer-facing cadence
under realistic eval settings to justify replacing the direct `e4ms2`
topology?"
