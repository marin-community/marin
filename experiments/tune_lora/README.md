# LoRA DPO Tuning Plan

These experiments use the executor framework on top of canonical `train_dpo` so
preemption, checkpoint discovery, and W&B resume all follow the same path as
other executor-managed runs.

The already-successful run `endlboq3` is intentionally excluded:

- Bloom SpecEval v2
- marin-8b-instruct
- `beta=0.1`
- `learning_rate=7.5e-6`
- `seed=2`
- `train_batch_size=64`

All experiments in this directory:

- use LoRA with `reference.type=adapter_base`
- use cached reference eval log-probs via `reference_eval_cache.mode=build_or_load`
- keep `r=64`, `alpha=64`, `dropout=0.0`, `zero_init_b=true`, `target_modules=null`
- keep `train_seq_len=max_seq_len=4096`
- run on `v5p-8` with batch size `64`

Primary runs:

- `beta0p1_lr5e6_seed2_b64.py`
  - Lower-LR neighbor around the current best point.
- `beta0p1_lr6p25e6_seed2_b64.py`
  - Denser local refinement below the current best point.
- `beta0p1_lr8p75e6_seed2_b64.py`
  - Denser local refinement above the current best point.
- `beta0p1_lr1e5_seed2_b64.py`
  - Higher-LR neighbor around the current best point.
- `beta0p1_lr7p5e6_seed0_b64.py`
  - One seed-robustness check at the current best LR without repeating `endlboq3`.

Deferred follow-up, not scripted here yet:

- a second seed check at the winning LR if the LR picture is still ambiguous
- batch-size reduction at the current best LR if the seed/LR runs still suggest
  a large-batch optimization penalty
- rank sweep at the winning LR, starting with `r=32` and `r=128`, only after LR
  and batch are settled
