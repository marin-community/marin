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
- resolve `num_epochs=1.0` to `1,700` train steps
- run DPO eval at step `0`, steps `425`, `850`, `1275`, and final
- run Paloma + Uncheatable LM evals on the same cadence under `lm_eval/...`

Scripted sweep grid:

- learning rates:
  - `1e-6`
  - `2.5e-6`
  - `3.75e-6`
  - `4.5e-6`
  - `5e-6`
  - `6.25e-6`
  - `7.5e-6`
  - `8.75e-6`
  - `1e-5`
- seeds:
  - `0`
  - `2`
- total wrappers:
  - `18`

Filename pattern:

- `beta0p1_lr<lr_tag>_seed<seed>_b64.py`

Deferred follow-up after this scripted grid:

- batch-size reduction at the winning LR if the seed/LR runs still suggest a large-batch optimization penalty
- rank sweep at the winning LR, starting with `r=32` and `r=128`, only after LR and batch are settled
