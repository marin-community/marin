# Debugging log for per-statement DPO v5p-8 vs v6e-8 divergence

Investigate why paired per-statement LoRA-DPO runs on `v5p-8` and `v6e-8` with nominally identical hyperparameters show very different training and validation losses.

## Initial status

User reported that these two runs should be comparable but their losses differ substantially:

- `smh_lr1em06_s140_v6e8-a6a62e`
- `smh_lr1em06_s140_v5p8-cc7957`

Relevant planning and experiment context lives in:

- `.agents/projects/continual_alignment_single_statement_dpo.md`
- `.agents/logbooks/continual-alignment-per-statement-dpo.md`

## Hypothesis 1

The runs are not actually taking the same training path because the TPU-specific config changes `per_device_parallelism`, which changes whether gradient accumulation is used.

## Changes to make

No code changes yet. Inspect:

- experiment config in `experiments/posttrain/per_stmt_dpo/common.py`
- trainer microbatch path in `lib/levanter/src/levanter/trainer.py`
- gradient accumulation implementation in `lib/levanter/src/levanter/grad_accum.py`
- W&B histories and run configs for paired runs

## Results

Confirmed the TPU-specific config is not identical:

- `v5p-8`: `per_device_parallelism=16`, `per_device_eval_parallelism=16`
- `v6e-8`: `per_device_parallelism=4`, `per_device_eval_parallelism=4`
- both: `train_batch_size=64`

Given the trainer semantics:

- `microbatch_size = per_device_parallelism * data_axis_size`
- `v5p-8` effectively runs with microbatch size `64`, so no grad accumulation
- `v6e-8` effectively runs with microbatch size `32`, so 2 microsteps / gradient accumulation

This means the two runs are not the same execution, even with the same global batch size and learning rate.

Across all 9 matched `exp1a` pairs, `v5p-8` is consistently worse than `v6e-8` on both training and evaluation metrics. This is systematic, not a one-off bad run.

Fresh 70-step paired runs diverge immediately after the first real update:

- both start at `0.693147`
- by `_step=2`, `v5p-8` remains near `0.695`
- by `_step=2`, `v6e-8` drops to about `0.467`

Initial gradients match almost exactly at `_step=0`, so the divergence does not look like an initialization mismatch.

## Hypothesis 2

The `v6e-8` microbatch / grad-accum path is not numerically equivalent to the non-microbatched path for this DPO + LoRA workload.

## Changes to make

Inspect:

- `lib/levanter/src/levanter/grad_accum.py`
- `lib/levanter/tests/test_grad_accum.py`
- LoRA code in `lib/levanter/src/levanter/lora.py`
- checkpoint / parameter norm traces in W&B

## Results

`grad_accum.py` divides accumulated grads by `num_micro_steps`, and the existing test suite checks equivalence for a simple MLP. However:

- the existing test does not cover the DPO + LoRA path
- the accumulation code uses the quantization overwrite helpers during gradient accumulation
- this leaves room for a bug that only appears on the real training tree

The surprising part is that sampled LoRA parameter norms remain very close between `v5p-8` and `v6e-8`, so this is not a simple exploding-update story. The models may be only modestly different while the per-statement DPO metrics are sensitive to small shifts.

## Hypothesis 3

The evaluation path is introducing the discrepancy.

## Changes to make

Inspect:

- validation hook implementation in `lib/levanter/src/levanter/callbacks/__init__.py`
- reference-eval cache building in `lib/levanter/src/levanter/dpo.py`
- initial eval metrics before any training steps

## Results

Evaluation is not the primary cause of the divergence:

- initial eval at `_step=0` is identical across `v5p-8` and `v6e-8`
- the divergence appears during training before later eval points

There is still an evaluation footgun:

- `eval_loss_loop` averages batch losses uniformly over batches
- the two TPU types use different eval batch sizes

That is worth cleaning up, but it does not explain the observed divergence because:

- `eval/*/dpo_loss` matches `eval/*/loss`
- `_step=0` eval matches exactly across TPU types

Reference eval caches are rebuilt by each run under the same cache key, so that code path is worth hardening, but it is not the first-order explanation for the training divergence.

## Future Work

- [ ] Add a targeted equivalence test for DPO + LoRA with and without microbatching
- [ ] Check whether the quantization overwrite path in `grad_accum.py` behaves identically on the real model tree
- [ ] Fix `eval_loss_loop` to aggregate by example count rather than by batch count
- [ ] Make reference-eval cache metadata serialization stable so runs do not rebuild the same cache repeatedly
