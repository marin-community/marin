# Qwen distillation: first-step device OOM

Validate an online Qwen3-32B to Qwen3-0.6B distillation step at sequence
length 2,048 and effective batch size 8 on one `GB200x4` node.

## Initial status

The 12-step `QD-C1` smoke staged both models, initialized four GB200 devices,
loaded the regional token cache, loaded all 17 teacher weight shards, traced
the train step, and lowered it to HLO. The first device execution failed on all
four devices while allocating 3.91 GiB. The run did not complete a step or
write a checkpoint.

- Job: `/power/qwen-distill-smoke-651629`
- Task: `run_levanter_distill_lm-6bf6582f`
- W&B: `QD-C1-smoke-seed-0`
- Failure: `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 3.91GiB`

## Hypothesis 1

The un-microbatched teacher and student forward pass, exact float32
full-vocabulary KL, and student backward pass exceed the device peak despite
the four-way model and vocabulary sharding. The failure occurs during the
compiled step rather than model staging or HLO lowering.

## Changes to make

Keep the effective batch size and sequence length fixed, but accumulate two
microbatches of four examples. This preserves the objective and stopping
condition while reducing activation and logit peak memory.

## Results

The `microbatch_size=4` smoke failed on the same 3.91 GiB allocation. The
allocation size and timing were unchanged, ruling out batch-dependent
activations as the primary cause.

## Hypothesis 2

The entry point constructs a concrete random 32B teacher before calling
`Trainer.initial_state`. The trainer wraps that object in a model factory, but
the original `initial_model` local remains live after the trainer creates its
mixed-precision state. Loading the checkpoint teacher therefore retains an
unneeded third teacher representation until training exits.

## Changes to make

Pass a lazy model factory to `Trainer.initial_state` and derive the trainable
filter from its shape. The factory closes over configurations and keys, not
concrete arrays, so replacing the random state teacher can release it before
the first compiled step.

## Results

The lazy-initialization smoke completed all 12 steps on one `GB200x4` node,
saved `checkpoints/step-11`, and exited successfully. The first training step
took 15.7 seconds including compilation. Subsequent training loss was finite.

- Job: `/power/qwen-distill-smoke-f8672e`
- Task: `run_levanter_distill_lm-25bf5d8d`
- Commit: `f8672e5abd`
- Checkpoint:
  `s3://marin-us-east-02a/marin/qwen-distillation/smoke/qd-c1-seed-0/2026.07.26.5/checkpoints/step-11`

This confirms that retaining the random 32B initialization, rather than the
online objective itself, caused the first-step OOM.

## Hypothesis 3

Training loss remained finite, but all validation metrics were `NaN`.
Validation batches contain padded positions whose loss weight is zero. The
evaluator multiplied per-position loss by its weight, so a nonfinite loss on a
padded position propagated through `NaN * 0`.

## Changes to make

Use an explicit zero-weight mask before accumulating tagged and labeled
evaluation losses. Add regression tests whose ignored positions contain
nonfinite losses, then repeat the smoke and require finite validation metrics.

## Results

The explicit evaluator mask passed its regression tests but the device smoke
still reported nonfinite validation loss. W&B retained finite student
parameter norms, finite student gradient norms, and a finite distillation loss
through step 11 (`9.1567`). This rules out student corruption and zero-weight
aggregation as the primary cause.

- Job: `/power/qwen-distill-smoke-c39033`
- Commit: `c3903335cb`
- Training throughput at step 11: 27,515 tokens/s
- Validation path: unreduced fused linear cross-entropy

## Hypothesis 4

The GPU fused cross-entropy path is nonfinite when asked for unreduced
per-position Qwen losses in this shape, while the independently computed
full-logit KL remains finite.

## Changes to make

Compute validation NLL from materialized float32 logits as
`logsumexp(logits) - logits[target]`. The online KL already materializes this
logit shape successfully, and the student-only evaluation has lower peak
memory than the training step. Keep explicit masking in the evaluator as a
separate correctness fix.

## Results

The stable validation path completed on device. NLL was finite at every
evaluation and decreased from `11.210` after the first update to `8.679` at
step 11. The run saved and committed its final checkpoint.

- Job: `/power/qwen-distill-smoke-8dd74b`
- Commit: `8dd74b043b`
- Checkpoint:
  `s3://marin-us-east-02a/marin/qwen-distillation/smoke/qd-c1-seed-0/2026.07.26.7/checkpoints/step-11`

The full experiment matrix uses the materialized validation NLL for both
hard-label controls as well, preserving a common evaluation implementation.

## Future work

- [ ] Record steady-state step throughput after the corrected evaluation
  smoke.
- [ ] Inspect whether excluding frozen teacher leaves from the differentiated
  model reduces compiler memory further.
