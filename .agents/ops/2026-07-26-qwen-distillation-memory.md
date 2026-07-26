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

Pending the lazy-initialization smoke.

## Future work

- [ ] Measure step throughput and peak device memory at microbatch sizes 4 and
  2 if size 4 remains over capacity.
- [ ] Inspect whether excluding frozen teacher leaves from the differentiated
  model reduces compiler memory further.
