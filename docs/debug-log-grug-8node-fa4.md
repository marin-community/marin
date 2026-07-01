# Debugging log for Grug 8-node FA4 profile

Goal: get the PR #6395 May d=2560 Grug MoE profile running on 8 CoreWeave H100
nodes with `processes_per_task=8`, AdamH, and synthetic data if tokenization
blocks the run.

## Initial status

The Slimg Pajama profile job `/dlwh/iris-run-job-20260701-182817` stalled in
tokenization at 8/48 shards complete with 40 shards in flight and 0.0/s progress.
It was stopped and replaced by the synthetic fallback job
`/dlwh/iris-run-job-20260701-185027`.

## Hypothesis 1

The synthetic fallback should bypass Zephyr and immediately run the GPU training
child. The launcher needed an in-memory dataset because the wrapper advertised
`--data synthetic` while the Python launcher only accepted `slimpajama`.

## Changes to make

Add a deterministic `SyntheticGrugDataset` to
`experiments/grug/moe/launch_cw_may_d2560.py`, route `MAY_DATA=synthetic` through
`DirectDatasetComponent`, and omit Slimg Pajama from the step dependencies in
synthetic mode.

## Results

Synthetic mode lowered with zero dataset dependencies, and the fallback job
started an 8-task GPU training child directly. The child then failed before step
1 with:

`TypeError: fa4_cute_attention_forward() missing 1 required positional argument: 'valid'`

## Hypothesis 2

The merged FA4 frontend/backend contract was inconsistent. The high-level
`gpu_fa4_cute_attention` path computed `valid` but the committed call did not pass
it through to `fa4_cute_attention_forward`.

## Changes to make

Pass `valid` from `_fa4_cute.py` into the custom-VJP boundary and into the
forward CUTLASS launcher as `valid.astype(jnp.int32)`. Remove the stray `valid`
reference from the non-SM90 backward fallback launcher so pyrefly sees a
consistent contract.

## Results

Focused local checks pass:

- `./infra/pre-commit.py --files ... --fix`
- `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -k "passes_valid_mask_to_backend or forward_backend_does_not_pass_valid"`

The next run `/dlwh/iris-run-job-20260701-190324` got past the frontend
signature error and failed in the CUTLASS JAX primitive with:

`ValueError: Must have the same number of specs (5) as tensors (4).`

The forward launcher input spec still expects `valid`, so the backend call now
passes `valid.astype(jnp.int32)` into `cutlass_call`; the "valid only for
backward" hypothesis was wrong for this version of the kernel boundary.

## Hypothesis 3

The 8-node synthetic job `/dlwh/iris-run-job-20260701-190730` got past FA4
signature/spec errors and selected fused CE `batched_xla`, but failed before any
completed step. The babysitter found the primary failure during `jit_train_step`
at `jax.block_until_ready(metrics["train/loss"])`: NCCL `ncclCommSplit` failed
with an unhandled CUDA error whose last NCCL warning was CUDA out-of-memory.
The later shutdown-barrier and peer-disconnect logs are fallout.

The profile default `MAY_REMAT=save_moe` may be saving too much forward state:
it keeps MoE dispatch input, expert hidden, dispatch output, and MoE output for
each block so backward avoids recomputing expert dispatch/collectives. At this
shape, that tradeoff can exceed HBM before step 1. `MAY_REMAT=recompute_all` is
the supported lower-memory mode.

## Changes to make

Change the d=2560 CoreWeave profile default remat mode to `recompute_all` in
the Python launcher and shell wrapper, then resubmit the same 8-node synthetic
probe with explicit `--remat recompute_all`.

## Results

The resubmitted full-batch job `/dlwh/iris-run-job-20260701-220304` repeated the
same first-step NCCL/CUDA OOM under `MAY_REMAT=recompute_all`, so remat alone is
not sufficient at the original batch/model layout.

A lower-memory diagnostic with `MAY_BATCH=32`, `MAY_EXPERT_AXIS=4`, and
`MAY_MODEL_AXIS=2` failed before reaching training. The failure was a
Levanter trainer validation bug rather than an OOM:

`ZeroDivisionError: integer modulo by zero`

The trainer validation mesh still treated all 64 devices as batch-parallel while
Grug's compact mesh excludes the `model` axis from batch sharding. Add a Grug
MoE-specific trainer validation mesh whose batch mapping is
`replica, data, expert`, leaving `model` out of the batch axis. For the active
diagnostic shape this gives `data=8`, `expert=4`, `model=2`, and 32 batch
shards, matching the run's global batch size.

The first version of this fix incorrectly pinned `replica_dcn=1` in
`TrainerConfig.mesh`; on CoreWeave the trainer sees `num_slices=8`, so
`/dlwh/iris-run-job-20260701-222846` failed during mesh validation with:

`ValueError: DCN product 1 must equal num_slices 8.`

Leave `replica_dcn=-1` as the DCN absorber and map the trainer batch axis over
`replica_dcn, data, expert`. With eight slices and eight devices per slice, the
lower-batch diagnostic shape gives `replica_dcn=8`, `data=1`, `expert=4`,
`model=2`, and 32 batch shards.

## Future work

- [ ] Capture W&B link, step time, throughput, MFU, and FA4/CE routing once the
      resubmitted synthetic run reaches training steps.
