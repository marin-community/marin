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

Pass `valid` from `_fa4_cute.py` into the custom-VJP boundary. Keep the forward
CUTLASS launcher at four runtime tensor arguments because invalid queries are
encoded in `lower_bounds`; the `valid` tensor is consumed by the backward sparse
metadata path. Remove the stray `valid` reference from the non-SM90 backward
fallback launcher so pyrefly sees a consistent contract.

## Results

Focused local checks pass:

- `./infra/pre-commit.py --files ... --fix`
- `uv run pytest lib/levanter/tests/grug/test_fa4_cute_attention.py -k "passes_valid_mask_to_backend or forward_backend_does_not_pass_valid"`

## Future work

- [ ] Capture W&B link, step time, throughput, MFU, and FA4/CE routing once the
      resubmitted synthetic run reaches training steps.
