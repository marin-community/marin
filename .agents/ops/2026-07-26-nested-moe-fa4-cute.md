# Grug MoE: FA4/CuTe import failure

Restore the preregistered nested-MoE smoke gate without changing its model,
data, optimizer, or routing treatments.

## Initial status

All four 64-GB200 smoke arms were admitted on `cw-us-east-08a`. The first two
arms to finish compilation failed before optimizer step 0. Every worker
reported the same backward-pass traceback while importing
`flash_attn.cute.flash_bwd_preprocess`:

```text
AttributeError: module 'cutlass.cute.core' has no attribute 'ThrMma'
```

W&B contained no history rows. The Iris jobs had no preemptions or hardware
diagnostics.

## Hypothesis 1

The `gpu_fa4_cute` backward path is incompatible with the `quack` and CUTLASS
packages in the current GB200 image. This is independent of the nested-expert
change because both the unmodified E256 and E128 controls failed in the same
import path.

## Changes to make

Use the existing `gpu_fa4_thd` Grug backend in the experiment launcher. Marin
already has a GPU lowering test for the model's GQA sharding with this backend.
Keep every preregistered experimental variable fixed.

## Results

Pending corrected smoke runs.

## Future work

- [ ] Repair or repin the FA4/CuTe dependency stack separately from this
      deadline-bound experiment.
