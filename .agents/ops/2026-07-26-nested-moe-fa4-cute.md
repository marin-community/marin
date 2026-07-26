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

Five focused nested/attention contract tests passed and the generated
experiment config resolved `gpu_fa4_thd`; the accelerator-only lowering test
was skipped on the local CPU host. Commit `03bcd5c74b` contains the fallback.
The four failed jobs were stopped after zero optimizer steps and the corrected
smokes were resubmitted as `r2` coordinators.

## Hypothesis 2

The `r2` jobs selected the THD backend but failed before compilation because
`Transformer.__call__` reconstructed its layer masks from `segment_ids` and
dropped the input mask's fixed-shape THD metadata. The prepacked dataset had
already created that metadata.

## Changes to make

Route layer-mask construction through `_layer_attention_masks`, preserving the
metadata while retaining the intended 2,048-token short layers and
full-causal long layers. Extend the contract test to assert both windows and
metadata identity.

## Results

Six focused mask/nested contract tests passed after the change. Pyrefly
reported only the two unchanged model-file errors recorded in the research
logbook. Corrected smoke runs remain pending.

## Hypothesis 3

The `r3` forward preserved its input mask correctly, but the text-LM mixture's
default `pack=None` selected `CausalLmDataset`, which does not attach static THD
segment metadata. Marin's THD canary explicitly applies `with_pack(data, 1)` to
select the fixed-shape one-document representation.

## Changes to make

Apply `with_pack(data, 1)` to the complete training/validation mixture. Add a
launcher contract test that materializes the resolved caches and checks every
component's pack setting.

## Results

Seven focused launcher/mask/nested tests passed. Corrected smoke runs remain
pending.

## Future work

- [ ] Repair or repin the FA4/CuTe dependency stack separately from this
      deadline-bound experiment.
