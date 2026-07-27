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

## Hypothesis 4

The `r4` jobs supplied valid THD metadata and reached the FA4 THD import, but
failed on the same missing `ThrMma` symbol. The current dependency lock combines
FlashAttention 4.0.0b16 and Quack 0.5.0 with CUTLASS DSL 4.6.0; the former two
still depend on the CUTLASS 4.5 API.

## Changes to make

Restore the previous `nvidia-cutlass-dsl[cu13]>=4.5.2,<4.6` constraints and
the root resolver override that suppresses the overlapping base wheel. Test the
exact import on one GB200 before another multi-rack allocation.

## Results

The single-GPU import canary passed under CUTLASS 4.5.2, but the four-arm `r5`
smoke failed during actual kernel compilation. CUTLASS 4.5 is not compatible
with the JAX 0.11 compiler integration.

## Hypothesis 5

The repository's CUTLASS 4.6 upgrade left Quack 0.5.0 pinned transitively.
Quack 0.5 imports the CUTLASS 4.5 `ThrMma` API, while Quack 0.6.1 explicitly
requires CUTLASS 4.6.0. FA4 4.0.0b16 permits Quack 0.6.1.

## Changes to make

Restore CUTLASS DSL 4.6.0 and require `quack-kernels>=0.6.1,<0.7` in the GPU
extras. Add a single-GPU integration test that compiles and executes both FA4
THD forward and backward kernels.

## Results

The corrected dependency pair passed import and lowering. A forward-only
one-GB200 canary compiled in 13 seconds and executed successfully with finite
output. The combined forward/backward canary compiled and dispatched but did
not return from device execution. Matching the upstream dense-backward
`subtile_factor=2` produced the same hang. Iris reported zero preemptions and
zero task failures while each process remained resident, so both exact canary
jobs were stopped.

The experiment launcher now permits the `reference` attention backend through
an explicit validated setting. All four scientific arms use that backend for
the remainder of this window. Nine focused Levanter attention tests, two
launcher tests, and the required pre-commit entry point passed in commit
`e2f4036439`.

## Hypothesis 6

The first reference-attention smoke returned a finite forward loss but
nonfinite gradients in the E256 control and both nested arms. The E128 control
showed the same signature. The reference runs had retained `pack=1`, although
that fixed-shape representation was introduced only for the THD kernel.
Padding query rows have no eligible keys; reference softmax evaluates an
all-negative-infinity row and contaminates backward.

## Changes to make

Apply `with_pack(data, 1)` only when the selected backend is
`gpu_fa4_thd`. Reference attention uses ordinary causal examples. Reduce the
fallback proxy to d768 and length 2,048 while retaining the original
2,097,152 tokens per step through a 1,024-sequence global batch.

## Results

Two materialized-launcher tests assert the THD and reference data contracts,
including the amended proxy overrides. The corrected GPU smoke is pending.

## Future work

- [ ] Keep Quack and CUTLASS release lines constrained as a tested pair.
- [ ] Reproduce the FA4 dense backward hang with the smallest SM100 tensor
  shape and collect a device-side trace.
- [ ] Re-enable FA4 for nested-MoE throughput measurements only after the
  backward canary completes.
