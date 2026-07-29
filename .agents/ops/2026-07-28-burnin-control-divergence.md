# Nested MoE burn-in: control loss divergence

Determine why the NEST-BURN control loss does not reproduce the
`aug-dk-d768-ev-sw2k-g4-nomtp-noconv-f1` reference, establish a reproducible
1,000-step control gate, and prevent the nested experiment from resuming until
that gate passes.

## Initial status

The reference reaches training loss near 3 by update 1,000 and 2.21 at update
16,839. NEST-BURN-001 remains near 6 at update 1,000 and finishes at 4.73.
NEST-BURN-002 shows the same bad regime. The active 100B control and queued
fixed25 continuation were stopped on 2026-07-28 at 23:14 UTC.

## Hypothesis 1: batch size, optimizer, or mixture weights differ

W&B configuration comparison rules this out for NEST-BURN-001:

- both use global batch 32, sequence length 8,192, seed 0, and 16,840 updates;
- both use the same 191 Datakit training/evaluation components and phase
  weights;
- MuonH and AdamH peak learning rates agree within rounding
  (`0.008379826`/`0.001933806` for the reference);
- beta values, epsilon, linear schedule, warmup, minimum LR ratio, and weight
  decay agree.

NEST-BURN-002 later changed global batch and optimizer horizon, but it already
inherited the NEST-BURN-001 control failure.

## Hypothesis 2: the data packing and model contracts differ

Confirmed by the serialized W&B configs and original Iris launch request.

The reference leaves component `pack` unset and keeps
`block_cross_document_attention=true`. NEST-BURN explicitly applies
`with_pack(data, 1)` to every component and disables cross-document attention.
The launcher comment states that `pack=1` permits at most one document per
example; padding is excluded from loss. Nominal `batch * sequence_length` token
accounting therefore overstates the useful training tokens seen by NEST-BURN.

The reference also uses the augmented Grug source bundle
`adc2aad8a60b45f4a105d4d6e4134cb7fff350caa77d7e56ab23fbe66bd3479b`
from dirty tree `43cb2e3738` on branch `AugQualityTests`, whereas NEST-BURN uses
the nested branch's model and router implementation. The reference contract
includes explicit GatedNorm, attention gates, XSA, QB routing, one shared
expert, global attention every four layers, and the CUDA 13/JAX 0.10.1 stack.
NEST-BURN's final per-layer router-bias norms are roughly 600--1,000, while the
reference's complete stacked router-bias norm is 22.73. This is an independent
warning that the control paths are not equivalent.

## Changes to make

1. Reuse the reference Iris bundle and exact launch request on one 8xH100 node.
2. Preserve the 16,840-step optimizer horizon; stop after update 1,000 rather
   than shortening the schedule.
3. Compare pointwise and windowed loss, LR, Paloma, throughput, and routing
   diagnostics against the reference.
4. Port nesting onto the validated augmented control and restore dense packing
   only after the reproduction passes.

## Reproduction gate

The immutable reference bundle was relaunched at 23:31 UTC as
`/power/nest-burn-control-augdk-repro1000-r1-coord`. Only the Iris job name,
W&B run ID, submission date, credentials, and priority band changed. The
training configuration and 16,840-step optimizer horizon are unchanged.

The gate passes only if all of the following hold through the update-1,000
evaluation:

- the median absolute pointwise training-loss difference is at most 0.05 nat;
- the median loss over updates 900--999 differs by at most 0.05 nat;
- the learning-rate schedule agrees to relative tolerance `1e-6`;
- Paloma macro loss at update 1,000 differs by at most 0.05 nat; and
- there is no NaN, task restart, collective stall, or sustained routing
  pathology.

These tolerances were recorded before observing the reproduction results.

## Results

The reproduction passed and was stopped after its update-1,000 evaluation.
W&B serialized configs differ only in run identity and output/checkpoint paths.
There were no training failures or task restarts.

| Metric | Reference | Reproduction | Difference |
| --- | ---: | ---: | ---: |
| Median training loss, updates 900--999 | 3.22680 | 3.22943 | +0.00263 |
| Median step duration, updates 900--999 | 0.45687 s | 0.45622 s | -0.00065 s |
| Median throughput, updates 900--999 | 573,781 tok/s | 574,595 tok/s | +0.14% |
| Paloma macro loss, update 1,000 | 4.22119 | 4.22487 | +0.00368 |
| Uncheatable macro loss, update 1,000 | 3.79053 | 3.78847 | -0.00206 |

Across the 999 matched training updates, median absolute pointwise loss error
was 0.00229 nat, p95 was 0.01381 nat, and the learning-rate schedules matched
exactly. The largest single-batch loss difference was 0.06139 nat; this was not
a preregistered gate because single-batch ordering is sensitive to asynchronous
loader timing.

The packing change is large enough to explain a major part of the invalid
control. Cache-ledger row and token counts give a phase-0 mixture-weighted mean
document length of 2,297.7 tokens. A one-document `pack=1` example can therefore
fill at most 28.05% of an 8,192-token sequence on average, meaning nominal token
accounting overstates useful targets by at least 3.57x. The bound is optimistic
because documents longer than the sequence length are truncated. It also does
not capture the change from flat-token sampling to document sampling.

The old control cannot isolate this data error from the simultaneous model and
router implementation change. The next arms must start from the reproduced
augmented bundle and alter only the preregistered nesting behavior.

## Corrected experiment gate

The fixed-prefix architecture was ported onto the immutable augmented source.
The matched control
[`nest-augdk-e256-4b-r2`](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-e256-4b-r2)
passed the update-1,000 gate:

| Metric | Reference | Corrected control | Difference |
| --- | ---: | ---: | ---: |
| Median training loss, updates 900--999 | 3.22680 | 3.23041 | +0.00361 |
| Paloma macro loss, update 1,000 | 4.22119 | 4.21913 | -0.00206 |
| Uncheatable macro loss, update 1,000 | 3.79053 | 3.78865 | -0.00187 |

Median absolute pointwise loss error over updates 2--1,000 is 0.00218 nat,
p95 is 0.01189 nat, and the LR schedule matches exactly. The paired
[`fixed25`](https://wandb.ai/marin-community/marin_moe/runs/nest-augdk-fixed25-4b-r2)
arm continues to the 4.42B-token endpoint.

## Future work

- [x] Add an operational control gate covering packing, source/model version,
      optimizer horizon, and batch semantics.
- [ ] Log valid non-padding target tokens per update instead of relying only on
      nominal token accounting.
- [x] Port nested routing to the augmented implementation with independent QB
      state for E256/E128/E16.
