# Nested MoE continuation: NaN after checkpoint resume

Continue the four-arm nested-MoE study from its step-8,192 checkpoints without
changing the paired architecture comparison.

## Initial status

All four `extend16b-r28` jobs loaded the intended full-state checkpoints.
E256 reported a NaN training loss at step 8,197 and E128 at step 8,194. The
jobs were stopped before treating the continuation as experimental evidence.

## Hypothesis 1

The optimizer schedule is rebuilt from `num_train_steps`. Extending that value
from 8,192 to 38,912 changed the learning rate at restored step 8,192 from the
old schedule's 5% floor to about 80% of peak:

- MuonH: `0.00019696` to `0.00315176`;
- Adam: `0.00004545` to `0.00072733`.

The 16x discontinuity applies to every arm and explains the immediate control
failures. It is independent of nested routing.

## Changes to make

Represent continuation as a second optimizer cycle with an explicit endpoint
at the restored step. Require the checkpoint step and rewarmup length whenever
`NESTED_RESUME_FROM` is set. A 512-step linear rewarmup begins at the exact
old-schedule floor, reaches the original peak, and then decays through the
remaining continuation.

Launch the replacement wave from the immutable step-8,192 checkpoints, not
the NaN-tainted `r28` outputs.

## Results

The schedule-continuous `r29` retry logged the intended low rates:

- MuonH `0.00019696` at step 8,192;
- Adam `0.00004545` at step 8,192.

E256 nevertheless became NaN at step 8,195 and E128 at step 8,194. Schedule
discontinuity was therefore a real defect in `r28`, but not the root cause of
the full-state continuation failure. The original checkpoints have finite
weights and validation loss; their restored optimizer state cannot be used for
this extension without a deeper optimizer/checkpoint investigation.

The replacement experiment will initialize same-architecture weights from the
original checkpoints and create fresh optimizer state. It uses 10% of the
pretraining peak learning rates with a 512-step warmup, then linear decay. This
changes the continuation question from seamless training-state resume to
paired continued pretraining from identical 4.295B-token endpoints.

## Future work

- [ ] Isolate which restored optimizer-state leaf first becomes non-finite.
- [ ] Confirm all four weights-only continuations survive their 512-step
  warmup and peak.
