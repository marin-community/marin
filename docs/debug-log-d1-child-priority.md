# Debugging log for D-1 child priority

Ensure the D-1 Grug training gangs enter Iris's production priority band.

## Initial status

The D-1 parents were submitted directly to `cw-us-east-08a` with
`--priority production`. The controller database persisted their tasks with
`priority_band=1`, but the Fray child gangs had `priority_band=2`. All 16 child
tasks remained Kueue-gated with zero workers assigned.

## Hypothesis 1

`dispatch_grug_training_run` constructs `JobRequest` without setting
`priority`. Fray therefore passes its default value, `0`, and Iris resolves the
child to the interactive band instead of inheriting the parent's production
band.

## Changes to make

Parse an explicit `GRUG_JOB_PRIORITY` value in the Grug dispatcher and pass the
corresponding Iris priority band into `JobRequest.priority`. Keep the default
unspecified so existing launches do not change.

## Results

Pending a direct-cluster submission with `GRUG_JOB_PRIORITY=production`. The
pass criterion is `priority_band=1` on all 16 persisted child tasks before any
training metric is accepted.

## Future work

- Decide whether Iris should inherit a parent job's priority when a child leaves
  its priority unspecified.
