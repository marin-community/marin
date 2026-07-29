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

The direct-cluster resubmissions set `GRUG_JOB_PRIORITY=production`. The
controller persisted all 16 tasks in each child gang with `priority_band=1`.
Kueue then reported active workload preemption rather than the previous
interactive-band capacity wait.

The corrected gangs were admitted, but D-1a attempts r5, r6, and r7 and D-1b
attempt r4 all failed before training. In every case, one member landed on node
`s6xvdgb4` and terminated with `Init:Error stage-workdir`; Iris atomically
bounced the other 15 members. Kubernetes reported clean node conditions before
r7, but r7 reproduced the same failure. The priority propagation fix is
validated independently of this node-local infrastructure blocker.

## Future work

- Decide whether Iris should inherit a parent job's priority when a child leaves
  its priority unspecified.
- Repair or exclude `s6xvdgb4` before rerunning the D-1 positive control.
