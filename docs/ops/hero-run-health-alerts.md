# Hero run health alerts

Three Grafana rules cover hero-run health beyond stepping
([`TrainingProgressStalled`](training-stall-alert-contract.md)) and loss
([`TrainingLossSpike`](training-loss-spike-alert.md)). They carry the checks the standalone hero
Pushover monitor applies.

| Rule | Route | Fires on |
|---|---|---|
| `TrainingTelemetryGone` | `notification=hero-run`: Slack, a Loom triage session, email | `telemetry_gone`, `run_down` |
| `TrainingOptimizerUnstable` | `notification=hero-run` | `loss_jump`, `grad_norm_high`, `steps_skipped` |
| `TrainingRunHealthDegraded` | `notification=slack`: one Slack message, no triage session | `token_drops`, `router_entropy`, `router_bias`, `throughput_low`, `mfu_low`, `eval_regressed`, `iris_state_stale`, `task_retried` |

Each rule evaluates once a minute and stays pending for five. Each firing check is its own alert
instance labeled with its `reason`; the hero-run route groups them by logical run.
Every `notification=hero-run` group, including `TrainingProgressStalled` and
`TrainingLossSpike`, carries `operator_behavior=hero`. The bridge resolves that
trusted behavior to the `operator:hero` channel on Loom's shared `ops` profile,
giving Hero operations a durable coordinator separate from generic Grafana incidents.
The Hero behavior gives that coordinator stable discovery and query guidance,
not a precomputed evidence snapshot. Starting from the alert's cluster, run, and
job labels, the coordinator queries current execution UIDs and telemetry, derives
prior coordinator roots for the same logical run, and then inspects exact Iris
state, event, and task-log prefixes. It compares tasks and ranks without a fixed
error-signature list and treats the first pass as lead gathering, not a diagnosis.

## Enrollment

A run is watched while either an `iris.task_state` row fresh within 90 seconds reports running tasks
(the [`TrainingProgressStalled`](training-stall-alert-contract.md) contract) or a `hero-`-prefixed
`run_id` published `phase` telemetry in the last hour. The Levanter side takes the longest
prefix of its `job_id` that is a hero coordinator root, so
`/rav/hero-20260819-coord/grug-train-hero-20260819` is watched as `/rav/hero-20260819-coord`.

The stall and loss rules enroll from the Iris side alone, so a break in that path stops them watching
a training run with no signal that it happened. That is what `iris_state_stale` reports.

## What fires each check

| Reason | Condition | Metric |
|---|---|---|
| `telemetry_gone` | Last phase was training, over 10 minutes old, while Iris still reports the tasks running | `phase` |
| `run_down` | The same, and Iris no longer reports them running | `phase` |
| `loss_jump` | Recent five-minute loss floor over 1.0 above the trailing floor of the same attempt, where the six-sigma band did not already catch it | `train_loss` |
| `grad_norm_high` | Newest value above 2.0 | `grad_norm_total` |
| `steps_skipped` | 3 or more skipped steps in 15 minutes | `optim_skipped_step` |
| `token_drops` | Newest value above 7% | `moe_drop_fraction` |
| `router_entropy` | Newest value below 5.92 | `train_router_routing_entropy_mean` |
| `router_bias` | Newest bound over 400 from zero | `train_router_bias_max`, `train_router_bias_min` |
| `throughput_low` | Most of 15 minutes below 2.0M tokens per second | `throughput_tokens_per_second` |
| `mfu_low` | Most of 15 minutes below 15% | `throughput_mfu` |
| `eval_regressed` | Newest evaluation, within 30 minutes, worse than the one before it | `eval_paloma_macro_loss` |
| `iris_state_stale` | Newest `iris.task_state` row for the root over 5 minutes old | `iris.task_state` |
| `task_retried` | A controller retry or gang requeue in the last 15 minutes | `iris.task_event` |

Uniform routing over the hero rung's experts is 5.951 entropy, so falling entropy is expert collapse.
The 7% drop limit sits above the intermittent 5% spikes a healthy MoE run shows.

One bounded `telemetry_v1` scan per bridge cache interval feeds all three rules, reduced over a
single execution: the newest attempt process zero reports. `loss_jump` reads its two loss windows
against each other, so it filters them to that same execution. A retry keeps the run ID and takes a
new `execution_uid`, so partitioning on the run alone would sum one attempt's skipped steps into the
next and compare evaluations across a restore that redid steps. Process zero is the stable choice
because Levanter publishes tracker metrics only from it. A check reads a newest sample only while it
is under 15 minutes old.

The throughput checks count how much of the window sat below the floor rather than averaging it —
the median comparison the Pushover monitor makes, which keeps one restart step at zero from reading
as a slow run. Fewer than 10 samples fires nothing.

Silence pages either way; the Iris state only picks the reason. `telemetry_gone` means Iris still
counts the tasks, so the telemetry path or the process is the suspect. `run_down` means it no longer
does, so the job probably exited — the most severe case, and one `TrainingProgressStalled` cannot
report, since its enrollment needs running tasks.

Both need the run's last phase to have been training. A finished tracker ended on purpose, and a run
still initializing is `TrainingProgressStalled`'s, which allows the full startup budget. That
exclusion carries most of the traffic: `hero-` names a smoke test as often as a production run, and
40 hours of production telemetry held one production run against about 25 short dev runs, every one
of which died during initialization.

The phase heartbeat is read over a day rather than the hour the other metrics use, because an outage
is measured from the last heartbeat however long ago it was. Levanter enrollment runs an hour, so a
silent run stays watched while the rule counts out its threshold and pending period.

Every other check needs fresh phase telemetry reporting the training phase, so an initializing,
finished, or silent attempt announces nothing. `iris_state_stale` additionally needs a row that went
stale rather than one that never existed: the GCE controllers publish no `iris.task_state` rollup at
all, and a broken rollup leaves its last row readable for an hour. A longer outage stops being
visible, which is the limit of this check.

Quiet runs emit zero-valued `healthy` rows and an empty fleet emits one `fleet` row, so
`noDataState: Alerting` stays reserved for a malformed or unavailable response.

## When one fires

1. Open the [Training run dashboard](https://grafana.oa.dev/d/marin-training) and select the run.
   Execution health carries the attempt age, Iris task counts, and retry events; Token drops and
   Router health draw the same limits these rules use.
2. `telemetry_gone` or `run_down`: start from `iris job describe` and the task logs. Healthy Iris
   task counts with no Levanter samples point at the telemetry path or a wedged process; no running
   tasks means the job exited and the question is why.
3. `grad_norm_high` or `steps_skipped`: read the Optimizer and Loss spike panels together. A gradient
   norm climbing while the schedule holds the learning rate flat precedes a spike. Skipped steps mean
   the optimizer rejected the update, so the weights did not take it.
4. `loss_jump`: date the change against Run progress. A mixture stage boundary, a resumed run reading
   a different config, and a checkpoint restore all shift the level legitimately.
5. `router_entropy` or `router_bias`: both move slowly, so the shape over hours is the evidence, not
   the distance from the limit.
6. `iris_state_stale`: the hero alerting path is degraded — `TrainingProgressStalled` and
   `TrainingLossSpike` no longer enroll this run. Check the controller and its state telemetry.
7. `task_retried` is information, not a fault. It explains a W&B gap while the new attempt redoes
   steps below the high-water mark.

Verify what a rule saw with a bounded Finelog query:

```sql
SELECT
  name,
  value,
  to_timestamp_millis(timestamp_ms) AS observed_at
FROM "telemetry_v1"
WHERE service = 'levanter'
  AND run_id = '<hero-run-id>'
  AND name IN ('phase', 'grad_norm_total', 'optim_skipped_step', 'moe_drop_fraction',
               'train_router_routing_entropy_mean', 'throughput_mfu')
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '15 minutes') * 1000 AS BIGINT)
ORDER BY timestamp_ms DESC, seq DESC
LIMIT 50;
```

## Tuning

Thresholds and windows are constants at the top of `infra/grafana/src/hero_health.py`;
`TELEMETRY_GONE_AGE` is in `hero_runs.py` because the stall rule defers on the same number. Changing
one takes a redeploy. `training.json` draws the drop and router limits as bands, and a test asserts
the two agree.

A benign firing that repeats across runs is the case for changing a constant. One run's stage
boundary is the case for silencing that run.
