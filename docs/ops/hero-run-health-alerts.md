# Hero run health alerts

Three Grafana alerts cover hero-run health beyond progress and loss:
[`TrainingProgressStalled`](training-stall-alert-contract.md) watches whether a run steps, and
[`TrainingLossSpike`](training-loss-spike-alert.md) watches what those steps produce. These watch
the telemetry path itself, the optimizer, MoE routing, throughput, evaluation, and Iris retries.
They carry the checks the standalone hero Pushover monitor applies, so an operator who reads Slack
sees what the on-call phone sees.

| Rule | Route | Fires on |
|---|---|---|
| `TrainingTelemetryGone` | `notification=hero-run`: Slack, a Loom triage session, and email | `telemetry_gone` |
| `TrainingOptimizerUnstable` | `notification=hero-run` | `loss_jump`, `grad_norm_high`, `steps_skipped` |
| `TrainingRunHealthDegraded` | `notification=slack`: one Slack message, no triage session | `token_drops`, `router_entropy`, `router_bias`, `throughput_low`, `mfu_low`, `eval_regressed`, `iris_state_stale`, `task_retried` |

The first two page because the run is at risk within the hour. The third announces because an
operator reads it beside the run dashboard and decides. Each rule evaluates once a minute and stays
pending for five minutes. Each firing check is its own alert instance, labeled with its `reason`,
and the hero-run route groups them by logical run.

## Enrollment

A run is watched while **either** side still reports it:

- Iris: a fresh `iris.task_state` row that reports running tasks, the
  [`TrainingProgressStalled`](training-stall-alert-contract.md) contract.
- Levanter: a `phase` sample from `service=levanter` telemetry within the last 15 minutes, for a
  `run_id` beginning with `hero-`.

The stall and loss rules enroll from the Iris side alone. A break in that path therefore stops them
watching a run that is plainly still training, with no signal that it happened. Watching the union
means one path can report on the other, which is what `iris_state_stale` reports.

The Levanter side names the root by taking the longest prefix of its telemetry `job_id` that is
still a hero coordinator root, so `/rav/hero-20260819-coord/grug-train-hero-20260819` is watched as
`/rav/hero-20260819-coord`.

## What fires each check

One bounded scan of `telemetry_v1` per bridge cache interval feeds all three rules. It reads the
newest sample of each metric, the one before it, and the reductions the 15-minute health window
needs. A check reads the newest sample only while that sample is under 15 minutes old, so a restart
cannot fire an alert from the previous attempt's last value.

| Reason | Condition | Metric |
|---|---|---|
| `telemetry_gone` | Newest sample over 10 minutes old while Iris still reports the tasks running | `phase` |
| `loss_jump` | Recent five-minute loss floor over 1.0 above the trailing floor, where the six-sigma band did not already catch it | `train_loss` |
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

Uniform routing over the hero rung's experts is 5.951 entropy, so falling entropy is expert
collapse. The 7% drop limit sits above the intermittent 5% spikes a healthy MoE run shows, and it is
the band the [Training run dashboard](https://grafana.oa.dev/d/marin-training) draws.

The throughput checks count how much of the window sat below the floor rather than averaging it.
That is the median comparison the Pushover monitor makes, and it is what keeps one restart step at
zero from reading as a slow run. A window with fewer than 10 samples says more about sampling than
about the run, and fires nothing.

`telemetry_gone` requires the Iris side to still report running tasks. Telemetry that stops when
Iris also stops counting the tasks is a run that ended, not an incident. A run that has published
nothing at all is left to `TrainingProgressStalled`, which allows the full initialization budget
before it fires — that is the "Levanter never started" case, not a lost telemetry path.

Every other check needs a fresh training phase from the run itself. An initializing run has
published none of these metrics, a finished one leaves its last samples behind for a while, and a
silent one is `TrainingTelemetryGone`'s, so none of the three announces. `iris_state_stale` also
needs a state row that went stale rather than one that never existed: the GCE controllers publish no
`iris.task_state` rollup at all, and a rollup that breaks leaves its last row readable for an hour.
An outage longer than that stops being visible here, which is the limit of this check.

Every quiet run emits a zero-valued `healthy` row and an empty fleet emits one `fleet` row, so a
resolved check clears its instance and `noDataState: Alerting` stays reserved for a malformed or
unavailable response.

## When one fires

1. Open the [Training run dashboard](https://grafana.oa.dev/d/marin-training) and select the run in
   the alert. Execution health carries the attempt age, Iris task counts, and retry events. Token
   drops and Router health carry the MoE signals against the same limits this rule uses.
2. For `telemetry_gone`, separate a dead run from a dead telemetry path. Iris task counts on the
   dashboard come from the same `iris.task_state` rollup that kept the run enrolled, so a healthy
   count there with no Levanter samples points at the telemetry path or a wedged process. Check
   `iris job describe` and the task logs.
3. For `grad_norm_high` or `steps_skipped`, read the Optimizer and Loss spike panels together. A
   gradient norm climbing while the schedule holds the learning rate flat is the shape that precedes
   a spike. Skipped steps mean the optimizer rejected the update, so the weights did not take it.
4. For `loss_jump`, date the change against Run progress. A mixture stage boundary, a resumed run
   reading a different config, and a checkpoint restore all shift the level legitimately. This
   check fires only where [`TrainingLossSpike`](training-loss-spike-alert.md) does not, so what it
   catches is the level shift a wide trailing spread hides rather than a spike out of a quiet run.
5. For `router_entropy` or `router_bias`, compare against the run's own history rather than the
   limit alone. Both move slowly, so the shape over hours is the evidence.
6. For `iris_state_stale`, treat the hero alerting path as degraded: `TrainingProgressStalled` and
   `TrainingLossSpike` no longer enroll this run. Check the controller and its state telemetry.
7. `task_retried` is information, not a fault. It explains a W&B gap and a loss discontinuity while
   the new attempt redoes steps below the high-water mark.

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

Every threshold and window is a constant at the top of `infra/grafana/src/hero_health.py`, and
`TELEMETRY_GONE_AGE` is in `infra/grafana/src/hero_runs.py` because the stall rule defers on the
same number. Changing one takes a redeploy of the Grafana service. Keep the drop and router limits
equal to the bands `infra/grafana/dashboards/training.json` draws, so the panel an operator opens
from the alert agrees with the alert.

A benign firing that repeats across runs is the case for changing a constant. One run's stage
boundary is the case for silencing that run.
