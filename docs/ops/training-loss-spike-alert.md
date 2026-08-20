# Hero training loss-spike alert

`TrainingLossSpike` is a critical Grafana alert for active Iris root jobs named `hero-*-coord`. It posts one Slack message per root job and opens a Loom triage session on that thread, the same `notification=hero-run` route [`TrainingProgressStalled`](training-stall-alert-contract.md) uses. The alert does not kick, restart, or reconfigure a job; the decision it asks for is a person's, which is to keep training or to stop and resume from an earlier checkpoint.

It pages because a hero run diverging unwatched costs more than a false page does, and a silence answers a false page. Expect benign firings at mixture stage boundaries; silence the run for that window rather than widening the band for every run.

Enrollment is the same contract as [`TrainingProgressStalled`](training-stall-alert-contract.md): a root job is eligible while its latest `iris.task_state` row is at most 90 seconds old, reports at least one running task, and matches `%/hero-%-coord`. Both rules read one enrollment query per bridge cache interval, so they share a single `iris.task_state` scan.

## What fires it

The bridge reads `train_loss` from `service=levanter` telemetry for the enrolled run IDs over one bounded hour and reduces it, in SQL, to two windows per run: a baseline covering `[now-60m, now-5m)` and a recent window covering the last five minutes.

The run alerts when either condition holds:

- The lowest `train_loss` in the recent window exceeds `mean(baseline) + max(0.05, 6 * stddev(baseline))`. Labeled `spiking`.
- Any reduction of the recent window is not finite, which is how a loss that has gone to NaN or infinity arrives. Labeled `not_finite`.

Six standard deviations is the band Levanter's `SkipStepConfig` rejects an individual step on, so a run with skip-step enabled and a run without it are judged against the same shape. The absolute floor of 0.05 keeps a very stable run from alerting on a rise too small to act on.

Reducing the recent window to its floor is the load-bearing choice. A single excursion, which skip-step already handles by discarding the step, raises the window's peak and its mean and leaves its minimum where it was. A level that shifts up and stays raises all three. The alert is therefore quiet for transients and fires for sustained divergence, at the cost of staying quiet for a loss that oscillates in and out of the band.

The baseline is the run's own trailing history, so the band moves with training. Early in a run, loss falls quickly and its trailing standard deviation is wide, which is when a lone excursion means least. Late in a run the band tightens and a smaller persistent rise clears it.

A run reports `warming_up` with fewer than 20 baseline or 5 recent samples, and `healthy` otherwise. Both are zero-valued rows, which resolves a firing instance. With no eligible roots the bridge returns an explicit zero-valued `fleet` row; `noDataState: Alerting` is reserved for a malformed or unavailable response.

`spiking` and `not_finite` are separate alert instances, but the route groups by root job and not by `reason`, so a run that spikes and then diverges to NaN threads under the message it already sent.

## When it fires

1. Open the [Training run dashboard](https://grafana.oa.dev/d/marin-training) and select the run named in the alert. Its loss panel is the same `train_loss` series the rule reduces, and the spike panel puts the peak beside the optimizer's own rejection threshold and its skipped-step count.
2. Separate a data cause from a numerical one. A mixture stage boundary, a resumed run reading a different config, or a checkpoint restore all shift the level legitimately, and the run progress and optimizer panels date the change. A gradient norm climbing into the spike points at the optimizer.
3. Check whether the optimizer absorbed it. Steps skipped during the window mean the update was rejected and the weights did not take the spike; a spike with no skipped steps entered the weights.
4. Decide. Resuming from the last checkpoint before the rise costs the steps since it; letting a diverged run continue costs everything after it. `manage-hero-run` covers the rollback.

Verify what the rule saw with a bounded Finelog query:

```sql
SELECT
  to_timestamp_millis(timestamp_ms) AS observed_at,
  value AS train_loss
FROM "telemetry_v1"
WHERE service = 'levanter'
  AND name = 'train_loss'
  AND run_id = '<hero-run-id>'
  AND timestamp_ms >= CAST(EXTRACT(EPOCH FROM now() - INTERVAL '60 minutes') * 1000 AS BIGINT)
ORDER BY timestamp_ms DESC
LIMIT 50;
```

## Tuning

The windows, the sigma factor, the absolute floor, and the sample minimums are constants at the top of `infra/grafana/src/loss_spikes.py`. Changing one takes a redeploy of the Grafana service.

A benign firing that repeats across runs, rather than at one run's stage boundary, is the case for changing a constant. One run's boundary is a case for silencing that run.
