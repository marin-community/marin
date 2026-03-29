# Debugging log for rollout transfer counter reset

Investigate why rollout-side transfer metrics such as `inference.successful_receives`
reset after rollout worker retries, and make the main W&B series resume-safe.

## Initial status

In the `iris-rl-e4ms2-500` rollout runs, the main receive counters are logged directly
from the weight-transfer client metrics object. Those counters are process-local, so
after a rollout retry the W&B series can drop back to a smaller value inside the same
resumed run.

## Hypothesis 1

Rollout-side receive counters are attempt-local metrics being logged as if they were
run-global cumulative metrics.

## Changes to make

None. Trace the rollout logging path and identify whether the counters are persisted
across rollout retries.

## Results

Confirmed.

The current code path is:

- the Arrow Flight client stores receive counters in an in-memory
  `WeightTransferClientMetrics` dataclass
- rollout logging calls `self._transfer_client.get_metrics()`
- those counters are written directly into W&B under:
  - `inference.total_polls`
  - `inference.successful_receives`
  - `inference.failed_receives`

So after a rollout retry, the run resumes under the same W&B id but the source counters
restart from zero. This is the same semantic issue the trainer-side
`weight_transfer/total_transfers` metric had.

## Hypothesis 2

The correct fix is to keep the attempt-local counters for debugging, but accumulate
their deltas in a coordinator-hosted actor so the main W&B series stays monotonic
across rollout retries.

## Changes to make

- Extend `RLRunState` with per-worker cumulative rollout transfer counters
- In `RolloutWorker`, track the last locally observed attempt counters
- On each log step:
  - compute non-negative deltas from the current local counters
  - add those deltas to `RLRunState`
  - log:
    - attempt-local counters under `attempt_*`
    - resume-safe cumulative counters under the original names

## Results

Implemented.

The new semantics are:

- `inference.attempt_total_polls`
- `inference.attempt_successful_receives`
- `inference.attempt_failed_receives`

remain process-local and are useful for debugging a single rollout attempt.

Meanwhile the main counters are now resume-safe:

- `inference.total_polls`
- `inference.successful_receives`
- `inference.failed_receives`

These are accumulated per `worker_index` inside `RLRunState`, so a rollout retry
contributes only its delta instead of resetting the chart.

## Future Work

- [ ] If any other rollout metrics are intended to be globally cumulative, audit them for the same attempt-local reset pattern
