# Debugging log for weight transfer metric reset

Investigate why `weight_transfer/total_transfers` resets in the resumed trainer W&B run
`iris-rl-e4ms2-500-train`, and determine whether this is a W&B resume problem or a
process-local metric that is not restored across trainer retries.

## Initial status

The trainer W&B run at
<https://wandb.ai/marin-community/marin_iris_rl_debug/runs/iris-rl-e4ms2-500-train?nw=nwuserahmedah>
shows `weight_transfer/total_transfers` dropping back near zero multiple times despite
the run resuming under a stable trainer run id.

## Hypothesis 1

W&B resume stitched multiple trainer attempts incorrectly and reset the metric series.

## Changes to make

None. Read the W&B history directly and compare reset points across trainer metrics.

## Results

Rejected.

Direct W&B history inspection shows the run id is stable (`iris-rl-e4ms2-500-train`) and
the reset points are limited to the two cumulative weight-transfer server counters:

- `weight_transfer/total_transfers`: drops at steps `67`, `143`, `281`, `405`
- `weight_transfer/successful_transfers`: drops at the same steps
- `weight_transfer/failed_transfers`: no drops

Other cumulative trainer metrics checked do not reset:

- `throughput/total_tokens`
- `throughput/total_gflops`

This means W&B run resumption is working for the trainer run as a whole.

## Hypothesis 2

The metric source is a fresh in-memory object created on each trainer retry, so the
resumed W&B run receives smaller process-local counter values after restart.

## Changes to make

None yet. Trace the logging path from trainer startup to weight-transfer metrics.

## Results

Confirmed.

The trainer creates a fresh transfer server in
`lib/marin/src/marin/rl/train_worker.py`, and that server owns a fresh
`WeightTransferServerMetrics` dataclass:

- `TrainWorker.__init__` sets `config.trainer.id = f"{config.run_id}-train"` and calls
  `weight_transfer.create_weight_transfer_server(...)`
- `weight_transfer_hook(...)` logs
  `dataclasses.asdict(self.transfer_server.get_metrics())` to W&B at each sync step
- `WeightTransferServerMetrics` is a plain in-memory dataclass with counters initialized
  to zero
- `ArrowFlightServer.__init__` sets `self.metrics = WeightTransferServerMetrics()`

Levanter W&B initialization uses the stable trainer id with `resume="allow"`, so the W&B
run resumes correctly. The metric object itself does not resume.

The metric resets are therefore expected under the current implementation whenever the
trainer child restarts and resumes into the same W&B run.

## Future Work

## Hypothesis 3

The simplest fix is to keep the process-local counters for debugging under explicit
`attempt_*` names, and publish the run-global cumulative counters from restored trainer
step state instead of from the transfer server object.

## Changes to make

- Update `TrainWorker.weight_transfer_hook(...)` to:
  - log `weight_transfer/attempt_*` directly from the transfer server metrics dataclass
  - derive monotonic `weight_transfer/total_transfers` and
    `weight_transfer/successful_transfers` from `info.step` and
    `sync_interval_steps`
- Add a focused unit test for the logging behavior in `tests/rl/test_train_worker.py`

## Results

Implemented.

The new run-global metrics are derived from the actual schedule of this trainer path:

- bootstrap transfer at step `-1`
- one transfer hook every `sync_interval_steps`, starting at step `0`

So the global cumulative count at a logged hook step is:

- `2 + step // sync_interval_steps`

For the observed run with `sync_interval_steps=1`, that yields:

- step `0` -> `2`
- step `67` -> `69`
- step `143` -> `145`

which matches the intended cumulative semantics and removes the retry-induced resets.

## Future Work

- [ ] If we ever need true global failure counts, persist them separately rather than reusing attempt-local counters
