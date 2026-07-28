# Training stall warning

`TrainingProgressStalled` is a passive, warning-only Grafana rule. It does not kick, restart, or profile a job. Capture `iris process profile threads -t <task>` for the affected tasks before deciding whether to intervene.

The rule evaluates once a minute and waits five minutes before notifying. A root job is eligible only while its latest `iris.task_state` row is at most 90 seconds old, reports at least one running task, and has emitted `levanter_phase` within the trailing 24 hours. This enrollment gate prevents long-running Zephyr, inference, and generic Iris jobs from being classified as training. An eligible job warns when either condition holds:

- Training has started and `levanter_progress_time_seconds` is at least 15 minutes old. An explicit training phase or a positive `levanter_step` identifies training.
- The job has remained running for at least 45 minutes without entering training. Missing progress after Levanter enrollment counts as absent progress rather than suppressing the warning.

The first case is labeled `optimizer_progress_stale` or `optimizer_progress_missing`; the second is `initializing_stale`. Finished and progressing jobs emit zero-valued rows so Grafana can resolve their warning state. An idle fleet also emits an explicit zero.

A job that emits `levanter_step` but no `levanter_phase` runs a producer older than both phase and progress, and reports a zero-valued `producer_missing` row instead of a warning. Enrollment keys on phase because `TelltaleTracker` publishes it as it is constructed, which marks the producer generation exactly.

The bridge joins the durable streams by `(cluster, root job ID)`: `iris.task_state.root_job_id` equals Telltale `job_id`, and the finelog forwarder stamps both with their origin `cluster`. The task-state query scans the trailing hour; the enrollment and progress query scans 24 hours. An older running job still has an inferred running age of at least one hour, which is sufficient for both thresholds. No task-to-node mapping or GPU-utilization condition is required.

The required Telltale producers are `levanter_step`, `levanter_progress_time_seconds`, and numeric `levanter_phase` (`initializing=0`, `training=1`, `finished=2`). `TelltaleTracker` initializes phase and progress, records wall time after its completed-step `train/loss` callback, and marks a finished run.

GPU utilization, power, and power-limit metrics remain diagnostic evidence available in finelog and the capture bundle; they do not gate or classify this warning. This avoids hiding a stalled job when node attribution is unavailable.

The rule currently covers CoreWeave controllers whose active root-job state is forwarded into the `marin` finelog hub. GCE controllers do not emit `iris.task_state`, so their jobs are not evaluated by this rule.
