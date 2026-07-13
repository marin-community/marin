---
name: dispatch-job
description: Execute, monitor, and recover exact Iris work orders without changing the logical trial or target. Use for a formal agent-to-agent handoff when an orchestrator needs a Dispatcher to run jobs and return normalized Iris, log, W&B, objective, and artifact facts.
---

# Dispatch Job

Own the execution loop between an orchestrator and Iris. Keep operational context and recover the exact assigned work, but own no experiment policy, candidate choice, target selection, or sweep state.

## Require A Work Order

Require:

- `action` as `launch` or `stop`, plus stable trial, regional run, and dispatch IDs.
- Resource-rung index, Iris config, region, TPU slice, and chips.
- Deterministic job name, source revision, exact single-job command, output/checkpoint identity, and resume rules.
- W&B matching, observation interval, objective metric, completion signals, and applicable command preferences.

The launch command must be restart-safe and use `--no-wait`.

For `stop`, also require the current canonical Iris job ID. Return `needs_decision` for missing, conflicting, or unauthorized fields. Never infer a different candidate, target, resource amount, relocation, checkpoint transfer, or experiment-semantic change.

Reject a launch or recovery command that would read or copy a checkpoint from another region. Cross-region recovery must start from the trial's initial state; an existing checkpoint is resumable only in its own region.

Validate that the command passes both runtime inputs to the script. A replacement TPU within the same region may retain the regional run, W&B, output, and checkpoint identities. A region change requires a new regional run identity and an empty resume source.

## Execute Idempotently

Construct the exact Iris command from the work order, including command preferences, verify it, and report its structure with secret values redacted. A repeated work order resumes its existing dispatch. A recovery submission is another attempt of that dispatch, not a duplicate logical trial.

On every submission, increment `submission_attempt` and record the Iris job ID and `submitted_at`. Emit an initial observation at submission with the regional run's current `run_progress`, or `null` when it is not yet observable. A numeric value becomes the baseline for target-throughput measurement. Only stop under explicit Orchestrator direction or as part of recovery.

## Monitor And Recover

Use the status, log, W&B identity, completion, small-fix, and stop-resubmit mechanics from `babysit-job`, with the work order's launch command as `resubmit_command`. Reuse its diagnostics, not its retry stopping criteria; the rules below control recovery. Maintain one monitoring owner and follow `observation_interval`.

For each dispatch:

1. Submit or reconcile the current Iris job, then observe Iris, logs, W&B, objective, and artifacts on cadence. Report `run_progress` as a monotonic fraction of the regional run's declared resource work.
2. Continue while pending or running. Capacity wait is not failure and does not authorize relocation.
3. On success, verify the requested completion signal, final objective, and artifacts, then return `succeeded`.
4. On terminal failure, report the raw Iris failure with dispatch state `retrying` and resubmit the same work by default. If the job previously made observable log, W&B, or checkpoint progress, always retry unless a clear gate below applies.
5. Before resubmission, fix an obvious local bug when the `babysit-job` small-fix rule applies, run a focused check, and report the patch or revision. Any fix that may change training semantics requires `needs_decision` instead.
6. Repeat without a fixed retry limit. Report every failed and replacement Iris job while keeping the stable dispatch ID.

Treat Iris RPC timeouts, Iris controller restarts, bad-node TPU/JAX environment failures, Hugging Face rate limits, GitHub or other network timeouts, and similarly transient service failures as retryable. Iris handles individual preemptions; do not model them separately.

Automatic recovery stays on the assigned target. A new TPU slice or region requires an explicit Orchestrator stop and replacement launch with a new dispatch ID.

Return `needs_decision` only when evidence identifies a non-trivial code/data/configuration bug, OOM or TPU/XLA resource mismatch requiring configuration changes, invalid credentials or permissions, corrupted required state, a policy gate, or an operator stop. An unfamiliar terminal message or repeated transient failure is not by itself a gate, and retry count is diagnostic rather than a stopping rule.

## Return Structured Facts

Return one record per submission, observation, or terminal outcome. Use RFC 3339 UTC timestamps and `null` for requested facts that are not yet known.

```yaml
identity:
  event_id: string
  event_type: submission | observation | terminal
  trial_id: string
  regional_run_id: string     # Stable across TPU changes, never across regions.
  dispatch_id: string         # Stable across automatic Iris resubmissions.
  submission_attempt: integer # Starts at 1 and increments per Iris submission.

trial:
  resource_rung: integer      # Zero-based index into the policy's resource levels.

job:
  iris_job_id: string | null
  state: submitted | running | retrying | succeeded | stopped | needs_decision
  iris_state: string | null      # Raw state of the current or most recent Iris job.
  submitted_at: timestamp | null  # Current Iris submission.
  target:
    region: string
    tpu_slice: string
    chips: integer

observation:
  observed_at: timestamp
  wandb_run_id: string | null
  run_progress: number | null
  objective_value: number | null
  log_signals:
    - {severity: info | warning | error, source: string, message: string}
  artifacts:
    - {name: string, uri: string | null, status: pending | available | missing}

execution:
  submitted_command: string | null # Command structure with secret values redacted.
  source_revision: string        # Revision used by the current submission.
  detail: string | null
```

Do not estimate throughput or wall time. Do not write the Orchestrator's SQLite database.
