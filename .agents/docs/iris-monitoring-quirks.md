# Iris Monitoring Quirks

Running notes from live monitoring sessions to improve Iris ergonomics and reliability.

## 2026-03-03

- Controller RPC flakiness during otherwise valid calls:
  - Observed `Connection reset by peer` and `Connection refused` on `job list --json` and occasionally `job run`.
  - Retries sometimes recover, but failures can happen in bursts.
  - During steady-state monitoring (every ~2-10 minutes), a full retry budget can still fail for `list_jobs`, forcing external retry loops in the monitor.
  - Failures can repeat across back-to-back monitor attempts (not just isolated spikes), temporarily blinding job-state polling.
  - In severe windows, repeated outer retries (e.g., 5 full `job list` invocations) can all fail, leaving no controller-state visibility.
- Tunnel contention when multiple Iris commands run concurrently:
  - Parallel `job list`/`job logs` checks increased connection instability.
  - Sequential polling is more stable than parallel polling.
- `job logs` polling is awkward for monitor loops:
  - In practice, bounded snapshots are difficult; commands can block longer than expected even when `--follow` is not set.
  - A first-class non-streaming "tail latest N lines" mode would simplify robust polling.
- Status payload inconsistency while running:
  - `state=JOB_STATE_RUNNING` can coexist with `task_state_counts.pending=1` and `completed_count=0`.
  - This is likely valid internally but surprising to operators reading JSON literally.
- `RUNNING` can be effectively log-stalled:
  - Job/task state remains `RUNNING` while task log object timestamp and size stay frozen for long intervals.
  - In one case, both task logs and worker-process logs stopped at `BUILDING→RUNNING` transition without further user-command output, but controller status stayed non-terminal.
- `pending_reason` can flap while still pending:
  - Repeated polls for the same pending job sometimes return `pending_reason=""` and sometimes return a concrete scale-up reason (e.g., `tpu_v6e_32-us-east1-d`).
  - The reported scale group zone can change across polls for one job (e.g., `us-east1-d` then `us-east5-b`), implying scheduler rebalance/fallback that is not surfaced as a dedicated event.
- Preemption can occur early without explicit failure:
  - `preemption_count=1` observed while job remained `JOB_STATE_RUNNING`.
- Job visibility can disappear abruptly:
  - A job previously observed as `JOB_STATE_RUNNING` later returned `Job ... not found` via `job bug-report`.
  - Subsequent `job list --json` succeeded but returned `TOTAL_JOBS 0`, leaving no terminal state trail for postmortem.
  - Even newly submitted jobs can disappear quickly from `job list --prefix` (empty result) before any W&B run appears.
  - Reproduced again on `/dlwh/iris-run-run_direct-20260304-002249` after fatal JAX/TF coordination error in task logs.
  - Sequence: `RUNNING` -> task log gets `DEADLINE_EXCEEDED` + `RPC: /tensorflow.CoordinationService/RegisterTask` -> `job list --prefix` returns `[]` -> `job bug-report` says job not found.
