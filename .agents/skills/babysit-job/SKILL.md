---
name: babysit-job
description: Monitor/babysit a job continuously and recover on failure. Use when asked to babysit, monitor, or watch a job, pipeline, workflow, or training run.
---

# Skill: Babysit Job

Generic job babysitting practices. Delegates to framework-specific skills:

- **Iris jobs** -> invoke **babysit-iris-job**
- **Zephyr pipelines** -> invoke **babysit-zephyr**

If the user says to babysit a job without specifying, assume **Iris** and invoke
**babysit-iris-job**. The rest of this document defines shared practices that
framework-specific skills follow.

## Shared Practices

### Scope

- Recovery is stop then resubmit at the job level.
- Cluster-level actions are out of scope. Do not restart, recreate, or otherwise
  mutate the cluster unless the user gives explicit consent in the current thread.
- For TPU bad-node errors, escalate to **debug-tpu**.

### Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Keep this loop running until one of the following:
  - the job reaches a terminal state and the user has acknowledged next action
  - a user-specified stopping point is reached
  - an unrecoverable error is found and reported to the user
- Do not stop early after seeing first loss lines, first eval, or first W&B link.
- Expect monitoring to commonly take 4-5 hours for ferry-scale runs.
- For GPT Codex specifically: unless otherwise directed, do not end your turn just
  to give a status update; keep monitoring until terminal state or until the user's
  goal is reached.
- If the user requests continuous monitoring, do not end the turn while monitoring
  is active; continue until terminal state, a user-specified stopping point, or an
  unrecoverable error.
- If handoff is needed, transfer ownership explicitly with: current `job_id`,
  latest error/signal, W&B link(s), and resubmission metadata.

### Cadence and Tooling Notes

- Cadence default after startup stabilization is `sleep 570`.
- Startup stabilization sequence (after submit/resubmit):
  - once the job is submitted, sleep `120` and check for immediate failure
  - if still alive, switch to the normal `570` cadence
- Tool-runtime workaround:
  - keep one long-running monitor process/session
  - poll the same session in ~30 second chunks as needed by tool runtime limits
  - repeated no-output polls are expected while waiting for the next 570-second check
- Single monitor process rule:
  - run only one active monitor loop per job to avoid duplicate SSH tunnel and
    port-binding conflicts
- Sleep must be foreground (max ~10 min due to tool timeout).
- Loop control is at agent level, not bash.

### State File

Write to `scratch/<create_timestamp>_monitoring_state.json`, create the `scratch`
directory if needed. `<create_timestamp>` should have format `YYYYMMDD-HHMM`.
Track `restart_count` to detect flapping. State file allows resume after context reset.

### Generic Loop Structure

Each framework skill implements this loop with framework-specific commands:

```
1. SLEEP (120s after submit, 570s steady-state)
2. CHECK LOGS (framework-specific command, grep for errors)
3. CHECK STATUS (framework-specific command, classify terminal vs non-terminal)
4. PRINT W&B RUN IDS/LINKS (once per training run)
5. REPORT PROGRESS (format: ~<current>/<exact_max>)
6. EVALUATE (terminal? error? stalled? -> recover or continue)
7. RECOVER (stop -> resubmit, update state file, restart loop)
```

### Fixing Small Bugs

When EVALUATE detects an error, before recovery:

1. Analyze logs — look for `Traceback`, `Error`, `Exception`. Identify file and line.
2. If small fix (typo, missing import, wrong variable name): fix it, then RECOVER.
3. If complex (architectural, unclear cause, broad investigation): report to user, exit loop.

Small-fix examples: `NameError`, `ImportError`, `SyntaxError`, obvious `KeyError`.

Complex examples: OOM, TPU/XLA HBM exhaustion, distributed training failures,
data loading issues, unclear multi-file stack traces.

### Error Patterns

- Treat TPU/XLA HBM reports as failure even without literal OOM:
  - `Program hbm requirement ...`
  - `Largest program allocations in hbm`
- If progress stalls across multiple intervals with `OwnerDiedError`, dead node,
  or unsatisfied resources -> mark `degraded` and notify user.
- If same error repeats after one fix attempt, do not retry blindly; report to user.

### When to Escalate

- Debug Zephyr pipeline issues -> **debug-zephyr-job**
- Debug TPU bad-node errors -> **debug-tpu**
