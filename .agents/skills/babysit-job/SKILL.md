---
name: babysit-job
description: Monitor/babysit a job continuously and recover on failure. Use when asked to babysit, monitor, or watch a job, pipeline, workflow, or training run.
---

# Skill: Babysit Job

Generic job babysitting practices. Delegates to framework-specific skills:

- **Iris jobs** -> invoke **babysit-iris-job**
- **Ray jobs** -> invoke **babysit-ray-job**
- **Zephyr pipelines** -> invoke **babysit-zephyr**

If the user doesn't specify, ask:

> "Should I `ray` or `iris` cluster?

Then invoke the appropriate skill. The rest of this document defines shared
practices that both framework skills follow.

## Shared Practices

### Scope

- Recovery is stop then resubmit at the job level.
- Cluster-level actions are out of scope. Do not restart, recreate, or otherwise
  mutate the cluster unless the user gives explicit consent in the current thread.
- For TPU bad-node errors, escalate to **debug-tpu**.

### Required Info: Ray Track

1. `job_id` — Ray job ID (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. `cluster` — cluster name/alias for `scripts/ray/cluster.py` (e.g., `us-east5-a`, `us-central2`)
3. `experiment` — script path used for resubmission (e.g., `experiments/isoflop_sweep.py`)

### Required Info: Iris Track

1. `job_id` — Iris job ID in canonical format `/<user>/<job>` (e.g., `/dlwh/iris-run-train_tiny_model_tpu-20260302-185630`)
2. `config` — Iris config path (e.g., `lib/iris/examples/marin.yaml`)
3. `resubmit_command` — exact Iris submit command for resubmission; must include `--no-wait`
4. For Marin TPU training jobs, use `--extra marin:tpu` (not `--extra marin:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example Iris resubmit command:
`uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python experiments/tutorials/train_tiny_model_tpu.py`

Iris `job run --` passes argv literally. Prefer `python -c '...'` for inline Python. Do not use a local heredoc such as `python - <<'PY'` unless you explicitly wrap a remote shell, because the heredoc body is consumed before Iris sees the command.

If any required field is missing, ask for it before proceeding.

## Monitoring Ownership and Duration

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
