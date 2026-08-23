---
name: debug
description: Diagnose an explicit code, JAX, Marin, Iris, Zephyr, or TPU fault or startup/performance regression; do not activate for ordinary implementation or optimization without a stated symptom.
---

# Debug

Keep working notes in the active task. Do not add repository debug-log files.
Use `consult-echo` only when repository policy requires prior-work search, and
use `write-ops-log` when an infrastructure or durable multi-step incident needs
a standalone Echo record.

## Infrastructure faults

Read `lib/iris/AGENTS.md` or `lib/zephyr/AGENTS.md` for context, then follow
the matching `OPS.md` section:

| Symptom | Read |
|---|---|
| Stuck job, scheduling failure, resource leak, controller stalled | `lib/iris/OPS.md` → SQL Queries, Process Inspection & Profiling, Known Bugs, Troubleshooting |
| Iris task misbehaving, container inspection, profiling a running task | `lib/iris/OPS.md` → Task Operations, Process Inspection & Profiling |
| Zephyr pipeline slow / stragglers / data skew / worker failures | `lib/zephyr/OPS.md` → Diagnostic Patterns, Observability |
| TPU bad node (`No accelerator found`, `FAILED_PRECONDITION`, `Device or resource busy`) | `lib/iris/OPS.md` → TPU Bad-Node Recovery |

Read the guardrails beside the commands. Never modify the controller database,
prefer `iris process profile` over SSH, and never run a full
`iris cluster restart` without approval. After a TPU recovery or Zephyr fix,
return to the active `babysit-job` or `babysit-zephyr` loop.

## Code bugs

For code bugs, reproduce the failure, identify the smallest falsifiable
hypothesis, change one cause at a time, and test the behavior that failed. Let
exceptions propagate unless added context changes the diagnosis.
