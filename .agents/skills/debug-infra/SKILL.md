---
name: debug-infra
description: Debug Marin infrastructure (Iris controller, Iris jobs/tasks, Zephyr pipelines, TPU bad nodes). Use when investigating stuck jobs, scheduling failures, slow pipelines, worker issues, profiling, or hardware faults in iris/zephyr/fray.
---

Debug Marin infrastructure issues.

Read first:

- `lib/iris/AGENTS.md` for iris-specific context
- `lib/zephyr/AGENTS.md` for zephyr-specific context

Then follow the relevant `OPS.md` section:

| Symptom | Read |
|---|---|
| Stuck job, scheduling failure, resource leak, controller stalled | `lib/iris/OPS.md` → "SQL Queries", "Process Inspection & Profiling", "Known Bugs", "Troubleshooting" |
| Iris task misbehaving, container inspection, profiling a running task | `lib/iris/OPS.md` → "Task Operations", "Process Inspection & Profiling" |
| Zephyr pipeline slow / stragglers / data skew / worker failures | `lib/zephyr/OPS.md` → "Diagnostic Patterns", "Observability" |
| TPU bad node (`No accelerator found`, `FAILED_PRECONDITION`, `Device or resource busy`) | `lib/iris/OPS.md` → "TPU Bad-Node Recovery" |

After a TPU recovery or zephyr fix, return to the active babysit loop (`babysit-job` or `babysit-zephyr`).

Operational guardrails (never modify the controller DB, prefer `iris process profile` over SSH, never run a full `iris cluster restart` without approval) live next to the relevant commands in `lib/iris/OPS.md` — read those sections, don't reinvent the rules here.
