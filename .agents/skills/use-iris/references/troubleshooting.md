# Diagnose an infrastructure fault

Keep working notes in the active task. Do not add repository debug-log files.
Use `use-echo` for prior-work search and its
[incident procedure](../../use-echo/references/incidents.md) for incident
classification and publication.

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
return to the active Iris or [Zephyr monitoring](monitor-zephyr.md) loop.

For an explicitly requested Gemini analysis of an oversized log, read
`docs/references/logscan.md` before sending data to the service.
