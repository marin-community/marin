# Diagnose an infrastructure fault

Keep working notes in the active task. Do not add repository debug-log files.
At the start of a fresh investigation, use `use-echo` when repository policy
requires prior-work search. During continued diagnosis or mitigation of the
same incident, reuse results already read instead of invoking `use-echo`
again unless the scope or freshness requirement changes or the user asks for a
new search. After diagnosing a live infrastructure incident, use
`use-echo`'s incident procedure to publish its standalone record and link it from the
associated PR or issue. A code bug or local debugging session is not an
incident unless it caused a service, production run, or shared operational
system to fail or degrade.

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

## Code bugs

For code bugs, reproduce the failure, identify the smallest falsifiable
hypothesis, change one cause at a time, and test the behavior that failed. Let
exceptions propagate unless added context changes the diagnosis.
