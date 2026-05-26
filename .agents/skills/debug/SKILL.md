---
name: debug
description: Debug code bugs or Iris/Zephyr/TPU infrastructure faults with a structured debug log.
---

# Skill: Debug

Systematic debugging for code-level bugs and Marin infrastructure faults.
For infrastructure symptoms, route to the right `OPS.md` section first; for
code bugs, keep a structured debug log.

## Infrastructure faults

Read `lib/iris/AGENTS.md` or `lib/zephyr/AGENTS.md` for context, then follow
the matching `OPS.md` section:

| Symptom | Read |
|---|---|
| Stuck job, scheduling failure, resource leak, controller stalled | `lib/iris/OPS.md` → SQL Queries, Process Inspection & Profiling, Known Bugs, Troubleshooting |
| Iris task misbehaving, container inspection, profiling a running task | `lib/iris/OPS.md` → Task Operations, Process Inspection & Profiling |
| Zephyr pipeline slow / stragglers / data skew / worker failures | `lib/zephyr/OPS.md` → Diagnostic Patterns, Observability |
| TPU bad node (`No accelerator found`, `FAILED_PRECONDITION`, `Device or resource busy`) | `lib/iris/OPS.md` → TPU Bad-Node Recovery |

Operational guardrails (never modify the controller DB, prefer
`iris process profile` over SSH, never run a full `iris cluster restart`
without approval) live next to the relevant commands in `OPS.md` — read those
sections. After a TPU recovery or zephyr fix, return to the active babysit
loop (`babysit-job` or `babysit-zephyr`).

## Code bugs

For code-level bugs that are not infrastructure faults, maintain a debug log
at `docs/debug-log-<task-name>.md`:

```
# Debugging log for <task>

<goal>

## Initial status
<initial status, as reported or observed>

## <Hypothesis N>
The suspected source of the bug, or a change needed to isolate it.

## Changes to make
Which files you are altering and how.

## Results
Test results and any new hypotheses. Repeat the Hypothesis/Results cycle as needed.

## Future work
- [ ] Cleanups observed along the way
```
