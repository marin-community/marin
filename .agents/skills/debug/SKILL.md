---
name: debug
description: Debug code bugs or Iris/Zephyr/TPU infrastructure faults with a structured incident record.
---

# Skill: Debug

Systematic debugging for code-level bugs and Marin infrastructure faults.
For infrastructure symptoms, route to the right `OPS.md` section first. Publish
durable investigation records to Echo with `write-ops-log`.

Do not add repository debug-log files. Use `write-ops-log` to finish an
infrastructure or other multi-step incident as a standalone Echo postmortem.

## Consult Echo

Invoke `consult-echo` at the start when prior discussions, decisions, or
incident patterns could materially shorten debugging. At resolution, always
invoke it to search before deciding whether the reusable lesson belongs in
`OPS.md`, `docs/`, the Echo incident record, or an existing or new synthesis.

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

For code-level bugs that are not infrastructure faults, keep working notes in
the active task. A contained fix may use the lightweight structure below.
Publish the complete `write-ops-log` structure to Echo when the investigation
exposes a durable operational lesson:

```
# <System or component>: <symptom>

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
