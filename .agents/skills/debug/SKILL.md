---
name: debug
description: Debug code bugs or Iris/Zephyr/TPU infrastructure faults with a structured incident record.
---

# Debug

For infrastructure symptoms, read the relevant `AGENTS.md` and `OPS.md` first;
publish durable lessons with `write-ops-log`. Do not create repository debug-log
files. Invoke `consult-echo` at the start when prior incidents may shorten the
investigation, and search it again at resolution before changing `OPS.md`,
`docs/`, or Echo.

Route by symptom:

| Symptom | Read |
|---|---|
| Stuck job, scheduling/resource leak, controller stall | `lib/iris/OPS.md` → SQL, process inspection, known bugs, troubleshooting |
| Misbehaving task, container inspection, running-task profiling | `lib/iris/OPS.md` → task operations, process inspection |
| Zephyr slowdown, straggler, skew, worker failure | `lib/zephyr/OPS.md` → diagnostic patterns, observability |
| TPU bad node (`No accelerator found`, `FAILED_PRECONDITION`, `Device or resource busy`) | `lib/iris/OPS.md` → TPU bad-node recovery |

Use the guardrails in those sections: never modify the controller DB, prefer
`iris process profile` over SSH, and never run a full cluster restart without
approval. Return to the active `babysit-job`/`babysit-zephyr` loop after recovery.

For a code bug, keep notes in the active task and use this compact structure:

```text
# <System>: <symptom>
<goal>
## Initial status
<observed state>
## Hypothesis
<suspected source or isolating change>
## Changes
<files and intended fix>
## Results
<tests, evidence, and next hypothesis>
```

Publish the full incident record with `write-ops-log` when the investigation
reveals a durable operational lesson.
