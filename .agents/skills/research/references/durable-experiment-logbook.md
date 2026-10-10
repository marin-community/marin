# Durable Experiment Logbook

This file defines when Marin work uses a public logbook and how evidence moves
from the detailed record into a coordinating issue. The
[`marin-logbooks`](https://github.com/marin-community/marin-logbooks)
repository's `AGENTS.md` defines repository-local write mechanics.

## When To Use A Logbook

Use a logbook for programs expected to require repeated launches or controlled
comparisons and produce evidence worth reusing. Default to one for kernel
tuning across configurations or shapes, evaluation of a new modeling or
training idea, and systems or evaluation experiments with several meaningful
runs. Also use an existing logbook or create one when the user explicitly
requests it.

Do not start a logbook for ordinary implementation, debugging, incident
response, a single reproduction or smoke test, or one-off validation. A task
lasting multiple sessions does not by itself need a logbook. Start one if the
work later becomes a qualifying experimental program.

Create a logbook by default when qualifying work has a public coordinating
issue. If the work is not already public, suggest the logbook and confirm the
publication surface before writing. Create a coordinating issue only when the
user explicitly requests one.

## Record Layers

Maintain three levels of detail:

1. Append decision-relevant attempts to the logbook. This is the high-fidelity
   experimental record.
2. Post coordinating issue comments for milestones and material status changes.
   This is the medium-fidelity progress record.
3. Update the issue body when the overall conclusion, baseline, confidence, or
   decision changes. This is the low-fidelity current summary.

## Logbook Entries

Record attempts that test a real hypothesis, produce a meaningful result or
negative result, change the baseline or next action, or invalidate earlier
evidence. Include the source revision, exact command or material configuration,
hardware when relevant, result, supporting artifact links, interpretation, and
next action.

Omit command mistakes, ordinary environment repair, transient operational
noise, evidence-free retries, and debugging whose only outcome is that the code
runs. Keep raw logs, dense telemetry, and large tables in their source systems
and link them from the entry.

## Publish And Synchronize

Read the logbook repository's `AGENTS.md` before writing. Entries may accumulate
locally during active work. Before posting a coordinating issue comment, commit
and push every entry that supports the comment, then link the exact logbook
commit. One commit may contain several entries.

Also commit and push pending entries before a cross-session handoff or when
stopping work. This synchronization does not require an issue comment.

Never publish secrets or private results, force-push the logbook repository, or
overwrite unexpected concurrent edits.
