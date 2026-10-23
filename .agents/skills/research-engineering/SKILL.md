---
name: research-engineering
description: Explore Marin designs, datasets, and experimental pipelines through iterative analysis and experiments; preserve a runnable research branch and notebook, then promote selected code into Marin PRs. Use for experimental programs, not bounded background research or routine debugging.
---

# Research engineering

Use this skill when answering a question requires exploring alternatives,
analyzing data or artifacts, or iterating on experiments before choosing an
implementation. Work can start before any run is launched. Background research
alone follows [experiment records](../../../docs/dev-guide/experiment-records.md).

Read [the research engineering guide](../../../docs/dev-guide/research-engineering.md)
for the record, branch, and promotion workflow. Read the current
[`marin-logbooks` guidelines](https://github.com/marin-community/marin-logbooks/blob/main/AGENTS.md)
before creating or updating its branches; research branches inherit Marin's
root instructions rather than the logbooks index branch's instructions.

## Coordinate the work

- Start from the decision, baseline, constraints, and useful stopping condition.
  Search prior work through `use-echo` and inspect the current checkout.
- Use a task issue in `marin-community/marin-logbooks` as the public entry point.
  Reuse an existing issue; when the task authorizes a new public research
  program, use `file-issue` to create its coordinating issue. An existing Marin
  issue can link to it. Do not publish private work without authorization.
- Work on a Marin-derived branch in logbooks. Preserve the notebook and
  supporting analysis alongside the actual experimental source changes.
- Record why each meaningful alternative was tried, the evidence it produced,
  what was learned, and the next decision. Commit and push supporting code and
  notebook entries before a milestone issue update or handoff.
- Keep the issue body current and its comments useful as a readable account of
  the important attempts, results, and changes of direction.
- Extract selected code into Marin contribution branches. Use `manage-pr` for
  their validation, publication, and feedback; keep the research branch intact.
  A useful negative result can finish a program without a code PR.

## Delegate execution

Use the domain skill for each concrete operation, such as `add-dataset`,
`write-pipeline`, `launch-rl`, `add-pallas-kernel`, or `profile-training`.
`use-iris` owns job operations and `query-finelog` owns telemetry queries.
This coordinating workflow does not authorize compute launches or recovery
actions beyond the task's scope. Record meaningful restarts under their actual
execution identities.

Babysit launched runs and PRs through a heartbeat: a scheduled follow-up in the
host application that resumes monitoring after the active turn. Keep one
monitor per task with its execution or PR identity, latest observation, owner,
and exit condition. Report meaningful changes and required action; stay quiet
while unchanged. Use `use-iris` for job completion checks and `manage-pr` for PR
feedback and exit conditions.
