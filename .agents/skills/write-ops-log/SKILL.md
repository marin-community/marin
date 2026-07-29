---
name: write-ops-log
description: Publish a tagged postmortem incident record to Echo. Use after an infrastructure or durable debugging session, then link the canonical Echo URL from the associated PR or issue.
---

# Skill: Write an Ops Log

Publish the incident record to Echo. Do not add a repository debug-log file.
The audience is a future engineer with no memory of the investigation who must
reconstruct what broke, what was tried, what changed the direction, what fixed
it, and which reusable guidance could have shortened the work.

## Search before writing

Invoke `consult-echo` and run its complete search-before-write sequence. Edit
the existing entry when it covers the same incident. Create a new entry for a
different incident even when the symptom resembles an older one; link related
incidents and create or extend a separate synthesis only when they establish a
reusable cross-incident pattern.

## Draft the Echo entry

Write an OKF document in a temporary file:

```markdown
---
type: wiki-note
title: "Incident YYYY-MM-DD: <system> — <symptom>"
use_when: when investigating <specific symptom or exact error>
tags:
  - incident
  - debugging
  - ops
  - <system>
  - <severity>
  - <resolution>
---

# <System or component>: <symptom>

## TL;DR

- <user-visible symptom>
- <root cause>
- <fix or mitigation>
- <remaining caveat>

## Original problem report

<What the user observed, including the exact error or dashboard text.>

## Investigation path

1. <What was checked, why, and what it established.>

## User course corrections

- <What direction changed, what the user supplied, and why it mattered.>

## Root cause

<One or two concrete paragraphs with code, query, log, or dashboard evidence.>

## Fix

<What changed. Separate code changes from live repair or migration steps.>

## How OPS.md could have shortened this

<The reusable procedure or diagnostic signal to add, or state that no generic
OPS.md change follows from this incident.>

## Artifacts

- <PR, issue, dashboard, durable log bundle, or source URL>
```

Use the incident's investigation date. Use lowercase kebab-case tags and no
more than 20. Always include `incident` and `debugging`; add `ops` for
infrastructure work, followed by the subsystem, severity, and resolution.

## Write the record

Create a new incident:

```bash
uv run infra/echo/cli.py wiki add --file incident.md
```

Continue the same incident:

```bash
uv run infra/echo/cli.py wiki show <id> > incident.md
# Edit incident.md.
uv run infra/echo/cli.py wiki edit <id> --file incident.md
```

Both commands print the canonical Echo URL. Return that URL and add it to the
associated PR description or issue. Do not commit the temporary OKF file.

## Keep the record useful

- Preserve exact error strings and canonical evidence URLs.
- Summarize decisions and dead ends; do not paste raw logs or narrate every tool
  call.
- Keep incident-specific detail here. Promote recurring procedures and
  guardrails to the relevant `OPS.md`.
- Update `docs/` when behavior or configuration guidance belongs with the
  repository.
- Record an unknown root cause as unknown; do not fill the gap with speculation.
