---
name: write-ops-log
description: Publish a tagged postmortem incident record to Echo. Use after an infrastructure or durable debugging session, then link the canonical Echo URL from the associated PR or issue.
---

# Skill: Write an Ops Log

Publish the incident record to Echo. Do not add a repository debug-log file.
The audience is a future engineer who must quickly decide what to check or do
next. Record the smallest set of facts that supports that action. Link raw
evidence and detailed chronology rather than reproducing them; Echo entries
consume limited model context whenever they are retrieved.

## Search before writing

Invoke `consult-echo` and run its complete search-before-write sequence. Edit
the existing entry when it covers the same incident. Start with a natural-language
`infra/echo/cli.py search`, fetch likely matches with `get <domain:id>`, and use
`grep` for the exact error or run identifier. Create a new entry for a different
incident even when the symptom resembles an older one; link related incidents
and create or extend a separate synthesis only when they establish a reusable
cross-incident pattern.

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
- <diagnostic discriminator or immediate action>
- <fix or remaining caveat>

## Response

<The shortest safe procedure for diagnosing, mitigating, or avoiding the
incident.>

## Cause and resolution

<Only the evidence needed to justify the response and explain the resolution.>

## Artifacts

- <Canonical PR, issue, dashboard, durable log bundle, or report>
```

Use the incident's investigation date. Use lowercase kebab-case tags and no
more than 20. Always include `incident` and `debugging`; add `ops` for
infrastructure work, followed by the subsystem, severity, and resolution.
Omit any body section that adds no actionable value. Add a compact original
report, investigation step, or course correction only when it changes how a
future reader should recognize or respond to the incident.

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
- Prefer decisions, discriminating evidence, and response steps over narrative.
  Do not paste raw logs, duplicate source artifacts, or narrate every tool call.
- Link readers to the external source of record when its detail is not needed
  to choose or carry out the action.
- Keep incident-specific detail here. Promote recurring procedures and
  guardrails to the relevant `OPS.md`.
- Update `docs/` when behavior or configuration guidance belongs with the
  repository.
- Record an unknown root cause as unknown; do not fill the gap with speculation.
