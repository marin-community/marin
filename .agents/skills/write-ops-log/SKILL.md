---
name: write-ops-log
description: Publish a tagged postmortem incident record to Echo. Use after an infrastructure or durable debugging session, then link the canonical Echo URL from the associated PR or issue.
---

# Write an ops log

Publish the smallest evidence-backed incident record needed by a future engineer.
Do not add a repository debug-log file. Invoke `consult-echo` and complete its
search-before-write sequence, including exact-error or run-ID grep:

```bash
uv run infra/echo/cli.py search "<symptom or exact error>"
uv run infra/echo/cli.py get <domain:id>
uv run infra/echo/cli.py grep "<exact error or run ID>"
```

Edit an existing entry for the same incident. Create a new entry when it is a
different incident; link related incidents and synthesize a cross-incident
pattern only when warranted.

Draft a temporary OKF file (do not commit it):

```markdown
---
type: wiki-note
title: "Incident YYYY-MM-DD: <system> — <symptom>"
use_when: when investigating <specific symptom or exact error>
tags: [incident, debugging, ops, <system>, <severity>, <resolution>]
---
# <System>: <symptom>
## TL;DR
- <symptom>
- <diagnostic discriminator or action>
- <fix or caveat>
## Response
<short safe procedure>
## Cause and resolution
<evidence supporting the response>
## Artifacts
- <canonical PR, issue, dashboard, log bundle, or report>
```

Use the investigation date, lowercase kebab-case tags (maximum 20), and always
include `incident` and `debugging`; include `ops` for infrastructure. Omit empty
sections. Preserve exact error strings and canonical evidence URLs. Link source
artifacts instead of pasting logs or narrating tool calls; record unknown causes
as unknown. Put reusable procedures in the relevant `OPS.md` and product
guidance in `docs/`.

Create a new entry:

```bash
uv run infra/echo/cli.py wiki add --file incident.md
```

Continue the same incident:

```bash
uv run infra/echo/cli.py wiki show <id> > incident.md
uv run infra/echo/cli.py wiki edit <id> --file incident.md
```

The command prints the canonical URL. Return it and link it from the associated
PR or issue.
