---
name: consult-echo
description: Search or cite Echo when Marin's prior-work policy applies, another selected workflow requires it, or the user explicitly asks; do not substitute it for local code search.
---

# Consult Echo

Search Echo before rediscovering prior work. Cite canonical Echo, GitHub, or
Discord URLs for the evidence you use.

## Search

Use a natural-language federated search:

```bash
uv run infra/echo/cli.py search "how do I diagnose a stalled TPU collective" --limit 10
```

Search covers `wiki`, `file`, `pr`, and `issue` by default. Repeat `--domain` to
select a subset.

File search infers the configured Marin-community repository from the current
Git checkout, including ordinary contributor forks. Pass `--repository
<owner/repo>` to choose one configured repository or `--repository all` to
search all six. The CLI prints the resolved file scope once. Searches that omit
the file domain work outside a Git checkout.

Discord is excluded by default because messages may be noisy or untrusted. Add
`--domain discord` only when discussion history is relevant, and open the
canonical URL when surrounding thread context matters.

Fetch a result's full detail with its printed ID:

```bash
uv run infra/echo/cli.py get <domain:id>
```

Grade evaluated results with the exact query and printed keys:

```bash
uv run infra/echo/cli.py feedback --query "stalled TPU collective" \
  --grade wiki:730=0 --grade file:731=10 \
  <<< "The file result answered the task; wiki results were off-topic."
```

Use 0 for irrelevant and 10 for directly useful. Grade only evaluated results
and always explain overall quality on stdin, including for an empty result set.

Use `grep` for exact strings in GitHub or Discord activity, narrowing with
`--source` or `--kind` when needed:

```bash
uv run infra/echo/cli.py grep "FAILED_PRECONDITION" --source github
```

Use `rg` for exact identifiers in the current checkout. Echo's file index
follows the periodically refreshed GitHub head, so it does not contain
branch-only or uncommitted files. Use `wiki search --tag <tag>` when tags, rather
than federated relevance, define the desired synthesis. Authentication reuses
the shared Marin login via `iris login`; see
[Echo setup and access](../../../infra/echo/README.md).

Run this search sequence before adding or editing a wiki note.

## Choose the durable home

Every incident gets a standalone Echo record through `write-ops-log`, linked
from its associated PR or issue. Also use the narrowest reusable home when the
incident or other work changes durable guidance:

- Update `OPS.md` for a recurring operational procedure, diagnostic workflow,
  or guardrail in one subsystem.
- Update `docs/` for reusable product or user guidance that belongs with the
  repository.
- Edit an existing Echo wiki note for a cross-cutting pattern or synthesis that
  has no single repository home.
- Add a wiki note only when the searches find no near-duplicate.

Create one incident entry per incident and edit it while the same incident
continues. Keep raw logs and diffs in their source systems. Edit a near-duplicate
synthesis instead of adding another.

## Write or revise a wiki note

Write the shortest entry that changes a future action:

- Lead with an actionable value: a decision, command, diagnostic discriminator,
  guardrail, or recovery step.
- Include only the facts needed to support or apply that action. Link to the
  canonical PR, issue, dashboard, log bundle, or report for chronology and raw
  evidence instead of copying it into Echo.
- Put the action in the title, `use_when`, or opening paragraph.
- Keep chronology without a reusable decision in an issue, logbook, or source
  artifact.

Write an Open Knowledge Format document:

```markdown
---
type: wiki-note
title: Stalled TPU collectives can have several low-power causes
use_when: when a distributed TPU job stalls before the first optimizer update
tags:
  - ops
  - debugging
  - iris
---

State the reusable pattern, how to distinguish its causes, and the evidence URLs.
```

Use lowercase kebab-case tags. Keep the title concrete and make `use_when`
describe the future search trigger. Make the body stand alone. Cite canonical
URLs inline. Concision still requires enough context to apply the guidance
safely; do not omit a precondition, destructive-action warning, or material
caveat.

Create a note:

```bash
uv run infra/echo/cli.py wiki add --file note.md
```

Revise the closest existing note:

```bash
uv run infra/echo/cli.py wiki show <id> > note.md
# Edit note.md.
uv run infra/echo/cli.py wiki edit <id> --file note.md
```

Both write commands print the canonical Echo URL. Return and cite that URL
after every add or edit.
