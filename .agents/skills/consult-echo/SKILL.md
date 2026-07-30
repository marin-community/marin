---
name: consult-echo
description: Search and cite Marin's Echo activity and wiki, then capture incident records or reusable shared knowledge. Use when looking for prior discussions, decisions, workflows, incident patterns, exact errors, tags, or identifiers; when prior context could shorten debugging; or when an investigation reaches resolution and its record or lessons may belong in Echo, docs, or OPS.md.
---

# Skill: Consult Echo

Search Echo before rediscovering prior work. Cite canonical Echo, GitHub, or
Discord URLs for the evidence you use.

## Search

Use federated search when prior decisions, incidents, workflows, current GitHub
work, or indexed repository documentation could inform the task. Natural-language
questions work:

```bash
uv run infra/echo/cli.py search "how do I diagnose a stalled TPU collective" --limit 10
```

Search covers `wiki`, `file`, `pr`, and `issue` by default. Repeat `--domain` to
select a subset:

```bash
uv run infra/echo/cli.py search "stalled TPU collective" \
  --domain wiki --domain file --domain issue
```

Discord is excluded by default because messages may be noisy or untrusted. Add
`--domain discord` only when discussion history is relevant, and open the
canonical URL when surrounding thread context matters.

Search output is deliberately compact. Fetch any result's complete indexed or
raw-source detail with the printed ID:

```bash
uv run infra/echo/cli.py get <domain:id>
```

Use `grep` for exact strings in GitHub or Discord activity, with `--source` or
`--kind` when needed:

```bash
uv run infra/echo/cli.py grep "FAILED_PRECONDITION" --source github
```

Use `rg` for exact identifiers in the current checkout. Echo's file index
follows the periodically refreshed GitHub head, so it does not contain
branch-only or uncommitted files. Use `wiki search --tag <tag>` when tags, rather
than federated relevance, define the desired synthesis. Authentication reuses
the shared Marin login via `iris login`; see
[Echo setup and access](../../../infra/echo/README.md).

Run this search sequence before adding or editing a wiki note. Use it at the
start of an investigation when prior context could materially shorten
debugging.

## Choose the durable home

At resolution, choose the narrowest durable home:

- Update `OPS.md` for a recurring operational procedure, diagnostic workflow,
  or guardrail in one subsystem.
- Update `docs/` for reusable product or user guidance that belongs with the
  repository.
- Add one Echo entry for the evidence, decisions, and outcome of a specific
  incident. Tag it `incident`, `debugging`, the subsystem, severity, and
  resolution; add `ops` for infrastructure incidents. Link its canonical Echo
  URL from the associated PR or issue.
- Edit an existing Echo wiki note for a cross-cutting pattern or synthesis that
  has no single repository home.
- Add a wiki note only when the searches find no near-duplicate.

Create one incident entry per incident, and edit that entry only when the same
incident continues. Keep raw logs and commit diffs in their source systems; the
incident entry records the investigation and links the evidence. A fixed bug can
also justify a separate cross-incident synthesis when several incidents support
a reusable diagnostic pattern. Edit a near-duplicate synthesis instead of
adding another.

## Write or revise a wiki note

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
URLs inline.

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
