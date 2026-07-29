---
name: consult-echo
description: Search and cite Marin's Echo activity and wiki, then capture incident records or reusable shared knowledge. Use when looking for prior discussions, decisions, workflows, incident patterns, exact errors, tags, or identifiers; when prior context could shorten debugging; or when an investigation reaches resolution and its record or lessons may belong in Echo, docs, or OPS.md.
---

# Skill: Consult Echo

Search Echo before rediscovering prior work. Cite canonical Echo, GitHub, or
Discord URLs for the evidence you use.

## Search

Run these searches in order before adding or editing a wiki note. Use the same
sequence at the start of an investigation when prior context could materially
shorten debugging.

1. Search wiki entries semantically for an existing synthesis:

   ```bash
   uv run infra/echo/cli.py wiki search "stalled TPU collective diagnosis" --tag ops
   ```

2. Search GitHub and Discord activity semantically for prior discussions and
   decisions:

   ```bash
   uv run infra/echo/cli.py search "stalled TPU collective diagnosis" --limit 10
   ```

3. Grep each exact error string, tag, run name, or identifier:

   ```bash
   uv run infra/echo/cli.py grep "FAILED_PRECONDITION"
   ```

Use `--source`, `--kind`, or `--since` to narrow semantic activity searches.
Use repeated `--tag` flags to require all named wiki tags.
Use `show <id>` for a complete activity hit and `wiki show <id>` for a complete
wiki note. Activity results print canonical URLs. Wiki exports include the
canonical Echo URL in their OKF frontmatter. Open Discord URLs when surrounding
thread context matters because Echo stores each Discord hit as one message.
Authentication reuses the shared Marin login via `iris login`; see [Echo setup and access](../../../infra/echo/README.md).

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
