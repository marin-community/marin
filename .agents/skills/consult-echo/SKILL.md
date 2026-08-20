---
name: consult-echo
description: Search and cite Marin's Echo activity and wiki, then capture incident records or reusable shared knowledge. Use when looking for prior discussions, decisions, workflows, incident patterns, exact errors, tags, or identifiers; when prior context could shorten debugging; or when an investigation reaches resolution and its record or lessons may belong in Echo, docs, or OPS.md.
---

# Consult Echo

Search Echo before rediscovering prior work and cite the canonical Echo,
GitHub, or Discord URLs for evidence. Use this at the start of an investigation
when prior context could materially shorten debugging and before editing a wiki
note.

## Search and inspect

~~~
uv run infra/echo/cli.py search "<natural-language question>" --limit 10
uv run infra/echo/cli.py get <domain:id>
~~~

Search covers wiki, file, pr, and issue by default. Add repeated --domain flags
to narrow it. Discord is excluded by default; add --domain discord only when
thread history is relevant, then open its canonical URL. Use grep for exact
strings in GitHub/Discord activity and rg for exact identifiers in the current
checkout. The Echo file index follows refreshed GitHub head, so it excludes
branch-only and uncommitted files. Use wiki search --tag <tag> when a tag defines
the desired synthesis. See [Echo setup and access](../../../infra/echo/README.md)
for authentication.

Grade only evaluated results when feedback will improve retrieval. Use 0 for
irrelevant and 10 for directly useful results, and always explain overall
quality on stdin, including for an empty result set:

~~~
uv run infra/echo/cli.py feedback --query "<query>" \
  --grade wiki:730=0 --grade file:731=10 \
  <<< "The file result answered the task; the wiki result was off-topic."
~~~

## Choose the durable home

At resolution, choose one narrow home:

- OPS.md: recurring subsystem procedure, diagnostic workflow, or guardrail.
- docs/: reusable product or user guidance.
- One Echo incident entry: evidence, decisions, and outcome for one incident;
  tag it incident, debugging, subsystem, severity, resolution, and ops for
  infrastructure incidents. Link its canonical URL from the related PR or issue.
- An existing wiki note: cross-cutting synthesis with no repository home.
- A new wiki note only when search finds no near-duplicate.

Keep raw logs and diffs in their source systems. Edit the same incident entry
while that incident continues; revise a near-duplicate synthesis instead of
adding another.

## Wiki notes

Write the shortest standalone note that changes a future action. Lead with a
decision, command, discriminator, guardrail, or recovery step; include only
supporting facts and canonical links. Do not omit a precondition, destructive
action warning, or material caveat.

Use Open Knowledge Format with lowercase kebab-case tags and a search-oriented
use_when:

~~~markdown
---
type: wiki-note
title: Stalled TPU collectives can have several low-power causes
use_when: when a distributed TPU job stalls before the first optimizer update
tags: [ops, debugging, iris]
---

State the reusable pattern, discriminators, and evidence URLs.
~~~

Create or revise the closest note:

~~~
uv run infra/echo/cli.py wiki add --file note.md
uv run infra/echo/cli.py wiki show <id> > note.md
uv run infra/echo/cli.py wiki edit <id> --file note.md
~~~

Both write commands print the canonical URL; return and cite it after every
add/edit.
