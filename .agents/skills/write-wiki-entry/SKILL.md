---
name: write-wiki-entry
description: Record a durable Echo wiki note when you hit a new undocumented workflow, set of operations, or subtle issue. Use like memory-writing, but for team-shared knowledge.
---

# Skill: Write a Wiki Entry

Echo's wiki (`wiki_entries` in the `context` database) is the team's shared,
searchable memory. Write an entry when you work out something a future agent or
teammate would otherwise have to rediscover. This is the shared counterpart to
your private memory files: memory is for you, a wiki entry is for everyone.

## When to Write

Add an entry when you encounter and resolve one of:

- **A new undocumented workflow** — the concrete steps to do something that no
  doc, `AGENTS.md`, or `OPS.md` already covers.
- **A set of operations** you had to assemble — the specific commands, flags, or
  query that got a job done, when finding them again would take real effort.
- **A subtle issue** — a non-obvious failure mode, a gotcha, or a fix whose
  reason is not evident from the code.

Do not write an entry for what the repo already records (code structure, an
existing doc, git history, a fixed bug with a clear commit) or for
conversation-specific ephemera. If the knowledge belongs in `AGENTS.md`, a doc,
or an `OPS.md`, put it there instead — the wiki is for cross-cutting, hard-won
knowledge that has no single home in the tree.

## Before You Write: Search

Duplicate entries fragment the wiki. Search first, and **edit** the existing
entry instead of adding a near-duplicate:

```bash
uv run infra/echo/cli.py wiki search "hnsw index unused"
```

## Shape of an Entry

Three fields, mirroring a memory's frontmatter and body:

- **title** — the specific thing, named concretely (`chunks.text needs a pg_trgm
  index for grep`, not `Database performance`).
- **use_when** — one sentence naming the situation in which an agent should load
  this entry. This is what semantic search matches against, so describe the
  *trigger*, not the content (`when grep or ILIKE queries over the corpus are
  slow`).
- **body** — concise markdown: what is true, why, and the concrete commands or
  code. Link evidence inline (issue, PR, file path, query). State caveats.

Write it so it stands alone without this conversation.

## Write It

```bash
uv run infra/echo/cli.py wiki add \
  --title "chunks.text needs a pg_trgm index for grep" \
  --use-when "when grep or ILIKE substring queries over the Echo corpus are slow" \
  --body body.md          # inline text, a file path, or - for stdin
```

The server embeds and attributes the note to your authenticated identity, and
the command prints the entry's URL. **Return that URL** so the reader can open
it. To revise an entry you found while searching, edit it in place:

```bash
uv run infra/echo/cli.py wiki edit <id> --title ... --use-when ... --body -
```

Authentication reuses the shared Marin login (`iris login`); see
`infra/echo/README.md`.
