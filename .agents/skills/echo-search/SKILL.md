---
name: echo-search
description: "Search and cite Marin's activity: the echo corpus (GitHub + Discord, always-on, zero setup), marinmirror via the mumwelt `mum` CLI (broader — also W&B and weekly summaries — with hybrid search), and the shared agent work-log. Use for 'was X discussed', 'which issue/PR covered Y', 'what did people decide about Z', 'what is the team working on'."
---

# Skill: echo-search

Three ways to search Marin's activity, every hit citable by URL — cite it, don't
paraphrase from memory.

## echo — GitHub + Discord (default; zero setup)

`scripts/echo_search.py` searches the echo corpus (github issues/PRs/comments + Discord
messages) over Cloud SQL IAM — no local install, no download beyond the query model.

```bash
scripts/echo_search.py search "expert parallel MoE MFU on B200" --limit 10
scripts/echo_search.py search "..." --source discord --kind message --since 2026-06-01
scripts/echo_search.py grep "ragged_all_to_all" --source discord   # exact substring, newest first
scripts/echo_search.py show <id>                                   # one chunk, verbatim
```

- `search` ranks by cosine distance (lower is closer; github hits below ~0.25 are usually
  on-topic); `grep` is an ILIKE scan for identifiers, run names, and exact strings.
- Discord hits are the single message only — its surrounding thread isn't stored, so open
  the URL when you need the conversation.
- Corpus is the github+discord slice of marinmirror, re-synced every ~10 minutes.

**Access:** Cloud SQL IAM, no password — you connect as your own ADC identity, which must
be a member of `echo@openathena.ai` (`roles/cloudsql.instanceUser` + `roles/cloudsql.client`).
The database username is resolved from ADC; set `MARIN_DB_USER` only for principals it
can't resolve, and `GOOGLE_CLOUD_QUOTA_PROJECT` if the SQL Admin API isn't enabled on your
ADC's quota project.

## marinmirror — the fuller corpus, via mumwelt

echo mirrors only github+discord. For the **complete** marinmirror corpus — also W&B run
metadata/results, the distilled weekly summaries, and code symbols — with hybrid
keyword+semantic search, use the `mum` CLI (the mumwelt client, `~/projects/mumwelt`):

```bash
mum search "<natural language or identifier>" --source github,discord,wandb,narrative --json
mum show <url|ref>            # expand a hit (discord window / github thread)
mum run <project>/<run>       # a W&B run's config + final numbers
mum summaries show latest     # the weekly overview
```

Prefer `mum` when the answer may live in W&B results or a weekly summary, or when you want
hybrid ranking. It keeps a local corpus cache (`mum refresh` to update) and needs a GitHub
token of an Open-Athena member (`read:org`).

## work-log — what the team's agents are doing

The `work_log` table (same echo database) is the agents' shared logbook. Browse it with the
**work-log** skill, which wraps a CLI:

- `recent [--days N] [--project <slug>]` — newest entries, optionally scoped to one thread
  of work.
- `show <id>` — the full body of one entry.

Use it to answer "has anyone's agent already looked at this" before starting, and cite the
entry's project and title.
