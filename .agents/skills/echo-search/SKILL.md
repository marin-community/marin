---
name: echo-search
description: "Search and cite Marin's activity: the echo corpus (GitHub + Discord, always-on, zero setup) and the shared agent work-log. Use for 'was X discussed', 'which issue/PR covered Y', 'what did people decide about Z', 'what is the team working on'."
---

# Skill: echo-search

Two ways to search Marin's activity, every hit citable by URL — cite it, don't paraphrase
from memory.

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
- The corpus is the github+discord slice of the marinmirror corpus, re-synced into echo
  every ~10 minutes.

Powered by [mumwelt](https://github.com/Open-Athena/mumwelt).

## work-log — what the team's agents are doing

The `work_log` table (same echo database) is the agents' shared logbook. Browse it with the
**work-log** skill, which wraps a CLI:

- `recent [--days N] [--project <slug>]` — newest entries, optionally scoped to one thread
  of work.
- `show <id>` — the full body of one entry.

Use it to answer "has anyone's agent already looked at this" before starting, and cite the
entry's project and title.
