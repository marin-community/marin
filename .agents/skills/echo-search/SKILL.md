---
name: echo-search
description: "Search and cite Marin's GitHub and Discord activity in Echo. Use for 'was X discussed', 'which issue/PR covered Y', and 'what did people decide about Z'."
---

# Skill: Echo Search

Search Marin's activity through the shared Echo corpus. Every hit has a canonical URL;
cite that URL instead of paraphrasing from memory.

## Search

`infra/echo/cli.py` queries GitHub issues, pull requests, comments, and Discord messages
through the `echo-api` service. Run it in the repo environment:

```bash
uv run infra/echo/cli.py search "expert parallel MoE MFU on B200" --limit 10
uv run infra/echo/cli.py search "..." --source discord --kind message --since 2026-06-01
uv run infra/echo/cli.py grep ragged_all_to_all --source discord
uv run infra/echo/cli.py show <id>
```

- `search` is hybrid lexical + semantic, ranked by a reciprocal-rank score (higher is
  better); `grep` is an ILIKE scan for identifiers, run names, and exact strings.
- Discord hits are the single message only — its surrounding thread isn't stored, so open
  the URL when you need the conversation.
- The corpus is the GitHub and Discord slice of the marinmirror corpus, re-synced into Echo
  every ~10 minutes.

Authentication reuses the shared Marin login: run `iris login` once, or provide
service-account credentials for an unattended caller. Follow
[`infra/echo/README.md`](../../../infra/echo/README.md) for setup, access details, and the
HTTP API. Search is powered by [mumwelt](https://github.com/Open-Athena/mumwelt).
