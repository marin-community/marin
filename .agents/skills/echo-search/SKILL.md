---
name: echo-search
description: "Search and cite Marin's GitHub and Discord activity in Echo. Use for 'was X discussed', 'which issue/PR covered Y', and 'what did people decide about Z'."
---

# Skill: Echo Search

Search Marin's activity through the shared Echo corpus. Every hit has a canonical URL;
cite that URL instead of paraphrasing from memory.

## Search

`infra/echo/cli/search.py` queries GitHub issues, pull requests, comments, and Discord
messages through Cloud SQL IAM:

```bash
infra/echo/cli/search.py search "expert parallel MoE MFU on B200" --limit 10
infra/echo/cli/search.py search "..." --source discord --kind message --since 2026-06-01
infra/echo/cli/search.py grep "ragged_all_to_all" --source discord
infra/echo/cli/search.py show <id>
```

- `search` ranks by cosine distance (lower is closer; GitHub hits below about 0.25 are usually
  on-topic); `grep` is an ILIKE scan for identifiers, run names, and exact strings.
- Discord hits are the single message only — its surrounding thread isn't stored, so open
  the URL when you need the conversation.
- The corpus is the GitHub and Discord slice of the marinmirror corpus, re-synced into Echo
  every ~10 minutes.

Access requires OpenAthena ADC and membership in the organization-wide database group.
Follow [`infra/echo/README.md`](../../../infra/echo/README.md) for setup, access details,
and the HTTP API. Search is powered by
[mumwelt](https://github.com/Open-Athena/mumwelt).
