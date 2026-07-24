---
name: context-search
description: "Semantic and substring search over the echo corpus — GitHub issues/PRs/comments and Discord messages, every hit citable by URL. Use for questions about Marin activity: was X discussed, which issue/PR covered Y, what did people decide about Z."
---

# Skill: Context Search

The `chunks` table on echo (the `context` database on the shared `marin-metadata` instance) mirrors the github+discord slice of the
marinmirror corpus — 73k+ chunks, re-synced every 10 minutes — with pgvector embeddings
for semantic search (see `infra/echo/README.md`). Every hit carries a canonical URL:
cite it, don't paraphrase from memory.

## When to use

- "Was this discussed / has anyone hit this before?" — search before re-deriving.
- "Which issue or PR covers X?" — faster and fresher than browsing GitHub.
- Answering the user about past decisions, experiments, or Discord threads — with links.

For live GitHub state (open PR status, CI, review threads) use `gh`; this corpus is for
finding and citing, at up-to-10-minutes staleness. For what other people's *agents* are
doing, use the `work-log` skill — same database, different table.

## Commands

```bash
scripts/context_search.py search "expert parallel MoE MFU on B200" --limit 10
scripts/context_search.py search "..." --source discord --kind message --since 2026-06-01
scripts/context_search.py grep "ragged_all_to_all" --source discord   # exact substring, newest first
scripts/context_search.py show <id>                                   # full text of one hit
```

- `search` embeds the query with the corpus's model (bge-small, downloaded on first use)
  and ranks by cosine distance; lower is closer, and github hits below ~0.25 are usually
  on-topic.
- `grep` is a plain ILIKE scan — use it for identifiers, run names, and exact strings;
  semantic search is for intent.
- Discord hits are noisier than GitHub ones: messages are embedded with their conversation
  context, but only the single message is displayed — `show` the id or open the URL before
  citing.

## Access

Cloud SQL IAM, no password: you connect as your own ADC identity, which must be a member
of `echo@openathena.ai` (granted `roles/cloudsql.instanceUser` + `roles/cloudsql.client`).
Read-only on `chunks`. If your ADC is a service-account key, set `MARIN_DB_USER` or grant
that principal directly.
