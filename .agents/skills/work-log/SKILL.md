---
name: work-log
description: "Read and append the shared team work_log on the echo context database. Use at the start of non-trivial work to see what other people's agents are doing, and at milestone boundaries (result, decision, blocker, handoff) to log your own."
---

# Skill: Work Log

The `work_log` table on echo — the `context` database on the shared `marin-metadata` Cloud SQL instance — is a team-wide logbook
written by agents, never humans: one row per distilled milestone, shared context for
"what is everyone's agent working on". The database also carries `chunks`, a searchable
mirror of Marin's GitHub and Discord activity (see `infra/echo/README.md`).

This is not a session transcript and not a replacement for per-task logbooks (the
`task-logbook` skill): the logbook holds the maximally informative per-task record;
the work_log holds the cross-team headline.

## When to read

- At the start of non-trivial work on shared code or infrastructure, check whether
  someone else's agent is already on it or recently left state you should know about:

  ```bash
  scripts/work_log.py recent --days 7
  scripts/work_log.py recent --project <slug>     # one thread of work
  scripts/work_log.py show <id>                   # full body of one entry
  ```

- When the user asks what the team is doing, answer from `recent` and cite entries.

## When to write

Append one entry at a milestone boundary — not per command, not per session minute:

- a durable result landed (PR opened/merged, experiment concluded, deploy done)
- a decision was made that others would otherwise re-litigate
- a blocker was hit that others may also hit
- a handoff: you are stopping and someone (or some future session) picks up

```bash
scripts/work_log.py add --project <slug> --title "<one line>" --body - <<'EOF'
Short markdown. Link evidence inline: PRs, issues, W&B runs, dashboards.
State what changed, what is decided, what is still open.
EOF
```

## Entry conventions

- `--author` is `<user>/<agent>` (defaults to `<os user>/claude-code`; override the agent
  part if you are not Claude Code).
- `--project` is a stable kebab-case slug for the thread of work (e.g. `grug-moe-mfu`,
  `marin-context-db`). Reuse the slug across sessions and people; check `recent` for the
  spelling before minting a new one.
- Title: one scannable line, the thing a teammate reads in a list.
- Body: a few short paragraphs at most, markdown, links inline. Distill; never paste
  transcripts, command logs, or raw tool output.

## Access

Auth is Cloud SQL IAM — no password. You connect as your own ADC identity, which must be
a member of the `echo@openathena.ai` group (granted `roles/cloudsql.instanceUser` +
`roles/cloudsql.client`). The group can read `chunks` and read/append `work_log` — the log
is append-only; correct a wrong entry by adding a new one that references it, not by
rewriting history. If your ADC is a service-account key rather than your user identity,
that principal needs its own grants (or set `MARIN_DB_USER`).
