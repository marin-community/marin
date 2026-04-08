---
name: debug-iris-controller
description: Debug Iris controller state using offline checkpoint snapshots and the process RPC. Use when investigating stuck jobs, resource leaks, scheduling failures, or worker issues.
---

Debug Iris controller issues. Follow the procedures in `lib/iris/OPS.md` — specifically the "SQL Queries", "Process Inspection & Profiling", "Known Bugs", and "Troubleshooting" sections.

Read first: `lib/iris/AGENTS.md`

## Guardrails

- **NEVER modify the database** without explicit user approval. Read-only queries on downloaded checkpoints only.
- **Prefer `iris process profile`** over SSH for profiling — it uses the `/system/process` RPC and avoids direct VM access. SSH is a fallback when the RPC doesn't cover your needs.
