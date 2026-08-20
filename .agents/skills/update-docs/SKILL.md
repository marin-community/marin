---
name: update-docs
description: "Update documentation, runbooks, and reusable task guidance when implementation work or experiments change behavior or operational practice."
---

# Update Docs

Use when implementation or experiments change behavior, reveal stale
instructions, create an operational pattern, or produce guidance that must
survive the issue thread.

Update the nearest existing home:

- user-facing docs for behavior/configuration;
- OPS/runbooks for commands, services, dashboards, alerts, or recovery;
- skill docs for reusable agent workflows;
- experiment reports or indexes for durable findings.

Add a document only when no natural home exists or an existing page would become
unfocused. Describe current behavior, include exact commands and paths for
operations, and correct stale instructions instead of adding caveats. Keep
broad docs concise; put detailed run notes in logbooks or issue comments.

Done means code and docs agree, the authoritative procedure is discoverable
without the issue thread, and reusable guidance is specific enough to act on.
Apply .agents/skills/writing-style/SKILL.md to the prose.
