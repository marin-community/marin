---
name: fix-docs
description: De-rot markdown docs in lib/iris, lib/zephyr, and lib/fray.
---

Fix the markdown docs within `lib/iris`, `lib/zephyr` and `lib/fray` so they
comply with the principles below. Do NOT touch docs outside those directories.

Output: dispatch sub-agents that (1) parse the code and docs and (2) make the
documentation changes locally. Commit the changes into a single local commit,
inform the user, and summarize what changed. Never push without explicit user
approval.

## Principles

- **Agents are the primary consumers of documentation.** Coding agents write
  most Marin code; humans use agents as "documentation brokers". Segregate docs
  by audience:
  - Human docs live in `docs/` — high-level, "getting started", concise.
  - Agent docs live in the directory tree next to the code they describe —
    token-efficient.
- **Agentic docs follow standards** so agents know where to look. Entry points:
  - `**/AGENTS.md` — targeted agent guidance for the subtree, plus a recursive
    ToC/index of subtree content.
  - `**/OPS.md` — field guide for gathering telemetry, running commands, and
    debugging the systems in the subtree.
  - `.agents/skills/*/SKILL.md` — prompts for common agentic workflows.
- **Avoid rot.** Rot confuses agents and wastes context. When a doc is out of
  date with the code: update it (if recent and the change is small) or archive
  it to `.agents/project/YYYYMMDD_filename.md` (if historical-only; date from
  the first commit via `git`). The code is the source of truth.
- **Context quality and quantity are paramount.** Model performance degrades on
  a full context window. Agent docs should focus on:
  - "Negative guidance": assume agents are smart; document sharp edges based on
    *observed* suboptimal agent behavior.
  - Using markdown as indexes/ToC for context efficiency ("for X, read file Y").

## Common issues

- Stale design docs that guided initial system design — move to
  `.agents/project/YYYYMMDD_filename.md`, dated from the first commit (`git`).
