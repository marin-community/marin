---
name: fix-docs
description: De-rot markdown docs in lib/iris, lib/zephyr, and lib/fray.
---

Fix only Markdown under lib/iris, lib/zephyr, and lib/fray. Dispatch agents to
compare code and docs, then make the edits locally. Commit all changes in one
local commit, report the result, and never push without explicit approval.

Agent-facing entry points are '**/AGENTS.md', '**/OPS.md', and
'.agents/skills/*/SKILL.md'; human-facing high-level guidance belongs in docs/.
Keep agent docs as concise indexes and record sharp edges based on observed
failures.

The code is authoritative. Update small, recent drift; archive historical-only
design docs at .agents/projects/<YYYYMMDD>_<filename>.md, using the first commit
date from git.
