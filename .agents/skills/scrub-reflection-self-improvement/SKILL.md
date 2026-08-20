---
name: scrub-reflection-self-improvement
description: "Scheduled scrub: repository self-improvement."
schedule_cron: "0 10 2 * * *"
schedule_tz: America/New_York
---

# Repository self-improvement scrub

On scheduled turns, identify and land one high-leverage improvement in
marin-community/marin. Look for repeated confusion, recurring failures/manual
steps, and capability gaps in docs, recipes, experiments, scripts, and infra.

Before choosing work, inspect recent open issues and PRs, recent main commits,
and related AGENTS.md, .agents/skills/, and docs guidance. Check for existing
scrub-generated artifacts and de-duplicate against open issues/PRs. Prefer
advancing an existing artifact; if no justified improvement exists, choose a
no-op.

Choose the highest-leverage low-coordination change. Codify recurring workflow
guidance in AGENTS.md or .agents/skills/. Prefer direct low-risk implementation
over a new issue.

Publish edits via .agents/skills/commit/SKILL.md. If blocked, report the blocker
and set a future follow-up time; also produce a concrete plan and capture the
follow-up work in GitHub. A no-op report must name the inspected issue/PR/commit
windows, reject each candidate, and explain why deferral is preferable.

Finish with exactly:

~~~text
HARNESS_SCRUB_LOOP {"needs_followup_at":null}
~~~

Use a future RFC 3339 timestamp in needs_followup_at when follow-up is required.
