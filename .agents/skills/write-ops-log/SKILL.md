---
name: write-ops-log
description: Write a postmortem ops log to .agents/ops/. Use after an infrastructure-debugging session.
---

# Skill: Ops Log

Summarize a debugging / incident-response conversation as a standalone
postmortem entry under `.agents/ops/`. The audience is a future sysops
engineer (probably another Claude session) with no memory of this conversation
who must reconstruct: what broke, what was tried, what the user steered, what
fixed it, and how OPS.md guidance could have shortened the investigation.

## Filename

`.agents/ops/YYYY-MM-DD-<slug>.md`

The `.agents/ops/` directory is checked into git; `.agents/ops/logs/` is
gitignored (matched by the global `logs/` pattern), so do not nest logs
under a `logs/` subdirectory.

- Use the date the issue was investigated, not today's arbitrary date.
- `<slug>` is 3-6 words, kebab-case, naming the system + symptom. Examples:
  `iris-scheduler-freeze`, `coreweave-nodepool-stuck-delete`,
  `zephyr-coordinator-oom`.
- If a log already exists for this incident, extend it with a new section
  rather than creating a parallel file.

## Structure

Write a single markdown file with these sections in order. Don't invent
extra sections; omit any section that genuinely has nothing to say.

### Frontmatter

```yaml
---
date: YYYY-MM-DD
system: iris | zephyr | fray | coreweave | gcp | <component>
severity: outage | degraded | near-miss | diagnostic-only
resolution: fixed | mitigated | wontfix | investigating
pr: <url or "none">
issue: <url or "none">
---
```

### TL;DR (3-6 bullets)

One screen. A future engineer should know from this alone whether the log is
relevant. Include: user-visible symptom, real root cause, fix applied, any
lingering caveat.

### Original problem report

Quote or paraphrase the user's opening message — what they *observed*, not the
real bug. This is the pattern-match hook: preserve the exact error string /
dashboard text / command the next engineer will grep for.

### Investigation path

Narrative, not a log dump. What was checked, in order, and why. Include dead
ends — they teach what *not* to spend time on. Cite file:line for code read,
commit/log timestamps for live data.

Format as a numbered list of short paragraphs. Five to twelve steps; longer
means you're narrating tool calls instead of decisions.

### User course corrections

Explicit list of points where the user redirected the investigation. Each
entry: what the model was about to do, what the user said instead, and why it
was right. The most load-bearing section — it captures judgment the model
lacked.

### Root cause

Tight technical description, one or two paragraphs. Cite the specific file:line
of the bug. If there's a class of bug (e.g. "invariant violation between tables
X and Y"), name it.

### Fix

What changed, with file paths and a short diff-style excerpt if subtle. Include
the migration / data repair step separately from the code change. If the fix is
deferred, say so and link the tracking issue.

### How OPS.md could have shortened this

Concrete, actionable suggestions for `lib/<component>/OPS.md` edits. Each
suggestion: the section to edit, the sentence or command to add, and the signal
it would have unblocked.

**Generic patterns only.** OPS.md is for recurring diagnostic workflows across
many incidents — not this one bug. Before writing a suggestion, ask: *would
this help an engineer debugging a completely different incident in the same
subsystem?* If not, drop it.

Good OPS.md addition: a recurring smell mapped to a class of cause — e.g. "same
pending-reason text on many jobs → diagnostic cache has stopped updating"
(applies to any future cache-update bug, not just this one). Also good: a new
tool/workflow the investigation relied on, or a default that burned time.

Bad OPS.md addition: a troubleshooting row keyed on the exact literal string
this incident produced — Known-Bugs material at best, noise after the fix
ships. Also bad: SQL queries that only detect this specific invariant
violation, or "watch out for bug X" entries that duplicate the git log.

If a lesson is genuinely incident-specific, put it in "Root cause"/"Fix" here;
don't propose it for OPS.md. Avoid vague "improve documentation" — name the
exact section and text to add. This section pays forward; take it seriously.

### Artifacts

Links or paths to supporting evidence, in order of usefulness:

- Local files staged during the investigation — only if still present; don't
  link to `/tmp` paths already gone.
- GCS/S3 paths to parquet / sqlite / log bundles.
- Grafana / dashboard URLs.
- Parent PR and follow-up issues.

## Writing style

- Past tense, third person; you're writing for someone who wasn't there.
- Short, dense sentences. The reader is busy.
- No praise, no "we" language, no apology.
- No emojis.
- Absolute file paths relative to the repo root (e.g.
  `lib/iris/src/iris/cluster/controller/transitions.py:2167`).
- Command-line snippets the next engineer can paste verbatim.
- If you cite a log message, preserve its exact text — that's the grep string.

## What to skip

- Minute-by-minute narration of tool calls.
- Code already obvious from the fix diff.
- Generic reminders ("remember to check logs first").
- Hedged conclusions. If the root cause is known, state it; if not, say so
  under "Investigating".

## After writing

Do not add an index file or update `MEMORY.md`. Logs are discoverable by
`ls .agents/ops/` and full-text search. If the directory gets unwieldy (>30
entries), flag it to the user.
