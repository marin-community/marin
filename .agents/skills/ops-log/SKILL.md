---
name: ops-log
description: Write a postmortem-style ops log for a debugging or incident-response conversation into `.agents/ops/`. Use when asked to "log this", "summarize for ops", "write a postmortem", or at the end of a live-infrastructure debugging session so future sysops engineers inherit the context.
---

# Skill: Ops Log

Summarize a debugging / incident-response conversation as a standalone
postmortem entry under `.agents/ops/`. The audience is a future sysops
engineer (probably another Claude session) who has no memory of this
conversation and needs to reconstruct: what broke, what was tried, what the
user steered, what fixed it, and how OPS.md guidance could have shortened the
investigation.

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

One screen. A future engineer reading just this should know whether this log
is relevant to their current incident. Include: the user-visible symptom,
the real root cause, the fix applied, and any lingering caveat.

### Original problem report

Quote or paraphrase the user's opening message. What they *observed*, not
what the real bug turned out to be. This is the pattern-match hook for
future searches — preserve the exact error string / dashboard text / command
that the next engineer will also be typing into ripgrep.

### Investigation path

Narrative, not a log dump. What was checked, in order, and why each step was
chosen. Include the dead ends — they teach the next engineer what *not* to
spend time on. Cite file paths and line numbers for code you read, and
commit/log timestamps for live data.

Format as a numbered list of short paragraphs. Five to twelve steps is
typical; if it's longer, you're narrating tool calls instead of decisions.

### User course corrections

Explicit list of points where the user redirected the investigation. Each
entry: what the model was about to do, what the user said instead, and why
it was the right call. This is the most load-bearing section — it captures
judgment the model didn't have.

### Root cause

Tight technical description. One or two paragraphs. Cite the specific
file:line of the bug. If there's a class of bug (e.g. "invariant
violation between tables X and Y"), name it.

### Fix

What was actually changed, with file paths and a short diff-style excerpt
if the change is subtle. Include the migration / data repair step
separately from the code change. If the fix is deferred, say so and link
the tracking issue.

### How OPS.md could have shortened this

Concrete, actionable suggestions for `lib/<component>/OPS.md` edits that
would have saved time. Each suggestion: the section to edit, the sentence
or command to add, and the signal it would have unblocked.

**Generic patterns only.** OPS.md is for recurring diagnostic workflows
across many incidents — not for this one bug. Before writing a
suggestion, ask: *would this have helped an engineer debugging a
completely different incident in the same subsystem?* If not, drop it.

Good OPS.md additions look like:
- A new tool/workflow the investigation relied on (e.g. "query parquet
  logs directly with duckdb + gcsfs" — applies to every future log dive).
- A recurring smell mapped to a class of cause (e.g. "same pending-reason
  text on many jobs → diagnostic cache has stopped updating" — applies
  across any future cache-update bug, not just this one).
- A default that burned time (e.g. "prefer copying an existing GCS
  checkpoint over triggering a new one when the controller is stuck" —
  applies to any stuck-controller incident).

Bad OPS.md additions look like:
- Troubleshooting rows keyed on the exact literal string this incident
  produced. That's Known-Bugs material at best, and after the fix ships,
  it's just noise.
- SQL queries that only detect the invariant violation this specific
  bug produced.
- "Watch out for bug X" entries that duplicate the git log.

If a lesson is genuinely incident-specific, put it in the "Root cause" /
"Fix" sections of this log; don't propose it for OPS.md. Avoid vague
"improve documentation" — name the exact section and the exact text to
add.

This is the section that pays forward. Take it seriously.

### Artifacts

Links or paths to supporting evidence, in order of usefulness:

- Local files staged during the investigation (checkpoints, logs) — only
  if still present; don't link to `/tmp` paths that are already gone.
- GCS/S3 paths to parquet / sqlite / log bundles.
- Grafana / dashboard URLs.
- Parent PR and follow-up issues.

## Writing style

- Past tense, third person. The conversation is over; you're writing for
  someone who wasn't there.
- Short, dense sentences. A sysops engineer reading this is busy.
- No praise, no "we" language, no apology. Just: what happened and what to
  do about it.
- No emojis.
- Absolute file paths relative to the repo root (e.g.
  `lib/iris/src/iris/cluster/controller/transitions.py:2167`).
- Command-line snippets that the next engineer can paste verbatim.
- If you cite a log message, preserve its exact text — that is the string
  they will grep for.

## What to skip

- Minute-by-minute narration of tool calls.
- Code that's already obvious from the fix diff.
- Generic reminders ("remember to check logs first"). OPS.md is for general
  guidance; this file is for this specific incident.
- Hedged conclusions. If the root cause is known, state it. If it isn't,
  say so explicitly under "Investigating".

## After writing

Do not add an index file or update `MEMORY.md`. The logs are discoverable
by `ls .agents/ops/` and by full-text search. If you notice the
directory is getting unwieldy (>30 entries), flag it to the user — they
may want an index.
