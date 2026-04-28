---
name: design-doc
description: Walk a user through producing a 1-page design doc, opening a PR with it, and pinging Discord for review. Use when asked to write a design doc, spec, or technical proposal.
---

# Skill: Design Doc Workflow

## Purpose

A design doc in Marin is a ~1-page document posted as a PR for early feedback. The goal is to surface design issues *before* implementation, not to gate work — area owners are expected to LGTM or comment quickly, and the author begins implementation in parallel. See [issue #5210](https://github.com/marin-community/marin/issues/5210) for the rationale.

This skill is the workflow for producing one. It is **interactive** — you ask the user questions when you genuinely don't know, but you make reasonable inferences and proceed when you can.

The template lives at `.agents/projects/design-template.md`. New docs go to `.agents/projects/<slug>.md` (slug only, no date prefix — `git log` already records the date).

## When to use this skill

- A task will likely take more than a day, or is load-bearing for other work.
- A change crosses subproject boundaries (e.g. iris ↔ levanter, marin ↔ zephyr).
- A change introduces a new service, package, or persistent data shape.

If none of those apply, just open the PR — don't manufacture a design doc for a 50-line bug fix.

---

# Workflow

Five phases. Confirm with the user at natural decision points (after Research, after Draft, before Publish), but don't ask permission to move forward when the next step is obvious.

## 1. Frame

The user starts the skill with a framing paragraph stating what they want to do and why. If they didn't, query them for it — or infer it from prior conversation context if that context is rich enough. A one-sentence "fix the foo thing" is *not* enough to proceed; push back and ask for the why.

**You infer the slug.** Short, lowercase, underscores (`finelog_lift`, `iris_autoscaler_refactor`). State it in one line ("I'll save this as `.agents/projects/<slug>.md`") and proceed — only stop if it collides with an existing file, in which case propose a disambiguator.

Once you have a framing paragraph, proceed directly to research. Don't batch a long list of questions up front; that's what Interrogate is for, after research has shown you what to ask.

## 2. Research

Spawn an `Explore` subagent (do not search yourself — the digest stays out of main context). Brief it with the framing paragraph and ask for:

- Relevant files with line numbers (the design doc must reference real code, not placeholders).
- Related designs already in `.agents/projects/` — read them, note overlap.
- Related GitHub issues/PRs via `gh` if the user named any, plus a quick `gh issue list --search` on the topic.
- Existing utilities or abstractions the proposal might reuse (per `AGENTS.md` "Code Reuse").

Return to the user with a bulleted digest: *"Here's what I found, here's what surprised me, here's what's still unclear."* Ask whether the framing should shift before drafting.

## 3. Interrogate

Ask 3–6 targeted questions in one batched message. Good questions surface things the doc *must* answer:

- Scope boundaries — what is explicitly **not** in this design?
- Testing — what's the smallest test that would catch a regression?
- Tradeoff decisions — when there are two reasonable approaches, which one and why?
- Unknowns the user wants reviewers to weigh in on (these become Open Questions).

Bad questions are ones research could have answered. Don't ask things you could grep for. **Don't ask about backwards compatibility unless you have specific reason to think it matters here** — Marin generally does not optimize for backwards compatibility (per `AGENTS.md` "Deprecation"), so default to assuming the proposal updates all call sites.

## 4. Draft

Read `.agents/projects/design-template.md`, fill in each section. Guidelines:

- **~500 words is the target, not a hard limit.** Concision is a virtue; spend words where they buy clarity. If you're at 700, look for cuts. If 600 is genuinely tighter than 500, ship 600.
- Reference real `file.py:line` paths from research, not placeholders.
- One small code snippet (10-30 lines) only if prose is genuinely worse. Default to no snippet.
- Open Questions section is non-empty — if the design has no unknowns, ask the user what they want feedback on.
- Don't add a backwards-compat section by default. Mention compat only if the change genuinely needs migration (persisted data, public API consumed externally, etc.).

Show the draft inline, accept edits in conversation. Iterate until the user says ship.

## 5. Stress-test (senior review)

Before publishing, hand the draft to a senior reviewer subagent (`Plan` agent — software architect) with a prompt like: *"Review this design doc thoroughly. Identify underspecified areas, weak motivation, missing tradeoffs, places where two reasonable engineers would implement different things, and concrete suggestions for tightening the proposal."*

When the reviewer returns:

- **Incorporate obvious improvements directly into the draft.** Tightening prose, fixing a confused tradeoff, adding a missing file:line reference, moving a clear ambiguity into Open Questions — just do it.
- **Query the user only on ambiguous or load-bearing decisions** — places where the reviewer surfaced a real tradeoff and you can't tell which way the user wants to go.

Show the user a brief summary: what you incorporated, what you're punting on, what needs their call.

## 6. Publish

Two actions, can run together. After this, the skill is done.

1. **Commit and PR** via the `pull-request` skill. Branch `design/<slug>`. Single commit adding `.agents/projects/<slug>.md`. PR title `[Design] <slug>`. PR body is the doc itself plus a one-line "Discussion welcome — see Open Questions." footer. Labels `design` and `agent-generated`.

2. **Discord ping.** Run `python scripts/ops/discord.py --channel code-review` with a 2-line message: PR title + URL + the framing paragraph (or a one-sentence compression of it). Send it; no need to confirm exact text unless the user asked.

Once both have happened, you're done. Feedback lives on the PR; the user can start implementation in parallel; the 1-pager is a snapshot, not a living doc.

---

# Notes for the agent running this skill

- **The template and canonical worked example live in `.agents/projects/design-template.md`.** Read it before drafting. Don't use other docs in `.agents/projects/` as style references — they predate this skill and are inconsistent.
- `agent-generated` and `design` labels: create the `design` label if it doesn't exist (`gh label create design --description "Design doc / 1-pager for review"`).
- If the user wants to skip a phase ("just write it, I know what I want"), honor that — but still produce the Open Questions section and still run the Stress-test in Phase 5. Those are the cheapest, highest-value steps.
- Implementation is out of scope. After Publish, the skill is done — hand off to `fix-issue` or `pull-request` for the work itself only if the user asks.
