---
name: design-doc
description: Walk a user through producing a 1-page design doc, opening a PR with it, and pinging Discord for review. Use when asked to write a design doc, spec, or technical proposal.
---

# Skill: Design Doc Workflow

## Purpose

A design doc in Marin is a **≤500-word 1-pager** posted as a PR for early feedback. The goal is to surface design issues *before* implementation, not to gate work — area owners are expected to LGTM or comment quickly, and the author begins implementation in parallel. See [issue #5210](https://github.com/marin-community/marin/issues/5210) for the rationale.

This skill is the workflow for producing one. It is **interactive** — you ask the user questions, you do not silently autocomplete a draft.

The template lives at `.agents/projects/design-template.md`. New docs go to `.agents/projects/<slug>.md` (slug only, no date prefix — `git log` already records the date).

## When to use this skill

- A task will likely take more than a day, or is load-bearing for other work.
- A change crosses subproject boundaries (e.g. iris ↔ levanter, marin ↔ zephyr).
- A change introduces a new service, package, or persistent data shape.

If none of those apply, just open the PR — don't manufacture a design doc for a 50-line bug fix.

---

# Workflow

The skill has five phases. **Confirm with the user before moving from one phase to the next.** Do not chain through silently.

## 1. Frame

Ask the user, in one batched message:

- One-sentence problem statement (what's broken or missing today?).
- A project slug — short, lowercase, underscores (`ferry_framework`, `iris_autoscaler_refactor`). Confirm the resulting path: `.agents/projects/<slug>.md`.
- Whether they already have prior context (an issue, a Slack thread, an earlier doc) you should read.

If the slug collides with an existing file in `.agents/projects/`, propose a disambiguating one.

## 2. Research

Spawn an `Explore` subagent (do not search yourself — the digest stays out of main context). Brief it with the problem statement and ask for:

- Relevant files with line numbers (the design doc must reference real code, not placeholders).
- Related designs already in `.agents/projects/` — read them, note overlap.
- Related GitHub issues/PRs via `gh` if the user named any, plus a quick `gh issue list --search` on the topic.
- Existing utilities or abstractions the proposal might reuse (per `AGENTS.md` "Code Reuse").

Return to the user with a bulleted digest: *"Here's what I found, here's what surprised me, here's what's still unclear."* Ask whether the framing should shift before drafting.

## 3. Interrogate

Ask 3–6 targeted questions in one batched message. Good questions surface things the doc *must* answer:

- Scope boundaries — what is explicitly **not** in this design?
- Backwards compatibility — does this break existing users/experiments? If so, what's the migration?
- Testing — what's the smallest test that would catch a regression?
- Tradeoff decisions — when there are two reasonable approaches, which one and why?
- Unknowns the user wants reviewers to weigh in on (these become Open Questions).

Bad questions are ones research could have answered. Don't ask things you could grep for.

## 4. Draft

Read `.agents/projects/design-template.md`, fill in each section. Hard rules:

- **≤500 words.** Count them. If you're over, cut — usually from Design (you're describing implementation, not motivation) or by removing throat-clearing.
- Reference real `file.py:line` paths from research, not placeholders.
- One small code snippet (10-30 lines) only if prose is genuinely worse. Default to no snippet.
- Open Questions section is non-empty — if the design has no unknowns, ask the user what they want feedback on.
- State backwards-compat posture in the Design section, not as a separate heading.

Show the draft inline, accept edits in conversation. Iterate until the user says ship.

## 5. Stress-test (gap check)

Before publishing, hand the draft to a fresh `Explore` subagent with a prompt like: *"Read this design doc cold. Identify underspecified areas — places where two reasonable engineers would implement different things. Don't propose fixes; just list ambiguities."*

Surface its findings to the user. Decide together: tighten the doc, or move ambiguities into Open Questions. This step is cheap and catches the gap-implementation problem agents fall into.

## 6. Publish

Three sub-steps, each gated on **explicit user approval** (per `AGENTS.md`'s rule on shared-state actions). Do not chain them.

1. **Commit and PR.** Branch named `design/<slug>`. Single commit adding `.agents/projects/<slug>.md`. PR title `[Design] <slug>`. PR body is the doc itself, plus a one-line "Discussion welcome — see Open Questions." footer. Apply labels `design` and `agent-generated`. Use the `pull-request` skill for the actual PR mechanics.

2. **Discord ping.** Run `python scripts/ops/discord.py --channel code-review` with a 2-line message: PR title + URL + the one-sentence problem statement from Phase 1. Confirm the exact message text with the user before sending.

3. **Hand off.** Tell the user: feedback comes on the PR; they can begin exploratory implementation in a separate branch in parallel; the design issue/PR closes when the implementing PR lands (the 1-pager is a snapshot, not a living doc).

---

# Notes for the agent running this skill

- **The template and canonical worked example live in `.agents/projects/design-template.md`.** Read it before drafting. Don't use other docs in `.agents/projects/` as style references — they predate this skill and are inconsistent.
- `agent-generated` and `design` labels: create the `design` label if it doesn't exist (`gh label create design --description "Design doc / 1-pager for review"`).
- If the user wants to skip a phase ("just write it, I know what I want"), honor that — but still produce the Open Questions section and still run the Stress-test in Phase 5. Those are the cheapest, highest-value steps.
- Implementation is out of scope for this skill. After the PR is open, hand off to `fix-issue` or `pull-request` for the work itself.

