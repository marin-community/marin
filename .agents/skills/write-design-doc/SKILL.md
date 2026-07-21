---
name: write-design-doc
description: Produce a 1-page design doc, open a PR, and ping Discord for review.
---

# Skill: Design Doc Workflow

## Purpose

A design doc in Marin is a ~1-page document posted as a PR for early feedback. It surfaces design issues *before* implementation, not to gate work — area owners LGTM or comment quickly, and the author implements in parallel. See [issue #5210](https://github.com/marin-community/marin/issues/5210) for rationale.

This skill is **interactive**: ask the user questions when you genuinely don't know, but make reasonable inferences and proceed when you can.

The template lives at `.agents/projects/design-template.md`. New docs go to a slug-named directory `.agents/projects/<slug>/` (slug only, no date prefix — `git log` records the date). Inside it:

- `design.md` — the 1-pager (always).
- `research.md` — in-repo refs, prior art, Q&A summary (always — even a short one).
- `spec.md` — concrete contracts, written *after* the design stabilises (always).

`spec.md` is the contract layer, not an implementation plan. It pins the public surface — `.proto` content, public class/function signatures with types, schema-registry `CREATE` tables, directory layout, file paths. It excludes algorithm pseudocode, sequenced steps, file-by-file plans, "how" rather than "what" — those belong in the PR description or get deleted in favor of writing the code. Even an "internal" refactor has a public surface (an imported function, a passed flag, a config key); the spec pins it so reviewers know what they're agreeing to. If you can't write a spec, you don't have a design yet.

## When to use this skill

- A task will likely take more than a day, or is load-bearing for other work.
- A change crosses subproject boundaries (e.g. iris ↔ levanter, marin ↔ zephyr).
- A change introduces a new service, package, or persistent data shape.

If none apply, just open the PR — don't manufacture a design doc for a 50-line bug fix.

---

# Workflow

Seven phases. Confirm with the user at natural decision points (after Research, Draft, Spec, before Publish), but don't ask permission when the next step is obvious.

## 1. Frame

The user starts the skill with a framing paragraph stating what they want and why. If they didn't, query them — or infer it from rich prior conversation context. A one-sentence "fix the foo thing" is *not* enough; push back and ask for the why.

**You infer the slug.** Short, lowercase, underscores (`finelog_lift`, `iris_autoscaler_refactor`). State it in one line ("I'll save this as `.agents/projects/<slug>/`") and proceed — only stop if it collides with an existing directory, then propose a disambiguator.

Once you have a framing paragraph, proceed directly to research. Don't batch a long question list up front; that's what Interrogate is for.

## 2. Research

Use the `background-research` skill in design-doc mode. Prefer an isolated
Explore subagent when available so the detailed digest stays out of main
context. Use `low` or `medium` effort by default.

The research output must cover in-repo findings, prior-art shape when relevant,
what surprised you, and what remains unclear. Ask whether the framing should
shift before drafting.

**Persist the research.** Save the full digest (in-repo findings with file:line
refs, prior-art digest, anything load-bearing that won't fit the 1-pager) to
`.agents/projects/<slug>/research.md`. The design doc gets a short
`## Background` section (3-5 sentences) linking to `research.md`.

## 3. Interrogate

Ask 3–6 targeted questions in one batched message. Good questions surface things the doc *must* answer:

- Scope boundaries — what is explicitly **not** in this design?
- Testing — the smallest test that would catch a regression.
- Tradeoff decisions — when there are two reasonable approaches, which and why.
- Unknowns the user wants reviewers to weigh in on (these become Open Questions).

Bad questions are ones research could have answered — don't ask things you could grep for. **Don't ask about backwards compatibility unless you have specific reason to think it matters** — Marin generally does not optimize for it (per `AGENTS.md` "Deprecation"); default to assuming the proposal updates all call sites.

## 4. Draft

Read `.agents/projects/design-template.md`, fill in each section. Guidelines:

- **~1000 words is the target, not a hard limit.** Spend words where they buy clarity. At 1200, look for cuts; if 800 is genuinely tighter than 600, ship 800. Goal is "good design" not "short doc".
- Reference real `file.py:line` paths from research, not placeholders. Convert load-bearing citations to commit-pinned permalinks before publishing — see "Linking conventions".
- One small code snippet (10-30 lines) only if prose is genuinely worse. Default to none.
- Open Questions section is non-empty — if the design has no unknowns, ask the user what they want feedback on.
- No backwards-compat section by default; mention compat only if the change genuinely needs migration (persisted data, externally-consumed public API).

Write the draft to `.agents/projects/<slug>/design.md` from the start and iterate on the file (it's the source of truth — conversation-only snippets are invisible to reviewers and diff tools). Summarize the diff inline as you go. Iterate until the user says the design is settled. **Do not write `spec.md` yet** — it derives from the stable design.

## 5. Spec

Once `design.md` is settled, write `.agents/projects/<slug>/spec.md`. This phase is non-negotiable; every design produces a spec. If you can't write one, the design isn't concrete enough — go back to Draft.

What goes in `spec.md`:

- **Public API**: full Python class/function signatures with parameter/return types and a one-paragraph contract docstring per symbol (behavior, edge cases, ordering guarantees). Not implementation bodies.
- **Proto definitions**: the full `.proto` content (name every RPC, message, field).
- **File paths**: where each new piece lives (e.g. "proto at `lib/finelog/src/finelog/proto/finelog_stats.proto`"). A summary table at the bottom is good.
- **Persisted shapes**: schema-registry `CREATE` statements, on-disk layout, file naming, JSON/proto envelope formats.
- **Errors**: every new error type with its triggering condition, plus behavioural changes in existing error paths.
- **Out of scope**: explicit list of related changes the design *doesn't* commit to — reviewers use this to know what *not* to push back on.

`spec.md` has no length cap. For genuinely tiny changes (one-function refactor, single config flag) it may be very short — a single signature with a docstring is a valid spec, but it still exists; writing it forces commitment to the surface. Write it to the file from the start, summarize contract decisions inline, iterate until settled.

## 6. Stress-test (senior review)

Before publishing, hand both `design.md` and `spec.md` to a `Plan` agent (software architect). Reviewer prompt: *"Review this design doc and its spec. Identify underspecified areas, weak motivation, missing tradeoffs, places where two reasonable engineers would implement different things, mismatches between the design's intent and the spec's contracts, and concrete suggestions for tightening the proposal."*

When the reviewer returns:

- **Incorporate obvious improvements directly into both files** — tightening prose, fixing a confused tradeoff, adding a missing file:line ref, sharpening a signature, moving an ambiguity into Open Questions.
- **Query the user only on ambiguous or load-bearing decisions** — real tradeoffs where you can't tell which way they want to go.

Show the user a brief summary: what you incorporated (design vs spec), what you're punting, what needs their call.

## 7. Publish

Two actions, can run together. After this, the skill is done.

1. **Commit and PR** via the `commit` skill. Branch `design/<slug>`. Use one
   commit to add `.agents/projects/<slug>/design.md`, `research.md`, and
   `spec.md`. Use an imperative PR title such as `[Design] Add finelog
   persistence`. State the proposed behavior and motivation in the PR body,
   then link all three files with absolute branch-rooted URLs
   (relative paths 404 from PR descriptions; see "Linking conventions"). Do not
   add a stock discussion footer; the design's Open Questions section identifies
   requested feedback. Follow
   `.agents/skills/writing-style/pull-requests.md` and
   `.agents/skills/writing-style/ai-writing-donts.md`. Labels `design` and
   `agent-generated`.

2. **Discord ping.** Run `python scripts/ops/discord.py --channel code-review` with a 2-line message: PR title + URL + the framing paragraph (or a one-sentence compression). Send it; no need to confirm exact text unless asked.

Once both happen, you're done. Feedback lives on the PR; the user starts implementation in parallel; the 1-pager is a snapshot, not a living doc.

---

# Linking conventions

- **Code citations inside the docs.** Use SHA-pinned permalinks for anything load-bearing: `https://github.com/marin-community/marin/blob/<sha>/path#Lnnn` (SHA from `git rev-parse main`). Plain `path:line` text drifts within days. Run a permalink pass at the end of Phase 4/5.
- **Sibling-file links from the PR body.** Use absolute branch-rooted URLs — `https://github.com/marin-community/marin/blob/design/<slug>/.agents/projects/<slug>/design.md`. Relative paths 404 from PR descriptions. Use the branch name (not a SHA) so the link follows future edits.

---

# Notes for the agent running this skill

- **The template and canonical worked example live in `.agents/projects/design-template.md`.** Read it before drafting. Don't use other docs in `.agents/projects/` as style references — they predate this skill.
- `design` label: create it if missing (`gh label create design --description "Design doc / 1-pager for review"`).
- If the user wants to skip a phase ("just write it, I know what I want"), honor that — but still produce the Open Questions section in `design.md`, still write `spec.md` (Phase 5), and still run the Stress-test (Phase 6).
- Implementation is out of scope. After Publish the skill is done — hand off to `fix-issue` or `commit` for the work itself only if the user asks.
