---
name: design-doc
description: Walk a user through producing a 1-page design doc, opening a PR with it, and pinging Discord for review. Use when asked to write a design doc, spec, or technical proposal.
---

# Skill: Design Doc Workflow

## Purpose

A design doc in Marin is a ~1-page document posted as a PR for early feedback. The goal is to surface design issues *before* implementation, not to gate work — area owners are expected to LGTM or comment quickly, and the author begins implementation in parallel. See [issue #5210](https://github.com/marin-community/marin/issues/5210) for the rationale.

This skill is the workflow for producing one. It is **interactive** — you ask the user questions when you genuinely don't know, but you make reasonable inferences and proceed when you can.

The template lives at `.agents/projects/design-template.md`. New docs go to a slug-named directory: `.agents/projects/<slug>/` (slug only, no date prefix — `git log` already records the date). Inside that directory:

- `design.md` — the 1-pager (always)
- `research.md` — in-repo refs, prior art, Q&A summary (always — even a short one)
- `spec.md` — concrete contracts: public function/class signatures, file paths, persisted shapes, error types, proto definitions (always — written *after* the design has stabilised so the contracts reflect the final decision)

`spec.md` is the contract layer, not an implementation plan. The full `.proto`, the public class signatures with types, the schema-registry table CREATE, the directory layout, the file paths where each piece will live — yes. Algorithm pseudocode, sequenced steps, file-by-file plans, "how" rather than "what" — no, those belong in the PR description or get deleted in favor of writing the code.

Even an "internal" refactor has a public surface — a function someone else will import, a flag someone else will pass, a config key someone else will set. The spec pins that surface explicitly so reviewers know what they're agreeing to. If you can't write a spec, you don't have a design yet.

## When to use this skill

- A task will likely take more than a day, or is load-bearing for other work.
- A change crosses subproject boundaries (e.g. iris ↔ levanter, marin ↔ zephyr).
- A change introduces a new service, package, or persistent data shape.

If none of those apply, just open the PR — don't manufacture a design doc for a 50-line bug fix.

---

# Workflow

Six phases. Confirm with the user at natural decision points (after Research, after Draft, after Spec, before Publish), but don't ask permission to move forward when the next step is obvious.

## 1. Frame

The user starts the skill with a framing paragraph stating what they want to do and why. If they didn't, query them for it — or infer it from prior conversation context if that context is rich enough. A one-sentence "fix the foo thing" is *not* enough to proceed; push back and ask for the why.

**You infer the slug.** Short, lowercase, underscores (`finelog_lift`, `iris_autoscaler_refactor`). State it in one line ("I'll save this as `.agents/projects/<slug>/`") and proceed — only stop if it collides with an existing directory, in which case propose a disambiguator.

Once you have a framing paragraph, proceed directly to research. Don't batch a long list of questions up front; that's what Interrogate is for, after research has shown you what to ask.

## 2. Research

Spawn an `Explore` subagent (do not search yourself — the digest stays out of main context). Brief it with the framing paragraph and ask for:

- Relevant files with line numbers (the design doc must reference real code, not placeholders).
- Related designs already in `.agents/projects/` — read them, note overlap.
- Related GitHub issues/PRs via `gh` if the user named any, plus a quick `gh issue list --search` on the topic.
- Existing utilities or abstractions the proposal might reuse (per `AGENTS.md` "Code Reuse").

**For proposals that reinvent a category of system** (a logger, a stats store, a queue, a scheduler, a KV, a service-discovery layer, a workflow engine, etc.), also do a **prior-art pass via web search** — in parallel with the in-repo Explore, since the two are independent. Spawn a `general-purpose` agent (it has WebSearch/WebFetch) with a focused brief: *what does the established shape of this kind of system look like, what are 2–4 representative implementations (OSS or well-known), and what design choices do they converge or disagree on?* Cap to ~5 results and ask for a short bulleted digest (under 200 words). The point is to surface obvious patterns we'd otherwise reinvent badly, and to give the design doc one or two reference points reviewers can compare against — not to write a literature review. Skip this pass for in-repo refactors, internal API tweaks, or anything where the category is novel to the user, not novel in the world.

Return to the user with a bulleted digest combining both passes: *"Here's what I found in-repo, here's what the prior art looks like, here's what surprised me, here's what's still unclear."* Ask whether the framing should shift before drafting.

**Persist the research as a sibling artifact.** Save the full digest (in-repo findings with file:line refs, prior-art digest, anything load-bearing that won't fit in the 1-pager) to `.agents/projects/<slug>/research.md`. The design doc itself gets a short `## Background` section — 3–5 sentences summarizing what we found and how it shapes the proposal — with a link to `research.md` for reviewers who want depth. Keep the design doc on-screen; let the research doc grow.

## 3. Interrogate

Ask 3–6 targeted questions in one batched message. Good questions surface things the doc *must* answer:

- Scope boundaries — what is explicitly **not** in this design?
- Testing — what's the smallest test that would catch a regression?
- Tradeoff decisions — when there are two reasonable approaches, which one and why?
- Unknowns the user wants reviewers to weigh in on (these become Open Questions).

Bad questions are ones research could have answered. Don't ask things you could grep for. **Don't ask about backwards compatibility unless you have specific reason to think it matters here** — Marin generally does not optimize for backwards compatibility (per `AGENTS.md` "Deprecation"), so default to assuming the proposal updates all call sites.

## 4. Draft

Read `.agents/projects/design-template.md`, fill in each section. Guidelines:

- **~1000 words is the target, not a hard limit.** Concision is a virtue; spend words where they buy clarity. If you're at 1200, look for cuts. If 800 is genuinely tighter than 600, ship 800. The goal is "good design" not "short doc" — but a doc that's mostly load-bearing detail is better than one that buries the decision in throat-clearing.
- Reference real `file.py:line` paths from research, not placeholders.
- One small code snippet (10-30 lines) only if prose is genuinely worse. Default to no snippet.
- Open Questions section is non-empty — if the design has no unknowns, ask the user what they want feedback on.
- Don't add a backwards-compat section by default. Mention compat only if the change genuinely needs migration (persisted data, public API consumed externally, etc.).

Show the draft inline, accept edits in conversation. Iterate until the user says the design is settled. **Do not write `spec.md` yet** — it derives from the stable design, and writing it before the design has converged means rewriting it.

## 5. Spec

Once `design.md` is settled, write `.agents/projects/<slug>/spec.md` — the contract layer the design implies. This phase is non-negotiable; every design produces a spec. If you find yourself unable to write one, the design isn't actually concrete enough and you should go back to Draft.

What goes in `spec.md`:

- **Public API**: full Python class/function signatures with parameter types, return types, and a one-paragraph contract docstring per symbol (what it does, edge cases, ordering guarantees). Not implementation bodies.
- **Proto definitions**: the full `.proto` file content (or close to it — name every RPC, message, and field).
- **File paths**: where each new piece lives (e.g. "proto at `lib/finelog/src/finelog/proto/stats.proto`, client at `lib/finelog/src/finelog/client/stats.py`"). A summary table at the bottom of the spec is good.
- **Persisted shapes**: schema-registry table `CREATE` statements, on-disk directory layout, file naming conventions, JSON / proto envelope formats.
- **Errors**: every new error type with the exact condition that triggers it, plus any behavioural changes in existing error paths.
- **Out of scope**: an explicit list of related changes the design *doesn't* commit to (e.g. "deletion of `executor_main` deferred to follow-up PR"). Reviewers use this to know what to *not* push back on.

What does **not** go in `spec.md`: algorithm pseudocode, sequenced implementation steps, file-by-file rollout plans, "how" rather than "what." Those belong in the PR description for the implementation, or get deleted in favor of just writing the code.

`spec.md` has no length cap — it should be exactly as long as the contracts. Reviewers should be able to read `design.md` for the why, then `spec.md` to check "would I actually build this exact API?"

For genuinely tiny changes — a one-function refactor, a single config flag — `spec.md` may be very short (a single function signature with a docstring is a valid spec). It still exists; the act of writing it forces you to commit to the surface.

Show the spec inline, accept edits, iterate until the user says it's settled.

## 6. Stress-test (senior review)

Before publishing, hand both `design.md` and `spec.md` to a senior reviewer subagent (`Plan` agent — software architect). Reviewer prompt: *"Review this design doc and its spec. Identify underspecified areas, weak motivation, missing tradeoffs, places where two reasonable engineers would implement different things, mismatches between the design's intent and the spec's contracts, and concrete suggestions for tightening the proposal."*

When the reviewer returns:

- **Incorporate obvious improvements directly into both files.** Tightening prose, fixing a confused tradeoff, adding a missing file:line reference, sharpening a function signature in the spec, moving a clear ambiguity into Open Questions — just do it.
- **Query the user only on ambiguous or load-bearing decisions** — places where the reviewer surfaced a real tradeoff and you can't tell which way the user wants to go.

Show the user a brief summary: what you incorporated (in design vs spec), what you're punting on, what needs their call.

## 7. Publish

Two actions, can run together. After this, the skill is done.

1. **Commit and PR** via the `pull-request` skill. Branch `design/<slug>`. Single commit adding the `.agents/projects/<slug>/` directory (design.md, research.md, and spec.md — all three are always present). PR title `[Design] <slug>`. PR body is `design.md` itself plus a one-line "Discussion welcome — see Open Questions." footer, with links to the sibling `research.md` and `spec.md` files. Labels `design` and `agent-generated`.

2. **Discord ping.** Run `python scripts/ops/discord.py --channel code-review` with a 2-line message: PR title + URL + the framing paragraph (or a one-sentence compression of it). Send it; no need to confirm exact text unless the user asked.

Once both have happened, you're done. Feedback lives on the PR; the user can start implementation in parallel; the 1-pager is a snapshot, not a living doc.

---

# Notes for the agent running this skill

- **The template and canonical worked example live in `.agents/projects/design-template.md`.** Read it before drafting. Don't use other docs in `.agents/projects/` as style references — they predate this skill and are inconsistent.
- `agent-generated` and `design` labels: create the `design` label if it doesn't exist (`gh label create design --description "Design doc / 1-pager for review"`).
- If the user wants to skip a phase ("just write it, I know what I want"), honor that — but still produce the Open Questions section in `design.md`, still write `spec.md` (Phase 5), and still run the Stress-test (Phase 6). Those are the cheapest, highest-value steps and the spec is what makes review meaningful.
- Implementation is out of scope. After Publish, the skill is done — hand off to `fix-issue` or `pull-request` for the work itself only if the user asks.
