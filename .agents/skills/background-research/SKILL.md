---
name: background-research
description: "Forage prior work before or during Marin research threads: search internal Marin artifacts and external literature/code; produce a cited brief with negative results and ranked experiment hypotheses."
---

# Skill: Background Research

Use this when a research, design, or experiment thread needs a compact prior-work
pass before choosing hypotheses, drafting a design, or launching runs.

## Effort

State the effort level at the top of the output. Stop when new sources no
longer change the ranked hypotheses; go longer only when the user asks or the
decision is expensive enough to justify it.

- `low` (3-7 min): current issue/logbook, obvious local refs, and a few external
  sources. Use for small follow-ups or when the user provides strong context.
- `medium` (10-15 min): default. Search internal Marin sources plus targeted
  external literature/code, include a contradiction pass, and produce a source
  ledger plus ranked next experiments.
- `high` (30-60 min): for expensive runs, architecture changes, data/eval
  decisions, or public claims. Use subagents for independent tracks when useful:
  internal Marin corpus, external literature/code, and W&B/report sources.
  Record query strings and rejected-source notes.

Effort changes breadth and provenance depth, not claim quality. Even `low`
effort must cite sources and distinguish evidence from speculation.

## Search Order

Search internal and external sources in parallel when useful, but do not skip
the internal pass. Prefer durable artifacts over transient conversation.

1. Current issue, PR, research logbook, or design file.
2. GitHub issues and PRs, especially experiment issues and linked comments.
3. `docs/reports/index.md`, `docs/reports/`, `docs/experiments/`, model cards.
4. Existing logbook files and `.agents/projects/*/research.md`.
5. Relevant code and experiment definitions under `experiments/`, `lib/`, and
   long-lived research branches or tags.
6. W&B reports/runs and Data Browser links surfaced from issues, reports, or
   logbooks.
7. External papers, blog posts, official docs, codebases, arXiv, OpenReview,
   Semantic Scholar, and cited references.

For external search, include at least one adversarial query family for
`medium`/`high`.

## Design-Doc Mode

When `write-design-doc` uses this skill, the output usually lands in
`.agents/projects/<slug>/research.md`. Keep it focused on design inputs, not
experiment execution.

Include:

- Relevant code paths. Use pinned GitHub links for claims tied to a fixed
  revision; use paths or section links for draft-local notes.
- Related designs in `.agents/projects/`, with overlap and differences.
- Related GitHub issues/PRs when known or discoverable.
- Existing utilities or abstractions the proposal might reuse.
- Prior-art shape when the proposal reinvents a category of system such as a
  logger, stats store, queue, scheduler, KV store, service-discovery layer, or
  workflow engine.
- What surprised you and what remains unclear.

For design-doc prior art, use `low` or `medium` effort by default. Skip
external prior art for narrow in-repo refactors, internal API tweaks, or designs
where the category is novel to this repo rather than novel to the world.

## Source Handling

- Keep raw sources as ground truth. The brief is derived and may be wrong.
- Prefer primary sources: papers, official docs, code, issues, W&B runs, reports.
- Record source version/date when it affects interpretation.
- Grade evidence by claim, not by source prestige. Directness to Marin's regime
  matters: model scale, hardware, data, objective, optimizer, context length,
  evaluation harness, and implementation constraints.
- Treat contradictions and negative results as first-class evidence.
- Record meaningful "not found" results when a search was expected to find
  something and did not.

## Output Contract

Write a compact brief in the research logbook, issue comment, or
`.agents/projects/<slug>/research.md`, depending on the parent workflow. If an
experiment issue exists, also provide a short issue-ready `Prior work` block.

```md
## Background Research Brief

- Effort:
- Stop rule:
- Date:

### Question

### Current Marin Context

### Internal Prior Work

### External Prior Art

### Negative / Failed Leads

### Evidence Map

#### Claim: <short claim>
- Support:
  - <source>: <one-line evidence>
- Contradictions:
  - <source>: <one-line caveat or failed result>
- Directness to Marin:
- Confidence:
- Action:

### Recommended Next Experiments

#### 1. <hypothesis>
- Minimum experiment:
- Baseline/control:
- Expected signal:
- Falsifier:
- Cost/risk:
- Sources:

### Hypothesis Queue Update
- Add:
- Revise:
- Falsify / stop:
- Promote:

### Source Ledger
| Source | Type | Location | Claim used for | Confidence | Notes |
|---|---|---|---|---|---|

### Handoff
- Suggested issue `Prior work` block:
- Suggested logbook entry:
- Open questions:
- Stop reason:
```

Use source types such as `paper`, `external code`, `official docs`, `GitHub
issue`, `PR`, `report`, `logbook`, `W&B`, `Data Browser`, and `Marin code`.

Use tables only for compact metadata. For claims, caveats, hypotheses, or any
cell likely to contain prose, use block-style cards so GitHub remains readable.

For research threads with a logbook, treat the hypothesis queue as a living
index derived from append-only entries. Link each queue change to the supporting
logbook entry, issue comment, W&B run, commit, tag, or pinned source.

## Hypothesis Quality

Recommended experiments should be actionable without re-reading the full source
set. Each candidate needs:

- A falsifiable hypothesis.
- The smallest experiment that could change the decision.
- Baseline or control.
- Primary metric and expected direction.
- Cost/risk.
- Source links.
- Confidence: `exploratory`, `replicated`, or `stable` when evidence supports
  those labels; otherwise say why confidence is weak.

## What To Skip

- Do not paste long paper summaries when a claim-level evidence table is enough.
- Do not use transient conversation as the durable record; file the synthesis in
  the issue/logbook/research file.
