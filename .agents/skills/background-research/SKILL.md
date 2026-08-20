---
name: background-research
description: "Forage prior work before or during Marin research threads: search internal Marin artifacts and external literature/code; produce a cited brief with negative results and ranked experiment hypotheses."
---

# Background Research

Use this for a compact prior-work pass before selecting hypotheses, designing a
system, or launching an expensive experiment. State `low`, `medium` (default),
or `high` effort and stop when new sources no longer change the ranked
hypotheses. All effort levels cite sources and separate evidence from
speculation; high effort may use independent internal, external, and W&B tracks.
Use approximate time caps of 3–7 minutes for low, 10–15 minutes for medium, and
30–60 minutes for high; exceed them only when the user asks or the decision's
cost justifies it.

## Search

Search in this order, using durable sources before transient conversation:

1. Current issue/PR, logbook, or design file.
2. GitHub issues and PRs.
3. `docs/reports/`, `docs/experiments/`, model cards, and existing logbooks.
4. Relevant code under `experiments/` and `lib/`.
5. W&B reports/runs linked from Marin artifacts.
6. Primary external papers, official docs, and code.

For medium/high effort, include an adversarial query family and record useful
“not found” or rejected-source results. Use `consult-echo` for internal Marin
activity when prior decisions or incidents matter.

## Evidence and hypotheses

Treat raw sources as ground truth. Record source version/date when material,
grade evidence by claim and directness to Marin's model, hardware, data,
objective, and implementation regime, and preserve contradictions/negative
results. Each recommended experiment needs a falsifiable hypothesis, smallest
decision-changing test, baseline/control, primary metric and direction,
falsifier, cost/risk, source links, and confidence (`exploratory`, `replicated`,
or `stable`, or an explanation for weaker confidence).

## Output

Write the synthesis in the parent workflow's logbook, issue, or
`.agents/projects/<slug>/research.md`; do not leave it only in conversation.
Use this compact structure:

```md
## Background Research Brief
- Effort: | Stop rule: | Date:
### Question
### Current Marin Context
### Internal Prior Work
### External Prior Art
### Negative / Failed Leads
### Evidence Map
### Recommended Next Experiments
### Hypothesis Queue Update
### Source Ledger
### Handoff
```

For design-doc mode, also record relevant code paths, related Marin designs,
reusable abstractions, and prior-art shape; skip external research for narrow
in-repo refactors unless requested. Keep claims/caveats in prose blocks and use
tables only for compact metadata. Link the brief's queue updates to evidence.
