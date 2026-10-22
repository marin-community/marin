---
name: nightshift-review
description: Review one project area holistically and implement its highest-value supported improvement when assigned a nightshift cleanup run.
---

# Nightshift area review

Choose one bounded subsystem, understand how it works, and implement the
highest-value improvement supported by that review. A justified no-change
result is a successful run.

## Choose an area

Read the applicable `AGENTS.md` files and search prior work with
`consult-echo`. Inspect the last 20 PRs labeled `nightshift`, including closed
PRs and their feedback:

```bash
gh pr list --state all --label nightshift --limit 20 \
  --json number,title,state,createdAt,url,files
```

Choose one subsystem under `lib/marin/src/marin`, `lib/iris/src/iris`,
`lib/zephyr/src/zephyr`, or `lib/levanter/src/levanter`. State its boundaries
and why it deserves review: recurring friction, recent changes, a critical
boundary, or an area the recent runs have neglected. Prefer a different area
from recent runs unless new evidence justifies returning. Do not scan all
four libraries looking for the easiest patch.

## Review the system

Use the whole-system questions in
[review-pr-holistic](../review-pr-holistic/SKILL.md) as review lenses: fit,
better abstractions, consequences across boundaries, hidden tests, claims
versus evidence, related PRs, test value, and scope. This is an area audit of
existing behavior; the PR-specific head, delta, posting, and telemetry
procedures in that skill apply only to PR reviews.

Before proposing edits, trace the area's entry points, callers, configuration
defaults, state and error transitions, neighboring abstractions, and tests.
Read the recent history of the area. Explain the system in a short walkthrough
so the choice follows from its behavior. Check whether failures become valid
empty results, whether defaults violate caller assumptions, and whether
duplicated policy or unnecessary work imposes a concrete recurring cost.
Read `TESTING.md` and module testing guidance before assessing test coverage.

Record up to three supported candidates in the existing Loom session channel
or artifact. Do not fill a quota. Each candidate needs:

- Exact source paths and lines, plus a current caller or supported public
  contract that reaches the problem. A possible input with no demonstrated
  use or meaningful consequence is insufficient.
- The affected user or operator and the observable consequence. Distinguish
  measured cost from an estimate; reproduce suspected defects where feasible.
- A concrete replacement, reusing repository concepts, and a falsifiable
  regression test or measurement that can run safely for this change.
- Any overlapping open PR or issue, and why this work is still needed.

List open PRs with their files and filter by the candidate's paths;
`gh pr list --search` does not search changed paths. Read relevant issue and
PR discussions before selecting work. Defer candidates already covered by an
open PR, rejected without new evidence, or requiring a maintainer decision.

## Rank eligible work

Use these ordinal scores to make the tradeoff explicit. Explain each score
with evidence; the arithmetic is a comparison aid, not a measured return.

| Dimension | 1 | 2 | 3 |
| --- | --- | --- | --- |
| Impact | Local maintenance benefit | Repeated user or operator cost | Correctness failure, data loss, outage, or measured major cost |
| Confidence | Hypothesis | Complete source-to-caller trace | Reproducer or measured evidence |
| Effort | Small change and local validation | Several modules within one PR | Multiple PRs or hardware-dependent validation |
| Risk | Narrow local behavior | Shared interface or state | Live migration or distributed semantics |

A candidate is eligible only when Impact and Confidence are at least 2,
Effort and Risk are at most 2, and a safe, meaningful validation plan exists.
Style changes, trivial extraction, speculative performance claims, and
cleanups without a concrete recurring consequence do not qualify.

Rank eligible candidates by `Impact * Confidence / (Effort + Risk)`. Break
ties in favor of stronger evidence, then smaller scope and lower risk. Include
the scores, evidence, and rejection reasons in a candidate table. If none
qualifies, report what was reviewed and why no change is warranted. Do not
switch areas repeatedly to force a PR.

## Implement one improvement

Validate the winning candidate's premise before changing code. If a probe or
caller review disproves it, record that result, drop the candidate, and rank
the remaining candidates again. Do not preserve an initial choice merely
because it scored well. Implement only the highest-ranked eligible candidate
as one coherent PR. Leave other findings in the session record.

Follow the applicable implementation and testing skills. Compare the changed
behavior with the prior behavior using the promised regression or measurement;
passing existing tests alone does not prove the improvement. Keep reusable
guidance current with `update-docs` when needed. Follow `commit` to publish a
PR with `agent-generated` and `nightshift` labels; do not merge it yourself.
Do not file new issues for unselected candidates during this run.

Publish a typed Loom `result` containing the source revision, reviewed area,
walkthrough, candidate table, selection or no-change rationale, validation
results and limits, and PR URL when one was opened. Keep the full review in
the session; keep the PR description focused on its resulting behavior.
