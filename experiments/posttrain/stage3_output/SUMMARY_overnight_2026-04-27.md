# Overnight working session summary — 2026-04-27 07:20 UTC onwards

Continuously updated as experiments land. See `executable_specs_claude.md`
session log for full timestamps + chronological detail.

## Question being answered

Is the **spec-as-source-of-truth** thesis viable for the rubric writer
layer? Specifically:

1. Does spec-edit propagation work? (Experiment 1)
2. Does the edit-and-regen loop converge? (Experiment 2)
3. Does cross-judge edit sharing help? (Experiment 3)
4. Is loading cross-cutting always-on (Option B) a viable alternative
   to spec-edit-per-statement (Option A)? (Experiment 4)
5. Does the pipeline generalize beyond the seed-22 sample? (Experiment 6)

## Experiments table (live)

| ID | Name | Status | Cost (GPT-5.1) | Key finding |
|---|---|---|---:|---|
| Exp 1 | Self-edit propagation diff | DONE 07:38 | $0 | 19/29 (66%) strong propagation, 19/29 cited (just below 70% threshold) |
| Exp 2a | Round-2 agent review (4× Opus) | DONE 08:20 | ~$15 (Opus, not GPT) | 25 round-2 edits; **all 4 judges fabricate spec quotes** (mechanical audit primitive insufficient) |
| Exp 2b | Round-1 vs Round-2 distribution | DONE 08:30 | $0 | NOT converging in target coverage — R2 surfaces 8 NEW target statements |
| Exp 2c | Round-2 cumulative propagation | PENDING | TBD ~$0.73 | (queued after v3) |
| Exp 3 | Cross-judge union spec | PENDING | TBD ~$0.73 | (queued after v3) |
| Exp 4 | Option B (cross-cutting alwayson) vs A | partial DONE | ~$1 | v3 removes "I am programmed" boilerplate but loses warm-pivot. Option A wins on rich behaviors. |
| Exp 6 | Atlas-scale generalization | NOT STARTED | TBD ~$0.50 | (stretch) |

## Headline findings (so far)

### 1. Mechanical citation audit is fooled by fabrication

The rationale field's `spec_clauses_anchored_on` was supposed to make
rubrics auditable. My v1 verbatim audit (case-insensitive substring
across all spec text) reported 95-99% verbatim rates per judge. But
round-2 multi-agent review found that **all 4 judges fabricate quotes**
that look spec-congruent and partial-match the spec, fooling the audit.

**Implication**: the audit primitive needs strengthening. Require
(target_statement, exact substring) match — not "anywhere in any
statement." Multi-agent qualitative review is necessary; mechanical
metrics underestimate fabrication rates.

### 2. Spec-edit propagation works, but partially (66% strong)

Across all 29 round-1 edits, 19/29 produced a STRONG propagation signal
(citation + significant text change). 0 produced no signal. Per-judge:
flash 75%, glm51 67%, gpt51 62%, pro 57%. Pro's lower rate consistent
with its overall pattern (forced reasoning mode + terse rubrics resist
spec edits).

**Implication**: M5 thesis partially validated. Spec edits do move
rubrics, but not all the way. Some edits influence text without
verbatim citation (10 AMBIGUOUS classifications) — could be paraphrase
or coincidental change.

### 3. Edit-and-regen does NOT cleanly converge

Round-2 review of with-self-edits rubrics found 25 NEW pathologies
across 15 unique target statements. R1 hit 9 unique targets; R2 hit
15. **8 new targets surfaced that R1 didn't touch.**

Per-judge: pro had ZERO overlap between R1 and R2 targets — completely
shifted pathology distribution. gpt51 had FULL overlap — R1 didn't fix
its pathologies. Rest in between.

**Implication**: M5's loop won't terminate in 1-2 rounds. The 22 cross-tier
pairs have more pathology surface than 1 round of review can find. Each
round of edits exposes the next layer. Convergence rate is a real
empirical question — likely 3-5+ rounds on this sample size, possibly
more on the full atlas.

### 4. Option A (per-statement examples) > Option B (cross-cutting always-on)

V3 architecture (loads 4 cross-cutting statements into every writer
call) successfully removes the "I am programmed" boilerplate pathology.
But it does NOT reproduce the warm-pivot positive behaviors that
with-self-edits achieves (e.g., "I'd be happy to help you write a
general post for a broad audience"). Cross-cutting style rules rule
things OUT effectively but don't supply positive examples.

**Implication**: User's choice of Option A is empirically right. A
hybrid (Option A + Option B) might be best of both — cross-cutting
to enforce style, per-statement examples for positive behaviors.

## Cost tracking

| category | spend |
|---|---:|
| GPT-5.1 pre-overnight | $0.80 |
| GPT-5.1 self-edit propagation | $0.40 |
| GPT-5.1 v3 alwayson | ~$0.50 |
| **GPT-5.1 cumulative** | **~$1.70** (target <$100, limit <$200) |
| Opus round-1 agents | ~$15 |
| Opus round-2 agents | ~$15 |
| **Opus cumulative** | **~$30** |
| Other (Flash, Pro, GLM-5.1) | basically free |

GPT-5.1 spend at 1.7% of target. Plenty of headroom for atlas-scale
generalization, more iteration rounds, etc.

## Pending work (overnight queue)

1. **Round-2 cumulative propagation** — apply r1+r2 edits, regen. Tests
   whether more edits actually fix more pathologies.
2. **Union-spec cross-judge runs** — does giving each judge ALL 29
   round-1 edits help vs just its own?
3. **Comparison of round-2 propagation** — does the next round of edits
   show similar 66% strong propagation rate? Does it stack with round-1?
4. **Atlas-scale generalization** — does the pipeline produce good
   rubrics on cross-tier pairs not in the seed-22 sample?
5. **Synthesis report + design doc updates** — fold all findings into
   `executable_specifications.md` for the M5 design.

## What's NOT being done tonight (and why)

- **M3 retraining** — out of scope, requires TPU compute and a clean
  spec the team agrees on. The user wants the rubric layer validated
  first.
- **M4 / M5 / M6 milestones** — design work, not experiments. Will be
  informed by tonight's findings.
- **LM compiler stub (M5 dependency)** — could fit if time allows; lower
  priority since tonight's findings inform what the compiler should
  produce.
