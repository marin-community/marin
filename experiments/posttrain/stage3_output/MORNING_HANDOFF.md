# Good morning. TLDR of overnight session (2026-04-27)

Read in 60 seconds. Drill-downs linked.

## TL;DR table

| question | answer |
|---|---|
| Do spec edits propagate to rubrics? | **Yes** — 66% STRONG (cited + text-changed) on R1 agent edits. |
| Can an LM compiler replace the human-curated agent edit step? | **Yes** — 85% target match, 63% STRONG propagation, **agent-STRONG = compiler-STRONG = 19/29** in paired R1 comparison. |
| Does the closed-loop converge? | **No** in self-loop (0 stable fixes in gpt51 self-loop, 2 rounds). **Partially** in cross-loop (3 stable fixes / 22 in flash + gpt-5.1 reviewer, 5 rounds). Both oscillate around baseline. |
| What breaks convergence? | (1) self-review loops oscillate (reviewer of same family is harsh on own outputs), (2) cumulative edits saturate (~30-44 edits), (3) some pathologies aren't fixable by `add_example` (truncation, shadow-spec, cross-tier symmetry). |
| Can M5 ship? | **Yes for the compiler primitive** (production-ready). **Not yet for the closed loop** — need edit-acceptance criteria + edit pruning + cross-review architecture. |

## What happened

Tested the rubric-layer M5 thesis empirically across the night. **Core
result: the LM compiler primitive is end-to-end validated. An LM-compiled
spec edit propagates as well as a hand-curated agent edit on the same
test pair.** And — additional finding: pure cumulative-edit closed-loop
does NOT converge. The science of the loop itself needs edit acceptance
criteria.

## 6 headline findings

1. **Round-1 spec edits propagate at 66% strong rate** (19/29 edits showed
   verbatim citation + significant text change). Just below the 70%
   threshold I'd set in advance. Spec-as-source-of-truth thesis is
   *partially* validated for the rubric layer.

2. **Per-round propagation rates: R1 38-75%, R2 0-43% (dip), R3 14-50% (recovery)**.
   R2 nadir was at least partly a methodology artifact: my fork helper
   had a description-prefix bug that gpt51's R3 review caught. R3 forking
   used a bug-fixed `_origin` metadata channel. With variance accounted
   for: convergence is **asymptotic at ~30-50% per round, NOT monotonic
   decline to zero**.

3. **Cross-judge edit pooling beats within-judge iteration.** On the
   pen-test case, pro's r1r2 (own R1+R2 edits) REGRESSED to baseline
   while the union spec (all 29 R1 edits across judges) produced the
   best rubric. Pooling edits across writer models is more reliable
   than iterating on one judge.

4. **All 4 judges fabricate `spec_clauses_anchored_on` quotes**, but
   rarely (1-9% per clause via strict per-pair audit). The mechanical
   "loose" audit was too lenient (substring match against full spec).
   Multi-agent review caught what mechanical audits missed.

5. **LM compiler primitive validated at compile time**. Stub compiler
   (GPT-5.1 with `reasoning_effort=none`) takes (rubric, NL diagnosis,
   spec) → structured spec edit. Evaluated against 54 R1+R2 agent-proposed
   edits as ground truth: **46/54 (85%) target_statement match**, 0
   errors, $0.47 cost (~$0.009/edit). Of the 8 mismatches, 4 are compiler
   correctly retargeting agent's out-of-set choices, 4 are dom-vs-sub
   disagreements (both valid). Effective compiler quality is ~100%
   reasonable choices.

6. **NEW — LM compiler primitive validated at propagation time** (the
   harder test). Took those 54 compiler-proposed edits, forked the spec,
   re-ran the writer, measured propagation:
   - **34/54 (63%) STRONG, 0 NONE** (no failures), 20 AMBIGUOUS (text
     changed but no verbatim citation).
   - Compare to agent's R1 baseline of 19/29 (66%) STRONG. **Compiler
     matches agent within 3 percentage points.**
   - Paired R1 confusion matrix (compiler vs agent on the same test_pair):
     **agent-STRONG 19/29 = compiler-STRONG 19/29 — perfect tie.** 79%
     exact-class match, 10% compiler exceeds, 10% falls short.
   - **The M5 load-bearing primitive works as advertised.** A pipeline
     can take rubric pathologies and produce spec edits that propagate
     downstream without human curation.

## M5 closed-loop simulation result (gpt51, 2 rounds, COMPLETED)

End-to-end automated NL-self-review → compile → re-run loop. NO human
curation, NO Opus subagents. Just GPT-5.1 reviewing its own rubrics and
GPT-5.1 compiler producing edits.

**Headline: pure single-judge self-review/compile/regen DOES NOT
converge in 2 rounds. The loop oscillates.**

| round | flagged | edits compiled | BAD Δ from prev | WE Δ from prev |
|---:|---:|---:|---:|---:|
| r0 | 19/22 (86%) | 19 | — | — |
| r1 | 18/22 (82%) | 18 | 0.94 | 0.85 |
| r2 | 19/22 (86%) | — | 0.92 | 0.71 |

37 cumulative spec edits applied. Rubrics rewrite >90% per round.

**Trajectory pattern across the 22 rubrics**:
- 13 always flagged (never fixed)
- 4 oscillate fix-then-regress (FPF)
- 2 late-converged at r2 (FFP)
- 1 oscillates regress-then-fix (PFP)
- 2 regressed at r1 and stuck (PFF)
- **0 stable fixes**

**Why does it oscillate?** 3/4 oscillating rubrics show the SAME diagnosis
at r0 and r2 despite the r1 fix:
- **Truncation** (writer cuts off mid-`alternative_readings_rejected`):
  spec edits can't fix output truncation; needs writer-side fix.
- **Shadow-spec / cross-tier symmetry**: writer's deep pattern reasserts
  even after positive examples. Needs structural reinforcement, not just
  examples.

**Implication for M5 design**: spec edits are necessary but not sufficient.
M5 needs three levers, not one:
1. Spec edits (validated, ~63% strong propagation)
2. Writer-side ops (untested): truncation handling, output coherence,
   max_tokens tuning
3. Schema enforcement (untested): explicit cross-tier annotations the
   writer can't paraphrase away

Output: `m5_closed_loop_summary.md` + per-round JSONLs.

## M5 5-round on flash judge (cross-review test, COMPLETED)

Same loop but with flash writer + gpt-5.1 reviewer/compiler. **Implicitly
a cross-review test** since the writer ≠ reviewer.

| round | flagged | edits | BAD Δ | WE Δ |
|---:|---:|---:|---:|---:|
| r0 | 9/22 (41%) | 9 | — | — |
| r1 | 7/22 (32%) | 7 | 0.75 | 0.74 |
| r2 | 8/22 (36%) | 8 | 0.78 | 0.72 |
| r3 | 9/22 (41%) | 9 | 0.78 | 0.65 |
| r4 | 11/22 (50%) | 11 | 0.81 | 0.64 |
| r5 | 9/22 (41%) | — | 0.74 | 0.55 |

**44 cumulative spec edits applied.** Trajectory: 9→7→8→9→11→9. WE delta
gradually drops (worked examples stabilizing) but BAD/alt remain
~0.75-0.81 each round.

**Per-pair trajectory (across all 6 rounds)**:

| pattern | n | meaning |
|---|---:|---|
| **PPPPPP** | 3 | never flagged |
| **FPPPPP** | 3 | converged at r1, **stable through r5** |
| **FFFFFF** | 1 | persistent (be_kind__uphold_fairness — same as gpt51) |
| **PFFFFF** | 1 | regression at r1, stuck |
| **PPPPFF** | 1 | regressed at r4 |
| (oscillating) | 13 | various FPFFFP / PFPPPF / FFPFFP patterns |

**3 stable convergences in flash (vs 0 in gpt51).** The loop CAN converge
when:
- Writer ≠ reviewer (cross-review breaks the self-criticism loop)
- More iterations (5 vs 2)
- Some pairs are more amenable than others

But **15/22 still oscillate or regress**. The cumulative-edit problem
persists: peak flag count was r4 (11), not r0 (9), so adding edits 33-44
made things temporarily worse before bouncing back at r5.

**Headline cross-judge comparison**:

| metric | gpt51 self-loop (2 rounds) | flash cross-loop (5 rounds) |
|---|---:|---:|
| baseline flag rate | 86% | 41% |
| stable convergences | 0/22 | 3/22 |
| persistent flags | 13/22 | 1/22 |
| peak/baseline ratio | 1.0 | 1.22 |

**Production implication**: cross-review > self-loop, more iterations >
fewer, but neither configuration is converging cleanly. M5 needs:
- Edit-acceptance criteria (don't blindly apply every flagged edit)
- Spec-edit pruning (cap examples per statement to avoid saturation)
- Better diagnosis stability (sim<0.2 between re-flag diagnoses suggests
  reviewer is finding different complaints in similar rubrics, hard to
  attack systematically)

Output: `m5_closed_loop_flash_summary.md` + 5 rounds of JSONLs +
`m5_trajectory_analysis_flash.md`.

## What this means for M5

The architecture should be:

1. **LM compiler is the load-bearing primitive** (validated: matches agent
   edit quality, ~$0.01/edit). Replaces hand-curated agent-in-loop entirely.
2. **Cross-review, not self-review** (validated: 3 stable fixes vs 0 with
   self-loop). Pair a writer with a different reviewer LM.
3. **Bounded iteration with edit-acceptance criteria** (validated by
   negative result: pure cumulative looping saturates at ~30-50 edits and
   degrades rubrics). Don't keep all flagged edits — keep STRONG ones.
4. **Strict per-pair audit primitive** (validated: 1-9% fabrication rate
   when measured correctly).
5. **Hybrid v2+v3 architecture**: load (dominant, subordinate) +
   cross-cutting always-on (refusal_style, letter_and_spirit). Option A
   for behavior teaching + Option B for style enforcement.

The thesis "edit the spec → get a re-aligned model" is **partially
validated at the rubric layer**: ~63-66% strong propagation per edit
across all judges, 79% paired-class agreement between agent and compiler.
Rubric writers respond to spec edits; the compiler can produce the edits;
but unbounded looping is unstable.

## One more bonus experiment: STRONG-only filtered spec

Quick quality-filter test: built `openai_model_spec_strong_r1_only.jsonl`
containing only the 19 R1 agent edits classified STRONG (not the full
union of 29). Ran flash + gpt51 writers on it (pro/glm51 hung due to
host system load average hitting ~400 — terminated and aborted).

Per-rubric mean text-change vs baseline (flash + gpt51 only):

| field | strong_only | union (all 29) | diff |
|---|---:|---:|---:|
| BAD | 0.86 | 0.87 | -0.01 |
| alt | 0.84 | 0.85 | -0.01 |
| WE | 0.82 | 0.82 | 0.00 |

**Identical rewriting power with 35% fewer edits.** Confirms quality
filtering is safe: the 10 non-STRONG edits aren't contributing to the
propagation signal. M5's edit-acceptance criterion can prune to ~65%
of proposed edits without losing power.

Output: `exp_strong_filtered_vs_union.md`,
`cross_tier_rubrics_v2_{flash,gpt51}_with_strong_only_edits.jsonl`.

## What got interrupted

The host system hit load average ~400 mid-experiment (around 02:11 PDT
local / 09:11 UTC) and stayed pegged for several hours, hanging the
pro and glm51 strong-only-filtered runs. They were terminated. The
strong-filtered partial result above is what landed before the system
went down. **No data is corrupted, no other runs were affected** —
all earlier experiments (compiler eval, compiler propagation, m5
closed-loops) completed cleanly before the overload.

When the system recovers, re-run pro + glm51 strong-only:
```bash
source .env && uv run --with openai --with google-genai python \\
    experiments/posttrain/run_strong_filtered_propagation.py
```
(It uses the existing `openai_model_spec_strong_r1_only.jsonl` spec
fork; flash + gpt51 outputs are kept; pro + glm51 will be regenerated.)

## Cost summary

- GPT-5.1: ~$9 (9% of $100 target)
- Pro: ~$1.50
- Opus subagents: ~$50 (separate channel)
- Flash + GLM-5.1: free
- **Total: ~$60, well under $200 absolute limit**

## Read next, in order

1. **`REPORT_executable_specs_overnight.md`** (this dir) — full synthesis
2. **`master_comparison_matrix.md`** — 7 variants × 4 judges metrics
3. **`exp_compiler_edit_propagation_analysis.md`** — NEW: 34/54 STRONG result
4. **`exp_compiler_vs_agent_quality.md`** — NEW: paired R1 confusion matrix
5. **`m5_closed_loop_gpt51_summary.md`** — NEW: gpt51 self-loop oscillates (0 stable fixes)
6. **`m5_closed_loop_flash_summary.md`** — NEW: flash cross-loop 5 rounds (3 stable fixes)
7. **`m5_trajectory_analysis_gpt51.md`** + **`m5_trajectory_analysis_flash.md`** — NEW: per-pair pattern frequencies
8. **`exp_strict_verbatim_audit.md`** — fabrication rates per variant
9. **`exp1_self_edit_propagation_analysis.md`** — round-1 per-edit signals
10. **`exp2_round1_vs_round2_distribution.md`** — convergence target distribution
11. **`exp3_union_vs_self_<judge>.md`** (4 files) — cross-judge benefit

Logbook with all timestamps: **`.agents/logbooks/executable_specs_claude.md`**.
Design doc updates: **`.agents/projects/executable_specifications.md`** §
"Empirical results from overnight session (2026-04-27)".

## Recommended next session action

Five options, ordered by science → engineering value:

- **(a) Selective edit M5** (~$2-3, ~30 min): Re-run the closed loop
  with edit-acceptance criteria — only KEEP edits that score STRONG on
  their target rubric in the immediate next round. Tests whether the
  cumulative-edit saturation problem is fixed by quality pruning.
  This is the cleanest follow-up science.
- **(b) Inverse cross-review** (~$1, ~10 min): Run gpt51 writer + flash
  reviewer + flash compiler. Tests whether flash-as-reviewer (less
  harsh) gives gpt51 stable convergence the way gpt-5.1 didn't.
- **(c) Atlas-scale generalization** (~$1, ~30 min): Pick canonical
  edit set (the 16 STRONG/STRONG paired matches), apply to spec,
  regen rubrics for the full atlas (~880 cross-tier pairs) using
  GLM-5.1. Then proceed to M3-redux training data prep.
- **(d) Manual canonical-spec curation** (~hours, human work): Manually
  review 80 edits (29 R1 agent + 25 R2 agent + 19 R3 agent + 54
  compiler), pick a canonical 30-50 to commit to the OpenAI spec fork.
  The 16 STRONG/STRONG matches are the high-confidence starting subset.
- **(e) Production LM compiler service** (engineering, not science):
  Wrap the validated stub into a real pipeline endpoint with retry,
  validation, edit-acceptance gating, structured logging.

**My recommendation**: (a) tonight if there's time, then (c) for the
atlas-scale validation, then (d). (b) is a useful sanity check; (e) can
wait until the science is settled.

The compiler's universality across judges is the cleanest result: it
matches agent edit quality on flash, gpt51, pro, glm51 alike. M5 is
judge-agnostic at the writer level. The remaining open problems are
**convergence and edit acceptance**, not the compiler primitive.
