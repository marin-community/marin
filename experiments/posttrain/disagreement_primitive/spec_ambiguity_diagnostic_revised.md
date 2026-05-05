# Spec ambiguity diagnostic — Tier 1 + 2 (revised after robustness validation)

**Revised 2026-05-01 22:15 UTC** following P1+P2+P3 validation pass. Original headline of '41 robust' is downgraded; Method C's signal turns out to measure 'frontier-LLM-prior convergence' rather than language precision, and has ~3-point run-to-run variance. The load-bearing actionable findings — 4 internally_inconsistent statements — survived the validation.

## Validation summary

| signal | what survived validation |
|---|---|
| Method D (text-vs-examples) | ✅ 4 flags real beyond ~1.7pt artifact baseline; corrected threshold <6 |
| Method C (multi-compiler) | ⚠️ Reframed: measures compiler-prior convergence, not language precision; ≥3pt run-to-run variance |
| Method B (judge enumeration) | ❌ Confirmed enumeration-biased twice (synthetic baseline + uncorrelated with C, Pearson=0.139) |
| Method A (bare-statement scores) | Weak; not useful as primary signal |

## Headline result — what we are confident in

**4 statements have meaningful prose-vs-examples internal inconsistency** (Method D < 6 after artifact correction). These are real spec-author repair candidates:

| statement | D (overall equiv) | core finding |
|---|---:|---|
| `avoid_abuse` | 6/10 | Rubric E explicitly allows and even expects participation in user-consented, non‑protected self-roasting and light teasing, and treats blanket refusal of such teasing as BAD, whereas Rubric T is more  |
| `be_engaging` | 4/10 | Rubric T emphasizes adapting style to the user’s goal, sometimes being concise and minimal for task-focused queries, while Rubric E consistently pushes for warm, conversational engagement and invitati |
| `letter_and_spirit` | 6/10 | Rubric T emphasizes inferring user goals, handling ambiguity, and avoiding real‑world autonomous actions or harmful compliance, while Rubric E focuses on respecting higher‑priority/earlier instruction |
| `refusal_style` | 4/10 | Text-based rubric T emphasizes ultra-brief, single-sentence refusals with a required apology and bans most explanation, while example-based rubric E allows neutral, possibly longer policy-based explan |

## Status of the 'language_robust' label

After robustness validation, this label means: **compilers from the same training paradigm (frontier LLMs) converge on the same operationalization of this language.** It does NOT mean: the language is operationally precise. A spec author should read this as 'frontier-LLM behavior will be consistent on this statement', not 'the language pins down a unique reading'.

42 of 46 statements meet this criterion (Method C ≥ 7 mean pairwise equivalence). However, Method C's run-to-run variance is ≥3 points on at least one statement (`be_rationally_optimistic` swung 6.67→10.0 across reruns), so individual close-to-threshold statements should be reverified before action.

## Reliability caveats per signal

- **Method D (the load-bearing signal)**: validated against intra-channel baseline. The 4 flagged statements drop 2-4 points BELOW the ~1.7-point cross-channel artifact offset. Reliable for sign-of-a-defect detection. Cannot detect inconsistency on the 11 statements with <2 examples (which were skipped).
- **Method C**: tested via P3 (4-compiler panel including Qwen). Adding Qwen drops mean equivalence by ~0.7-1 point but doesn't separate synthetic clear vs ambiguous. Run-to-run variance ≥3 points on borderline cases. **Single-run flags should be reverified**.
- **Method B**: not used in the synthesis. Enumeration-biased and uncorrelated with C across 46 statements (Pearson 0.139).
- **Method A**: not used in the synthesis. Score variance weak in our setup; might be salvageable with score-stratified analysis (P4 deferred).

## Statements not evaluated for internal consistency (Method D skipped)

11 statements have fewer than 2 examples and were skipped by Method D. They are:
- `avoid_being_condescending` (1 examples)
- `avoid_overstepping` (1 examples)
- `be_rationally_optimistic` (1 examples)
- `be_thorough_but_efficient` (1 examples)
- `comply_with_laws` (0 examples)
- `do_not_encourage_self_harm` (1 examples)
- `formatting` (1 examples)
- `no_agenda` (0 examples)
- `respect_creators` (1 examples)
- `sexual_content_involving_minors` (1 examples)
- `uphold_fairness` (1 examples)

Without Method D, we have only Method C signal for these — which we now know is unreliable single-shot. Recommend running Method C with K=3 reruns on these statements + spot-check rubric inspection before acting.

## Cross-validation against pair-level findings

11 scenarios from the existing Phase 4 pair-level data are flagged with `compliance_ambiguity` / `activation_ambiguity` / `inherent_subtlety` despite **both constituent statements** being `language_robust` at the spec level. This is *consistent* with the layer separation: per-statement language can be robust while cross-statement tension emerges from interactions. Examples:
- `be_clear × be_creative` (3 variants) — both robust at spec level; pair flagged inherent_subtlety
- `do_not_facilitate_illicit_behavior × transformation_exception` — both robust; pair flagged compliance_ambiguity
- `ask_clarifying_questions × avoid_overstepping` — both robust; pair flagged compliance_ambiguity

Per-statement diagnostic (this epic) and per-pair diagnostic (existing Phase 4) are correctly attributing issues to different layers.

## What spec author should review (the actionable list)

**Priority 1** — 4 statements with internal prose-vs-examples inconsistency (validated): `avoid_abuse`, `be_engaging`, `letter_and_spirit`, `refusal_style`.

**Priority 2** — 11 statements not assessable via Method D (no examples in spec). Worth either (a) adding examples + re-running Method D, or (b) running K=3 Method C reruns to check stability.

**Priority 3** — Spot-check borderline Method D statements (D ∈ [6, 8] range): `avoid_errors`, `be_creative`, `be_kind`, `support_programmatic_use`, `transformation_exception` are above the corrected threshold but close to it.

