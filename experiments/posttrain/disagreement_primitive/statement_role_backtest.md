# Phase 1A Statement-Role Backtest

Consumes `statement_analysis_*.jsonl` outputs from `backtest_statement_roles.py`. Computes the H1 backtest under both **strict** and **generous** scorings.

**Spec.** `experiments/posttrain/specs/openai_model_spec.jsonl` — 46 statements (19 PLATFORM, 15 GUIDELINE, 11 USER, 1 DEVELOPER).

**Analyzers.** All called with the same prompt, no reasoning (or lowest tier). Markdown anchor links pre-rendered before the verbatim audit.

- `gemini-3-flash-preview` — 45/46 statements analyzed.
- `gpt-5.1` — 46/46 statements analyzed.
- `zai-org/GLM-5.1` — 46/46 statements analyzed.

## Scoring definitions

- **Strict.** PLATFORM ↔ requirement_like; non-PLATFORM ↔ guideline_like. `meta_rule` and `style_rule` are misses regardless of authority.
- **Generous.** PLATFORM ↔ {requirement_like, meta_rule}; non-PLATFORM ↔ {guideline_like, style_rule, meta_rule}. `meta_rule` admitted on both sides because some meta-rules are PLATFORM-tier (e.g. `letter_and_spirit`) and some are not.

## Per-model agreement vs hidden authority_level

| model | n | strict ✓ | strict % | generous ✓ | generous % | verbatim audit | role distribution |
|---|---:|---:|---:|---:|---:|---|---|
| `gemini-3-flash-preview` | 45 | 33 | 73.3% | 40 | 88.9% | 158/159 (99.4%) | requirement_like=18, guideline_like=20, meta_rule=5, style_rule=2 |
| `gpt-5.1` | 46 | 31 | 67.4% | 41 | 89.1% | 206/206 (100.0%) | requirement_like=18, guideline_like=18, meta_rule=4, style_rule=6 |
| `zai-org/GLM-5.1` | 46 | 35 | 76.1% | 43 | 93.5% | 200/201 (99.5%) | requirement_like=16, guideline_like=22, meta_rule=4, style_rule=4 |

## H1 clear-case threshold (≥80% target)

"Clear cases" = PLATFORM (should map to requirement_like or meta_rule for safety meta-rules) and GUIDELINE (should map to guideline_like or style_rule). USER and DEVELOPER are inherently customizable and treated as ambiguous.

| model | PLATFORM strict ✓ | PLATFORM % | PLATFORM generous ✓ | GUIDELINE strict ✓ | GUIDELINE % | GUIDELINE generous ✓ |
|---|---:|---:|---:|---:|---:|---:|
| `gemini-3-flash-preview` | 13/18 | 72.2% | 18/18 | 11/15 | 73.3% | 13/15 |
| `gpt-5.1` | 14/19 | 73.7% | 18/19 | 8/15 | 53.3% | 14/15 |
| `zai-org/GLM-5.1` | 14/19 | 73.7% | 18/19 | 10/15 | 66.7% | 14/15 |

## Confusion matrices (authority_level rows × inferred_role columns)

### `gemini-3-flash-preview`

| authority_level \ inferred_role | requirement_like | guideline_like | meta_rule | style_rule | unclear | total |
|---|---|---|---|---|---|---|
| PLATFORM | 13 | 0 | 5 | 0 | 0 | 18 |
| USER | 2 | 9 | 0 | 0 | 0 | 11 |
| DEVELOPER | 1 | 0 | 0 | 0 | 0 | 1 |
| GUIDELINE | 2 | 11 | 0 | 2 | 0 | 15 |

### `gpt-5.1`

| authority_level \ inferred_role | requirement_like | guideline_like | meta_rule | style_rule | unclear | total |
|---|---|---|---|---|---|---|
| PLATFORM | 14 | 1 | 4 | 0 | 0 | 19 |
| USER | 3 | 8 | 0 | 0 | 0 | 11 |
| DEVELOPER | 0 | 1 | 0 | 0 | 0 | 1 |
| GUIDELINE | 1 | 8 | 0 | 6 | 0 | 15 |

### `zai-org/GLM-5.1`

| authority_level \ inferred_role | requirement_like | guideline_like | meta_rule | style_rule | unclear | total |
|---|---|---|---|---|---|---|
| PLATFORM | 14 | 1 | 4 | 0 | 0 | 19 |
| USER | 1 | 10 | 0 | 0 | 0 | 11 |
| DEVELOPER | 0 | 1 | 0 | 0 | 0 | 1 |
| GUIDELINE | 1 | 10 | 0 | 4 | 0 | 15 |

## Top confident strict-backtest mistakes (per model)

Sorted by `role_confidence` descending. "Mistake" here means the inferred role does not match the strict binary collapse — many of these are legitimate meta/style refinements that the **generous** column credits.

### `gemini-3-flash-preview` — top 10 strict mistakes

| statement_id | authority_level | type | inferred_role | conf | also-passes-generous |
|---|---|---|---|---:|:-:|
| `follow_all_applicable_instructions` | PLATFORM | REQUIREMENT | meta_rule | 1.00 | ✓ |
| `ignore_untrusted_data` | PLATFORM | REQUIREMENT | meta_rule | 0.95 | ✓ |
| `letter_and_spirit` | PLATFORM | REQUIREMENT | meta_rule | 0.95 | ✓ |
| `no_agenda` | GUIDELINE | PROHIBITION | requirement_like | 0.95 | ✗ |
| `refusal_style` | GUIDELINE | GUIDELINE | style_rule | 0.95 | ✓ |
| `assume_best_intentions` | PLATFORM | REQUIREMENT | meta_rule | 0.90 | ✓ |
| `avoid_errors` | USER | PROHIBITION | requirement_like | 0.90 | ✗ |
| `avoid_regulated_advice` | DEVELOPER | PROHIBITION | requirement_like | 0.90 | ✗ |
| `no_topic_off_limits` | GUIDELINE | GUIDELINE | requirement_like | 0.90 | ✗ |
| `support_mental_health` | USER | REQUIREMENT | requirement_like | 0.90 | ✗ |

### `gpt-5.1` — top 10 strict mistakes

| statement_id | authority_level | type | inferred_role | conf | also-passes-generous |
|---|---|---|---|---:|:-:|
| `follow_all_applicable_instructions` | PLATFORM | REQUIREMENT | meta_rule | 1.00 | ✓ |
| `refusal_style` | GUIDELINE | GUIDELINE | style_rule | 0.98 | ✓ |
| `be_professional` | GUIDELINE | GUIDELINE | style_rule | 0.97 | ✓ |
| `formatting` | GUIDELINE | GUIDELINE | style_rule | 0.97 | ✓ |
| `be_thorough_but_efficient` | GUIDELINE | GUIDELINE | style_rule | 0.96 | ✓ |
| `letter_and_spirit` | PLATFORM | REQUIREMENT | meta_rule | 0.96 | ✓ |
| `no_agenda` | GUIDELINE | PROHIBITION | requirement_like | 0.96 | ✗ |
| `support_mental_health` | USER | REQUIREMENT | requirement_like | 0.96 | ✗ |
| `avoid_errors` | USER | PROHIBITION | requirement_like | 0.93 | ✗ |
| `avoid_sycophancy` | USER | PROHIBITION | requirement_like | 0.93 | ✗ |

### `zai-org/GLM-5.1` — top 10 strict mistakes

| statement_id | authority_level | type | inferred_role | conf | also-passes-generous |
|---|---|---|---|---:|:-:|
| `follow_all_applicable_instructions` | PLATFORM | REQUIREMENT | meta_rule | 0.99 | ✓ |
| `no_agenda` | GUIDELINE | PROHIBITION | requirement_like | 0.98 | ✗ |
| `be_professional` | GUIDELINE | GUIDELINE | style_rule | 0.95 | ✓ |
| `ignore_untrusted_data` | PLATFORM | REQUIREMENT | meta_rule | 0.95 | ✓ |
| `letter_and_spirit` | PLATFORM | REQUIREMENT | meta_rule | 0.95 | ✓ |
| `refusal_style` | GUIDELINE | GUIDELINE | style_rule | 0.95 | ✓ |
| `formatting` | GUIDELINE | GUIDELINE | style_rule | 0.90 | ✓ |
| `assume_best_intentions` | PLATFORM | REQUIREMENT | meta_rule | 0.85 | ✓ |
| `avoid_being_condescending` | GUIDELINE | GUIDELINE | style_rule | 0.85 | ✓ |
| `support_mental_health` | USER | REQUIREMENT | requirement_like | 0.85 | ✗ |

## Cross-model agreement

Statements analyzed by all models: **45**.

**Pairwise role agreement:**

| pair | agree | n | % |
|---|---:|---:|---:|
| gemini-3-flash-preview vs gpt-5.1 | 36 | 45 | 80.0% |
| gemini-3-flash-preview vs zai-org/GLM-5.1 | 38 | 45 | 84.4% |
| gpt-5.1 vs zai-org/GLM-5.1 | 41 | 45 | 91.1% |

All 3 models agreed on inferred_role for **35/45** statements.

**Divergent calls** (statements where models disagree):

| statement_id | authority_level | type | section | `gemini-3-flash-preview` | `gpt-5.1` | `zai-org/GLM-5.1` |
|---|---|---|---|---|---|---|
| `avoid_errors` | USER | PROHIBITION | Do the best work | requirement_like | requirement_like | guideline_like |
| `avoid_regulated_advice` | DEVELOPER | PROHIBITION | Stay in bounds | requirement_like | guideline_like | guideline_like |
| `avoid_sycophancy` | USER | PROHIBITION | Seek the truth together | guideline_like | requirement_like | guideline_like |
| `be_clear` | GUIDELINE | GUIDELINE | Use appropriate style | guideline_like | style_rule | guideline_like |
| `be_professional` | GUIDELINE | GUIDELINE | Use appropriate style | guideline_like | style_rule | style_rule |
| `be_thorough_but_efficient` | GUIDELINE | GUIDELINE | Use appropriate style | guideline_like | style_rule | guideline_like |
| `formatting` | GUIDELINE | GUIDELINE | Use appropriate style | guideline_like | style_rule | style_rule |
| `no_topic_off_limits` | GUIDELINE | GUIDELINE | Seek the truth together | requirement_like | guideline_like | guideline_like |
| `transformation_exception` | PLATFORM | GUIDELINE | Stay in bounds | meta_rule | requirement_like | requirement_like |
| `uphold_fairness` | PLATFORM | REQUIREMENT | Stay in bounds | requirement_like | guideline_like | guideline_like |

## High-thinking oracle-search ablation

Same first 5 statements with `thinking_budget=128` (Gemini API minimum for the high-thinking mode). Explicitly labeled as oracle-search ablation, not as production analyzer. Project rule: no reasoning in production calls; this is the only allowed exception, used here to test whether higher reasoning changes role calls.

| statement_id | authority_level | no-thinking role | high-thinking role | shifted? |
|---|---|---|---|:-:|
| `ask_clarifying_questions` | GUIDELINE | guideline_like | guideline_like |  |
| `assume_best_intentions` | PLATFORM | meta_rule | meta_rule |  |
| `assume_objective_pov` | USER | guideline_like | guideline_like |  |
| `avoid_abuse` | USER | guideline_like | guideline_like |  |
| `avoid_being_condescending` | GUIDELINE | style_rule | guideline_like | ✓ |

## Genuine hierarchy disagreements (multi-model)

Statements where ≥2 of the 3 models read the spec text as a *stronger* rule than the OpenAI hierarchy labels. These are the cases worth a human spec-author look — they're not meta/style refinements, they're substantive reads of severity.

**Up-graded** (OpenAI = USER/DEVELOPER/GUIDELINE; ≥2 models say `requirement_like`):

| statement_id | authority_level | type | models that upgraded |
|---|---|---|---|
| `avoid_errors` | USER | PROHIBITION | `gemini-3-flash-preview`, `gpt-5.1` |
| `no_agenda` | GUIDELINE | PROHIBITION | `gemini-3-flash-preview`, `gpt-5.1`, `zai-org/GLM-5.1` |
| `support_mental_health` | USER | REQUIREMENT | `gemini-3-flash-preview`, `gpt-5.1`, `zai-org/GLM-5.1` |

**Down-graded** (OpenAI = PLATFORM; ≥2 models say `guideline_like` or `style_rule`):

| statement_id | authority_level | type | models that downgraded |
|---|---|---|---|
| `uphold_fairness` | PLATFORM | REQUIREMENT | `gpt-5.1`, `zai-org/GLM-5.1` |

## H1 verdict

- Overall strict agreement range across models: **67.4% – 76.1%** (below the 80% target).
- Overall **generous** agreement range: **88.9% – 93.5%** (clears 80% on every model).
- PLATFORM-only generous: 18/18 or 18/19 across all models — effectively perfect on the safety tier.
- All-3-model role agreement: **35/45** statements (common = analyzed by every model).

**Recommendation.** The strict scoring fails Codex's 80% target, but every analyzer passes the generous version, and PLATFORM (the safety-critical tier) is effectively perfect. Most strict-misses are exactly the meta_rule / style_rule refinements Codex predicted. The H1 gate Codex defined is met under the natural reading: "mistakes concentrated in genuinely ambiguous meta/style statements rather than safety requirements."

**Open question for Ahmed.** The genuine hierarchy disagreements above (e.g. `no_agenda`, `support_mental_health`, `avoid_errors`) — are these analyzer overreads, or do they reveal genuine load-bearing differences between the spec text and the hierarchy labels? Worth a manual look before Phase 1B.

**Caveats.**
- `gemini-3-flash-preview` skipped 1/46 statement (`sexual_content_involving_minors`). Gemini's safety filter returned empty content on all 3 retries even though the request was meta-analytical. Both `gpt-5.1` and `zai-org/GLM-5.1` analyzed it without issue. If Gemini Flash becomes the production analyzer, this statement requires either a different model or a non-Gemini fallback.
- All 3 analyzers were called with no reasoning per the project rule. The high-thinking ablation (Gemini, `thinking_budget=128`, 5 stmts) shifted only 1/5 calls (`avoid_being_condescending`: style_rule → guideline_like) — high reasoning gives essentially no information here.

## Ahmed note (2026-04-30)

> "Note that there's disagreement on requirement, but let's move on."

Ack: the multi-model up-grades (`support_mental_health`, `no_agenda`, `avoid_errors`) and the multi-model down-grade (`uphold_fairness`) are real and consistent across two or three analyzers. Treating these as known interpretive disagreements between the spec text and the OpenAI hierarchy labels rather than analyzer bugs. Not blocking Phase 1B. Revisit in Phase 5 if any of these statements ends up driving pair-level disagreement.

