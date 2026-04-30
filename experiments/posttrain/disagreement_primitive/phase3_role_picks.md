# Phase 3 — Role picks

Locked in 2026-04-30 by Ahmed (post-H1, post-H2 review). Per the Codex
plan, Phase 3 is the formal "choose generator, judge, and compiler
roles" gate; this file documents the working defaults the rest of the
pipeline runs against. All choices honor the project-wide
no-reasoning rule.

## Compiler — single model

**`gpt-5.1` with `reasoning_effort="none"`** (OpenAI Chat Completions API).

- Default for: statement role analysis, pair classification, scenario
  generation, edit proposals (Phase 5).
- Settings: `temperature=0.2`, `response_format={"type":"json_object"}`,
  `max_completion_tokens=3000–4000`.
- Backup / second opinion: **`zai-org/GLM-5.1`** via Together
  (OpenAI-compat). Use only when a specific call is load-bearing and
  cross-compiler agreement is the desired robustness check. Together
  rate-limits Together to ~1 call/sec serial-equivalent; not
  workable as a primary driver.
- **Not** Gemini Flash as primary — its safety filter blocks legitimate
  meta-analysis on sensitive statements (`sexual_content_involving_minors`
  was unanalyzable in Phase 1A).

Project memory: [`project_lm_compiler_is_gpt51.md`](file:///Users/ahmed/.claude/projects/-Users-ahmed-code-marin/memory/project_lm_compiler_is_gpt51.md).

## Judges — REQUIRED ensemble of 3

Every judge call (Phase 4 zero-shot eval, Phase 5 refinement loop,
Phase 6 calibration probe) goes through **all three** judges in
parallel:

1. **`gpt-5.1`** with `reasoning_effort="none"`
2. **`zai-org/GLM-5.1`** (no reasoning toggle)
3. **`gemini-3-flash-preview`** with `thinking_budget=0`

- Same system prompt across all three judges.
- Compute Fleiss κ on (compliance score, controlling-statement) pairs.
- Surface judge disagreement as the *primary signal*, not as noise to be
  averaged. Compliance disagreement ⇒ spec ambiguity (label #3).
  Activation disagreement ⇒ priority / cross-tension repair needed.
- Single-judge fallback is **not allowed**, regardless of cost. If a
  judge becomes systematically noisy, demote to analysis-only — keep
  the ensemble shape.

Project memory: [`project_judge_ensemble_required.md`](file:///Users/ahmed/.claude/projects/-Users-ahmed-code-marin/memory/project_judge_ensemble_required.md).

## Generators — three strong oracles for Phase 4

Per Codex Phase 3A, the generator panel for the zero-shot
disagreement-primitive eval (Phase 4):

1. **`gpt-5.1`** with `reasoning_effort="none"` (strong, fast, cheap)
2. **`zai-org/GLM-5.1`** (strong open-weight surrogate; different
   training pedigree)
3. **`gemini-3-flash-preview`** with `thinking_budget=0` (third
   independent family)

- Optional weak/target generator for Phase 4's #1 ablation (model
  behavior / training failures): the SFT or M2/M3 trained model.
  Used **only** for the explicit "training_issue" ablation, not as
  part of the strong-oracle panel.
- High-thinking oracle-search ablation (`gemini-3-flash-preview` with
  `thinking_budget=128`) is allowed on a small subset, explicitly
  labeled as oracle-search rather than as a production generator.

## Bookkeeping rules across all roles

- Always record `reasoning_setting` and `temperature` on each output
  record.
- Never silently retry into a different model — if a model fails after
  retries, log the error and propagate.
- Reasoning tokens must be 0 on every production call. Audits should
  check this from `usage.completion_tokens_details.reasoning_tokens`
  for OpenAI calls.

## Cost expectations

- Phase 2 (scenario generation): 65 pairs × 1 call ≈ **$0.50** (done).
- Phase 4 first cut (65 pairs × 3 scenarios × 3 generators × 3 judges):
  - 195 generator calls × 3 = **585 generator calls**
  - 195 × 3 generators × 3 judges = **1,755 judge calls**
  - GPT-5.1 + GLM-5.1 + Gemini Flash mix ≈ **$30–80** total
- Phase 5 pilot (25 pairs × bounded loop, max 3 iterations): ~$5–15.
- Phase 6 calibration UI: code-only.
- Phase 7 scale-up + Demo A training: separate budget conversation.

## Signed off

- Compiler choice: 2026-04-30 ~04:43 UTC
- Judge ensemble requirement: 2026-04-30 ~05:00 UTC
- Generator panel: 2026-04-30 ~05:13 UTC (default; Phase 4 pre-spend
  Gate H3 confirmation pending)
