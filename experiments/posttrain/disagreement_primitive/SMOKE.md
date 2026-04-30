# Phase 1A Smoke Test — Statement Role Analyzer

**Run timestamp.** 2026-04-30 03:42 UTC
**Model.** `gemini-3-flash-preview` with `thinking_budget=0`, `temperature=0.2`, `response_mime_type="application/json"`.
**Statements.** First 5 of `experiments/posttrain/specs/openai_model_spec.jsonl` (alphabetically: `ask_clarifying_questions`, `assume_best_intentions`, `assume_objective_pov`, `avoid_abuse`, `avoid_being_condescending`).
**Wall time.** ~5 s for 5 stmts at 4 workers.
**Spend.** 1 call/stmt, ~2400 input tok + ~460 output tok ≈ 14k tok total. <$0.005 at gemini-3-flash-preview rates.

## What ran

`uv` is not the entry point in this worktree; `.venv/bin/python` is. The Gemini SDK was missing from this worktree's venv (`google-genai` not installed), so installed locally:

```
.venv/bin/pip install google-genai
# pulled google-genai 1.74.0, upgraded google-auth 2.47.0 → 2.49.2
```

Smoke command:

```
source .env && .venv/bin/python -m experiments.posttrain.backtest_statement_roles \
    --model gemini-3-flash-preview --limit 5 --max-workers 4 --max-retries 2
```

Outputs:
- `experiments/posttrain/disagreement_primitive/statement_analysis_gemini-3-flash-preview.jsonl` — 5 `StatementAnalysis` rows.
- `experiments/posttrain/disagreement_primitive/statement_analysis_gemini-3-flash-preview_audit.jsonl` — diag + verbatim-audit sidecar.

## Schema validation

5/5 rows parsed JSON cleanly on the first attempt. 5/5 schema-valid (all required keys present, `inferred_role` in allowed vocabulary, `role_confidence ∈ [0,1]`, list fields are lists). Zero retries hit.

## Verbatim audit (rationale_quotes)

**Aggregate.** 18/19 quotes (94.7%) appear character-for-character in the rendered statement corpus.

**The one failure.** `assume_objective_pov` quote 4:

```
"If the user asks the assistant to roleplay or assist with creative endeavors,
 the assistant should comply without attempting to impose an objective point of view."
```

The spec text contains markdown-link syntax: `to [roleplay](#present_perspectives) or assist with [creative endeavors](#be_creative)`. The model stripped the `[text](#anchor)` link wrapper to a clean string. This is a generic markdown-normalization pathology, not an analyzer bug.

**Two ways to fix in the full run.**
1. **Pre-render markdown links** in `render_statement_for_analyzer()` — convert `[text](#anchor)` → `text` once at corpus construction so model output matches strict audit.
2. **Tighten prompt** — explicitly forbid this normalization. Less reliable than (1).

Recommend (1) for the full run. Cheap, deterministic, and reflects the actual reading of the spec.

## Role distribution (5 smoke stmts)

```
guideline_like : 3   (ask_clarifying_questions, assume_objective_pov, avoid_abuse)
meta_rule      : 1   (assume_best_intentions)
style_rule     : 1   (avoid_being_condescending)
```

Non-degenerate. No `unclear` calls.

## Mini-backtest vs hidden `authority_level` (5 stmts)

| statement_id | hidden `authority_level` | `type` | `inferred_role` | conf | binary collapse match |
|---|---|---|---|:-:|:-:|
| ask_clarifying_questions | GUIDELINE | GUIDELINE | guideline_like | 0.90 | ✓ |
| assume_best_intentions | PLATFORM | REQUIREMENT | meta_rule | 0.90 | ✗ |
| assume_objective_pov | USER | REQUIREMENT | guideline_like | 0.85 | ✓ |
| avoid_abuse | USER | PROHIBITION | guideline_like | 0.90 | ✓ |
| avoid_being_condescending | GUIDELINE | GUIDELINE | style_rule | 0.85 | ✗ |

**Reading.** 3/5 match a strict binary collapse (PLATFORM → requirement_like, non-PLATFORM → guideline_like). The 2/5 "misses" are not obvious analyzer mistakes:

- `assume_best_intentions` is genuinely a meta-rule (text says how to interpret user prompts and developer instructions). The model picking `meta_rule` over `requirement_like` is a refinement of the binary collapse, not a wrong call. The Codex plan's role taxonomy explicitly admits `meta_rule` as a fifth category alongside the binary axis.
- `avoid_being_condescending` is a style/tone rule. Model picking `style_rule` is again a refinement, not a miss.

Both are exactly the "mistakes concentrated in genuinely ambiguous meta/style statements" pattern Codex flagged in Gate H1. This is a 5-row sample, so don't over-interpret — but the early signal is that the analyzer is producing semantically reasonable roles, including legitimately calling out meta- and style-rules as such.

For Gate H1 we'll need to define the right backtest scoring before computing 46-row numbers. Two options:
- **Strict binary backtest.** PLATFORM ↔ requirement_like; everything else ↔ guideline_like. `meta_rule` and `style_rule` count as misses on PLATFORM. Tougher, will likely show ~60-80% agreement.
- **Generous backtest.** PLATFORM ↔ {requirement_like, meta_rule}; non-PLATFORM ↔ {guideline_like, style_rule, meta_rule}. Will likely show ~85-95% agreement.

Plan to report both in the full backtest table and let Ahmed pick the gate criterion.

## Sample full record

```jsonc
{
  "statement_id": "ask_clarifying_questions",
  "summary": "The assistant should balance the efficiency of guessing user intent against the risks of making incorrect assumptions, generally attempting a response while stating assumptions unless the ambiguity is too high or the consequences of an error are severe.",
  "inferred_role": "guideline_like",
  "role_confidence": 0.9,
  "non_negotiables": [],
  "soft_preferences": [
    "articulating and/or confirming any assumptions",
    "weighing the costs of making the wrong assumption vs. asking for additional input",
    "taking a stab at fulfilling the request while mentioning missing information",
    "avoiding trivial questions that waste user cognitive bandwidth"
  ],
  "examples_used": [
    "Example of asking clarifying questions for a personal request",
    "ambiguous message from user, where the assistant should guess and state its assumptions",
    "ambiguous question that merits a clarifying question or comprehensive answer",
    "ambiguous task from developer; clarifying question avoided by default"
  ],
  "likely_tension_targets": [
    "conciseness_rules",
    "direct_answer_requirements",
    "task_completion_speed"
  ],
  "likely_supersedes": [
    "style_rules_regarding_brevity"
  ],
  "likely_subordinated_by": [
    "safety_rules_regarding_irreversible_actions",
    "legal_and_financial_accuracy_requirements"
  ],
  "rationale_quotes": [
    "the assistant should weigh the costs of making the wrong assumption vs. asking for additional input.",
    "Unless the cost of making the wrong assumption is too high or it's completely unclear what the user wants",
    "the assistant typically should take a stab at fulfilling the request and tell the user that it could be more helpful with certain information.",
    "Trivial questions may waste the user's time and cognitive bandwidth, and may be better if stated as an assumption that the user can correct."
  ],
  "analyzer_model": "gemini-3-flash-preview",
  "reasoning_setting": "thinking_budget=0",
  "temperature": 0.2
}
```

Note that `likely_tension_targets`, `likely_supersedes`, `likely_subordinated_by` are *conceptual descriptors* (per the prompt instruction), not statement_ids. Mapping descriptors → statement_ids is Phase 1B's job.

## Estimated cost for the authorized full run

If we proceed with all 46 statements × 3 analyzer models {GLM-5.1, GPT-5.1 reasoning_effort=none, Gemini 3 Flash}:

| model | per-stmt input tok | per-stmt output tok | per-stmt cost | 46-stmt cost |
|---|---:|---:|---:|---:|
| Gemini 3 Flash | ~2400 | ~460 | ~$0.001 | ~$0.05 |
| GPT-5.1 (no reasoning) | ~2400 | ~460 | ~$0.018 | ~$0.83 |
| GLM-5.1 (Together) | ~2400 | ~460 | ~$0.002 | ~$0.10 |

**Full 3-model batch budget: <$1.50.** Comfortably under the $5 cap from the plan.

If we add the optional high-thinking oracle-search ablation on a 5-stmt subset with Gemini 3 Pro (or Flash with `thinking_budget=128`), add another ~$0.50.

## Decision points for Ahmed

1. **Prompt template.** Sample record above + the system prompt in `backtest_statement_roles.py:55-90`. OK as-is, or revise before the full run? In particular:
   - Is the 5-label role taxonomy (requirement_like / guideline_like / meta_rule / style_rule / unclear) the right granularity, or should we collapse meta_rule + style_rule into the binary axis?
   - `likely_tension_targets` etc. are conceptual descriptors. Want them as descriptors or as guesses at concrete `statement_id`s?
2. **Schemas.** `experiments/posttrain/disagreement_primitive/schemas.py`. Field names track the Codex plan exactly. Anything to add/remove?
3. **Markdown-link normalization.** Apply fix (1) before the full run? (Recommend yes.)
4. **Authorize full run** on 46 stmts × {Gemini 3 Flash, GPT-5.1 no-reasoning, GLM-5.1}? Est <$1.50.
5. **Oracle-search ablation.** Run a 5-stmt subset with Gemini 3 Pro at `thinking_budget=128` as the explicit "high-thinking" condition? +$0.50.
6. **Backtest scoring.** Strict binary or both strict + generous? (Recommend both, surface in the gate report.)

## What's already on disk

```
experiments/posttrain/
├── backtest_statement_roles.py                        ← analyzer entry point
└── disagreement_primitive/
    ├── schemas.py                                     ← Phase 0 dataclasses
    ├── SMOKE.md                                       ← this file
    ├── statement_analysis_gemini-3-flash-preview.jsonl
    └── statement_analysis_gemini-3-flash-preview_audit.jsonl
```
