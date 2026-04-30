# Open-Weight Judge Faithfulness Review

## Goal

Produce a per-statement qualitative comparison between `gpt-5.1` and four
open-weight / Together-hosted LM-as-a-judge models:

- `zai-org/GLM-5.1` (`glm51`)
- `zai-org/GLM-5` (`glm5`)
- `MiniMaxAI/MiniMax-M2.5` (`mm25`)
- `Qwen/Qwen3-235B-A22B-Instruct-2507-tput` (`qwen235`)

The point is not just correlation. Each comparison should check whether the
model's scores and explanations are faithful to:

1. the OpenAI Model Spec statement text,
2. the per-item evaluation rubric,
3. the prompt / response pair being judged.

## Output Contract

Write exactly one markdown file per `(statement, model)` pair at:

`codex_subagents/{SPEC_STATEMENT}/{MODEL_X}_gpt5.md`

Examples:

- `codex_subagents/ask_clarifying_questions/glm51_gpt5.md`
- `codex_subagents/refusal_style/mm25_gpt5.md`

Expected total: `46 statements * 4 models = 184 markdown files`.

## Data Sources

### Statement inventory and canonical spec snapshot

- `experiments/posttrain/specs/openai_model_spec.jsonl`
- official public entry points:
  - `https://model-spec.openai.com/`
  - `https://openai.com/index/sharing-the-latest-model-spec/`

Use the local JSONL as the exact statement snapshot for this evaluation. Use the
official URLs as public pointers to the Model Spec.

### Judge result inputs

All judge results are in:

- `~/judge_correlations/inputs/gpt51/{target}/judged_results.jsonl`
- `~/judge_correlations/inputs/glm51/{target}/judged_results.jsonl`
- `~/judge_correlations/inputs/glm5/{target}/judged_results.jsonl`
- `~/judge_correlations/inputs/mm25/{target}/judged_results.jsonl`
- `~/judge_correlations/inputs/qwen235/{target}/judged_results.jsonl`

Targets available for all five judges:

- `sft`
- `full_dpo_beta01_b64_step1699`
- `lora_lr1e5_b64_step1699`
- `lora_lr5e6_b64_step1699`
- `gpt41_target`

### Prompt / rubric / source-question context

Use Bloom prompt metadata from:

- `~/judge_correlations/inputs/bloom_judge/gpt-4.1-target/inference/run_20260324_122545_518a0f141314/indexes/prompt_catalog.jsonl`

Relevant fields:

- `statement_id`
- `question_id`
- `question`
- `source_rubric`
- `source_system_prompt`
- `source_tags`
- `source_axis_config`

## Analysis Requirements

For each `(statement, model)` markdown file:

1. State the statement ID, section, subsection, type, and authority level.
2. Quote or paraphrase the statement text from
   `openai_model_spec.jsonl`.
3. Mention the public Model Spec pointer(s).
4. Compare `gpt-5.1` vs the target model using the full available target set.
5. Use actual judged examples, not just aggregate Spearman.
6. Inspect both scores and explanations / reasoning.
7. Decide which judge is more faithful to the rubric / prompt on this
   statement, or say it is mixed / unclear.
8. Call out recurring failure modes:
   - over-refusal
   - under-refusal
   - rubric literalism vs rubric neglect
   - missing factual checks
   - tone/style over-weighting
   - explanation quality drift
   - calibration mismatch that still preserves ordering
9. Include enough concrete evidence that a reader can audit the conclusion.

## Recommended Per-File Structure

```md
# {statement_id}: {model_label} vs gpt-5.1

## Statement
- Type / authority / section / subsection
- Model Spec pointer
- Statement text

## Data Reviewed
- Targets covered
- Number of paired items
- Any dropped / missing items

## Quantitative Snapshot
- Mean / median score comparison if useful
- Notable agreement / disagreement patterns

## Faithfulness Review
- Cases where gpt-5.1 is more faithful
- Cases where {model_label} is more faithful
- Whether disagreements are rubric-faithful or just calibration differences

## Representative Items
- 3-8 concrete prompt/response pairs with score + explanation deltas

## Verdict
- `gpt-5.1 better`
- `{model_label} better`
- `mixed`

## Confidence
- short note on confidence and residual ambiguity
```

## Quality Bar

- Do not write generic summaries from correlation alone.
- Prefer representative high-signal examples over exhaustive dumps.
- Be explicit when low correlation is mostly ceiling compression rather than
  genuine construct mismatch.
- If a statement appears structurally broken or underspecified, say so.
