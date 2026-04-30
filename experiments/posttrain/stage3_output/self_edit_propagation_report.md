# Self-edit propagation run report

Each LM judge re-ran cross-tier rubric generation on a spec forked
with that judge's own proposed edits applied. Each judge sees only
its own forked spec (no cross-judge contamination).

## Per-judge results

| judge | edits applied | forked spec | output | rows | schema_ok | elapsed |
|---|---:|---|---|---:|---|---:|
| flash | 8 | openai_model_spec_flash_self_edits.jsonl | cross_tier_rubrics_v2_flash_with_self_edits.jsonl | 22 | 22/22 | 40.0s |
| gpt51 | 8 | openai_model_spec_gpt51_self_edits.jsonl | cross_tier_rubrics_v2_gpt51_with_self_edits.jsonl | 22 | 22/22 | 97.1s |
| pro | 7 | openai_model_spec_pro_self_edits.jsonl | cross_tier_rubrics_v2_pro_with_self_edits.jsonl | 22 | 22/22 | 192.7s |
| glm51 | 6 | openai_model_spec_glm51_self_edits.jsonl | cross_tier_rubrics_v2_glm51_with_self_edits.jsonl | 22 | 22/22 | 237.6s |

## Source edits

- **flash**: 8 edits from `experiments/posttrain/lm_judge_edits/flash/proposed_edits/`
- **gpt51**: 8 edits from `experiments/posttrain/lm_judge_edits/gpt51/proposed_edits/`
- **pro**: 7 edits from `experiments/posttrain/lm_judge_edits/pro/proposed_edits/`
- **glm51**: 6 edits from `experiments/posttrain/lm_judge_edits/glm51/proposed_edits/`

## Output rubric files

- **flash**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_flash_with_self_edits.jsonl`
- **gpt51**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_gpt51_with_self_edits.jsonl`
- **pro**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_pro_with_self_edits.jsonl`
- **glm51**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl`

## Forked specs

- **flash**: `experiments/posttrain/specs/openai_model_spec_flash_self_edits.jsonl`
- **gpt51**: `experiments/posttrain/specs/openai_model_spec_gpt51_self_edits.jsonl`
- **pro**: `experiments/posttrain/specs/openai_model_spec_pro_self_edits.jsonl`
- **glm51**: `experiments/posttrain/specs/openai_model_spec_glm51_self_edits.jsonl`

## Next step

Compare each judge's `with_self_edits` rubrics against the original
`cross_tier_rubrics_v2_<judge>.jsonl` to measure whether the proposed
edits actually moved the rubrics in the predicted direction.
