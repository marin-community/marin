# Round-2 (cumulative r1+r2) propagation run report

Each judge ran rubric regeneration against a spec with BOTH its round-1
AND round-2 proposed edits applied cumulatively. Tests convergence of
the edit-and-regen loop.

## Results

| judge | r1+r2 edits | rows | schema_ok | elapsed |
|---|---:|---:|---|---:|
| glm51 | 13 | 22 | 22/22 | 375.2s |

## Output files

- **glm51**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_r1r2_edits.jsonl`
  - forked spec: `experiments/posttrain/specs/openai_model_spec_glm51_r1r2_edits.jsonl`

## Next step

Compare each judge's `with_r1r2_edits` rubrics to:
- baseline (no edits): cumulative effect of 2 rounds
- `with_self_edits` (round-1 only): marginal effect of round-2