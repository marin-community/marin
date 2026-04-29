# Rubric set comparison: v2_baseline vs v3_alwayson

**Baseline**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51.jsonl` (22 rows)
**Treatment**: `experiments/posttrain/stage3_output/cross_tier_rubrics_v3_alwayson_glm51.jsonl` (22 rows)
**Common pairs**: 22

## Per-field aggregate change

Mean text change (1 - SequenceMatcher.ratio) across all common pairs. Higher = more changed. Threshold for 'significant change' = 0.3.

| field | mean Δ | median Δ | max Δ | n significant (>0.3) |
|---|---:|---:|---:|---:|
| dominant_rubric.GOOD | 0.777 | 0.830 | 0.965 | 22/22 |
| dominant_rubric.BAD | 0.843 | 0.859 | 0.946 | 22/22 |
| dominant_rubric.KEY_TENSION | 0.927 | 0.945 | 0.987 | 22/22 |
| non_leakage_rubric.GOOD | 0.883 | 0.889 | 0.959 | 22/22 |
| non_leakage_rubric.BAD | 0.776 | 0.789 | 0.877 | 22/22 |
| non_leakage_rubric.KEY_TENSION | 0.889 | 0.884 | 0.982 | 22/22 |
| worked_example.spec_compliant | 0.795 | 0.842 | 0.977 | 22/22 |
| worked_example.subordinate_over_satisfaction | 0.823 | 0.843 | 0.985 | 22/22 |
| worked_example.pathological_refusal | 0.659 | 0.657 | 0.962 | 22/22 |
| rationale.interpretive_choices_made | 0.875 | 0.916 | 0.965 | 22/22 |
| rationale.alternative_readings_rejected | 0.885 | 0.909 | 0.986 | 22/22 |

## Rationale clause changes

- Avg `spec_clauses_anchored_on` count in v2_baseline: 4.3
- Avg `spec_clauses_anchored_on` count in v3_alwayson: 4.9
- Avg new clauses (in treatment, not baseline): 2.8
- Avg removed clauses (in baseline, not treatment): 2.2

## Per-pair changes

| pair | dom.BAD Δ | non_leak.BAD Δ | spec_compliant Δ | alt_read Δ | clauses Δ |
|---|---:|---:|---:|---:|---:|
| `assume_objective_pov__do_not_encourage_self_harm tp=0` | 0.88 | 0.82 | 0.88 | 0.96 | +0 |
| `assume_objective_pov__do_not_encourage_self_harm tp=3` | 0.93 | 0.81 | 0.63 | 0.93 | +1 |
| `assume_objective_pov__do_not_facilitate_illicit_behavior tp=5` | 0.90 | 0.80 | 0.84 | 0.89 | +0 |
| `avoid_abuse__letter_and_spirit tp=2` | 0.90 | 0.88 | 0.92 | 0.83 | +1 |
| `avoid_being_condescending__letter_and_spirit tp=5` | 0.82 | 0.83 | 0.91 | 0.83 | +0 |
| `avoid_errors__avoid_info_hazards tp=2` | 0.84 | 0.80 | 0.90 | 0.94 | +1 |
| `avoid_errors__letter_and_spirit tp=4` | 0.95 | 0.79 | 0.76 | 0.94 | +1 |
| `avoid_hateful_content__be_rationally_optimistic tp=5` | 0.86 | 0.71 | 0.71 | 0.72 | -1 |
| `avoid_hateful_content__present_perspectives tp=3` | 0.93 | 0.75 | 0.69 | 0.96 | +2 |
| `avoid_targeted_political_manipulation__be_kind tp=0` | 0.87 | 0.81 | 0.53 | 0.91 | -1 |
| `avoid_targeted_political_manipulation__do_not_lie tp=0` | 0.69 | 0.72 | 0.90 | 0.99 | +0 |
| `be_kind__uphold_fairness tp=1` | 0.92 | 0.69 | 0.81 | 0.66 | +0 |
| `comply_with_laws__no_agenda tp=5` | 0.65 | 0.78 | 0.90 | 0.97 | +2 |
| `comply_with_laws__no_agenda tp=6` | 0.89 | 0.84 | 0.77 | 0.89 | +3 |
| `do_not_facilitate_illicit_behavior__no_agenda tp=0` | 0.83 | 0.83 | 0.94 | 0.89 | +1 |
| `express_uncertainty__prevent_imminent_harm tp=5` | 0.85 | 0.77 | 0.98 | 0.91 | +1 |
| `formatting__letter_and_spirit tp=4` | 0.82 | 0.64 | 0.89 | 0.88 | -1 |
| `no_agenda__respect_creators tp=3` | 0.77 | 0.75 | 0.81 | 0.78 | +0 |
| `no_agenda__respect_creators tp=4` | 0.78 | 0.80 | 0.72 | 0.73 | +1 |
| `no_topic_off_limits__respect_creators tp=4` | 0.86 | 0.71 | 0.88 | 0.95 | +1 |
| `prevent_imminent_harm__support_programmatic_use tp=1` | 0.94 | 0.73 | 0.54 | 0.97 | +0 |
| `protect_privacy__support_mental_health tp=4` | 0.67 | 0.79 | 0.57 | 0.94 | +1 |